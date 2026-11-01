"""Materialize the parquet ImageNet into an ImageFolder tree (dataloaders/ImageNet.py).

Cluster-aware: paths come from ``utils.common_libs`` so the same command works
on Delta (shared /work/hdd/bibd) and on the vision cluster, where both the
source parquet and the extracted tree live on node-local scratch. On vision the
output is therefore per-node -- run this once on each node you plan to train on
(vision-c21 / c22 / c23).

Images are decoded, converted to RGB and re-encoded at quality=95 at their
original resolution, so the tree matches the one already extracted on Delta
byte-for-byte. Note that this inflates the ~q85-90 ImageNet JPEGs by roughly
1.5-2x; budget ~200 GB rather than the parquet's ~145 GB. Set
EXTRACT_COPY_BYTES=1 to instead copy the original JPEG bytes through untouched,
which is lossless, ~30% smaller and much faster, but produces different (better)
pixels than the Delta tree -- do not mix the two in one dataset.

The parquet shards are read directly with pyarrow rather than through
``load_dataset``, which would first materialize a second full-size copy of the
dataset as Arrow under ``processed_cache`` -- ~144 GB that this script never
needs and that will not fit alongside the output on vision-c21.

Overrides:
    EXTRACT_DATA_DIR      source dir holding data/*.parquet
    EXTRACT_OUTPUT_ROOT   destination for the <split>/<class>/<idx>.jpg tree
    EXTRACT_NUM_PROC      worker processes (default: min(shards, 16))
    EXTRACT_JPEG_QUALITY  PIL JPEG quality (default 95, matching Delta)
    EXTRACT_COPY_BYTES    set to 1 to copy original bytes instead of re-encoding
    EXTRACT_SKIP_PREFLIGHT set to 1 to bypass the disk-space check
"""

import io
import os
import shutil
from glob import glob
from multiprocessing import Pool

import pyarrow.parquet as pq
from datasets.features import Features
from PIL import Image

from utils.common_libs import get_data_dir, get_hostname, get_raw_data_dir

# Refuse to start unless the projected output leaves this much headroom, since
# the scratch filesystems are shared with other users' jobs.
FREE_SPACE_MARGIN_GB = 20
# Images sampled to project the on-disk cost of re-encoding before the real run.
PREFLIGHT_SAMPLES = 200


def _num_proc(num_shards):
    override = os.environ.get('EXTRACT_NUM_PROC') or os.environ.get('SLURM_CPUS_PER_TASK')
    if override:
        return max(1, min(int(override), num_shards))
    # Condor here does not pin the job to its requested cores, so affinity
    # reports every core on the machine (96 on vision-c21). This workload is
    # pure IO, so more writers past a point only contend on the disk and crowd
    # out other users' jobs.
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(available, 16, num_shards))


def encode(image_bytes):
    """Produce the bytes to write for one source image.

    The re-encode path mirrors what the original load_dataset version of this
    script did (PIL decode -> convert('RGB') -> save at quality=95), so the
    output matches the tree already extracted on Delta.
    """
    if COPY_BYTES:
        return image_bytes
    buffer = io.BytesIO()
    Image.open(io.BytesIO(image_bytes)).convert('RGB').save(
        buffer, format='JPEG', quality=JPEG_QUALITY
    )
    return buffer.getvalue()


DATA_DIR = os.environ.get('EXTRACT_DATA_DIR') or get_data_dir()
OUTPUT_ROOT = os.environ.get('EXTRACT_OUTPUT_ROOT') or get_raw_data_dir()
JPEG_QUALITY = int(os.environ.get('EXTRACT_JPEG_QUALITY', '95'))
COPY_BYTES = os.environ.get('EXTRACT_COPY_BYTES') == '1'

if not OUTPUT_ROOT:
    raise ValueError(
        f'No extracted-ImageNet directory is defined for host {get_hostname()}. '
        'Set EXTRACT_OUTPUT_ROOT, or add this host to utils.common_libs.get_raw_data_dir.'
    )

if os.path.realpath(OUTPUT_ROOT) == os.path.realpath(DATA_DIR):
    # On Delta the extracted tree replaced the parquet dir, so the two resolve
    # to the same path; writing there would nest a second copy inside itself.
    raise ValueError(
        f'Output root {OUTPUT_ROOT} is the same directory as the parquet source. '
        'The extraction has likely already been done -- set EXTRACT_OUTPUT_ROOT '
        'if you really want to rebuild it somewhere else.'
    )


def shard_paths(split_name):
    paths = sorted(glob(os.path.join(DATA_DIR, 'data', f'{split_name}*.parquet')))
    if not paths:
        raise FileNotFoundError(f'No {split_name}*.parquet shards under {DATA_DIR}/data')
    return paths


def label_names(shard_path):
    """Class names from the HuggingFace feature metadata carried in the schema.

    ``load_dataset`` would surface these as ``features['label'].names``; reading
    the parquet directly means decoding the same metadata by hand.
    """
    features = Features.from_arrow_schema(pq.ParquetFile(shard_path).schema_arrow)
    return features['label'].names


def shard_offsets(paths):
    """Row index each shard starts at, so filenames match the whole-split index.

    Only parquet footers are read here, not the data. Keeping the global index
    as the filename means a tree written by the previous (load_dataset) version
    of this script can still be resumed rather than duplicated.
    """
    offsets, total = [], 0
    for path in paths:
        offsets.append(total)
        total += pq.ParquetFile(path).metadata.num_rows
    return offsets, total


TRAIN_SHARDS = shard_paths('train')
CLASS_NAMES = label_names(TRAIN_SHARDS[0])


def export_shard(job):
    """Write every row of one parquet shard. Runs in a worker process."""
    path, split_name, start_idx = job
    written = 0
    parquet_file = pq.ParquetFile(path)
    idx = start_idx

    for batch in parquet_file.iter_batches(batch_size=64, columns=['image', 'label']):
        images = batch.column('image').to_pylist()
        labels = batch.column('label').to_pylist()
        for image, label in zip(images, labels):
            class_dir = os.path.join(OUTPUT_ROOT, split_name, CLASS_NAMES[label])
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, f'{idx}.jpg')
            idx += 1
            # Skip work already on disk so an interrupted or preempted job can
            # be resumed by simply resubmitting it on the same node.
            if os.path.exists(save_path):
                continue
            if not image or image.get('bytes') is None:
                # An Image feature can reference an external file instead of
                # inlining bytes; these shards are self-contained, so this
                # would mean the source is not what we assume.
                raise ValueError(f'Row {idx - 1} of {path} has no inline image bytes: {image}')
            # Write to a temp name first: a job killed mid-write would
            # otherwise leave a truncated file that the resume check accepts.
            tmp_path = f'{save_path}.partial'
            with open(tmp_path, 'wb') as f:
                f.write(encode(image['bytes']))
            os.replace(tmp_path, save_path)
            written += 1

    return os.path.basename(path), written


def project_output_gb(paths, total_rows):
    """Estimate the tree's size by encoding a sample the same way the run will.

    Re-encoding inflates the source JPEGs by a factor that depends on the
    images, so unlike a byte-for-byte copy the parquet size is not the answer.
    Sampling from shards spread across the split avoids biasing the estimate
    with a single class, since the shards are ordered by label.
    """
    stride = max(1, len(paths) // 5)
    sample_shards = paths[::stride][:5]
    per_shard = max(1, PREFLIGHT_SAMPLES // len(sample_shards))

    encoded, sampled = 0, 0
    for path in sample_shards:
        batches = pq.ParquetFile(path).iter_batches(batch_size=per_shard, columns=['image'])
        for image in next(batches).column('image').to_pylist():
            encoded += len(encode(image['bytes']))
            sampled += 1

    avg_kb = encoded / sampled / 1024
    print(
        f'[extract_imagenet] {avg_kb:.1f} KB/image over {sampled} sampled images',
        flush=True,
    )
    return avg_kb * total_rows / 1024 ** 2


def preflight(paths, total_rows):
    projected_gb = project_output_gb(paths, total_rows)
    free_gb = shutil.disk_usage(OUTPUT_ROOT).free / 1024 ** 3
    mode = 'copying bytes' if COPY_BYTES else f'quality={JPEG_QUALITY}'
    print(
        f'[extract_imagenet] projected output ({mode}): {projected_gb:.0f} GB, '
        f'free: {free_gb:.0f} GB',
        flush=True,
    )

    if os.environ.get('EXTRACT_SKIP_PREFLIGHT') == '1':
        return
    if projected_gb + FREE_SPACE_MARGIN_GB > free_gb:
        raise RuntimeError(
            f'Extracting needs about {projected_gb:.0f} GB (+{FREE_SPACE_MARGIN_GB} GB headroom) but '
            f'only {free_gb:.0f} GB is free on {OUTPUT_ROOT}.\n'
            f'  - {os.path.join(DATA_DIR, "processed_cache")} is a second full copy of the dataset '
            'left over from when the dataloaders read parquet through load_dataset. Nothing in this '
            'repo reads it any more (dataloaders/ImageNet.py is ImageFolder-backed), so deleting it '
            'frees roughly the size of the parquet. It is shared, though, so check with other users '
            'of the node in case their checkout still predates that change.\n'
            '  - Or extract on a node with a larger scratch (vision-c22 / c23 use /scratch-spin).\n'
            '  - EXTRACT_COPY_BYTES=1 skips the re-encode and is ~30% smaller, but its images '
            'will not match the tree already extracted on Delta.\n'
            'Set EXTRACT_SKIP_PREFLIGHT=1 to override.'
        )


def export_split(split_name, paths, offsets, total_rows, num_proc):
    print(
        f'[extract_imagenet] starting {split_name}: {total_rows} examples across '
        f'{len(paths)} shards, num_proc={num_proc}',
        flush=True,
    )
    jobs = [(path, split_name, offset) for path, offset in zip(paths, offsets)]
    written = done = 0
    with Pool(num_proc) as pool:
        for shard_name, shard_written in pool.imap_unordered(export_shard, jobs):
            done += 1
            written += shard_written
            print(
                f'[extract_imagenet] {split_name} {done}/{len(paths)} shards '
                f'({shard_name}: {shard_written} written)',
                flush=True,
            )
    print(f'[extract_imagenet] finished {split_name}: {written} new files', flush=True)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f'[extract_imagenet] host={get_hostname()}', flush=True)
    print(f'[extract_imagenet] source={DATA_DIR}', flush=True)
    print(f'[extract_imagenet] output={OUTPUT_ROOT}', flush=True)

    splits = {}
    for split_name, paths in (('train', TRAIN_SHARDS), ('validation', shard_paths('validation'))):
        offsets, total_rows = shard_offsets(paths)
        splits[split_name] = (paths, offsets, total_rows)

    preflight(TRAIN_SHARDS, sum(total for _, _, total in splits.values()))

    for split_name, (paths, offsets, total_rows) in splits.items():
        export_split(split_name, paths, offsets, total_rows, _num_proc(len(paths)))

    print('[extract_imagenet] all splits done', flush=True)


if __name__ == '__main__':
    main()
