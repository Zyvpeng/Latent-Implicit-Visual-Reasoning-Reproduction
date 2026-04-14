from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from livr.utils import save_jsonl


@dataclass
class Row:
    source_split: str
    source_index: int
    source_url: str
    source_label: str
    source_count: int
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Build a paper-faithful PixMo-Count split for LIVR. '\
            'Pipeline: valid URLs only -> count range 2..10 for train -> '\
            'CLIP prefilter against official test -> pHash + SSIM confirm -> '\
            'sample 1,000 train examples with approximately uniform count distribution.'
        )
    )
    parser.add_argument('--input-dir', default='data/pixmo_count_official')
    parser.add_argument('--output-dir', default='data/pixmo_count_livr_paper')
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--max-count', type=int, default=10)
    parser.add_argument(
        '--clip-model-name',
        default='google/siglip-base-patch16-224',
        help=(
            'Image similarity model used for the CLIP-style embedding prefilter. '
            'Default uses a safetensors-backed SigLIP checkpoint to avoid the '
            'torch<2.6 + pytorch_model.bin loading restriction in recent transformers.'
        ),
    )
    parser.add_argument('--clip-batch-size', type=int, default=32)
    parser.add_argument(
        '--clip-cache-dir',
        default=None,
        help='Optional directory used to cache CLIP embeddings as .pt files.',
    )
    parser.add_argument('--clip-top-k', type=int, default=10)
    parser.add_argument('--clip-cosine-threshold', type=float, default=0.92)
    parser.add_argument('--phash-threshold', type=int, default=8)
    parser.add_argument('--ssim-threshold', type=float, default=0.90)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_prompt(label: str) -> str:
    return f'How many {label} are there in this image?'


def image_path_for_url(images_dir: Path, url: str) -> Path:
    return images_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.jpg"


def load_split(meta_path: Path, images_dir: Path, split: str) -> list[Row]:
    rows: list[Row] = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            url = r.get('image_url')
            label = r.get('label')
            count = r.get('count')
            if url is None or label is None or count is None:
                continue
            img_path = image_path_for_url(images_dir, url)
            if not img_path.exists() or img_path.stat().st_size <= 0:
                continue
            rows.append(
                Row(
                    source_split=split,
                    source_index=idx,
                    source_url=url,
                    source_label=str(label),
                    source_count=int(count),
                    image_path=str(img_path),
                )
            )
    return rows


def distributed_enabled() -> bool:
    return int(os.environ.get('WORLD_SIZE', '1')) > 1


def distributed_rank() -> int:
    return int(os.environ.get('RANK', '0'))


def distributed_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def distributed_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', '1'))


def is_main_process() -> bool:
    return distributed_rank() == 0


def maybe_init_distributed(device: str) -> None:
    if not distributed_enabled() or dist.is_initialized():
        return
    backend = 'nccl' if device.startswith('cuda') else 'gloo'
    if backend == 'nccl':
        torch.cuda.set_device(distributed_local_rank())
    dist.init_process_group(backend=backend)


def maybe_destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def resolve_worker_device(device: str) -> str:
    if not distributed_enabled():
        return device
    if device.startswith('cuda'):
        return f'cuda:{distributed_local_rank()}'
    return device


def clip_cache_file(cache_dir: str | None, model_name: str, image_paths: list[str]) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha1()
    digest.update(model_name.encode('utf-8'))
    digest.update(b'\0')
    for image_path in image_paths:
        digest.update(str(Path(image_path).resolve()).encode('utf-8'))
        digest.update(b'\0')
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f'{digest.hexdigest()}.pt'


def compute_clip_embeddings(
    image_paths: list[str],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: str | None = None,
) -> torch.Tensor:
    cache_file = clip_cache_file(cache_dir, model_name, image_paths)
    if cache_file is not None and cache_file.exists():
        return torch.load(cache_file, map_location='cpu')

    maybe_init_distributed(device)
    worker_device = resolve_worker_device(device)
    rank = distributed_rank()
    world_size = distributed_world_size()

    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, use_safetensors=True).eval().to(worker_device)
    except Exception as exc:
        raise ValueError(
            'Failed to load the image similarity model with safetensors. '
            'On torch<2.6, recent transformers blocks loading .bin checkpoints via torch.load. '
            'Use a safetensors-backed checkpoint such as '
            '`google/siglip-base-patch16-224`, or upgrade torch to >=2.6.'
        ) from exc

    local_indices = list(range(rank, len(image_paths), world_size))
    local_paths = [image_paths[idx] for idx in local_indices]
    outputs: list[torch.Tensor] = []
    iterator = range(0, len(local_paths), batch_size)
    if not distributed_enabled() or is_main_process():
        iterator = tqdm(iterator, desc='clip_embed', leave=False)
    for start in iterator:
        batch_paths = local_paths[start:start + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=images, return_tensors='pt')
        inputs = {k: v.to(worker_device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        outputs.append(image_features.cpu())
    local_embeddings = torch.cat(outputs, dim=0) if outputs else None

    if not distributed_enabled():
        if local_embeddings is None:
            raise ValueError('No embeddings were produced for the provided image paths.')
        if cache_file is not None:
            torch.save(local_embeddings, cache_file)
        return local_embeddings

    payload = {
        'indices': local_indices,
        'embeddings': local_embeddings,
    }
    gathered: list[dict[str, Any]] | None = [None] * world_size if is_main_process() else None
    dist.gather_object(payload, gathered, dst=0)

    if is_main_process():
        first_nonempty = next(
            item for item in gathered if item['embeddings'] is not None and item['embeddings'].numel() > 0
        )
        dim = int(first_nonempty['embeddings'].shape[1])
        full_embeddings = torch.empty((len(image_paths), dim), dtype=first_nonempty['embeddings'].dtype)
        for item in gathered:
            embeddings = item['embeddings']
            if embeddings is None or embeddings.numel() == 0:
                continue
            full_embeddings[item['indices']] = embeddings
        if cache_file is not None:
            torch.save(full_embeddings, cache_file)
    distributed_barrier()
    if cache_file is None:
        raise ValueError('Distributed CLIP extraction requires clip_cache_dir to be set.')
    return torch.load(cache_file, map_location='cpu')


def _dct_matrix(n: int) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float32)
    factor = math.pi / (2.0 * n)
    scale0 = math.sqrt(1.0 / n)
    scale = math.sqrt(2.0 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            mat[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return mat


_DCT32 = _dct_matrix(32)


def phash(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    dct = _DCT32 @ pixels @ _DCT32.T
    low = dct[:8, :8]
    med = np.median(low[1:, :].reshape(-1))
    return (low > med).astype(np.uint8).reshape(-1)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def grayscale_array(image_path: str, blurred: bool = False, size: int = 256) -> np.ndarray:
    image = Image.open(image_path).convert('L').resize((size, size), Image.Resampling.LANCZOS)
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    if den == 0:
        return 1.0
    return num / den


def find_near_duplicates(
    candidate_rows: list[Row],
    test_rows: list[Row],
    clip_model_name: str,
    clip_batch_size: int,
    device: str,
    clip_cache_dir: str | None,
    clip_top_k: int,
    clip_cosine_threshold: float,
    phash_threshold: int,
    ssim_threshold: float,
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    if not candidate_rows or not test_rows:
        return set(), []

    candidate_emb = compute_clip_embeddings(
        [r.image_path for r in candidate_rows],
        clip_model_name,
        clip_batch_size,
        device,
        cache_dir=clip_cache_dir,
    )
    test_emb = compute_clip_embeddings(
        [r.image_path for r in test_rows],
        clip_model_name,
        clip_batch_size,
        device,
        cache_dir=clip_cache_dir,
    )
    sims = candidate_emb @ test_emb.T
    top_k = min(int(clip_top_k), test_emb.shape[0])
    top_vals, top_idx = torch.topk(sims, k=top_k, dim=1)

    test_phashes = [phash(r.image_path) for r in tqdm(test_rows, desc='test_phash', leave=False)]
    duplicate_keys: set[tuple[str, int]] = set()
    reports: list[dict[str, Any]] = []

    for cand_i, cand_row in enumerate(tqdm(candidate_rows, desc='dedup_check', leave=False)):
        cand_hash = phash(cand_row.image_path)
        cand_gray = None
        cand_blur = None
        vals = top_vals[cand_i].tolist()
        idxs = top_idx[cand_i].tolist()
        for sim, test_i in zip(vals, idxs):
            if sim < clip_cosine_threshold:
                continue
            test_row = test_rows[test_i]
            hash_dist = hamming_distance(cand_hash, test_phashes[test_i])
            if hash_dist > phash_threshold:
                continue
            if cand_gray is None:
                cand_gray = grayscale_array(cand_row.image_path, blurred=False)
                cand_blur = grayscale_array(cand_row.image_path, blurred=True)
            test_gray = grayscale_array(test_row.image_path, blurred=False)
            test_blur = grayscale_array(test_row.image_path, blurred=True)
            ssim_raw = ssim_score(cand_gray, test_gray)
            ssim_blur = ssim_score(cand_blur, test_blur)
            if max(ssim_raw, ssim_blur) >= ssim_threshold:
                key = (cand_row.source_split, cand_row.source_index)
                duplicate_keys.add(key)
                reports.append(
                    {
                        'candidate_split': cand_row.source_split,
                        'candidate_index': cand_row.source_index,
                        'candidate_url': cand_row.source_url,
                        'candidate_label': cand_row.source_label,
                        'candidate_count': cand_row.source_count,
                        'test_split': test_row.source_split,
                        'test_index': test_row.source_index,
                        'test_url': test_row.source_url,
                        'test_label': test_row.source_label,
                        'test_count': test_row.source_count,
                        'clip_cosine': float(sim),
                        'phash_distance': int(hash_dist),
                        'ssim_raw': float(ssim_raw),
                        'ssim_blur': float(ssim_blur),
                    }
                )
                break
    return duplicate_keys, reports


def sample_train(rows: list[Row], train_size: int, min_count: int, max_count: int, seed: int) -> list[Row]:
    eligible = [r for r in rows if min_count <= r.source_count <= max_count]
    by_count: dict[int, list[Row]] = defaultdict(list)
    for r in eligible:
        by_count[r.source_count].append(r)
    rng = random.Random(seed)
    for bucket in by_count.values():
        rng.shuffle(bucket)
    counts = list(range(min_count, max_count + 1))
    sampled: list[Row] = []
    used = set()
    while len(sampled) < train_size:
        progressed = False
        for c in counts:
            bucket = by_count.get(c, [])
            while bucket and (bucket[-1].source_split, bucket[-1].source_index) in used:
                bucket.pop()
            if not bucket:
                continue
            row = bucket.pop()
            key = (row.source_split, row.source_index)
            if key in used:
                continue
            used.add(key)
            sampled.append(row)
            progressed = True
            if len(sampled) >= train_size:
                break
        if not progressed:
            break
    return sampled


def convert_rows(rows: list[Row], split_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        out.append(
            {
                'id': f'{split_name}-{i}',
                'images': [str(Path('pixmo_count_official/images') / Path(row.image_path).name)],
                'prompt': build_prompt(row.source_label.strip()),
                'target': str(int(row.source_count)),
                'task': 'counting',
                'object_name': row.source_label.strip(),
                'source_split': row.source_split,
                'source_index': row.source_index,
                'source_url': row.source_url,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_cache_dir = args.clip_cache_dir or str(output_dir / 'clip_cache')
    images_dir = input_dir / 'images'

    train_rows = load_split(input_dir / 'train_metadata.jsonl', images_dir, 'train')
    val_rows = load_split(input_dir / 'validation_metadata.jsonl', images_dir, 'validation')
    test_rows = load_split(input_dir / 'test_metadata.jsonl', images_dir, 'test')

    duplicate_keys, duplicate_reports = find_near_duplicates(
        candidate_rows=train_rows + val_rows,
        test_rows=test_rows,
        clip_model_name=args.clip_model_name,
        clip_batch_size=args.clip_batch_size,
        device=args.device,
        clip_cache_dir=clip_cache_dir,
        clip_top_k=args.clip_top_k,
        clip_cosine_threshold=args.clip_cosine_threshold,
        phash_threshold=args.phash_threshold,
        ssim_threshold=args.ssim_threshold,
    )

    filtered_train = [r for r in train_rows if (r.source_split, r.source_index) not in duplicate_keys]
    filtered_val = [r for r in val_rows if (r.source_split, r.source_index) not in duplicate_keys]
    sampled_train = sample_train(filtered_train, args.train_size, args.min_count, args.max_count, args.seed)

    train_jsonl = convert_rows(sampled_train, 'train')
    val_jsonl = convert_rows(filtered_val, 'validation')
    test_jsonl = convert_rows(test_rows, 'test')

    if is_main_process():
        save_jsonl(str(output_dir / 'counting_train.jsonl'), train_jsonl)
        save_jsonl(str(output_dir / 'counting_val.jsonl'), val_jsonl)
        save_jsonl(str(output_dir / 'counting_test.jsonl'), test_jsonl)
        save_jsonl(str(output_dir / 'dedup_reports.jsonl'), duplicate_reports)

        by_count = defaultdict(int)
        for row in train_jsonl:
            by_count[int(row['target'])] += 1

        manifest = {
            'dataset': 'allenai/pixmo-count',
            'paper_alignment': {
                'train_source': 'official train split',
                'validation_source': 'official validation split',
                'test_source': 'official test split',
                'train_count_range': [args.min_count, args.max_count],
                'train_sampling': 'approximately uniform over counts',
                'deduplication': 'CLIP prefilter + pHash + SSIM against official test',
            },
            'source_dir': str(input_dir),
            'clip_model_name': args.clip_model_name,
            'clip_cache_dir': clip_cache_dir,
            'clip_top_k': args.clip_top_k,
            'clip_cosine_threshold': args.clip_cosine_threshold,
            'phash_threshold': args.phash_threshold,
            'ssim_threshold': args.ssim_threshold,
            'train_candidates_before_dedup': len(train_rows),
            'validation_before_dedup': len(val_rows),
            'test_size': len(test_jsonl),
            'duplicates_removed_train_plus_val': len(duplicate_keys),
            'train_size': len(train_jsonl),
            'val_size': len(val_jsonl),
            'test_size_after_url_filter': len(test_jsonl),
            'train_count_histogram': {str(k): by_count[k] for k in sorted(by_count)},
            'seed': args.seed,
        }
        with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    maybe_destroy_distributed()


if __name__ == '__main__':
    main()
