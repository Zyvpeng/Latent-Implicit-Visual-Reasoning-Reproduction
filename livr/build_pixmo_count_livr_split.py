from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from livr.utils import save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/pixmo_count_official')
    parser.add_argument('--output-dir', default='data/pixmo_count_livr')
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--max-count', type=int, default=10)
    return parser.parse_args()


def build_prompt(label: str) -> str:
    return f'Count the number of {label} in the image. Answer with a single integer only.'


def image_path_for_url(images_dir: Path, url: str) -> Path:
    return images_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.jpg"


def load_split(meta_path: Path, images_dir: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                {
                    'source_split': split,
                    'source_index': idx,
                    'source_url': url,
                    'source_label': label,
                    'source_count': int(count),
                    'image_path': str(img_path),
                }
            )
    return rows


def sample_train(rows: list[dict[str, Any]], train_size: int, min_count: int, max_count: int, seed: int) -> list[dict[str, Any]]:
    eligible = [r for r in rows if min_count <= r['source_count'] <= max_count]
    by_count_label: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in eligible:
        by_count_label[int(r['source_count'])][str(r['source_label'])].append(r)

    rng = random.Random(seed)
    for label_buckets in by_count_label.values():
        for bucket in label_buckets.values():
            rng.shuffle(bucket)

    counts = list(range(min_count, max_count + 1))
    label_orders: dict[int, list[str]] = {c: list(by_count_label.get(c, {}).keys()) for c in counts}
    for labels in label_orders.values():
        rng.shuffle(labels)

    label_ptr = {c: 0 for c in counts}
    sampled: list[dict[str, Any]] = []
    used = set()
    while len(sampled) < train_size:
        progressed = False
        for c in counts:
            labels = label_orders.get(c, [])
            if not labels:
                continue
            picked = None
            for _ in range(len(labels)):
                label = labels[label_ptr[c] % len(labels)]
                label_ptr[c] += 1
                bucket = by_count_label[c][label]
                while bucket and (bucket[-1]['source_split'], bucket[-1]['source_index']) in used:
                    bucket.pop()
                if bucket:
                    picked = bucket.pop()
                    break
            if picked is None:
                continue
            key = (picked['source_split'], picked['source_index'])
            if key in used:
                continue
            used.add(key)
            sampled.append(picked)
            progressed = True
            if len(sampled) >= train_size:
                break
        if not progressed:
            break
    return sampled


def convert_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        label = str(row['source_label']).strip()
        count = int(row['source_count'])
        out.append(
            {
                'id': f'{split_name}-{i}',
                'images': [str(Path('pixmo_count_official/images') / Path(row['image_path']).name)],
                'prompt': build_prompt(label),
                'target': str(count),
                'task': 'counting',
                'object_name': label,
                'source_split': row['source_split'],
                'source_index': row['source_index'],
                'source_url': row['source_url'],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = input_dir / 'images'

    train_rows = load_split(input_dir / 'train_metadata.jsonl', images_dir, 'train')
    val_rows = load_split(input_dir / 'validation_metadata.jsonl', images_dir, 'validation')
    test_rows = load_split(input_dir / 'test_metadata.jsonl', images_dir, 'test')

    sampled_train = sample_train(
        train_rows,
        train_size=args.train_size,
        min_count=args.min_count,
        max_count=args.max_count,
        seed=args.seed,
    )

    train_jsonl = convert_rows(sampled_train, 'train')
    val_jsonl = convert_rows(val_rows, 'validation')
    test_jsonl = convert_rows(test_rows, 'test')

    save_jsonl(str(output_dir / 'counting_train.jsonl'), train_jsonl)
    save_jsonl(str(output_dir / 'counting_val.jsonl'), val_jsonl)
    save_jsonl(str(output_dir / 'counting_test.jsonl'), test_jsonl)

    by_count = defaultdict(int)
    for row in train_jsonl:
        by_count[int(row['target'])] += 1

    by_label = defaultdict(int)
    for row in train_jsonl:
        by_label[row['object_name']] += 1

    manifest = {
        'dataset': 'allenai/pixmo-count',
        'source_dir': str(input_dir),
        'train_size': len(train_jsonl),
        'val_size': len(val_jsonl),
        'test_size': len(test_jsonl),
        'train_count_range': [args.min_count, args.max_count],
        'train_sampling': 'count_balanced_label_round_robin',
        'train_count_histogram': {str(k): by_count[k] for k in sorted(by_count)},
        'train_top_labels': dict(sorted(by_label.items(), key=lambda kv: (-kv[1], kv[0]))[:25]),
        'seed': args.seed,
    }
    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
