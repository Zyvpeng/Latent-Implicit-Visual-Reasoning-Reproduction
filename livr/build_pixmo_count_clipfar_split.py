from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from livr.build_pixmo_count_livr_paper_split import (
    Row,
    compute_clip_embeddings,
    convert_rows,
    find_near_duplicates,
    is_main_process,
    load_split,
    maybe_destroy_distributed,
)
from livr.utils import save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PixMo-Count split whose train set is intentionally far from the official test set in CLIP space. "
            "Pipeline: valid URLs only -> optional paper-style near-duplicate removal -> "
            "score each train image by its maximum cosine similarity to test -> "
            "sample 1,000 train examples with approximately uniform count distribution, preferring the lowest-similarity rows."
        )
    )
    parser.add_argument("--input-dir", default="data/pixmo_count_official")
    parser.add_argument("--output-dir", default="data/pixmo_count_clipfar")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument(
        "--clip-model-name",
        default="google/siglip-base-patch16-224",
        help="Image similarity model used to score train-to-test distance.",
    )
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument(
        "--clip-cache-dir",
        default=None,
        help="Optional directory used to cache CLIP embeddings as .pt files.",
    )
    parser.add_argument(
        "--clip-top-k",
        type=int,
        default=16,
        help="Store the top-k test similarities per train image for summary statistics.",
    )
    parser.add_argument(
        "--remove-near-duplicates",
        action="store_true",
        help="Apply the same paper-style CLIP + pHash + SSIM deduplication against test before CLIP-far selection.",
    )
    parser.add_argument("--clip-cosine-threshold", type=float, default=0.92)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--ssim-threshold", type=float, default=0.90)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def score_train_rows_against_test(
    train_rows: list[Row],
    test_rows: list[Row],
    clip_model_name: str,
    clip_batch_size: int,
    device: str,
    clip_top_k: int,
    clip_cache_dir: str | None,
) -> list[dict[str, Any]]:
    if not train_rows:
        return []
    if not test_rows:
        return [
            {
                "row": row,
                "max_test_similarity": float("-inf"),
                "mean_topk_similarity": float("-inf"),
                "topk_test_similarities": [],
            }
            for row in train_rows
        ]

    train_emb = compute_clip_embeddings(
        [row.image_path for row in train_rows],
        clip_model_name,
        clip_batch_size,
        device,
        cache_dir=clip_cache_dir,
    )
    test_emb = compute_clip_embeddings(
        [row.image_path for row in test_rows],
        clip_model_name,
        clip_batch_size,
        device,
        cache_dir=clip_cache_dir,
    )
    sims = train_emb @ test_emb.T

    k = max(1, min(int(clip_top_k), test_emb.shape[0]))
    top_vals, _ = torch.topk(sims, k=k, dim=1)

    scored_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(train_rows):
        vals = top_vals[row_idx].tolist()
        scored_rows.append(
            {
                "row": row,
                "max_test_similarity": float(vals[0]),
                "mean_topk_similarity": float(sum(vals) / len(vals)),
                "topk_test_similarities": [float(v) for v in vals],
            }
        )
    return scored_rows


def select_clipfar_train(
    scored_rows: list[dict[str, Any]],
    train_size: int,
    min_count: int,
    max_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        item
        for item in scored_rows
        if min_count <= item["row"].source_count <= max_count
    ]
    by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_count[item["row"].source_count].append(item)

    rng = random.Random(seed)
    for count, bucket in by_count.items():
        rng.shuffle(bucket)
        bucket.sort(
            key=lambda item: (
                item["max_test_similarity"],
                item["mean_topk_similarity"],
                item["row"].source_index,
            )
        )
        by_count[count] = bucket

    counts = list(range(min_count, max_count + 1))
    pointers = {count: 0 for count in counts}
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, int]] = set()

    while len(selected) < train_size:
        progressed = False
        for count in counts:
            bucket = by_count.get(count, [])
            while pointers[count] < len(bucket):
                item = bucket[pointers[count]]
                pointers[count] += 1
                key = (item["row"].source_split, item["row"].source_index)
                if key in used_keys:
                    continue
                used_keys.add(key)
                selected.append(item)
                progressed = True
                break
            if len(selected) >= train_size:
                break
        if not progressed:
            break
    return selected


def build_similarity_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for item in rows:
        row = item["row"]
        report.append(
            {
                "source_split": row.source_split,
                "source_index": row.source_index,
                "source_url": row.source_url,
                "label": row.source_label,
                "count": row.source_count,
                "image_path": row.image_path,
                "max_test_similarity": item["max_test_similarity"],
                "mean_topk_similarity": item["mean_topk_similarity"],
                "topk_test_similarities": item["topk_test_similarities"],
            }
        )
    return report


def summarize_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count_histogram": {},
            "max_similarity_min": None,
            "max_similarity_mean": None,
            "max_similarity_max": None,
            "mean_topk_similarity_mean": None,
        }
    count_histogram: dict[str, int] = defaultdict(int)
    max_sims = []
    mean_topk_sims = []
    for item in rows:
        count_histogram[str(item["row"].source_count)] += 1
        max_sims.append(float(item["max_test_similarity"]))
        mean_topk_sims.append(float(item["mean_topk_similarity"]))
    return {
        "count_histogram": {k: count_histogram[k] for k in sorted(count_histogram, key=int)},
        "max_similarity_min": min(max_sims),
        "max_similarity_mean": sum(max_sims) / len(max_sims),
        "max_similarity_max": max(max_sims),
        "mean_topk_similarity_mean": sum(mean_topk_sims) / len(mean_topk_sims),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_cache_dir = args.clip_cache_dir or str(output_dir / "clip_cache")
    images_dir = input_dir / "images"

    train_rows = load_split(input_dir / "train_metadata.jsonl", images_dir, "train")
    val_rows = load_split(input_dir / "validation_metadata.jsonl", images_dir, "validation")
    test_rows = load_split(input_dir / "test_metadata.jsonl", images_dir, "test")

    duplicate_keys: set[tuple[str, int]] = set()
    duplicate_reports: list[dict[str, Any]] = []
    if args.remove_near_duplicates:
        duplicate_keys, duplicate_reports = find_near_duplicates(
            candidate_rows=train_rows + val_rows,
            test_rows=test_rows,
            clip_model_name=args.clip_model_name,
            clip_batch_size=args.clip_batch_size,
            device=args.device,
            clip_cache_dir=clip_cache_dir,
            clip_top_k=max(1, min(args.clip_top_k, 10)),
            clip_cosine_threshold=args.clip_cosine_threshold,
            phash_threshold=args.phash_threshold,
            ssim_threshold=args.ssim_threshold,
        )

    filtered_train = [row for row in train_rows if (row.source_split, row.source_index) not in duplicate_keys]
    filtered_val = [row for row in val_rows if (row.source_split, row.source_index) not in duplicate_keys]

    scored_train = score_train_rows_against_test(
        train_rows=filtered_train,
        test_rows=test_rows,
        clip_model_name=args.clip_model_name,
        clip_batch_size=args.clip_batch_size,
        device=args.device,
        clip_top_k=args.clip_top_k,
        clip_cache_dir=clip_cache_dir,
    )
    selected_train = select_clipfar_train(
        scored_rows=scored_train,
        train_size=args.train_size,
        min_count=args.min_count,
        max_count=args.max_count,
        seed=args.seed,
    )

    train_jsonl = convert_rows([item["row"] for item in selected_train], "train")
    val_jsonl = convert_rows(filtered_val, "validation")
    test_jsonl = convert_rows(test_rows, "test")

    if is_main_process():
        save_jsonl(str(output_dir / "counting_train.jsonl"), train_jsonl)
        save_jsonl(str(output_dir / "counting_val.jsonl"), val_jsonl)
        save_jsonl(str(output_dir / "counting_test.jsonl"), test_jsonl)
        save_jsonl(str(output_dir / "train_similarity_scores.jsonl"), build_similarity_report(selected_train))
        if duplicate_reports:
            save_jsonl(str(output_dir / "dedup_reports.jsonl"), duplicate_reports)

    selected_summary = summarize_selected(selected_train)
    if is_main_process():
        manifest = {
            "dataset": "allenai/pixmo-count",
            "selection_policy": {
                "train_source": "official train split",
                "validation_source": "official validation split",
                "test_source": "official test split",
                "objective": "Select train examples with the lowest CLIP similarity to the official test set",
                "train_count_range": [args.min_count, args.max_count],
                "train_sampling": "approximately uniform over counts, prioritized by ascending max test similarity",
                "remove_near_duplicates": bool(args.remove_near_duplicates),
            },
            "source_dir": str(input_dir),
            "clip_model_name": args.clip_model_name,
            "clip_cache_dir": clip_cache_dir,
            "clip_batch_size": args.clip_batch_size,
            "clip_top_k": args.clip_top_k,
            "train_candidates_before_filters": len(train_rows),
            "validation_before_filters": len(val_rows),
            "test_size_after_url_filter": len(test_rows),
            "duplicates_removed_train_plus_val": len(duplicate_keys),
            "filtered_train_candidates": len(filtered_train),
            "selected_train_size": len(train_jsonl),
            "val_size": len(val_jsonl),
            "test_size": len(test_jsonl),
            "selected_train_summary": selected_summary,
            "seed": args.seed,
        }
        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
