from __future__ import annotations

import argparse
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from livr.data import build_counting_prompt
from livr.utils import ensure_dir, save_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/pixmo_count")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def fetch_image(url: str, dest: Path, timeout: int = 20) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.save(dest)
        return True
    except Exception:
        return False


def resolve_count(row: dict[str, Any]) -> int | None:
    for key in ["count", "answer", "label", "number", "number_answer"]:
        if key in row and row[key] is not None:
            try:
                return int(row[key])
            except Exception:
                continue
    return None


def resolve_object_name(row: dict[str, Any]) -> str:
    for key in ["object", "object_name", "category", "label_name"]:
        if key in row and row[key]:
            return str(row[key])
    return "objects"


def resolve_url(row: dict[str, Any]) -> str | None:
    for key in ["image_url", "url", "image"]:
        if key in row and row[key]:
            return str(row[key])
    return None


def convert_row(row: dict[str, Any], image_path: str, count: int, object_name: str, split: str, index: int) -> dict[str, Any]:
    return {
        "id": row.get("id", f"{split}-{index}"),
        "images": [image_path],
        "prompt": build_counting_prompt(object_name),
        "target": str(count),
        "task": "counting",
        "object_name": object_name,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    ensure_dir(str(images_dir))

    ds = load_dataset("allenai/pixmo-count")
    rng = random.Random(args.seed)

    def candidate_rows(split_name: str, filter_train_range: bool) -> list[tuple[int, dict[str, Any], int, str]]:
        rows = list(ds[split_name])
        rng.shuffle(rows)
        candidates: list[tuple[int, dict[str, Any], int, str]] = []
        for index, row in enumerate(rows):
            count = resolve_count(row)
            url = resolve_url(row)
            if count is None or url is None:
                continue
            if filter_train_range and not (args.min_count <= count <= args.max_count):
                continue
            candidates.append((index, row, count, url))
        return candidates

    def prepare_split(split_name: str, limit: int | None, filter_train_range: bool) -> list[dict[str, Any]]:
        candidates = candidate_rows(split_name, filter_train_range)
        if limit is not None:
            candidates = candidates[: limit * 4]

        prepared: list[dict[str, Any]] = []

        def work(item: tuple[int, dict[str, Any], int, str]) -> dict[str, Any] | None:
            index, row, count, url = item
            image_name = f"{split_name}_{index}.jpg"
            image_path = images_dir / image_name
            if not image_path.exists():
                ok = fetch_image(url, image_path)
                if not ok:
                    return None
            return convert_row(
                row=row,
                image_path=str(image_path.relative_to(output_dir.parent)),
                count=count,
                object_name=resolve_object_name(row),
                split=split_name,
                index=index,
            )

        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(work, item) for item in candidates]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"prepare-{split_name}"):
                example = future.result()
                if example is None:
                    continue
                prepared.append(example)
                if limit is not None and len(prepared) >= limit:
                    break
        return prepared

    train_rows = prepare_split("train", limit=args.train_size, filter_train_range=True)
    val_rows = prepare_split("validation", limit=None, filter_train_range=False)

    save_jsonl(str(output_dir / "counting_train.jsonl"), train_rows)
    save_jsonl(str(output_dir / "counting_val.jsonl"), val_rows)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "allenai/pixmo-count",
                "train_size": len(train_rows),
                "val_size": len(val_rows),
                "count_range_train": [args.min_count, args.max_count],
                "seed": args.seed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved train={len(train_rows)} val={len(val_rows)} to {output_dir}")


if __name__ == "__main__":
    main()
