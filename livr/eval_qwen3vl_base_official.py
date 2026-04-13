from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from livr.data import LIVRJsonlDataset
from livr.model import resolve_model_class
from livr.utils import load_yaml, normalize_count_prediction, normalize_mcq_prediction, save_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/counting_base_official_eval")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_image(repo_root: Path, rel_path: str) -> Image.Image:
    path = repo_root / "data" / rel_path
    return Image.open(path).convert("RGB")


def normalize_prediction(text: str, target: str) -> str:
    if target.strip().isdigit():
        return normalize_count_prediction(text)
    return normalize_mcq_prediction(text)


def adjust_base_prompt(prompt: str, target: str) -> str:
    if not target.strip().isdigit():
        return prompt
    suffix = " Answer using a single integer only."
    if prompt.endswith(suffix):
        return prompt
    return prompt + suffix


def collate_identity(rows):
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True)
    model_cls = resolve_model_class(cfg["model_name"])
    model = model_cls.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    eval_file = cfg.get("test_file", cfg["val_file"])
    dataset = LIVRJsonlDataset(eval_file)
    if args.limit is not None:
        dataset.rows = dataset.rows[: args.limit]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_identity)

    rows = []
    correct = 0
    total = 0
    progress = tqdm(dataloader, desc="eval_base_official", leave=True)
    for batch_rows in progress:
        row = batch_rows[0]
        image = load_image(repo_root, row["images"][0])
        prompt = adjust_base_prompt(str(row["prompt"]).strip(), row["target"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=cfg.get("eval_max_new_tokens", 8))
        prompt_len = inputs["input_ids"].shape[1]
        raw_text = processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)
        text = raw_text.split("<|im_end|>", 1)[0].strip()
        pred = normalize_prediction(text, row["target"])
        is_correct = pred == row["target"]
        correct += int(is_correct)
        total += 1
        rows.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "target": row["target"],
                "raw_pred": text,
                "pred": pred,
                "correct": is_correct,
            }
        )
        print(
            "sample",
            row["id"],
            "target=",
            row["target"],
            "pred=",
            pred,
            "raw_pred=",
            repr(text),
            "correct=",
            is_correct,
        )
        progress.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    save_jsonl(str(output_path), rows)
    print(f"accuracy={correct / max(total, 1):.4f}")
    print(f"predictions={output_path}")


if __name__ == "__main__":
    main()
