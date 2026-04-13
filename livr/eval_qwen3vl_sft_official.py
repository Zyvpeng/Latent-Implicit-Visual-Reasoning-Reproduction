from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from livr.model import load_model_bundle
from livr.sft_official import OfficialSFTBatchBuilder
from livr.data import LIVRJsonlDataset
from livr.utils import load_yaml, normalize_count_prediction, normalize_mcq_prediction, save_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def normalize_prediction(text: str, target: str) -> str:
    if target.strip().isdigit():
        return normalize_count_prediction(text)
    return normalize_mcq_prediction(text)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    ckpt = args.checkpoint or cfg.get("init_checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model_bundle(
        cfg,
        init_checkpoint=ckpt,
        device=device,
        verbose=False,
        is_trainable=False,
        attach_lora=ckpt is not None,
    )
    model = bundle.model.eval()

    eval_file = cfg.get("test_file", cfg["val_file"])
    dataset = LIVRJsonlDataset(eval_file)
    batch_builder = OfficialSFTBatchBuilder(
        processor=bundle.processor,
        tokenizer=bundle.tokenizer,
        max_length=cfg["max_length"],
        label_assistant_end=cfg.get("label_assistant_end", True),
        image_min_pixels=cfg.get("image_min_pixels"),
        image_max_pixels=cfg.get("image_max_pixels"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["per_device_batch_size"],
        shuffle=False,
        collate_fn=batch_builder.collate_eval,
    )

    rows: list[dict[str, Any]] = []
    correct = 0
    total = 0
    progress = tqdm(dataloader, desc="eval_sft_official", leave=True)
    for batch in progress:
        inputs: dict[str, Any] = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "max_new_tokens": cfg.get("eval_max_new_tokens", 200),
        }
        if "pixel_values" in batch:
            inputs["pixel_values"] = batch["pixel_values"].to(device)
        if "image_grid_thw" in batch:
            inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        stop_token_ids = batch.get("stop_token_ids", [])
        if stop_token_ids:
            inputs["eos_token_id"] = stop_token_ids
        with torch.no_grad():
            generations = model.generate(**inputs)
        prompt_len = batch["input_ids"].shape[1]
        for i in range(generations.shape[0]):
            raw_text = bundle.tokenizer.decode(generations[i][prompt_len:], skip_special_tokens=False)
            text = raw_text.split("<|im_end|>", 1)[0].strip()
            pred = normalize_prediction(text, batch["targets"][i])
            is_correct = pred == batch["targets"][i]
            correct += int(is_correct)
            total += 1
            rows.append(
                {
                    "id": batch["ids"][i],
                    "prompt": batch["prompts"][i],
                    "target": batch["targets"][i],
                    "raw_pred": text,
                    "pred": pred,
                    "correct": is_correct,
                }
            )
            print(
                "sample",
                batch["ids"][i],
                "target=",
                batch["targets"][i],
                "pred=",
                pred,
                "raw_pred=",
                repr(text),
                "correct=",
                is_correct,
            )
        progress.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    save_jsonl(str(output_path), rows)
    print(f"accuracy={correct / max(total, 1):.4f}")
    print(f"predictions={output_path}")


if __name__ == "__main__":
    main()
