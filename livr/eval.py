from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from livr.data import LIVRBatchBuilder, LIVRJsonlDataset
from livr.model import LIVRModelWrapper, load_model_bundle
from livr.utils import load_yaml, normalize_count_prediction, normalize_mcq_prediction, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def include_latents(stage: str) -> bool:
    return stage in {"livr_stage1", "livr_stage2"}


def normalize_prediction(text: str, task: str) -> str:
    if task == "counting":
        return normalize_count_prediction(text)
    return normalize_mcq_prediction(text)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
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
    model_wrapper = LIVRModelWrapper(bundle.model.eval(), stage=cfg["stage"])

    eval_file = cfg.get("test_file", cfg["val_file"])
    dataset = LIVRJsonlDataset(eval_file)
    batch_builder = LIVRBatchBuilder(
        processor=bundle.processor,
        tokenizer=bundle.tokenizer,
        latent_tokens=bundle.latent_info.tokens,
        latent_token_ids=bundle.latent_info.token_ids,
        max_length=cfg["max_length"],
        label_assistant_end=cfg.get("label_assistant_end", True),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["per_device_batch_size"],
        shuffle=False,
        collate_fn=lambda rows: batch_builder.collate_eval(rows, include_latents=include_latents(cfg["stage"])),
    )

    rows = []
    correct = 0
    total = 0
    progress = tqdm(dataloader, desc="eval", leave=True)
    for batch in progress:
        # batch['prompts'] = ["请描述图片"]
        generations = model_wrapper.generate(batch, max_new_tokens=cfg.get("eval_max_new_tokens", 8))
        prompt_len = batch["input_ids"].shape[1]
        for i in range(generations.shape[0]):
            raw_text = bundle.tokenizer.decode(generations[i][prompt_len:], skip_special_tokens=False)
            text = raw_text.split("<|im_end|>", 1)[0].strip()
            task = "counting" if batch["targets"][i].strip().isdigit() else "localization"
            pred = normalize_prediction(text, task)
            target = batch["targets"][i]
            is_correct = pred == target
            correct += int(is_correct)
            total += 1
            rows.append(
                {
                    "id": batch["ids"][i],
                    "prompt": batch["prompts"][i],
                    "target": target,
                    "pred": pred,
                    "correct": is_correct,
                }
            )
            print(
                "sample",
                batch["ids"][i],
                "target=",
                target,
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
    acc = correct / max(total, 1)
    print(f"accuracy={acc:.4f}")
    print(f"predictions={output_path}")


if __name__ == "__main__":
    main()
