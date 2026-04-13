from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
    parser.add_argument("--save-latent-attn", action="store_true")
    parser.add_argument("--latent-attn-dir", default=None)
    return parser.parse_args()


def include_latents(stage: str) -> bool:
    return stage in {"livr_stage1", "livr_stage2"}


def _save_latent_attention_maps(
    save_dir: Path,
    batch: dict[str, Any],
    attentions: tuple[torch.Tensor, ...],
    tokenizer,
    merge_size: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    layer_stack = torch.stack(attentions, dim=0)
    for i, sample_id in enumerate(batch["ids"]):
        latent_start, latent_end = batch["latent_spans"][i]
        image_start, image_end = batch["image_spans"][i]
        if latent_end <= latent_start or image_end <= image_start:
            continue
        sample_attn = layer_stack[:, i, :, latent_start:latent_end, image_start:image_end]
        mean_heads = sample_attn.mean(dim=1)
        last_layer = mean_heads[-1]
        avg_layers = mean_heads.mean(dim=0)
        token_ids = batch["input_ids"][i, latent_start:latent_end].tolist()
        token_text = tokenizer.convert_ids_to_tokens(token_ids)
        image_grid = None
        visual_grid = None
        last_layer_2d = None
        avg_layers_2d = None
        if "image_grid_thw" in batch:
            image_grid = [int(x) for x in batch["image_grid_thw"][i].tolist()]
            t, h, w = image_grid
            if merge_size > 0 and h % merge_size == 0 and w % merge_size == 0:
                visual_grid = [t, h // merge_size, w // merge_size]
                expected = visual_grid[0] * visual_grid[1] * visual_grid[2]
                if expected == (image_end - image_start):
                    last_layer_2d = last_layer.reshape(len(token_ids), visual_grid[0], visual_grid[1], visual_grid[2])
                    avg_layers_2d = avg_layers.reshape(len(token_ids), visual_grid[0], visual_grid[1], visual_grid[2])
        payload = {
            "id": sample_id,
            "prompt": batch["prompts"][i],
            "target": batch["targets"][i],
            "image_span": batch["image_spans"][i],
            "latent_span": batch["latent_spans"][i],
            "latent_token_ids": token_ids,
            "latent_token_text": token_text,
            "image_grid_thw": image_grid,
            "merge_size": merge_size,
            "visual_grid": visual_grid,
            "last_layer_mean_heads": last_layer,
            "avg_layers_mean_heads": avg_layers,
        }
        if last_layer_2d is not None:
            payload["last_layer_mean_heads_2d"] = last_layer_2d
            payload["avg_layers_mean_heads_2d"] = avg_layers_2d
        torch.save(payload, save_dir / f"{sample_id}.pt")


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
        image_min_pixels=cfg.get("image_min_pixels"),
        image_max_pixels=cfg.get("image_max_pixels"),
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
    latent_attn_dir = None
    if args.save_latent_attn:
        latent_attn_dir = Path(args.latent_attn_dir) if args.latent_attn_dir is not None else ((Path(args.output_dir) if args.output_dir is not None else Path(cfg["output_dir"])) / "latent_attn")
        latent_attn_dir.mkdir(parents=True, exist_ok=True)
        print(f"latent_attn_dir={latent_attn_dir}")
    merge_size = int(getattr(bundle.processor.image_processor, "merge_size", 1))
    progress = tqdm(dataloader, desc="eval", leave=True)
    for batch in progress:
        if latent_attn_dir is not None:
            attentions = model_wrapper.collect_attentions(batch)
            if attentions is not None:
                _save_latent_attention_maps(latent_attn_dir, batch, attentions, bundle.tokenizer, merge_size)
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
