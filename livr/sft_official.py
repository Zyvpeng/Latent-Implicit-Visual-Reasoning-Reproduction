from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from livr.data import LIVRJsonlDataset, build_labels, load_image
from livr.model import load_model_bundle, save_livr_checkpoint
from livr.utils import (
    ensure_dir,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    load_yaml,
    normalize_count_prediction,
    normalize_mcq_prediction,
    save_jsonl,
    set_seed,
    validate_qwen3vl_vision_batch,
)


ASSISTANT_END_MARKER = "<|im_end|>\n"


@dataclass
class OfficialSFTSample:
    example_id: str
    prompt: str
    target: str
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    distributed = is_distributed()
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    if distributed:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            timeout_minutes = int(os.environ.get("LIVR_DDP_TIMEOUT_MINUTES", "180"))
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
            )
    return distributed, world_size, rank, local_rank


def _candidate_max_lengths(max_length: int) -> list[int]:
    candidates = [max_length, max(max_length, 4096), max(max_length, 8192)]
    out: list[int] = []
    for value in candidates:
        if value not in out:
            out.append(value)
    return out


def _build_sft_prompt(prompt: str) -> str:
    return prompt.strip()


class OfficialSFTBatchBuilder:
    def __init__(
        self,
        processor,
        tokenizer,
        max_length: int,
        label_assistant_end: bool = True,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_assistant_end = label_assistant_end
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

    def _render_generation_prefix(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _encode_with_retry(self, text: str, image) -> dict[str, torch.Tensor]:
        last_error: Exception | None = None
        for max_length in _candidate_max_lengths(self.max_length):
            try:
                kwargs: dict[str, Any] = {
                    "text": [text],
                    "images": [image],
                    "padding": False,
                    "truncation": True,
                    "max_length": max_length,
                    "return_tensors": "pt",
                }
                images_kwargs = {}
                if self.image_min_pixels is not None:
                    images_kwargs["min_pixels"] = int(self.image_min_pixels)
                if self.image_max_pixels is not None:
                    images_kwargs["max_pixels"] = int(self.image_max_pixels)
                if images_kwargs:
                    kwargs["images_kwargs"] = images_kwargs
                return self.processor(**kwargs)
            except ValueError as exc:
                if "Mismatch in `image` token count" not in str(exc):
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to encode SFT sample.")

    def encode_example(self, row: dict[str, Any], with_answer: bool) -> OfficialSFTSample:
        prompt = _build_sft_prompt(str(row["prompt"]))
        image = load_image(row["images"][0])
        prefix = self._render_generation_prefix(prompt)
        answer_only = prefix + str(row["target"])
        full_text = answer_only + ASSISTANT_END_MARKER
        text = full_text if with_answer else prefix

        encoded = self._encode_with_retry(text, image)
        prefix_encoded = self._encode_with_retry(prefix, image)
        answer_only_encoded = self._encode_with_retry(answer_only, image) if with_answer else prefix_encoded

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        prefix_input_ids = prefix_encoded["input_ids"][0]
        answer_only_input_ids = answer_only_encoded["input_ids"][0]

        answer_start = len(prefix_input_ids)
        answer_end = len(answer_only_input_ids)
        label_end = len(input_ids) if (with_answer and self.label_assistant_end) else answer_end
        labels = build_labels(input_ids=input_ids, answer_span=(answer_start, label_end)) if with_answer else torch.full_like(input_ids, -100)

        return OfficialSFTSample(
            example_id=str(row["id"]),
            prompt=prompt,
            target=str(row["target"]),
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=encoded.get("pixel_values", None) if encoded.get("pixel_values", None) is not None else None,
            image_grid_thw=encoded.get("image_grid_thw", None)[0] if encoded.get("image_grid_thw", None) is not None else None,
        )

    def collate_train(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._pack_batch([self.encode_example(row, with_answer=True) for row in rows])

    def collate_eval(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._pack_batch([self.encode_example(row, with_answer=False) for row in rows])

    def _pack_batch(self, rows: list[OfficialSFTSample]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(row.input_ids.shape[0] for row in rows)

        input_ids = []
        labels = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - row.input_ids.shape[0]
            input_ids.append(torch.nn.functional.pad(row.input_ids, (0, pad_len), value=pad_id))
            labels.append(torch.nn.functional.pad(row.labels, (0, pad_len), value=-100))
            attention_mask.append(torch.nn.functional.pad(row.attention_mask, (0, pad_len), value=0))

        batch: dict[str, Any] = {
            "ids": [row.example_id for row in rows],
            "prompts": [row.prompt for row in rows],
            "targets": [row.target for row in rows],
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "stop_token_ids": [
                token_id
                for token_id in [
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    self.tokenizer.eos_token_id,
                ]
                if token_id is not None and token_id >= 0
            ],
        }
        if rows[0].pixel_values is not None:
            batch["pixel_values"] = torch.cat([row.pixel_values for row in rows], dim=0)
        if rows[0].image_grid_thw is not None:
            batch["image_grid_thw"] = torch.stack([row.image_grid_thw for row in rows], dim=0)
        validate_qwen3vl_vision_batch(
            batch.get("pixel_values"),
            batch.get("image_grid_thw"),
            where="OfficialSFTBatchBuilder._pack_batch",
        )
        return batch


def build_dataloaders(
    cfg: dict[str, Any],
    bundle,
    distributed: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, OfficialSFTBatchBuilder, DistributedSampler | None]:
    train_ds = LIVRJsonlDataset(cfg["train_file"])
    val_ds = LIVRJsonlDataset(cfg["val_file"])
    val_subset_size = cfg.get("train_val_subset_size")
    if val_subset_size is not None:
        val_ds = Subset(val_ds, list(range(min(int(val_subset_size), len(val_ds)))))
    batch_builder = OfficialSFTBatchBuilder(
        processor=bundle.processor,
        tokenizer=bundle.tokenizer,
        max_length=cfg["max_length"],
        label_assistant_end=cfg.get("label_assistant_end", True),
        image_min_pixels=cfg.get("image_min_pixels"),
        image_max_pixels=cfg.get("image_max_pixels"),
    )
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if distributed else None
    loader_kwargs = {
        "batch_size": cfg["per_device_batch_size"],
        "num_workers": cfg.get("num_workers", 0),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.get("num_workers", 0) > 0,
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=batch_builder.collate_train,
        **loader_kwargs,
    )
    val_loader = None
    val_eval_loader = None
    if not distributed or rank == 0:
        val_loader = DataLoader(val_ds, shuffle=False, collate_fn=batch_builder.collate_train, **loader_kwargs)
        val_eval_loader = DataLoader(val_ds, shuffle=False, collate_fn=batch_builder.collate_eval, **loader_kwargs)
    return train_loader, val_loader, val_eval_loader, batch_builder, train_sampler


def evaluate_loss(model, dataloader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            inputs: dict[str, Any] = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if "pixel_values" in batch:
                inputs["pixel_values"] = batch["pixel_values"].to(device)
            if "image_grid_thw" in batch:
                inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
            outputs = model(**inputs)
            losses.append(float(outputs.loss.item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def _normalize_prediction(text: str, target: str) -> str:
    if target.strip().isdigit():
        return normalize_count_prediction(text)
    return normalize_mcq_prediction(text)


def evaluate_accuracy(model, dataloader, tokenizer, max_new_tokens: int, device: torch.device, prediction_output_path: str | None = None) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    rows: list[dict[str, object]] = []
    correct = 0
    total = 0
    progress = tqdm(dataloader, disable=not is_main_process(), desc="val_acc", leave=False)
    with torch.no_grad():
        for batch in progress:
            inputs: dict[str, Any] = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "max_new_tokens": max_new_tokens,
            }
            if "pixel_values" in batch:
                inputs["pixel_values"] = batch["pixel_values"].to(device)
            if "image_grid_thw" in batch:
                inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
            stop_token_ids = batch.get("stop_token_ids", [])
            if stop_token_ids:
                inputs["eos_token_id"] = stop_token_ids
            generations = model.generate(**inputs)
            prompt_len = batch["input_ids"].shape[1]
            for i in range(generations.shape[0]):
                raw_text = tokenizer.decode(generations[i][prompt_len:], skip_special_tokens=False)
                text = raw_text.split("<|im_end|>", 1)[0].strip()
                pred = _normalize_prediction(text, batch["targets"][i])
                is_correct = pred == batch["targets"][i]
                if is_correct:
                    correct += 1
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
            if is_main_process():
                progress.set_postfix(acc=f"{correct / max(total, 1):.4f}")
    model.train()
    if prediction_output_path is not None:
        save_jsonl(prediction_output_path, rows)
    return correct / max(total, 1), rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = args.init_checkpoint

    set_seed(cfg.get("seed", 42))
    distributed, world_size, rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        ensure_dir(cfg["output_dir"])

    bundle = load_model_bundle(
        cfg,
        init_checkpoint=cfg.get("init_checkpoint"),
        device=device,
        verbose=is_main_process(),
    )
    train_loader, val_loader, val_eval_loader, _batch_builder, train_sampler = build_dataloaders(
        cfg,
        bundle,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    model = bundle.model
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = math.ceil(len(train_loader) / cfg["grad_accum_steps"]) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    raw_model = model.module if isinstance(model, DDP) else model
    best_metric = float("-inf")
    for epoch in range(cfg["num_epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        raw_model.train()
        progress = tqdm(train_loader, disable=not is_main_process(), desc=f"epoch {epoch}")
        for step, batch in enumerate(progress):
            is_update_step = (step + 1) % cfg["grad_accum_steps"] == 0 or (step + 1) == len(train_loader)
            sync_context = nullcontext() if not distributed or is_update_step else model.no_sync()
            with sync_context:
                inputs: dict[str, Any] = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }
                if "pixel_values" in batch:
                    inputs["pixel_values"] = batch["pixel_values"].to(device)
                if "image_grid_thw" in batch:
                    inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
                outputs = model(**inputs)
                loss = outputs.loss / cfg["grad_accum_steps"]
                loss.backward()
            if is_update_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if is_main_process():
                progress.set_postfix(loss=float(loss.item() * cfg["grad_accum_steps"]))

        if distributed:
            dist.barrier()

        if is_main_process() and val_loader is not None:
            val_loss = evaluate_loss(raw_model, val_loader, device)
            print(f"epoch={epoch} val_loss={val_loss:.6f}")
            val_acc = 0.0
            if cfg.get("compute_val_accuracy", True) and val_eval_loader is not None:
                print(f"epoch={epoch} computing_val_acc...")
                prediction_path = str(Path(cfg["output_dir"]) / f"val_predictions_epoch_{epoch}.jsonl")
                val_acc, prediction_rows = evaluate_accuracy(
                    raw_model,
                    val_eval_loader,
                    bundle.tokenizer,
                    max_new_tokens=cfg.get("eval_max_new_tokens", 200),
                    device=device,
                    prediction_output_path=prediction_path,
                )
                print(f"epoch={epoch} val_acc={val_acc:.6f}")
                print(f"epoch={epoch} val_predictions={prediction_path}")
                for sample in prediction_rows:
                    print(
                        "sample",
                        sample["id"],
                        "target=",
                        sample["target"],
                        "pred=",
                        sample["pred"],
                        "raw_pred=",
                        repr(sample["raw_pred"]),
                        "correct=",
                        sample["correct"],
                    )
            epoch_dir = Path(cfg["output_dir"]) / f"epoch_{epoch}"
            save_livr_checkpoint(raw_model, str(epoch_dir), bundle.latent_info.token_ids)
            bundle.processor.save_pretrained(epoch_dir)
            if (not cfg.get("compute_val_accuracy", True) and best_metric == float("-inf")) or val_acc >= best_metric:
                best_metric = val_acc
                best_dir = Path(cfg["output_dir"]) / "best"
                save_livr_checkpoint(raw_model, str(best_dir), bundle.latent_info.token_ids)
                bundle.processor.save_pretrained(best_dir)
                with open(Path(cfg["output_dir"]) / "best_metric.txt", "w", encoding="utf-8") as f:
                    f.write(f"epoch={epoch}\nval_acc={val_acc:.6f}\nval_loss={val_loss:.6f}\n")

        if distributed:
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
