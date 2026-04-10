from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from livr.data import LIVRBatchBuilder, LIVRJsonlDataset, decode_debug_example
from livr.model import LIVRModelWrapper, load_model_bundle, save_livr_checkpoint
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
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def include_latents(stage: str) -> bool:
    return stage in {"livr_stage1", "livr_stage2"}


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


def build_dataloaders(
    cfg: dict,
    bundle,
    distributed: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, LIVRBatchBuilder, DistributedSampler | None]:
    train_ds = LIVRJsonlDataset(cfg["train_file"])
    val_ds = LIVRJsonlDataset(cfg["val_file"])
    batch_builder = LIVRBatchBuilder(
        processor=bundle.processor,
        tokenizer=bundle.tokenizer,
        latent_tokens=bundle.latent_info.tokens,
        latent_token_ids=bundle.latent_info.token_ids,
        max_length=cfg["max_length"],
        label_assistant_end=cfg.get("label_assistant_end", True),
    )
    use_latents = include_latents(cfg["stage"])
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["per_device_batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.get("num_workers", 0) > 0,
        collate_fn=lambda rows: batch_builder.collate_train(rows, include_latents=use_latents),
    )
    val_loader = None
    val_eval_loader = None
    if not distributed or rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["per_device_batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.get("num_workers", 0) > 0,
            collate_fn=lambda rows: batch_builder.collate_train(rows, include_latents=use_latents),
        )
        val_eval_loader = DataLoader(
            val_ds,
            batch_size=cfg["per_device_batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.get("num_workers", 0) > 0,
            collate_fn=lambda rows: batch_builder.collate_eval(rows, include_latents=use_latents),
        )
    return train_loader, val_loader, val_eval_loader, batch_builder, train_sampler


def evaluate_loss(model_wrapper, dataloader) -> float:
    model_wrapper.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model_wrapper(batch)
            loss = outputs.loss.detach()
            losses.append(float(loss.item()))
    model_wrapper.train()
    return float(sum(losses) / max(len(losses), 1))


def _normalize_prediction(text: str, target: str) -> str:
    if target.strip().isdigit():
        return normalize_count_prediction(text)
    return normalize_mcq_prediction(text)


def evaluate_accuracy(model_wrapper, dataloader, tokenizer, max_new_tokens: int) -> float:
    model_wrapper.eval()
    correct = 0
    total = 0
    progress = tqdm(dataloader, disable=not is_main_process(), desc="val_acc", leave=False)
    with torch.no_grad():
        for batch in progress:
            generations = model_wrapper.generate(batch, max_new_tokens=max_new_tokens)
            prompt_len = batch["input_ids"].shape[1]
            for i in range(generations.shape[0]):
                raw_text = tokenizer.decode(generations[i][prompt_len:], skip_special_tokens=False)
                text = raw_text.split("<|im_end|>", 1)[0].strip()
                pred = _normalize_prediction(text, batch["targets"][i])
                if pred == batch["targets"][i]:
                    correct += 1
                total += 1
            if is_main_process():
                progress.set_postfix(acc=f"{correct / max(total, 1):.4f}")
    model_wrapper.train()
    return correct / max(total, 1)


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
    model_wrapper = LIVRModelWrapper(bundle.model, stage=cfg["stage"])
    train_loader, val_loader, val_eval_loader, batch_builder, train_sampler = build_dataloaders(
        cfg,
        bundle,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if distributed:
        model_wrapper = DDP(
            model_wrapper,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = AdamW(
        [param for param in model_wrapper.parameters() if param.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = math.ceil(len(train_loader) / cfg["grad_accum_steps"]) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    raw_wrapper = model_wrapper.module if isinstance(model_wrapper, DDP) else model_wrapper
    best_metric = float("-inf")
    for epoch in range(cfg["num_epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        raw_wrapper.train()
        progress = tqdm(train_loader, disable=not is_main_process(), desc=f"epoch {epoch}")
        for step, batch in enumerate(progress):
            is_update_step = (step + 1) % cfg["grad_accum_steps"] == 0 or (step + 1) == len(train_loader)
            sync_context = (
                nullcontext()
                if not distributed or is_update_step
                else model_wrapper.no_sync()
            )
            with sync_context:
                outputs = model_wrapper(batch)
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
            val_loss = evaluate_loss(raw_wrapper, val_loader)
            print(f"epoch={epoch} val_loss={val_loss:.6f}")
            val_acc = 0.0
            if cfg.get("compute_val_accuracy", True) and val_eval_loader is not None:
                print(f"epoch={epoch} computing_val_acc...")
                val_acc = evaluate_accuracy(
                    raw_wrapper,
                    val_eval_loader,
                    bundle.tokenizer,
                    max_new_tokens=cfg.get("eval_max_new_tokens", 8),
                )
                print(f"epoch={epoch} val_acc={val_acc:.6f}")
            debug_path = Path(cfg["output_dir"]) / f"debug_epoch_{epoch}.txt"
            first_batch = next(iter(train_loader))
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(decode_debug_example(bundle.tokenizer, first_batch, 0))
                f.write("\n\n")
                f.write(str({
                    "image_span": first_batch["image_spans"][0],
                    "prompt_span": first_batch["prompt_spans"][0],
                    "latent_span": first_batch["latent_spans"][0],
                    "answer_query_span": first_batch["answer_query_spans"][0],
                    "answer_span": first_batch["answer_spans"][0],
                    "has_latents": first_batch["has_latents"][0],
                }))

            epoch_dir = Path(cfg["output_dir"]) / f"epoch_{epoch}"
            save_livr_checkpoint(raw_wrapper.model, str(epoch_dir), bundle.latent_info.token_ids)
            bundle.processor.save_pretrained(epoch_dir)
            if (not cfg.get("compute_val_accuracy", True) and best_metric == float("-inf")) or val_acc >= best_metric:
                best_metric = val_acc
                best_dir = Path(cfg["output_dir"]) / "best"
                save_livr_checkpoint(raw_wrapper.model, str(best_dir), bundle.latent_info.token_ids)
                bundle.processor.save_pretrained(best_dir)
                with open(Path(cfg["output_dir"]) / "best_metric.txt", "w", encoding="utf-8") as f:
                    f.write(f"epoch={epoch}\nval_acc={val_acc:.6f}\nval_loss={val_loss:.6f}\n")

        if distributed:
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
