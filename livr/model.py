from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from safetensors import safe_open

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

from livr.attention_mask import build_livr_attention_mask
from livr.latent_tokens import (
    LatentTokenInfo,
    add_latent_tokens,
    mark_only_latent_rows_trainable,
)
from livr.utils import is_main_process, print_trainable_parameters


LANGUAGE_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class LIVRBundle:
    model: torch.nn.Module
    processor: Any
    tokenizer: Any
    latent_info: LatentTokenInfo


def resolve_model_class(model_name: str):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration
    if model_type == "qwen2_5_vl" and Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration
    return AutoModelForImageTextToText


def _stage_uses_latents(stage: str) -> bool:
    return stage in {"stage1", "stage2", "livr_stage1", "livr_stage2"}


def _freeze_vision_and_projector(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        lowered = name.lower()
        if "visual" in lowered or "vision" in lowered or "multi_modal_projector" in lowered or "merger" in lowered:
            param.requires_grad = False


def _resolve_torch_dtype(bf16: bool) -> torch.dtype:
    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_lora_model(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.nn.Module:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=LANGUAGE_LORA_TARGETS,
    )
    return get_peft_model(model, peft_config)


def _load_latent_rows_if_available(model: torch.nn.Module, init_checkpoint: str | None) -> None:
    if not init_checkpoint:
        return
    latent_path = Path(init_checkpoint) / "latent_rows.pt"
    if not latent_path.exists():
        return
    payload = torch.load(latent_path, map_location="cpu", weights_only=True)
    latent_rows = payload["latent_rows"]
    input_embeddings = model.get_input_embeddings()
    token_ids = payload.get("token_ids", [])
    with torch.no_grad():
        if hasattr(input_embeddings, "latent_rows"):
            input_embeddings.latent_rows.copy_(latent_rows.to(device=input_embeddings.latent_rows.device))
        else:
            index = torch.tensor(token_ids, dtype=torch.long, device=input_embeddings.weight.device)
            input_embeddings.weight.index_copy_(0, index, latent_rows.to(device=input_embeddings.weight.device))
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and output_embeddings is not input_embeddings:
                output_embeddings.weight.index_copy_(0, index, latent_rows.to(device=output_embeddings.weight.device))


def _checkpoint_uses_nested_peft(init_checkpoint: str | None) -> bool:
    if not init_checkpoint:
        return False
    adapter_path = Path(init_checkpoint) / "adapter_model.safetensors"
    if not adapter_path.exists():
        return False
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            return key.startswith("base_model.model.base_model.")
    return False


def save_livr_checkpoint(model: torch.nn.Module, save_dir: str, latent_token_ids: list[int]) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    peft_model = model
    if hasattr(peft_model, "module"):
        peft_model = peft_model.module
    if not hasattr(peft_model, "save_pretrained") and hasattr(peft_model, "model"):
        peft_model = peft_model.model

    if hasattr(peft_model, "save_pretrained"):
        peft_model.save_pretrained(save_path)

    input_embeddings = peft_model.get_input_embeddings()
    if hasattr(input_embeddings, "latent_rows"):
        latent_rows = input_embeddings.latent_rows.detach().cpu()
    else:
        index = torch.tensor(latent_token_ids, dtype=torch.long, device=input_embeddings.weight.device)
        latent_rows = input_embeddings.weight.detach().index_select(0, index).cpu()

    torch.save(
        {
            "token_ids": list(latent_token_ids),
            "latent_rows": latent_rows,
        },
        save_path / "latent_rows.pt",
    )


def load_model_bundle(
    cfg: dict[str, Any],
    init_checkpoint: str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    is_trainable: bool = True,
    attach_lora: bool = True,
) -> LIVRBundle:
    processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer = processor.tokenizer
    if _stage_uses_latents(cfg["stage"]):
        latent_info = add_latent_tokens(tokenizer=tokenizer, num_latents=cfg["num_latents"])
    else:
        latent_info = LatentTokenInfo(tokens=[], token_ids=[])

    torch_dtype = _resolve_torch_dtype(cfg.get("bf16", True))
    model_cls = resolve_model_class(cfg["model_name"])
    model = model_cls.from_pretrained(
        cfg["model_name"],
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if latent_info.token_ids:
        model.resize_token_embeddings(len(tokenizer))
    _freeze_vision_and_projector(model)
    if latent_info.token_ids:
        mark_only_latent_rows_trainable(model, latent_info.token_ids)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if init_checkpoint:
        if _checkpoint_uses_nested_peft(init_checkpoint):
            model = build_lora_model(model, cfg)
        model = PeftModel.from_pretrained(model, init_checkpoint, is_trainable=is_trainable)
        _load_latent_rows_if_available(model, init_checkpoint)
    elif attach_lora:
        model = build_lora_model(model, cfg)

    if device is not None:
        model = model.to(device)

    if verbose and is_main_process():
        print_trainable_parameters(model)
    return LIVRBundle(model=model, processor=processor, tokenizer=tokenizer, latent_info=latent_info)


class LIVRModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, stage: str) -> None:
        super().__init__()
        self.model = model
        self.stage = stage

    def _model_device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _model_dtype(self) -> torch.dtype:
        for param in self.model.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    def _eos_token_ids(self, batch: dict[str, Any] | None = None) -> list[int]:
        extra_ids: list[int] = []
        if batch is not None:
            extra_ids = [int(token_id) for token_id in batch.get("stop_token_ids", []) if token_id is not None]
        eos_token_id = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if eos_token_id is None:
            return list(dict.fromkeys(extra_ids))
        if isinstance(eos_token_id, int):
            return list(dict.fromkeys(extra_ids + [eos_token_id]))
        return list(dict.fromkeys(extra_ids + [int(token_id) for token_id in eos_token_id]))

    def _build_attention_mask(self, input_ids: torch.Tensor, batch: dict[str, Any], model_dtype: torch.dtype) -> torch.Tensor:
        masks = []
        seq_len = input_ids.shape[1]
        for i in range(input_ids.shape[0]):
            answer_query_start = batch["answer_query_spans"][i][0]
            answer_query_span = (answer_query_start, seq_len)
            masks.append(
                build_livr_attention_mask(
                    seq_len=seq_len,
                    image_span=batch["image_spans"][i],
                    prompt_span=batch["prompt_spans"][i],
                    latent_span=batch["latent_spans"][i],
                    answer_span=answer_query_span,
                    stage=self.stage,
                    device=input_ids.device,
                    dtype=model_dtype if model_dtype.is_floating_point else torch.float32,
                )
            )
        return torch.cat(masks, dim=0)

    def forward(self, batch: dict[str, Any]) -> Any:
        device = self._model_device()
        model_dtype = self._model_dtype()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = self._build_attention_mask(input_ids, batch, model_dtype)

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if "pixel_values" in batch:
            model_inputs["pixel_values"] = batch["pixel_values"].to(device)
        if "image_grid_thw" in batch:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        return self.model(**model_inputs)

    @torch.no_grad()
    def collect_attentions(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...] | None:
        device = self._model_device()
        model_dtype = self._model_dtype()
        input_ids = batch["input_ids"].to(device)
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "output_attentions": True,
            "return_dict": True,
            "use_cache": False,
        }
        if self.stage in {"sft", "direct_sft", "stage2", "livr_stage2"}:
            model_inputs["attention_mask"] = batch["attention_mask_2d"].to(device)
        else:
            model_inputs["attention_mask"] = self._build_attention_mask(input_ids, batch, model_dtype)
        if "pixel_values" in batch:
            model_inputs["pixel_values"] = batch["pixel_values"].to(device)
        if "image_grid_thw" in batch:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        outputs = self.model(**model_inputs)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            return None
        return tuple(att.detach().to(device="cpu", dtype=torch.float32) for att in attentions)

    @torch.no_grad()
    def generate(self, batch: dict[str, Any], max_new_tokens: int) -> torch.Tensor:
        device = self._model_device()
        model_dtype = self._model_dtype()
        input_ids = batch["input_ids"].to(device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
        }
        if "pixel_values" in batch:
            kwargs["pixel_values"] = batch["pixel_values"].to(device)
        if "image_grid_thw" in batch:
            kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

        eos_token_ids = self._eos_token_ids(batch)

        if self.stage in {"sft", "direct_sft", "stage2", "livr_stage2"}:
            kwargs["attention_mask"] = batch["attention_mask_2d"].to(device)
            if eos_token_ids:
                kwargs["eos_token_id"] = eos_token_ids
            return self.model.generate(**kwargs)

        fill_token_id = eos_token_ids[0] if eos_token_ids else 0
        generated = input_ids
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            attention_mask = self._build_attention_mask(generated, batch, model_dtype)
            step_inputs: dict[str, Any] = {
                "input_ids": generated,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if "pixel_values" in kwargs:
                step_inputs["pixel_values"] = kwargs["pixel_values"]
            if "image_grid_thw" in kwargs:
                step_inputs["image_grid_thw"] = kwargs["image_grid_thw"]
            outputs = self.model(**step_inputs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, fill_token_id), next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if eos_token_ids:
                for eos_token_id in eos_token_ids:
                    finished |= next_token.eq(eos_token_id)
                if finished.all():
                    break
        return generated
