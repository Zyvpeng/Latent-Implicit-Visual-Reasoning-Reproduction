from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from livr.latent_tokens import latent_token_text
from livr.utils import build_localization_prompt, load_jsonl, validate_qwen3vl_vision_batch


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_END_MARKER = "<|im_end|>\n"


@dataclass
class EncodedSample:
    example_id: str
    prompt: str
    target: str
    input_ids: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    attention_mask_2d: torch.Tensor
    image_span: tuple[int, int]
    prompt_span: tuple[int, int]
    latent_span: tuple[int, int]
    answer_query_span: tuple[int, int]
    answer_span: tuple[int, int]
    has_latents: bool


class LIVRJsonlDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows = load_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def load_image(path: str) -> Image.Image:
    candidates = [
        Path(path),
        REPO_ROOT / path,
        REPO_ROOT / "data" / path,
        REPO_ROOT / "data" / "pixmo_count" / Path(path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            image = Image.open(candidate)
            return image.convert("RGB")
    raise FileNotFoundError(f"Could not resolve image path: {path}")


def build_counting_prompt(object_name: str) -> str:
    return f"How many {object_name} are there in this image?"


def build_labels(input_ids: torch.Tensor, answer_span: tuple[int, int], ignore_index: int = -100) -> torch.Tensor:
    labels = torch.full_like(input_ids, ignore_index)
    start, end = answer_span
    labels[start:end] = input_ids[start:end]
    return labels


def find_contiguous_span(indices: torch.Tensor, name: str) -> tuple[int, int]:
    if indices.numel() == 0:
        raise ValueError(f"Could not locate span for {name}.")
    return int(indices.min().item()), int(indices.max().item()) + 1


class LIVRBatchBuilder:
    def __init__(
        self,
        processor,
        tokenizer,
        latent_tokens: Sequence[str],
        latent_token_ids: Sequence[int],
        max_length: int,
        label_assistant_end: bool = True,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.latent_tokens = list(latent_tokens)
        self.latent_token_ids = list(latent_token_ids)
        self.max_length = max_length
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.label_assistant_end = label_assistant_end
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

    def _build_task_prompt(self, row: dict[str, Any]) -> str:
        prompt = str(row.get("prompt", "")).strip()
        if prompt:
            return prompt
        task = row.get("task", "")
        if task == "counting":
            object_name = row.get("object_name", "objects")
            return build_counting_prompt(object_name)
        return build_localization_prompt(prompt)

    def _render_assistant_generation_prefix(self, prompt: str) -> str:
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

    def _assistant_prefix_text(self, include_latents: bool) -> str:
        if not include_latents:
            return ""
        return latent_token_text(self.latent_tokens) + "\n"

    def _candidate_max_lengths(self) -> list[int]:
        candidates = [
            self.max_length,
            max(self.max_length, 4096),
            max(self.max_length, 8192),
        ]
        deduped: list[int] = []
        for value in candidates:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _encode_with_retry(self, text: str, image: Image.Image) -> dict[str, torch.Tensor]:
        last_error: Exception | None = None
        for max_length in self._candidate_max_lengths():
            try:
                processor_kwargs = {
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
                    processor_kwargs["images_kwargs"] = images_kwargs
                return self.processor(**processor_kwargs)
            except ValueError as exc:
                if "Mismatch in `image` token count" not in str(exc):
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to encode example.")

    def encode_example(self, row: dict[str, Any], include_latents: bool, with_answer: bool) -> EncodedSample:
        image_path = row["images"][0]
        image = load_image(image_path)
        task_prompt = self._build_task_prompt(row)
        assistant_prompt = self._render_assistant_generation_prefix(task_prompt)
        assistant_prefix = self._assistant_prefix_text(include_latents)
        prefix_text = assistant_prompt + assistant_prefix
        answer_only_text = prefix_text + row["target"]
        full_text = answer_only_text + ASSISTANT_END_MARKER
        text = full_text if with_answer else prefix_text
        encoded = self._encode_with_retry(text, image)
        prefix_encoded = self._encode_with_retry(prefix_text, image)
        answer_only_encoded = self._encode_with_retry(answer_only_text, image) if with_answer else prefix_encoded

        input_ids = encoded["input_ids"][0]
        prefix_input_ids = prefix_encoded["input_ids"][0]
        answer_only_input_ids = answer_only_encoded["input_ids"][0]
        answer_start = len(prefix_input_ids)
        answer_end = len(answer_only_input_ids)
        label_end = len(input_ids) if (with_answer and self.label_assistant_end) else answer_end

        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=False).flatten()
        image_span = find_contiguous_span(image_positions, "image")

        if include_latents:
            latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for latent_id in self.latent_token_ids:
                latent_mask |= input_ids == latent_id
            latent_positions = latent_mask.nonzero(as_tuple=False).flatten()
            latent_span = find_contiguous_span(latent_positions, "latent")
        else:
            latent_span = (answer_start, answer_start)

        prompt_span = (image_span[1], latent_span[0] if include_latents else answer_start)
        answer_query_start = latent_span[1] if include_latents else prompt_span[1]
        answer_query_span = (answer_query_start, label_end if with_answer else answer_end)
        answer_span = (answer_start, label_end)
        labels = build_labels(input_ids=input_ids, answer_span=answer_span) if with_answer else torch.full_like(input_ids, -100)
        attention_mask_2d = torch.ones_like(input_ids)

        return EncodedSample(
            example_id=row["id"],
            prompt=task_prompt,
            target=row["target"],
            input_ids=input_ids,
            labels=labels,
            pixel_values=encoded.get("pixel_values", None) if encoded.get("pixel_values", None) is not None else None,
            image_grid_thw=encoded.get("image_grid_thw", None)[0] if encoded.get("image_grid_thw", None) is not None else None,
            attention_mask_2d=attention_mask_2d,
            image_span=image_span,
            prompt_span=prompt_span,
            latent_span=latent_span,
            answer_query_span=answer_query_span,
            answer_span=answer_span,
            has_latents=include_latents,
        )

    def collate_train(self, rows: list[dict[str, Any]], include_latents: bool) -> dict[str, Any]:
        encoded_rows = [self.encode_example(row, include_latents=include_latents, with_answer=True) for row in rows]
        return self._pack_batch(encoded_rows)

    def collate_eval(self, rows: list[dict[str, Any]], include_latents: bool) -> dict[str, Any]:
        encoded_rows = [self.encode_example(row, include_latents=include_latents, with_answer=False) for row in rows]
        return self._pack_batch(encoded_rows)

    def _pack_batch(self, rows: list[EncodedSample]) -> dict[str, Any]:
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
            attention_mask.append(torch.nn.functional.pad(row.attention_mask_2d, (0, pad_len), value=0))

        batch: dict[str, Any] = {
            "ids": [row.example_id for row in rows],
            "prompts": [row.prompt for row in rows],
            "targets": [row.target for row in rows],
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask_2d": torch.stack(attention_mask, dim=0),
            "image_spans": [row.image_span for row in rows],
            "prompt_spans": [row.prompt_span for row in rows],
            "latent_spans": [row.latent_span for row in rows],
            "answer_query_spans": [row.answer_query_span for row in rows],
            "answer_spans": [row.answer_span for row in rows],
            "has_latents": [row.has_latents for row in rows],
            "stop_token_ids": [token_id for token_id in [self.im_end_token_id, self.tokenizer.eos_token_id] if token_id is not None],
        }

        if rows[0].pixel_values is not None:
            batch["pixel_values"] = torch.cat([row.pixel_values for row in rows], dim=0)
        if rows[0].image_grid_thw is not None:
            batch["image_grid_thw"] = torch.stack([row.image_grid_thw for row in rows], dim=0)
        validate_qwen3vl_vision_batch(
            batch.get("pixel_values"),
            batch.get("image_grid_thw"),
            where="LIVRBatchBuilder._pack_batch",
        )
        return batch


def decode_debug_example(tokenizer, batch: dict[str, Any], index: int = 0) -> str:
    ids = batch["input_ids"][index]
    return tokenizer.decode(ids, skip_special_tokens=False)
