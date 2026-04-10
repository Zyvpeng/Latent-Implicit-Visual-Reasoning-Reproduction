from __future__ import annotations

from typing import Tuple

import torch


Span = Tuple[int, int]


def span_indices(span: Span) -> range:
    return range(span[0], span[1])


def build_standard_causal_mask(
    seq_len: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    blocked = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask[:, :, blocked] = torch.finfo(dtype).min
    return mask


def _block_span_to_span(mask: torch.Tensor, query_span: Span, key_span: Span) -> None:
    q0, q1 = query_span
    k0, k1 = key_span
    mask[:, :, q0:q1, k0:k1] = torch.finfo(mask.dtype).min


def build_livr_attention_mask(
    seq_len: int,
    image_span: Span,
    prompt_span: Span,
    latent_span: Span,
    answer_span: Span,
    stage: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = build_standard_causal_mask(seq_len=seq_len, device=device, dtype=dtype)

    if stage in {"sft", "direct_sft", "stage2", "livr_stage2"}:
        return mask

    if stage not in {"stage1", "livr_stage1"}:
        raise ValueError(f"Unsupported stage for attention mask: {stage}")

    _block_span_to_span(mask, prompt_span, image_span)
    _block_span_to_span(mask, answer_span, image_span)

    # Keep causal semantics intact for every query by re-applying the standard future block.
    future_block = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask[:, :, future_block] = torch.finfo(dtype).min
    return mask
