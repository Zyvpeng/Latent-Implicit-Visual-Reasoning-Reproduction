from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def build_latent_token_strings(num_latents: int) -> list[str]:
    return [f"<livr_{idx}>" for idx in range(num_latents)]


@dataclass
class LatentTokenInfo:
    tokens: list[str]
    token_ids: list[int]


def add_latent_tokens(tokenizer, num_latents: int) -> LatentTokenInfo:
    tokens = build_latent_token_strings(num_latents)
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return LatentTokenInfo(tokens=tokens, token_ids=token_ids)


def latent_token_text(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def register_row_mask_hook(module: torch.nn.Module, trainable_rows: Iterable[int]) -> None:
    row_ids = sorted(set(int(row) for row in trainable_rows))
    if not row_ids:
        return

    index = torch.tensor(row_ids, dtype=torch.long)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        local_index = index.to(device=grad.device)
        mask.index_fill_(0, local_index, 1.0)
        return grad * mask

    module.weight.requires_grad_(True)
    module.weight.register_hook(_mask_grad)


class TrainableLatentEmbedding(nn.Module):
    """
    Wraps a frozen embedding table and adds a tiny trainable delta for selected token rows.

    This keeps only the latent token rows trainable without exposing the full embedding matrix
    to the optimizer.
    """

    def __init__(self, base_embedding: nn.Module, token_ids: Sequence[int]) -> None:
        super().__init__()
        if not hasattr(base_embedding, "weight"):
            raise TypeError("base_embedding must expose a weight parameter")

        self.base_embedding = base_embedding
        self.token_ids = [int(token_id) for token_id in token_ids]
        self.token_to_index = {token_id: idx for idx, token_id in enumerate(self.token_ids)}
        self.base_embedding.weight.requires_grad_(False)

        init_rows = self.base_embedding.weight.detach().index_select(
            0, torch.tensor(self.token_ids, dtype=torch.long, device=self.base_embedding.weight.device)
        )
        self.latent_rows = nn.Parameter(init_rows.clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.base_embedding(input_ids)
        if not self.token_ids:
            return embeddings

        updated = embeddings.clone()
        for row_idx, token_id in enumerate(self.token_ids):
            token_mask = input_ids == token_id
            if token_mask.any():
                updated[token_mask] = self.latent_rows[row_idx]
        return updated

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_embedding.weight


def replace_embedding_with_trainable_latents(model, latent_token_ids: Sequence[int]) -> None:
    input_embeddings = model.get_input_embeddings()
    wrapped = TrainableLatentEmbedding(input_embeddings, latent_token_ids)
    model.set_input_embeddings(wrapped)


def mark_only_latent_rows_trainable(model, latent_token_ids: Sequence[int]) -> None:
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad_(False)
    register_row_mask_hook(input_embeddings, latent_token_ids)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings is not input_embeddings:
        output_embeddings.weight.requires_grad_(False)
        register_row_mask_hook(output_embeddings, latent_token_ids)
