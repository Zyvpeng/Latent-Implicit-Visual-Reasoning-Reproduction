from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_qwen3vl_vision_batch(
    pixel_values: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    where: str,
) -> None:
    if pixel_values is None or image_grid_thw is None:
        return
    if pixel_values.ndim != 2:
        raise ValueError(f"{where}: expected pixel_values to have shape (num_visual_tokens, hidden_dim), got {tuple(pixel_values.shape)}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"{where}: expected image_grid_thw to have shape (batch, 3), got {tuple(image_grid_thw.shape)}")
    expected_tokens = int(image_grid_thw.to(dtype=torch.long).prod(dim=1).sum().item())
    actual_tokens = int(pixel_values.shape[0])
    if actual_tokens != expected_tokens:
        raise ValueError(
            f"{where}: pixel_values first dimension should equal total visual tokens from image_grid_thw; "
            f"got pixel_values.shape[0]={actual_tokens}, expected={expected_tokens}"
        )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def build_localization_prompt(raw_prompt: str) -> str:
    return raw_prompt.strip()


def normalize_mcq_prediction(text: str) -> str:
    normalized = text.strip().upper()
    aliases = {
        "(A)": "A",
        "(B)": "B",
        "(C)": "C",
        "(D)": "D",
        "BOX A": "A",
        "BOX B": "B",
        "BOX C": "C",
        "BOX D": "D",
        "POINT A": "A",
        "POINT B": "B",
        "POINT C": "C",
        "POINT D": "D",
        "OPTION A": "A",
        "OPTION B": "B",
        "OPTION C": "C",
        "OPTION D": "D",
    }
    if normalized in {"A", "B", "C", "D"}:
        return normalized
    if normalized in aliases:
        return aliases[normalized]
    for letter in ["A", "B", "C", "D"]:
        if letter in normalized:
            return letter
    return normalized


def normalize_count_prediction(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"\d+", stripped)
    if match:
        return str(int(match.group(0)))
    word_to_num = {
        "ZERO": "0",
        "ONE": "1",
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
        "TEN": "10",
    }
    upper = stripped.upper()
    for word, num in word_to_num.items():
        if re.search(rf"\b{re.escape(word)}\b", upper):
            return num
    return stripped


def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            print(f"  {name}: {count}")
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable} / {total} ({pct:.4f}%)")
