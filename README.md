# LIVR Reproduction

非官方复现项目：本仓库是对论文 **Latent Implicit Visual Reasoning (LIVR)** 的工程复现，不是论文作者官方代码。  
Unofficial reproduction: this repository is an engineering reproduction of **Latent Implicit Visual Reasoning (LIVR)**, not the official code from the paper authors.

当前基座模型为 `Qwen3-VL-4B-Instruct`。  
The current base model is `Qwen3-VL-4B-Instruct`.

当前优先支持的任务是 `Counting / PixMo-Count`。  
The current primary supported task is `Counting / PixMo-Count`.

## 项目目标 | Goal

- 用尽量轻量、可调试的方式复现 LIVR 的核心机制。  
  Reproduce the core LIVR mechanism in a lightweight and debuggable way.
- 优先保证训练逻辑、mask 机制、latent token 训练方式和 SFT 范式正确。  
  Prioritize correct training logic, mask behavior, latent-token training, and SFT protocol.
- 先跑通单任务 counting，再扩展到其他任务。  
  First make single-task counting work end-to-end, then extend to more tasks.

## 当前实现范围 | Current Scope

- 基座模型：`Qwen3-VL-4B-Instruct`  
  Base model: `Qwen3-VL-4B-Instruct`
- 任务：`Counting / PixMo-Count`  
  Task: `Counting / PixMo-Count`
- 训练模式：
  - `direct_sft`
  - `livr_stage1`
  - `livr_stage2`

## 与论文对齐的核心点 | Paper-Faithful Core

已实现的核心机制：  
Implemented core mechanisms:

1. latent tokens 追加在输入中，不是自回归生成。  
   Latent tokens are appended to the input and are not autoregressively generated.
2. Stage 1 使用 bottleneck attention mask：  
   Stage 1 uses the bottleneck attention mask:
   - prompt queries 不能看 image keys  
     prompt queries cannot attend to image keys
   - answer queries 不能看 image keys  
     answer queries cannot attend to image keys
   - latent queries 仍可看 image keys  
     latent queries can still attend to image keys
3. Stage 2 恢复标准 causal mask。  
   Stage 2 restores the standard causal mask.
4. loss 只在 assistant answer span 上计算。  
   Loss is computed only on the assistant answer span.
5. 冻结 vision encoder 和 multimodal projector。  
   The vision encoder and multimodal projector are frozen.
6. LoRA 只打在 language backbone 的 attention + MLP 模块上。  
   LoRA is applied only to language-backbone attention and MLP modules.
7. 只训练 latent token 对应的 embedding rows。  
   Only the embedding rows corresponding to latent tokens are trainable.
8. 默认 latent 配置为 `K=16`，非共享 latent tokens：`<livr_0> ... <livr_15>`。  
   The default latent setup is `K=16` with unshared latent tokens: `<livr_0> ... <livr_15>`.

## SFT 范式说明 | SFT Protocol

当前训练和推理都统一使用 `Qwen3-VL` 的 chat template。  
Training and inference now consistently use the `Qwen3-VL` chat template.

- 不再使用额外自定义 system prompt。  
  No extra custom system prompt is injected.
- 训练输入与推理输入保持前缀一致。  
  The training input and inference input share the same prefix format.
- 训练时监督 assistant answer 以及结束标记 `<|im_end|>`。  
  During training, the assistant answer and the `<|im_end|>` end marker are supervised.

## 目录结构 | File Tree

```text
LIVR/
├── README.md
├── requirements.txt
├── configs/
│   ├── counting_qwen3vl_sft.yaml
│   ├── counting_qwen3vl_livr_stage1.yaml
│   ├── counting_qwen3vl_livr_stage2.yaml
│   ├── localization_qwen3vl_sft.yaml
│   └── localization_qwen3vl_livr.yaml
├── data/
│   └── pixmo_count/
├── livr/
│   ├── attention_mask.py
│   ├── data.py
│   ├── eval.py
│   ├── latent_tokens.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── scripts/
│   ├── train_counting_qwen3vl_sft_torchrun.sh
│   ├── train_counting_qwen3vl_livr_stage1_torchrun.sh
│   ├── train_counting_qwen3vl_livr_stage2_torchrun.sh
│   ├── eval_counting_qwen3vl_base.sh
│   ├── eval_counting_qwen3vl_sft.sh
│   └── ...
└── tests/
    ├── test_mask.py
    ├── test_loss.py
    └── test_latent_rows_trainable.py
```

## 环境安装 | Install

```bash
pip install -r requirements.txt
```

## 数据准备 | Data Preparation

当前 counting 实验使用 `PixMo-Count` 整理后的 JSONL 格式数据。  
Current counting experiments use `PixMo-Count` data organized in JSONL format.

示例格式：  
Example format:

```json
{
  "id": "train-0",
  "images": ["pixmo_count/images/train_0.jpg"],
  "prompt": "Count the number of objects in the image. Answer with a single integer only.",
  "target": "4",
  "task": "counting",
  "object_name": "objects"
}
```

## 训练命令 | Training

### 1. Direct SFT

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29511 \
bash scripts/train_counting_qwen3vl_sft_torchrun.sh
```

### 2. LIVR Stage 1

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29511 \
bash scripts/train_counting_qwen3vl_livr_stage1_torchrun.sh
```

### 3. LIVR Stage 2

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29511 \
bash scripts/train_counting_qwen3vl_livr_stage2_torchrun.sh
```

说明：  
Notes:

- single-task setting 使用 `1k` 训练样本。  
  The single-task setting uses `1k` training examples.
- SFT 默认训练 `10` 个 epochs。  
  SFT runs for `10` epochs by default.
- LIVR 默认训练 `4 + 6` 个 epochs。  
  LIVR runs for `4 + 6` epochs by default.
- 当前训练默认会计算 `val_loss` 和 `val_acc`，并保存 `best` checkpoint。  
  Training computes `val_loss` and `val_acc` by default, and saves the `best` checkpoint.

## 评测命令 | Evaluation

### 原始 base model

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_base.sh
```

### SFT checkpoint

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_sft.sh
```

指定 checkpoint：  
Specify a checkpoint:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_sft.sh outputs/counting_sft/best
```

### LIVR Stage 1 checkpoint

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_stage1.sh
```

指定 checkpoint：  
Specify a checkpoint:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_stage1.sh outputs/counting_livr_stage1/best
```

### LIVR Stage 2 checkpoint

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_stage2.sh
```

指定 checkpoint：  
Specify a checkpoint:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_stage2.sh outputs/counting_livr_stage2/best
```

如果你想固定评测某一轮，比如 `epoch_5`：  
If you want to evaluate a specific epoch such as `epoch_5`:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_stage2.sh outputs/counting_livr_stage2/epoch_5
```

当前 `livr.eval` 会输出：  
Current `livr.eval` writes:

- `accuracy=...`
- `predictions.jsonl`

## 当前已知说明 | Current Notes

- 本仓库仍处于持续整理阶段。  
  This repository is still being actively cleaned up.
- counting 是当前主要验证任务，localization 配置仍保留为后续扩展入口。  
  Counting is the main validated task at the moment; localization configs remain as future extensions.
- 若使用 DDP 训练，CPU 图像预处理可能成为瓶颈。  
  With DDP training, CPU-side image preprocessing may become a bottleneck.
- 若多卡训练出现同步超时，可优先增大 DDP timeout、提高 dataloader workers，或预缓存预处理结果。  
  If multi-GPU training hits synchronization timeouts, first increase the DDP timeout, raise dataloader workers, or pre-cache preprocessing results.

## 尚未完全锁死的超参数 | Configurable Hyperparameters

以下参数仍保持 YAML 可配置：  
The following hyperparameters remain configurable in YAML:

- `max_length`
- `num_latents`
- `learning_rate`
- `weight_decay`
- `per_device_batch_size`
- `num_workers`
- `grad_accum_steps`
- `num_epochs`
- `warmup_ratio`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `bf16`
- `gradient_checkpointing`
- `eval_max_new_tokens`
- `compute_val_accuracy`
- `init_checkpoint`

## 声明 | Disclaimer

这是一个非官方复现仓库。  
This is an unofficial reproduction repository.

如果你需要严格对照论文结果，请同时关注：  
If you need strict result matching with the paper, you should additionally check:

- 数据构造是否与论文原始版本完全一致  
  whether the data construction exactly matches the paper
- 训练超参数是否与论文附录完全一致  
  whether the optimization hyperparameters exactly match the appendix
- 评测协议是否与论文原始脚本完全一致  
  whether the evaluation protocol exactly matches the original scripts
