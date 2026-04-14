# LIVR Reproduction on Qwen3-VL

非官方复现项目：本仓库是对论文 **Latent Implicit Visual Reasoning (LIVR)** 的工程复现，不是论文作者官方代码。  
Unofficial reproduction: this repository is an engineering reproduction of **Latent Implicit Visual Reasoning (LIVR)**, not the official implementation from the paper authors.

当前主线基座模型为 `Qwen3-VL-4B-Instruct`，当前重点任务为 `Counting / PixMo-Count`。  
The current main backbone is `Qwen3-VL-4B-Instruct`, and the current primary task is `Counting / PixMo-Count`.

## 项目目标 | Goal

- 用尽量轻量、可调试的 PyTorch + Hugging Face + PEFT 代码复现 LIVR 核心机制。  
  Reproduce the core LIVR mechanism with a lightweight and debuggable PyTorch + Hugging Face + PEFT codebase.
- 优先保证训练协议、latent token 机制、mask 逻辑和 SFT 范式正确。  
  Prioritize correct training protocol, latent-token behavior, mask logic, and SFT protocol.
- 先在单任务 counting 上跑通，再扩展到其他任务。  
  First make single-task counting work end-to-end, then extend to other tasks.

## 当前范围 | Current Scope

- 基座模型：`Qwen3-VL-4B-Instruct`  
  Base model: `Qwen3-VL-4B-Instruct`
- 当前主任务：`Counting / PixMo-Count`  
  Current main task: `Counting / PixMo-Count`
- 训练模式：
  - `direct_sft`
  - `livr_stage1`
  - `livr_stage2`
- 评测入口：
  - 原始官方风格 Qwen3-VL baseline
  - 官方风格 SFT checkpoint
  - LIVR Stage 1 checkpoint
  - LIVR Stage 2 checkpoint

## 与论文对齐的核心点 | Paper-Faithful Core

已实现的 LIVR 核心机制：  
Implemented LIVR core mechanisms:

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
4. loss 在 assistant answer span 上计算，并监督结束标记 `<|im_end|>`。  
   Loss is computed on the assistant answer span, including the `<|im_end|>` end marker.
5. 冻结 vision encoder 和 multimodal projector。  
   The vision encoder and multimodal projector are frozen.
6. LoRA 只打在 language backbone 的 attention + MLP 模块上。  
   LoRA is applied only to language-backbone attention and MLP modules.
7. 只训练 latent token 对应的 embedding rows。  
   Only the embedding rows corresponding to latent tokens are trainable.
8. 默认 latent 配置为 `K=16`，非共享 latent tokens：`<livr_0> ... <livr_15>`。  
   The default latent setup is `K=16` with unshared latent tokens: `<livr_0> ... <livr_15>`.

## SFT 范式 | SFT Protocol

当前 SFT 训练和推理已经拆成独立的 official-style 路径；LIVR Stage 1/2 继续使用通用 LIVR 路径。  
SFT training and inference now use a dedicated official-style path; LIVR Stage 1/2 continue to use the shared LIVR path.

- 不额外注入自定义 system prompt。  
  No extra custom system prompt is injected.
- 训练输入和推理输入保持前缀一致。  
  Training and inference share the same prompt prefix.
- 训练监督 assistant answer 和 `<|im_end|>`。  
  Training supervises both the assistant answer and `<|im_end|>`.
- `direct_sft` 不使用 latent tokens，并走标准 2D `attention_mask` + `model.generate(...)`。  
  `direct_sft` does not use latent tokens, and uses standard 2D `attention_mask` + `model.generate(...)`.
- `livr_stage1 / livr_stage2` 使用 `<livr_0> ... <livr_15>`，且 latent token 之间不插入空格。  
  `livr_stage1 / livr_stage2` use `<livr_0> ... <livr_15>` with no spaces inserted between latent tokens.

## 目录结构 | File Tree

```text
LIVR/
├── README.md
├── requirements.txt
├── 2512.21218v1.pdf
├── configs/
│   ├── counting_qwen3vl_sft.yaml
│   ├── counting_qwen3vl_livr_stage1.yaml
│   ├── counting_qwen3vl_livr_stage2.yaml
│   ├── localization_qwen3vl_sft.yaml
│   └── localization_qwen3vl_livr.yaml
├── data/
│   ├── pixmo_count_official/        # 官方 metadata + 原始图片下载目录
│   ├── pixmo_count_livr/            # 当前默认 counting split
│   ├── pixmo_count_livr_paper/      # 预留：更接近论文的去重 split
│   ├── localization_train.jsonl
│   └── localization_val.jsonl
├── livr/
│   ├── attention_mask.py
│   ├── build_pixmo_count_livr_split.py
│   ├── build_pixmo_count_livr_paper_split.py
│   ├── data.py
│   ├── eval.py
│   ├── eval_qwen3vl_base_official.py
│   ├── eval_qwen3vl_sft_official.py
│   ├── latent_tokens.py
│   ├── model.py
│   ├── prepare_pixmo_count.py
│   ├── sft_official.py
│   ├── train.py
│   └── utils.py
├── scripts/
│   ├── build_pixmo_count_clipfar_split_torchrun.sh
│   ├── build_pixmo_count_livr_paper_split_torchrun.sh
│   ├── train_counting_qwen3vl_sft_torchrun.sh
│   ├── train_counting_qwen3vl_livr_stage1_torchrun.sh
│   ├── train_counting_qwen3vl_livr_stage2_torchrun.sh
│   ├── eval_counting_qwen3vl_base.sh
│   ├── eval_counting_qwen3vl_base_official.sh
│   ├── eval_counting_qwen3vl_sft.sh
│   ├── eval_counting_qwen3vl_stage1.sh
│   └── eval_counting_qwen3vl_stage2.sh
└── tests/
    ├── test_mask.py
    ├── test_loss.py
    └── test_latent_rows_trainable.py
```

## 环境安装 | Environment

### Python 依赖

```bash
pip install -r requirements.txt
```

### 与当前 `agent` 环境对齐的 PyTorch 安装

当前对齐版本是 `torch 2.4.0 + CUDA 12.1`。  
The currently aligned version is `torch 2.4.0 + CUDA 12.1`.

```bash
/home/ypzheng/miniconda3/bin/conda create -n livr python=3.10 -y
/home/ypzheng/miniconda3/bin/conda install -n livr \
  pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 \
  -c pytorch -c nvidia -y
```

## 数据目录与当前默认 split | Data Layout and Current Default Split

### 1. 官方下载目录

`data/pixmo_count_official/` 用于保存官方 metadata 和按 URL 下载的图片。  
`data/pixmo_count_official/` stores the official metadata and URL-downloaded images.

### 2. 当前默认训练/评测 split

当前 counting 配置默认使用：  
The current counting configs use:

- `data/pixmo_count_livr_paper/counting_train.jsonl`
- `data/pixmo_count_livr_paper/counting_val.jsonl`
- `data/pixmo_count_livr_paper/counting_test.jsonl`

当前这套 split 的特点：  
Characteristics of the current split:

- `train = 1000`
- `val = 524`
- `test = 527`
- train 在 `count ∈ [2, 10]` 上近均衡  
  the train split is nearly balanced over `count ∈ [2, 10]`
- train 使用 paper-style 的按 count 抽样，不再额外做 label 均衡  
  the train split uses paper-style count-balanced sampling, without extra label balancing

### 3. 更接近论文的去重 split

`livr/build_pixmo_count_livr_paper_split.py` 提供了更接近论文描述的脚本：  
`livr/build_pixmo_count_livr_paper_split.py` provides a more paper-aligned split builder:

- 读取官方 `train / validation / test`
- 只保留本地实际下载成功的样本
- 对 `train + validation` 和官方 `test` 做近重复过滤
- 过滤手段包括：CLIP embeddings、pHash、SSIM
- 从过滤后的 `train` 中抽取 `1000` 条训练样本

运行示例：  
Example:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
python -m livr.build_pixmo_count_livr_paper_split \
  --input-dir data/pixmo_count_official \
  --output-dir data/pixmo_count_livr_paper \
  --train-size 1000 \
  --seed 42
```

### 4. 用 `cuda:8,9` 多卡提取 CLIP embedding

现在 `build_pixmo_count_livr_paper_split.py` 和 `build_pixmo_count_clipfar_split.py` 都支持：

- `torchrun` 多进程并行提取 CLIP embedding
- embedding 缓存到 `.pt`
- 下一次相同图像列表 + 相同 CLIP 模型时直接复用缓存

#### 4.1 构建 `clipfar` split

```bash
cd /home/ypzheng/latent_reasoning/LIVR
conda activate agent
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29631 \
bash scripts/build_pixmo_count_clipfar_split_torchrun.sh
```

这条脚本默认等价于：

```bash
torchrun \
  --nproc_per_node=2 \
  --master_port=29631 \
  -m livr.build_pixmo_count_clipfar_split \
  --input-dir data/pixmo_count_official \
  --output-dir data/pixmo_count_clipfar \
  --clip-cache-dir /tmp/livr_clip_cache/pixmo_count_clipfar \
  --train-size 1000 \
  --seed 42 \
  --min-count 2 \
  --max-count 10 \
  --remove-near-duplicates
```

#### 4.2 重建论文风格 split

```bash
cd /home/ypzheng/latent_reasoning/LIVR
conda activate agent
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29632 \
bash scripts/build_pixmo_count_livr_paper_split_torchrun.sh
```

这条脚本默认等价于：

```bash
torchrun \
  --nproc_per_node=2 \
  --master_port=29632 \
  -m livr.build_pixmo_count_livr_paper_split \
  --input-dir data/pixmo_count_official \
  --output-dir data/pixmo_count_livr_paper \
  --clip-cache-dir /tmp/livr_clip_cache/pixmo_count_livr_paper \
  --train-size 1000 \
  --seed 42 \
  --min-count 2 \
  --max-count 10
```

#### 4.3 这些参数分别是什么意思

- `CUDA_VISIBLE_DEVICES=8,9`
  只让当前命令看到物理 GPU `8` 和 `9`。脚本内部的两个进程会分别使用可见设备里的 `cuda:0` 和 `cuda:1`，也就是物理 `8` 和 `9`。
- `MASTER_PORT=29631` 或 `29632`
  给 `torchrun` 的分布式通信指定端口。同一台机器上不要和别的 `torchrun` 任务冲突；冲突就换一个没被占用的端口。
- `--nproc_per_node=2`
  启动 `2` 个进程，对应 `2` 张可见 GPU。如果你只给一张卡，就把它改成 `1`。
- `--input-dir data/pixmo_count_official`
  官方 metadata 和图片所在目录。这里必须包含：
  `train_metadata.jsonl`、`validation_metadata.jsonl`、`test_metadata.jsonl` 和 `images/`。
- `--output-dir`
  最终 split 的输出目录。
  `clipfar` 默认写到 `data/pixmo_count_clipfar`；
  paper split 默认写到 `data/pixmo_count_livr_paper`。
- `--clip-cache-dir`
  embedding 缓存目录。第一次会写入 `.pt`，后面只要图像列表和模型名没变，就直接读缓存，不再重提。
- `--train-size 1000`
  最终训练集大小。
- `--seed 42`
  采样随机种子。
- `--min-count 2 --max-count 10`
  只从计数区间 `[2, 10]` 里选训练样本。
- `--remove-near-duplicates`
  只在 `clipfar` 脚本里默认打开。会先做一轮 paper-style `CLIP + pHash + SSIM` 去重，再从剩余 train 里挑和 test 最远的样本。

#### 4.4 如果你想改默认值

脚本里这几个环境变量都可以直接覆写：

```bash
cd /home/ypzheng/latent_reasoning/LIVR
conda activate agent
CUDA_VISIBLE_DEVICES=8,9 \
MASTER_PORT=29641 \
NPROC_PER_NODE=2 \
CACHE_DIR=/tmp/my_clip_cache \
bash scripts/build_pixmo_count_clipfar_split_torchrun.sh
```

含义：

- `NPROC_PER_NODE`
  覆盖脚本里的 `--nproc_per_node`
- `MASTER_PORT`
  覆盖脚本里的分布式端口
- `CACHE_DIR`
  覆盖脚本里的 `--clip-cache-dir`

## 训练配置 | Default Counting Config

当前 counting 配置默认包含：  
Current counting configs include:

- `max_length: 4096`
- `image_min_pixels: 3136`
- `image_max_pixels: 1048576`
- `per_device_batch_size: 1`
- `grad_accum_steps: 8`
- `bf16: true`
- `gradient_checkpointing: true`
- `train_val_subset_size: null`
- `compute_val_accuracy: true`

说明：  
Notes:

- `image_min_pixels / image_max_pixels` 用于控制 Qwen3-VL 视觉 token 长度，避免图像 token 过长触发 processor truncation mismatch。  
  `image_min_pixels / image_max_pixels` are used to cap Qwen3-VL visual token length and avoid processor truncation mismatch.
- 论文对齐配置下，训练时每个 epoch 会在完整 validation set 上计算 `val_loss` 和 `val_acc`，并逐条打印预测。  
  Under the paper-aligned setup, each epoch evaluates `val_loss` and `val_acc` on the full validation set and prints each prediction.

## 训练 | Training

### Direct SFT

`direct_sft` 现在使用独立的 official-style SFT 训练入口，不再和 `base eval` 或 LIVR 通用 wrapper 共用一条训练实现。  
`direct_sft` now uses a dedicated official-style SFT training path instead of sharing the same training implementation as `base eval` or the generic LIVR wrapper.

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29511 \
bash scripts/train_counting_qwen3vl_sft_torchrun.sh
```

### LIVR Stage 1

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29511 \
bash scripts/train_counting_qwen3vl_livr_stage1_torchrun.sh
```

### LIVR Stage 2

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8,9 MASTER_PORT=29511 \
bash scripts/train_counting_qwen3vl_livr_stage2_torchrun.sh
```

训练 schedule：  
Training schedule:

- single-task setting 使用 `1000` 条训练样本  
  the single-task setting uses `1000` training examples
- `direct_sft` 默认训练 `10` 个 epochs  
  `direct_sft` runs for `10` epochs by default
- `livr_stage1` 默认训练 `4` 个 epochs  
  `livr_stage1` runs for `4` epochs by default
- `livr_stage2` 默认训练 `6` 个 epochs  
  `livr_stage2` runs for `6` epochs by default

## 评测 | Evaluation

### 1. 原始 Qwen3-VL baseline

这是完全绕开 LIVR wrapper 的官方风格 baseline。  
This is the official-style baseline that bypasses the LIVR wrapper entirely.

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_base_official.sh
```

### 2. 当前通用 base eval

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_base.sh
```

### 3. SFT checkpoint

SFT 评测现在也走独立的 official-style 路径，不再借助通用 `eval.py`。  
SFT evaluation now also uses a dedicated official-style path instead of the shared `eval.py`.

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

### 4. LIVR Stage 1 checkpoint

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 bash scripts/eval_counting_qwen3vl_stage1.sh
```

### 5. LIVR Stage 2 checkpoint

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

固定评测某一轮，例如 `epoch_5`：  
Evaluate a specific epoch, for example `epoch_5`:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_stage2.sh outputs/counting_livr_stage2/epoch_5
```

默认评测会：
- 打印每条样本的 `id / target / pred / raw_pred / correct`
- 显示实时 `tqdm` 进度和累计 `acc`
- 输出 `predictions.jsonl`

By default, evaluation:
- prints `id / target / pred / raw_pred / correct` for each sample
- shows live `tqdm` progress and running `acc`
- writes `predictions.jsonl`

## Stage 2 可选 attention 导出 | Optional Stage 2 Attention Export

`stage2` 评测支持可选地实时保存每个 latent token 对视觉 token 的 attention map。  
Stage-2 evaluation optionally supports real-time export of each latent token's attention map over visual tokens.

默认关闭。开启方式：  
It is disabled by default. Enable it with:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
SAVE_LATENT_ATTN=1 CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_stage2.sh
```

指定保存目录：  
Specify a custom save directory:

```bash
cd /home/ypzheng/latent_reasoning/LIVR
SAVE_LATENT_ATTN=1 LATENT_ATTN_DIR=outputs/counting_livr_stage2_eval/my_latent_attn \
CUDA_VISIBLE_DEVICES=8 \
bash scripts/eval_counting_qwen3vl_stage2.sh
```

保存内容包括：  
Saved payload includes:

- `latent_token_ids`
- `latent_token_text`
- `image_span`
- `latent_span`
- `image_grid_thw`
- `last_layer_mean_heads`
- `avg_layers_mean_heads`
- 以及可恢复时的 2D attention maps  
  and 2D attention maps when recoverable

## 测试 | Tests

```bash
cd /home/ypzheng/latent_reasoning/LIVR
python -m pytest tests
```

当前测试覆盖：
- `test_mask.py`
- `test_loss.py`
- `test_latent_rows_trainable.py`

## 当前注意事项 | Current Notes

- 旧 checkpoint 中，凡是在修复 `pixel_values` 维度 bug 之前训练得到的结果都不可信。  
  Any checkpoint trained before the `pixel_values` shape fix should be treated as invalid.
- 当前默认 counting split 是 `data/pixmo_count_livr_paper/`，不是最早的旧 `pixmo_count/` 目录。  
  The current default counting split is `data/pixmo_count_livr_paper/`, not the older `pixmo_count/` directory.
- `pixmo-count` 官方有一部分 URL 已失效，因此最终可用的 `val/test` 样本数会小于官方 metadata 总数。  
  Some official `pixmo-count` URLs are dead, so the final usable `val/test` sizes are smaller than the raw metadata counts.
- 当前 paper split 目录已经包含自包含的本地图片副本：`data/pixmo_count_livr_paper/images/`。  
  The current paper split directory already contains a self-contained local image copy under `data/pixmo_count_livr_paper/images/`.
- 多卡训练时，CPU 侧图像预处理仍可能成为瓶颈。  
  CPU-side image preprocessing can still be a bottleneck during multi-GPU training.

## 可配置超参数 | Configurable Hyperparameters

以下参数保持 YAML 可配置：  
The following remain configurable in YAML:

- `model_name`
- `max_length`
- `image_min_pixels`
- `image_max_pixels`
- `num_latents`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `learning_rate`
- `weight_decay`
- `per_device_batch_size`
- `num_workers`
- `grad_accum_steps`
- `num_epochs`
- `warmup_ratio`
- `stage`
- `train_file`
- `val_file`
- `test_file`
- `output_dir`
- `bf16`
- `gradient_checkpointing`
- `eval_max_new_tokens`
- `compute_val_accuracy`
- `train_val_subset_size`
- `label_assistant_end`
