# Beyond Exact Match: Task-Aware GRPO for Cross-Domain PCBA Visual Question Answering

Official code for our **ACM MM 2026** paper on visual question answering for Printed Circuit Board Assembly (PCBA) inspection.


**[Paper](https://arxiv.org/abs/2609.21276)** 

---

## Project Structure

```text
BeyondExactMatch/
├── sft_dataset/                     # SFT training data, test prompts, and distillation artifacts
│   ├── train_standard.jsonl
│   ├── train_realworld.jsonl
│   ├── train_pcbbench.jsonl
│   ├── train_fix_data.jsonl          # Merged SFT training set
│   └── test.jsonl
├── grpo_dataset/                    # Converted training data with reward metadata
│   ├── grpo_fromsft.jsonl
│   └── grpo_nothink.jsonl            # Prompts with Explanation / Answer instructions
├── pcb_bench_additional_data/       # Auxiliary PCB-Bench annotations
├── ms_swift_sft_code/               # Raw annotation conversion and SFT dataset merging
├── sft_think_data/                  # Teacher-model request construction
├── ms_swift_grpo_code/              # SFT-to-GRPO data conversion utilities
├── grpo_plugin/
│   └── pcba_grpo_plugin.py          # Task-aware reward functions for ms-swift
├── infer/
│   ├── infer_sft_system_answer_only.py
│   ├── infer_test_batch_27b_grpo.py
│   └── infer_vote.py                # Greedy decoding and self-consistency voting
├── post_process/
│   └── fix_res_test.py              # Single-example counting verification
├── submit_code/                     # Prediction-to-CSV conversion utilities
├── qwen35_27B_ori_r16_lr5e5_ep2_cp1258_result/
│   └── result.jsonl                 # Included prediction artifact
├── qwen35_27B_ori_r16_lr5e5_ep2_cp1258_submit/
│   ├── submission.csv
│   └── submission.zip
└── README.md
```

---

## Environment Setup

### Requirements

- Python 3.10 or later
- PyTorch with CUDA support
- [ms-swift](https://github.com/modelscope/ms-swift) with Qwen3.5 support, plus PEFT and Transformers
- NVIDIA GPUs with sufficient memory for the 27B model; the paper uses **4 H100 GPUs** for experiments

### Install

From the project root, create an environment and install a CUDA-compatible PyTorch build, then install the framework dependencies:

```bash
conda create -n beyondexactmatch python=3.10 -y
conda activate beyondexactmatch

# Install PyTorch for your CUDA environment first.
pip install -U ms-swift peft
```

The original package versions are not recorded in this release. Follow the [ms-swift installation guide](https://github.com/modelscope/ms-swift#%EF%B8%8F-installation) for compatible training dependencies. The command examples below use Bash syntax on a Linux GPU machine.

### Configure Local Paths

The scripts and JSONL files retain paths from the original training server, including `/data/lijia/PCBA_2026/`. Before running them:

1. Obtain the PCBA challenge images and auxiliary PCB-Bench images, and update the `images` entries in the JSONL files to your local locations.
2. Set the input and output paths in each script to the corresponding files under this project.
3. Set the model and adapter paths, `CUDA_VISIBLE_DEVICES`, and batch size for your machine.

**Raw images, base model weights, merged SFT weights, and LoRA checkpoints are not included in this directory.** The result folders contain predictions and submission files.

---

## Data Preparation

### 1. SFT Data

The included training files use the ms-swift multimodal conversation format with `messages`, `images`, `task`, and `answer_type` fields.

| File | Samples | Description |
|------|--------:|-------------|
| `sft_dataset/train_standard.jsonl` | 1,600 | Standards-derived training samples |
| `sft_dataset/train_realworld.jsonl` | 6,600 | Real-world training samples |
| `sft_dataset/train_pcbbench.jsonl` | 2,193 | Auxiliary PCB-Bench training samples |
| `sft_dataset/train_fix_data.jsonl` | 10,393 | Merged SFT training set |
| `sft_dataset/test.jsonl` | 8,200 | Test prompts without reference answers |

To rebuild these files from the source annotations, configure the input, output, and image paths in the following scripts, then run:

```bash
python ms_swift_sft_code/convert_sft_system_answer_only.py
python ms_swift_sft_code/merge_datasets.py
python ms_swift_sft_code/convert_test_system_answer_only.py
```

For teacher-model request construction, configure and run `sft_think_data/make_pcba_distill_requests.py`. This script prepares requests; teacher inference and reasoning-trace verification are separate steps. The included `distill_pcba_raw_result.jsonl` contains 2,815 records, a subset of the 10,393 requests.

### 2. GRPO Data

Both included GRPO files contain **10,393 samples** with fields such as `solution`, `answer_type`, `choices`, and `solution_text`. `grpo_nothink.jsonl` includes explicit `Explanation:` / `Answer:` output instructions.

Use the included GRPO files directly after updating their image paths. To rebuild GRPO data, first prepare structured SFT responses containing an `Answer:` field, optionally preceded by `Explanation:`. The included answer-only `sft_dataset/train_fix_data.jsonl` cannot be passed directly to these converters.

For your own structured SFT file, replace the input placeholder below and run:

```bash
python ms_swift_grpo_code/convert_sft_to_grpo_ori.py \
    --input /path/to/structured_sft.jsonl \
    --output grpo_dataset/grpo_converted.jsonl \
    --failed-output grpo_dataset/conversion_failures.jsonl
```

This converter removes recognized XML-format instruction suffixes but does not automatically append the reward plugin's full output instructions. Before training, align the prompts with the format below, retain the reward metadata columns, and check the reported converted/failed counts. `convert_grpo_data_forsft.py` provides an alternative converter that appends XML reasoning and answer instructions.

---

## Training

Training follows **SFT → Task-Aware GRPO**. Use the ms-swift training CLI with the prepared datasets and your local model paths. This release includes the data utilities and reward plugin; the original training launch scripts are not included.

### Paper Settings

| Setting | SFT | GRPO |
|---------|-----|------|
| Backbone | Qwen3.5-27B | SFT-initialized Qwen3.5-27B |
| LoRA rank | 16 | 8 |
| LoRA alpha | 32 | 32 |
| Learning rate | `5e-5` | `5e-7` |
| Epochs | 2 | 1 |
| Rollout group size | — | 8 |
| Sampling temperature | — | 1.0 |

These settings are reported in the [paper](https://arxiv.org/html/2609.21276v1#S4.SS2). Configure batch size, sequence length, distributed training, and the rollout backend for your hardware using the [ms-swift documentation](https://github.com/modelscope/ms-swift).

### Task-Aware Rewards

The plugin in `grpo_plugin/pcba_grpo_plugin.py` registers:

| Reward | Behavior |
|--------|----------|
| `pcba_accuracy` | Choice reward: answer correctness (0.70), correct option text (0.20), and answer–explanation consistency (0.10). Counting reward: exact match (1.0), error of one (0.8), error of two (0.6), then decaying credit. |
| `pcba_format` | Checks the explanation field and valid answer format. |
| `pcba_strict_accuracy` | Optional strict diagnostic for choice-based outputs. |

The counting values above describe the active implementation. Text comparisons for choice explanations normalize whitespace and remain case-sensitive.

Add the following arguments to your `swift rlhf --rlhf_type grpo` command to load the plugin:

```bash
--external_plugins grpo_plugin/pcba_grpo_plugin.py \
--reward_funcs pcba_accuracy pcba_format \
--reward_weights 1.0 0.1
```

Use prompts that request this completion format, with a direct uppercase option letter for choice questions or an integer for counting questions:

```text
Explanation: exact selected option text or counted integer

Answer: A
```

For GRPO inference, `infer_test_batch_27b_grpo.py` loads a **merged SFT model** as its base and applies a **GRPO LoRA adapter**. Prepare both artifacts and update their paths before inference.

---

## Inference & Submission

### 1. Configure Inference

Choose the script matching your checkpoint and prompt format:

| Script | Use |
|--------|-----|
| `infer/infer_sft_system_answer_only.py` | SFT answer-only inference; preserves system and user messages. |
| `infer/infer_test_batch_27b_grpo.py` | GRPO batch inference; expects a user-first prompt and writes `qid` / `content` records. |
| `infer/infer_vote.py` | SFT answer-only inference with one greedy pass and eight stochastic samples. |

Set `input_file`, `output_file`, model and adapter paths, and GPU settings inside your chosen script. For GRPO, supply test prompts matching the training format; the included `sft_dataset/test.jsonl` is system-first and should not be passed unchanged to the user-first GRPO script.

After configuration, run the chosen script, for example:

```bash
python infer/infer_vote.py
```

The voting script uses temperature **0.5**, top-p **0.9**, and a confidence threshold of **0.55** over valid sampled answers. Low-confidence disagreement falls back to the greedy answer. Its output includes the selected `answer`, sampled answers, and voting confidence. `post_process/fix_res_test.py` provides a configurable single-example counting verifier; a full multi-model arbitration pipeline is not included.

### 2. Convert Predictions

Use `convert_result_to_system.py` for voting outputs: it prioritizes the selected `answer` field over the greedy `content`. The function can also process the included result artifact without loading a model:

```bash
python - <<'PY'
from submit_code.convert_result_to_system import convert_result_to_submission

convert_result_to_submission(
    "qwen35_27B_ori_r16_lr5e5_ep2_cp1258_result/result.jsonl",
    "outputs/submission.csv",
)
PY
```

For a new inference run, replace the input path with your generated `result.jsonl`. The other converters, `convert_result_to_submission_ori.py` and `convert_result_to_submission_ori_think.py`, extract answers from generated content.

### 3. Output

Submission files use the following CSV format:

```csv
qid,answer
1,A
```

Check that the submission contains all **8,200 test question IDs** and no unresolved `?` answers. A pre-generated CSV and ZIP are available in `qwen35_27B_ori_r16_lr5e5_ep2_cp1258_submit/`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{li2026beyondexactmatch,
  title         = {Beyond Exact Match: Task-Aware GRPO for Cross-Domain PCBA Visual Question Answering},
  author        = {Jia Li and Li Dai and Peng Jia and Zhenzhen Hu and Chee Seng Chan and Bingkun Bao and Richang Hong},
  year          = {2026},
  eprint        = {2609.21276},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.21276}
}
```

