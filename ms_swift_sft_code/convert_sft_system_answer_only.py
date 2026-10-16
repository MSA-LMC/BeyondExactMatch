import json
import os
import re
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = (
    "You are a careful PCB visual inspection assistant. "
    "Inspect the image carefully and answer according to the given question. "
    "For multiple-choice questions, output only one option letter. "
    "For counting questions, output only one integer."
)


def is_integer_answer(answer: str) -> bool:
    """Return True when the answer is an integer such as 0, 12, or -1."""
    return re.fullmatch(r"-?\d+", answer.strip()) is not None


def infer_task(question: str, options: Dict[str, str], answer: str) -> str:
    """Infer task type from answer/options."""
    answer = str(answer).strip()
    options = options or {}

    if is_integer_answer(answer):
        return "count"

    if len(options) == 2:
        values = [str(v).lower().strip() for v in options.values()]
        if "yes" in values and "no" in values:
            return "yes_no"

    return "multiple_choice"


def format_option_letters(options: Dict[str, str]) -> str:
    """Format option keys as 'A or B' or 'A, B, C, or D'."""
    letters = list(options.keys())
    if not letters:
        return ""
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def build_prompt(question: str, options: Dict[str, str], task_type: str) -> str:
    """Build user prompt. Keep the output instruction in user prompt for stable SFT/GRPO behavior."""
    question = str(question).strip()
    options = options or {}

    parts: List[str] = [question]

    if options:
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        parts.append(f"Options:\n{options_text}")

    if task_type == "count":
        parts.append("Answer only with one integer.")
    else:
        valid_letters = format_option_letters(options)
        parts.append(f"Answer only with one option letter: {valid_letters}.")

    return "\n\n".join(parts)


def load_json_or_jsonl(input_file: str) -> List[Dict[str, Any]]:
    """Load either a JSON list file or a JSONL file."""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        return json.loads(text)

    items = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at {input_file}:{line_no}: {e}") from e
    return items


def resolve_image_paths(image_paths: Iterable[str], image_base_path: str) -> List[str]:
    """Resolve relative image paths to absolute paths and keep only existing files."""
    abs_paths: List[str] = []

    for p in image_paths or []:
        p = str(p)
        if os.path.isabs(p):
            abs_path = p
        else:
            abs_path = os.path.abspath(os.path.join(image_base_path, p))

        if os.path.exists(abs_path):
            abs_paths.append(abs_path)
        else:
            print(f"[WARN] Missing image, skipped path: {abs_path}")

    return abs_paths


def convert_dataset(
    input_file: str,
    output_file: str,
    image_base_path: str,
    *,
    skip_samples_with_missing_images: bool = True,
) -> None:
    data = load_json_or_jsonl(input_file)

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for item in data:
            question = item["question"]
            options = item.get("options", {}) or {}
            answer = str(item["answer"]).strip()
            image_paths = item.get("image_paths", []) or []

            abs_paths = resolve_image_paths(image_paths, image_base_path)

            if image_paths and len(abs_paths) != len(image_paths):
                skipped += 1
                if skip_samples_with_missing_images:
                    qid = item.get("qid", "unknown")
                    print(f"[WARN] Skip qid={qid}: image count mismatch.")
                    continue

            task_type = infer_task(question, options, answer)
            user_content = build_prompt(question, options, task_type)

            # Add the same number of <image> tokens as existing image files.
            # Use a newline after image tokens to keep the text prompt clean.
            if abs_paths:
                image_tokens = "".join(["<image>"] * len(abs_paths))
                user_content = image_tokens + "\n" + user_content

            # Important:
            # Do NOT manually add '<think>\n\n</think>\n\n' here.
            # Keep assistant answer clean because the training command already uses:
            #   --add_non_thinking_prefix true
            #   --loss_scale ignore_empty_think
            assistant_content = answer

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ]

            output_item = {
                "messages": messages,
                "task": task_type,
                "answer_type": "integer" if task_type == "count" else "option",
            }

            # Keep qid for later debugging/evaluation. ms-swift will ignore unused fields.
            if "qid" in item:
                output_item["qid"] = item["qid"]

            if abs_paths:
                output_item["images"] = abs_paths

            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done! Input samples: {len(data)}")
    print(f"Written samples: {written}")
    print(f"Skipped samples: {skipped if skip_samples_with_missing_images else 0}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    datasets = [
        {
            "input": "/data/lijia/PCBA_2026/datasets/Train/Standard/standard_mm_vqa_train_public.json",
            "output": "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_standard.jsonl",
            "base": "/data/lijia/PCBA_2026/datasets/Train/Standard/",
        },
        {
            "input": "/data/lijia/PCBA_2026/datasets/Train/RealWorld/realworld_mm_vqa_train_public.json",
            "output": "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_realworld.jsonl",
            "base": "/data/lijia/PCBA_2026/datasets/Train/RealWorld/",
        },
        {
            "input": "/data/lijia/PCBA_2026/pcb_bench/pcb_bench.json",
            "output": "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_pcbbench.jsonl",
            "base": "/data/lijia/PCBA_2026/pcb_bench/PCB-Bench/Task2-Image-Text-QA_and_CQ_evaluation/",
        },
    ]

    for ds in datasets:
        print(f"\nProcessing: {ds['input']}")
        convert_dataset(
            ds["input"],
            ds["output"],
            ds["base"],
        )
