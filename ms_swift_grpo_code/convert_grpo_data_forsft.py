import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CHOICE_PROMPT = """
Respond strictly using the following XML format.

<think>
Brief reasoning.
</think>

<answer>
Explanation: exact selected option text

Answer: single uppercase option letter
</answer>
"""

BINARY_PROMPT = """
Respond strictly using the following XML format.

<think>
Brief reasoning.
</think>

<answer>
Explanation: exact selected option text

Answer: single uppercase option letter
</answer>
"""

INTEGER_PROMPT = """
Respond strictly using the following XML format.

<think>
Brief counting reasoning.
</think>

<answer>
Explanation: counted integer

Answer: integer only
</answer>
"""

TEXT_PROMPT = """
Respond strictly using the following XML format.

<think>
Brief reasoning.
</think>

<answer>
Explanation: brief explanation

Answer: final answer
</answer>
"""


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_role_message(messages: List[dict], role: str) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == role:
            return msg.get("content", "")
    return None


def normalize_answer_text(ans: str) -> str:
    ans = ans.strip()

    ans = re.sub(r"^[\(\[\{]\s*", "", ans)
    ans = re.sub(r"\s*[\)\]\}]\s*$", "", ans)
    ans = ans.strip().rstrip(".")

    m = re.match(r"^([A-Za-z])\s*[:\.\)]", ans)
    if m:
        return m.group(1).upper()

    if re.fullmatch(r"[A-Za-z]", ans):
        return ans.upper()

    m = re.search(r"\bOption\s+([A-Za-z])\b", ans, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ans.strip()


def extract_explanation_and_answer(assistant_content: str) -> Tuple[str, str]:
    """
    Supports:

    Old SFT:
        Explanation: xxx
        Answer: A

    New GRPO:
        <answer>
        Explanation: xxx

        Answer: A
        </answer>
    """

    text = assistant_content.strip()

    exp = ""
    ans = ""

    # First try XML <answer> block
    xml_answer = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if xml_answer:
        answer_block = xml_answer.group(1).strip()

        nested_exp = re.search(
            r"(?is)Explanation\s*[:：]\s*(.*?)(?=\n\s*Answer\s*[:：]|\Z)",
            answer_block,
        )

        if nested_exp:
            exp = nested_exp.group(1).strip()

        nested_ans = re.search(
            r"(?im)^\s*Answer\s*[:：]\s*(.+?)\s*$",
            answer_block,
        )

        if nested_ans:
            ans = nested_ans.group(1).strip()

    # Fallback to old SFT format
    if not exp:
        exp_match = re.search(
            r"(?is)(?:^|\n)\s*(?:Explanation|解释)\s*[:：]\s*(.*?)(?=\n\s*(?:Answer|答案)\s*[:：]|\Z)",
            text,
        )
        if exp_match:
            exp = exp_match.group(1).strip()

    if not ans:
        ans_match = re.search(
            r"(?im)^\s*(?:Answer|答案)\s*[:：]\s*(.+?)\s*$",
            text,
        )
        if ans_match:
            ans = ans_match.group(1).strip()

    ans = normalize_answer_text(ans)
    return exp, ans


def parse_choices(user_content: str) -> Dict[str, str]:
    """
    Parse options like:
      Options:
      A: xxx
      B: xxx
      C. xxx
      D) xxx

    Supports multi-line option text.
    """

    choices: Dict[str, str] = {}

    opt_start = re.search(
        r"\bOptions\s*[:：]\s*",
        user_content,
        flags=re.IGNORECASE,
    )

    if opt_start:
        block = user_content[opt_start.end():]
    else:
        block = user_content

    lines = block.splitlines()

    current_label = None
    current_text_parts = []

    option_line_pattern = re.compile(
        r"^\s*([A-H])\s*[:：\.\)]\s*(.*)$"
    )

    for line in lines:
        match = option_line_pattern.match(line)

        if match:
            if current_label is not None:
                choices[current_label] = " ".join(current_text_parts).strip()

            current_label = match.group(1).upper()
            current_text_parts = [match.group(2).strip()]
        else:
            if current_label is not None:
                stripped = line.strip()
                if stripped:
                    current_text_parts.append(stripped)

    if current_label is not None:
        choices[current_label] = " ".join(current_text_parts).strip()

    choices = {k: v for k, v in choices.items() if v}
    return choices


def infer_answer_type(
    solution: str,
    choices: Dict[str, str],
    user_content: str,
) -> str:
    sol = solution.strip()

    # Integer
    if re.fullmatch(r"-?\d+", sol):
        return "integer"

    # Binary
    if choices:
        normalized_choices = {v.strip().lower() for v in choices.values()}
        if normalized_choices.issubset({"yes", "no", "true", "false"}):
            return "binary"

    # Multiple choice
    if choices and re.fullmatch(r"[A-Z]", sol.upper()):
        return "choice"

    return "text"


def infer_task(user_content: str, images: List[str]) -> str:
    """
    Heuristic task inference.

    Current supported count tasks:
      - count_component   -> normalization_constant = 138.0
      - count_pin_lead    -> normalization_constant = 260.0
    """
    text = user_content.lower()

    pin_lead_markers = [
        "pin",
        "pins",
        "lead",
        "leads",
    ]

    count_markers = [
        "how many",
        "number of",
        "count",
        "quantity",
        "precise",
        "quantity of",
        "pin quantity",
        "pin count",
        "lead quantity",
        "lead count",
    ]

    # Strong pin/lead counting signals
    if any(k in text for k in pin_lead_markers):
        if any(k in text for k in count_markers):
            return "count_pin_lead"

        # Common pin/lead phrasing without explicit count words
        if any(
            k in text
            for k in [
                "pin quantity",
                "pin count",
                "number of pins",
                "how many pins",
                "count the pins",
                "precise pin quantity",
                "lead quantity",
                "lead count",
                "number of leads",
                "how many leads",
                "count the leads",
            ]
        ):
            return "count_pin_lead"

    # General component counting
    if any(k in text for k in ["how many", "number of", "count", "quantity"]):
        return "count_component"

    if "does the image show a defect" in text:
        return "defect_existence"

    if "defect type" in text:
        return "defect_type"

    if "component type" in text:
        return "component_type"

    if "mount side" in text:
        return "mount_side"

    if "shape" in text or "attribute" in text:
        return "attribute_reasoning"

    if images and any(
        "/Standard/" in p or "/standard/" in p
        for p in images
    ):
        return "standard_knowledge"

    return "unknown"


def get_normalization_constant(task: str) -> Optional[float]:
    if task == "count_component":
        return 138.0

    if task == "count_pin_lead":
        return 260.0

    return None


def build_extra_prompt(answer_type: str) -> str:
    if answer_type == "choice":
        return CHOICE_PROMPT

    if answer_type == "binary":
        return BINARY_PROMPT

    if answer_type == "integer":
        return INTEGER_PROMPT

    return TEXT_PROMPT


def has_grpo_format_instruction(user_content: str) -> bool:
    return (
        "<think>" in user_content
        or "<answer>" in user_content
    )


def build_answer_block(
    explanation: str,
    answer: str,
) -> str:
    explanation = explanation.strip()
    answer = answer.strip()

    return (
        f"Explanation: {explanation}\n\n"
        f"Answer: {answer}"
    )


def convert_one(example: dict, add_task: bool = True) -> dict:
    messages = example.get("messages", [])
    images = example.get("images", [])

    user_content = get_role_message(messages, "user")
    assistant_content = get_role_message(messages, "assistant")

    if user_content is None:
        raise ValueError("Missing user message")

    if assistant_content is None:
        raise ValueError("Missing assistant message")

    reference_explanation, solution = extract_explanation_and_answer(
        assistant_content
    )

    if not solution:
        raise ValueError(
            f"Cannot extract Answer from assistant content: {assistant_content}"
        )

    choices = parse_choices(user_content)

    answer_type = infer_answer_type(
        solution,
        choices,
        user_content,
    )

    solution_text = ""

    # Multiple Choice
    if answer_type == "choice":
        solution_letter = normalize_answer_text(solution)
        solution = solution_letter

        if choices and solution in choices:
            solution_text = choices[solution]
        else:
            solution_text = reference_explanation

    # Integer
    elif answer_type == "integer":
        solution_text = solution

    # Binary
    elif answer_type == "binary":
        solution_letter = normalize_answer_text(solution)
        solution = solution_letter

        if choices and solution in choices:
            solution_text = choices[solution]
        else:
            solution_text = reference_explanation

    # Text
    else:
        solution_text = reference_explanation or solution

    # Add GRPO prompt
    new_user_content = user_content.strip()

    if not has_grpo_format_instruction(new_user_content):
        new_user_content = (
            new_user_content.rstrip()
            + "\n\n"
            + build_extra_prompt(answer_type).strip()
        )

    # Build answer block
    answer_block = build_answer_block(
        solution_text,
        solution,
    )

    new_example = {
        "messages": [
            {
                "role": "user",
                "content": new_user_content,
            }
        ],
        "images": images,
        "solution": solution,
        "answer_block": answer_block,
        "answer_type": answer_type,
    }

    if solution_text:
        new_example["solution_text"] = solution_text

    if choices:
        new_example["choices"] = choices

    if reference_explanation:
        new_example["reference_explanation"] = reference_explanation

    # Task
    if add_task:
        task = infer_task(user_content, images)

        # Fallback for integer tasks to avoid missing normalization_constant
        if task == "unknown" and answer_type == "integer":
            t = user_content.lower()
            if any(k in t for k in ["pin", "lead"]):
                task = "count_pin_lead"
            elif any(k in t for k in ["count", "quantity", "number of", "how many"]):
                task = "count_component"

        new_example["task"] = task

        norm_c = get_normalization_constant(task)
        if norm_c is not None:
            new_example["normalization_constant"] = norm_c

    # Preserve metadata
    for key in [
        "id",
        "question_id",
        "image_id",
        "category",
        "source",
    ]:
        if key in example:
            new_example[key] = example[key]

    return new_example


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="/home/u2024170840/code/PCBA_2026/sft_dataset_5_22_ori/train_pcbbench_fix.jsonl",
        help="Input SFT JSONL file",
    )

    parser.add_argument(
        "--output",
        default="/home/u2024170840/code/PCBA_2026/grpo_dataset_5_22/grpo_fromsft.jsonl",
        help="Output GRPO JSONL file",
    )

    parser.add_argument(
        "--no-task",
        action="store_true",
        help="Do not add weakly inferred task field",
    )

    parser.add_argument(
        "--failed-output",
        default=None,
        help="Optional path to save failed lines",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    failed_output_path = (
        Path(args.failed_output)
        if args.failed_output
        else None
    )

    converted_rows = []
    failed_rows = []

    stats_answer_type = {}
    stats_task = {}

    total = 0
    failed = 0

    for line_no, example in read_jsonl(input_path):
        total += 1

        try:
            converted = convert_one(
                example,
                add_task=not args.no_task,
            )

            converted_rows.append(converted)

            answer_type = converted.get(
                "answer_type",
                "unknown",
            )

            stats_answer_type[answer_type] = (
                stats_answer_type.get(answer_type, 0) + 1
            )

            task = converted.get("task", "no_task")

            stats_task[task] = (
                stats_task.get(task, 0) + 1
            )

        except Exception as e:
            failed += 1

            failed_rows.append(
                {
                    "line_no": line_no,
                    "error": str(e),
                    "raw": example,
                }
            )

            print(f"[ERROR] line={line_no}: {e}")

    write_jsonl(output_path, converted_rows)

    if failed_output_path is not None and failed_rows:
        write_jsonl(failed_output_path, failed_rows)

    print("Done.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Total: {total}")
    print(f"Converted: {len(converted_rows)}")
    print(f"Failed: {failed}")
    print(f"Answer type stats: {stats_answer_type}")
    print(f"Task stats: {stats_task}")

    if failed_output_path is not None:
        print(f"Failed output: {failed_output_path}")


if __name__ == "__main__":
    main()