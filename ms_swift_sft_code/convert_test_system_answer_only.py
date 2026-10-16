import json
import os


SYSTEM_PROMPT = (
    "You are a careful PCB visual inspection assistant. "
    "Inspect the image carefully and answer according to the given question. "
    "For multiple-choice questions, output only one option letter. "
    "For counting questions, output only one integer."
)


def has_valid_options(options):
    return bool(options) and any(str(v).strip() != "" for v in options.values())


def infer_task(question, options):
    """
    Infer task type for test samples.

    - yes_no: two options whose texts are Yes/No
    - multiple_choice: non-empty options
    - count: no valid options
    """
    if has_valid_options(options):
        values = [str(v).strip().lower() for v in options.values()]

        if len(options) == 2 and set(values) == {"yes", "no"}:
            return "yes_no"

        return "multiple_choice"

    return "count"


def build_prompt(question, options, task_type):
    """
    Build user prompt in the same style as the SFT answer-only format.
    """
    question = str(question).strip()
    parts = [question]

    if has_valid_options(options):
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        parts.append(f"Options:\n{options_text}")

    if task_type == "count":
        parts.append("Answer only with one integer.")
    else:
        valid_letters = list(options.keys())

        if len(valid_letters) == 2:
            letters_text = " or ".join(valid_letters)
        else:
            letters_text = ", ".join(valid_letters[:-1]) + f", or {valid_letters[-1]}"

        parts.append(f"Answer only with one option letter: {letters_text}.")

    return "\n\n".join(parts)


def resolve_image_paths(image_paths, image_base_path):
    """
    Resolve relative image paths to absolute paths and keep only existing files.
    """
    abs_paths = []

    for p in image_paths:
        abs_path = os.path.abspath(os.path.join(image_base_path, p))

        if os.path.exists(abs_path):
            abs_paths.append(abs_path)
        else:
            print(f"[WARNING] Missing image: {abs_path}")

    return abs_paths


def convert_test_dataset(input_file, output_file, image_base_path, skip_missing_image=True):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for item in data:
            total += 1

            qid = item["qid"]
            question = item["question"]
            options = item.get("options", {})
            image_paths = item.get("image_paths", [])

            task_type = infer_task(question, options)

            abs_paths = resolve_image_paths(image_paths, image_base_path)

            if image_paths and len(abs_paths) != len(image_paths):
                if skip_missing_image:
                    skipped += 1
                    print(f"[WARNING] Skip qid={qid}, missing image(s).")
                    continue

            user_content = build_prompt(question, options, task_type)

            if abs_paths:
                image_tokens = "".join(["<image>"] * len(abs_paths))
                user_content = image_tokens + "\n" + user_content

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            output_item = {
                "qid": qid,
                "messages": messages,
                "task": task_type,
                "answer_type": "integer" if task_type == "count" else "option"
            }

            if abs_paths:
                output_item["images"] = abs_paths

            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done! Total test samples: {total}")
    print(f"Written samples: {written}")
    print(f"Skipped samples: {skipped}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    input_file = "/data/lijia/PCBA_2026/datasets/Test/vqa_test_public.json"

    output_file = "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/test.jsonl"

    image_base_path = "/data/lijia/PCBA_2026/datasets/Test/"

    convert_test_dataset(
        input_file=input_file,
        output_file=output_file,
        image_base_path=image_base_path,
        skip_missing_image=True
    )
