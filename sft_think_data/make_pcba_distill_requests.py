import os
import json
import re
import random
import textwrap


input_file = "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_fix_data.jsonl"

output_file = "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/distill_pcba_requests.jsonl"

# 如果想先测试，可以改成 100 或 500。
# 全量生成用 None。
MAX_SAMPLES = None

RANDOM_SEED = 42


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                data.append(json.loads(line))

    return data


def extract_gt_answer(sample):
    if "solution" in sample:
        return str(sample["solution"]).strip()

    if "answer" in sample:
        return str(sample["answer"]).strip()

    messages = sample.get("messages", [])

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = str(msg.get("content", "")).strip()

            # 去掉可能存在的 think
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

            # <answer>A</answer>
            m = re.search(r"<answer>\s*([^<]+?)\s*</answer>", text, flags=re.I)
            if m:
                return m.group(1).strip()

            # 选项题 A-F
            m = re.search(r"\b([A-F])\b", text.upper())
            if m:
                return m.group(1)

            # 计数题整数
            m = re.search(r"-?\d+", text)
            if m:
                return m.group(0)

    return None


def strip_answer_only_instruction(text):
    text = str(text)

    patterns = [
        r"\n\s*Answer only with[^\n]*\.?\s*$",
        r"\n\s*Output only[^\n]*\.?\s*$",
        r"\n\s*Only output[^\n]*\.?\s*$",
        r"\n\s*Please answer only[^\n]*\.?\s*$",
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.I)

    return text.strip()


def get_user_message(sample):
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return {
                "role": "user",
                "content": msg.get("content", "")
            }

    raise ValueError(f"qid={sample.get('qid')} has no user message")


def build_sample(sample):
    qid = sample.get("qid")
    answer_type = sample.get("answer_type", "option")
    images = sample.get("images", [])

    gt_answer = extract_gt_answer(sample)

    if gt_answer is None:
        raise ValueError(f"qid={qid} has no gt answer")

    user_msg = get_user_message(sample)

    base_user_content = strip_answer_only_instruction(user_msg["content"])

    # 蒸馏专用 system prompt，避免和 answer-only system 冲突
    distill_system_msg = {
        "role": "system",
        "content": (
            "You are a careful PCB visual inspection assistant. "
            "Inspect the image carefully, write one brief visual reasoning sentence, "
            "then give the final answer."
        )
    }

    # 给教师模型看的 prompt：带正确答案，让它生成简短 think
    # 这个是给教师模型看的：带标准答案，要求生成视觉推理。
    teacher_user_content = f"""{base_user_content}

The correct final answer is: {gt_answer}

Write one or two concise reasoning sentences.
For image-based samples, base the reasoning only on visible PCB evidence.
For text-only samples, base the reasoning on the question and options.
Do not compare all options one by one.
Do not mention "the user".
Do not mention "known correct answer".
Do not output extra explanation after the final answer.

Output exactly in this format:
<think>
one or two concise reasoning sentences
</think>

{gt_answer}"""

    # 这个是后面真正用于 SFT 训练的 user prompt：不泄露标准答案。
    train_user_content = f"""{base_user_content}

Think briefly, then give the final answer."""

    infer_messages = [
        distill_system_msg,
        {
            "role": "user",
            "content": teacher_user_content
        }
    ]

    train_messages = [
        distill_system_msg,
        {
            "role": "user",
            "content": train_user_content
        }
    ]

    return {
        "qid": qid,
        "messages": infer_messages,
        "train_messages": train_messages,
        "gt_answer": gt_answer,
        "answer_type": answer_type,
        "task": sample.get("task"),
        "images": images
    }


def main():
    samples = load_jsonl(input_file)

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    ok = 0
    bad = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            try:
                item = build_sample(sample)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                bad += 1
                print(f"[SKIP] qid={sample.get('qid')}: {e}")

    print(f"saved: {output_file}")
    print(f"ok={ok}, bad={bad}")


if __name__ == "__main__":
    main()