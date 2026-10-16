import os
import json
import re
from collections import Counter

# 多卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["IMAGE_MAX_TOKEN_NUM"] = "1024"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "128"
os.environ["FPS_MAX_FRAMES"] = "16"

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig


# ======================
# 输入输出路径
# ======================

input_file = "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/test.jsonl"

output_file = (
    "/data/lijia/PCBA_2026/result_sft/"
    "SFT_H100_data_5_28/qwen35_27B_ori_r16_lr5e5_ep2_cp1258_sample8_vote/"
    "result.jsonl"
)

adapter_dir = "/data/lijia/PCBA_2026/models_sft/SFT_H100_data_5_28/qwen35_27B_ori_r16_lr5e5_ep2/v0-20260528-070337/checkpoint-1258"

base_model = "/data/lijia/PCBA_2026/models/qwen35_27B"


# ======================
# 推理配置
# ======================

BATCH_SIZE = 16

USE_MULTI_GPU = True
DEVICE_MAP = "auto"

enable_thinking = False

# 多次采样次数
NUM_SAMPLES = 8

# greedy 推理：确定性答案
greedy_config = RequestConfig(
    max_tokens=32,
    temperature=0
)

# sampling 推理：用于投票
sample_config = RequestConfig(
    max_tokens=32,
    temperature=0.5,
    top_p=0.9
)


def load_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except Exception as e:
                print(f"[WARNING] JSON parse failed at line {line_no}: {e}")

    return samples


def remove_think(text):
    text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def get_valid_option_letters(sample):
    """
    从 user prompt 里解析合法选项字母。
    默认支持 A-F。
    """
    messages = sample.get("messages", [])
    text = "\n".join([m.get("content", "") for m in messages])

    letters = re.findall(r"^\s*([A-F])\s*:", text, flags=re.M)

    if letters:
        return sorted(set(letters))

    return ["A", "B", "C", "D", "E", "F"]


def extract_answer(text, answer_type, sample=None):
    """
    从模型输出中提取最终答案。
    - option: 提取合法选项字母
    - integer: 提取整数
    """
    text = remove_think(text)

    if not text:
        return None

    # 先处理 <answer>A</answer>
    m = re.search(r"<answer>\s*([A-Fa-f]|\d+)\s*</answer>", text)
    if m:
        ans = m.group(1)
        return ans.upper() if ans.isalpha() else ans

    if answer_type == "integer":
        # Answer: 12
        m = re.search(r"Answer\s*:\s*(-?\d+)", text, flags=re.I)
        if m:
            return m.group(1)

        # 纯数字
        m = re.fullmatch(r"\s*(-?\d+)\s*", text)
        if m:
            return m.group(1)

        # 兜底：第一个整数
        m = re.search(r"-?\d+", text)
        if m:
            return m.group(0)

        return None

    # option
    valid_letters = get_valid_option_letters(sample) if sample is not None else ["A", "B", "C", "D", "E", "F"]
    valid_set = set(valid_letters)

    # Answer: A
    m = re.search(r"Answer\s*:\s*([A-Fa-f])", text, flags=re.I)
    if m:
        ans = m.group(1).upper()
        if ans in valid_set:
            return ans

    # 纯字母 A
    m = re.fullmatch(r"\s*([A-Fa-f])\s*", text)
    if m:
        ans = m.group(1).upper()
        if ans in valid_set:
            return ans

    # A: xxx
    m = re.search(r"^\s*([A-Fa-f])\s*:", text)
    if m:
        ans = m.group(1).upper()
        if ans in valid_set:
            return ans

    # 兜底：找第一个合法独立字母
    for ans in re.findall(r"\b([A-Fa-f])\b", text):
        ans = ans.upper()
        if ans in valid_set:
            return ans

    return None


def vote_answer(sample_answers, greedy_answer, answer_type):
    """
    多次采样投票。
    如果采样结果不稳定，则回退到 greedy_answer。
    """
    valid_answers = [
        str(a).strip()
        for a in sample_answers
        if a is not None and str(a).strip() != ""
    ]

    if not valid_answers:
        return greedy_answer, 0.0, {}

    counter = Counter(valid_answers)
    best_answer, best_count = counter.most_common(1)[0]
    confidence = best_count / len(valid_answers)

    if answer_type == "integer":
        threshold = 0.55
    else:
        threshold = 0.55

    # 如果采样多数票不够明显，保留 greedy
    if best_answer != greedy_answer and confidence < threshold:
        return greedy_answer, confidence, dict(counter)

    return best_answer, confidence, dict(counter)


def build_infer_request(sample):
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    if not messages:
        raise ValueError(f"qid={sample.get('qid')} has empty messages.")

    infer_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role in {"system", "user"}:
            infer_messages.append({
                "role": role,
                "content": content
            })

    if not any(m["role"] == "user" for m in infer_messages):
        raise ValueError(f"qid={sample.get('qid')} has no user message.")

    return InferRequest(
        messages=infer_messages,
        images=images
    )


def load_completed_qids(output_file):
    completed_qids = set()

    if not os.path.exists(output_file):
        return completed_qids

    print("检测到已存在的结果文件，正在读取已完成的 qid...")

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                result = json.loads(line)
                completed_qids.add(result["qid"])
            except Exception:
                pass

    print(f"已完成 {len(completed_qids)} 条推理")
    return completed_qids


def main():
    print("正在加载模型...")

    if USE_MULTI_GPU:
        print(f"使用多卡推理，CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

        import torch

        max_memory = {
            i: f"{torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)}GB"
            for i in range(torch.cuda.device_count())
        }

        print(f"每卡最大内存: {max_memory}")

        model, processor = get_model_processor(
            base_model,
            device_map=DEVICE_MAP,
            max_memory=max_memory
        )
    else:
        model, processor = get_model_processor(base_model)

    model = PeftModel.from_pretrained(model, adapter_dir)

    template = get_template(
        processor,
        enable_thinking=enable_thinking
    )

    engine = TransformersEngine(
        model,
        template=template,
        max_batch_size=BATCH_SIZE
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    completed_qids = load_completed_qids(output_file)

    print("正在读取测试数据...")
    test_samples = load_jsonl(input_file)

    print(f"共 {len(test_samples)} 条测试数据")

    pending_samples = [
        s for s in test_samples
        if s.get("qid") not in completed_qids
    ]

    print(f"待处理 {len(pending_samples)} 条数据")

    if len(pending_samples) == 0:
        print("所有数据已处理完成！")
        return

    print(
        f"开始批量推理：batch_size={BATCH_SIZE}, "
        f"num_samples={NUM_SAMPLES}, sample_temperature=0.6"
    )

    with open(output_file, "a", encoding="utf-8") as out_f:
        for batch_start in range(0, len(pending_samples), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(pending_samples))
            batch_samples = pending_samples[batch_start:batch_end]

            valid_samples = []
            infer_requests = []

            for sample in batch_samples:
                try:
                    infer_requests.append(build_infer_request(sample))
                    valid_samples.append(sample)
                except Exception as e:
                    qid = sample.get("qid")
                    print(f"[ERROR] build request failed, qid={qid}: {e}")

                    result = {
                        "qid": qid,
                        "content": "",
                        "answer": "",
                        "error": str(e)
                    }

                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()

            if not infer_requests:
                continue

            # ======================
            # 1. greedy 推理
            # ======================
            greedy_resp_list = engine.infer(
                infer_requests,
                request_config=greedy_config
            )

            greedy_contents = []
            greedy_answers = []

            for sample, resp in zip(valid_samples, greedy_resp_list):
                response = resp.choices[0].message.content
                answer_type = sample.get("answer_type", "option")
                answer = extract_answer(response, answer_type, sample)

                greedy_contents.append(response)
                greedy_answers.append(answer)

            # ======================
            # 2. 多次 sampling 推理
            # ======================
            all_sample_contents = [[] for _ in valid_samples]
            all_sample_answers = [[] for _ in valid_samples]

            for sample_round in range(NUM_SAMPLES):
                sample_resp_list = engine.infer(
                    infer_requests,
                    request_config=sample_config
                )

                for i, (sample, resp) in enumerate(zip(valid_samples, sample_resp_list)):
                    response = resp.choices[0].message.content
                    answer_type = sample.get("answer_type", "option")
                    answer = extract_answer(response, answer_type, sample)

                    all_sample_contents[i].append(response)
                    all_sample_answers[i].append(answer)

            # ======================
            # 3. 投票
            # ======================
            for i, sample in enumerate(valid_samples):
                qid = sample["qid"]
                answer_type = sample.get("answer_type", "option")

                final_answer, vote_confidence, vote_counts = vote_answer(
                    sample_answers=all_sample_answers[i],
                    greedy_answer=greedy_answers[i],
                    answer_type=answer_type
                )

                result = {
                    "qid": qid,
                    "content": greedy_contents[i],
                    "answer": final_answer,
                    "greedy_answer": greedy_answers[i],
                    "sample_answers": all_sample_answers[i],
                    "vote_counts": vote_counts,
                    "vote_confidence": vote_confidence,
                    "sample_contents": all_sample_contents[i]
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

            processed_count = len(completed_qids) + batch_end
            total_count = len(test_samples)
            print(f"已处理 {processed_count}/{total_count}")

    print("推理完成！")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()