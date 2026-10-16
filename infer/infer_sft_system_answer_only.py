import os
import json
import re

# 多卡设置：指定使用的GPU编号（0,1 表示2张卡）
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

# 新测试集格式：
# {
#   "qid": 1,
#   "messages": [
#       {"role": "system", "content": "..."},
#       {"role": "user", "content": "<image>\n..."}
#   ],
#   "task": "yes_no",
#   "answer_type": "option",
#   "images": [...]
# }
input_file = "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/test.jsonl"

output_file = (
    "/data/lijia/PCBA_2026/result_sft/"
    "SFT_H100_data_5_28/qwen35_27B_ori_r16_lr5e5_ep2_cp1000/"
    "result.jsonl"
)

adapter_dir = "/data/lijia/PCBA_2026/models_sft/SFT_H100_data_5_28/qwen35_27B_ori_r16_lr5e5_ep2/v0-20260528-070337/checkpoint-1000"

base_model = "/data/lijia/PCBA_2026/models/qwen35_27B"


# ======================
# 推理配置
# ======================

BATCH_SIZE = 16

USE_MULTI_GPU = True
DEVICE_MAP = "auto"

# 你的 SFT 数据是 answer-only，并且训练时用了 add_non_thinking_prefix
# 推理时建议关闭 thinking，让模型直接输出 A/B/C/D 或整数
enable_thinking = False

# answer-only 任务不需要太长输出
request_config = RequestConfig(
    max_tokens=128,
    temperature=0
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


def extract_answer(text, answer_type):
    """
    从模型输出中提取最终答案，方便后续生成 submission。
    - option: 提取第一个 A/B/C/D
    - integer: 提取第一个整数
    """
    text = str(text).strip()

    # 去掉可能出现的 think
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    if answer_type == "integer":
        m = re.search(r"-?\d+", text)
        return m.group(0) if m else text

    # option
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else text


def build_infer_request(sample):
    """
    适配 system + user 的新测试格式。

    旧代码的问题：
        user_message = messages[0]["content"]

    但新格式中 messages[0] 是 system，messages[1] 才是 user。
    所以这里直接把 sample["messages"] 中的 system/user 原样传给 InferRequest。
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    if not messages:
        raise ValueError(f"qid={sample.get('qid')} has empty messages.")

    # 确保只保留 system/user，不传 assistant
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

    template = get_template(processor, enable_thinking=enable_thinking)

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

    print(f"开始批量推理（batch_size={BATCH_SIZE}）...")

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

            resp_list = engine.infer(
                infer_requests,
                request_config=request_config
            )

            for sample, resp in zip(valid_samples, resp_list):
                qid = sample["qid"]
                response = resp.choices[0].message.content
                answer_type = sample.get("answer_type", "option")
                answer = extract_answer(response, answer_type)

                result = {
                    "qid": qid,
                    "content": response,
                    "answer": answer
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
