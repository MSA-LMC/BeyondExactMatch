import json
import re
import os


def clean_think(content):
    """移除 <think>...</think> 内容"""
    if content is None:
        return ""
    content = str(content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def extract_answer_from_content(content):
    """
    从模型生成的 content 中提取答案。

    支持：
    - {"content": "<think>...</think>\\n\\nA"}
    - "A"
    - "Answer: A"
    - "<answer>A</answer>"
    - "0"
    - "Answer: 0"
    """
    content = clean_think(content)

    if not content:
        return None

    # 匹配 <answer>A</answer> 或 <answer>0</answer>
    match = re.search(r"<answer>\s*([A-Fa-f]|\d+)\s*</answer>", content)
    if match:
        return match.group(1).upper()

    # 匹配 Answer: A
    match = re.search(r"Answer\s*:\s*([A-Fa-f])", content)
    if match:
        return match.group(1).upper()

    # 匹配 Answer: 12
    match = re.search(r"Answer\s*:\s*(-?\d+)", content)
    if match:
        return match.group(1)

    # 匹配纯字母答案：A / B / C / D
    match = re.fullmatch(r"\s*([A-Fa-f])\s*", content)
    if match:
        return match.group(1).upper()

    # 匹配纯数字答案
    match = re.fullmatch(r"\s*(-?\d+)\s*", content)
    if match:
        return match.group(1)

    # 匹配 A: xxx 格式
    match = re.search(r"^\s*([A-Fa-f])\s*:", content)
    if match:
        return match.group(1).upper()

    # 兜底：找第一个独立的大写/小写选项字母
    match = re.search(r"\b([A-Fa-f])\b", content)
    if match:
        return match.group(1).upper()

    # 兜底：找第一个整数
    match = re.search(r"-?\d+", content)
    if match:
        return match.group(0)

    return None


def normalize_answer(answer):
    """
    标准化 answer 字段。
    适配 result.jsonl 中已有的 "answer": "A" 或 "answer": "0"。
    """
    if answer is None:
        return None

    answer = str(answer).strip()

    if not answer:
        return None

    # 去掉可能残留的 think
    answer = clean_think(answer)

    # 如果是单个选项字母
    if re.fullmatch(r"[A-Fa-f]", answer):
        return answer.upper()

    # 如果是整数
    if re.fullmatch(r"-?\d+", answer):
        return answer

    # 如果 answer 字段里不是纯答案，再走 content 提取逻辑
    return extract_answer_from_content(answer)


def convert_result_to_submission(result_file, output_file):
    """
    将推理结果 result.jsonl 转换为 submission.csv。

    优先使用 result["answer"]。
    如果没有 answer 字段，或者 answer 为空，再从 result["content"] 中提取。
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    failed_count = 0
    total_count = 0

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            total_count += 1

            try:
                result = json.loads(line)

                qid = result["qid"]

                # 关键修改：优先读取 result.jsonl 里的 answer 字段
                answer = normalize_answer(result.get("answer"))

                # 如果 answer 字段不存在或为空，再从 content 提取
                if answer is None:
                    content = result.get("content", "")
                    answer = extract_answer_from_content(content)

                if answer is None:
                    content = result.get("content", "")
                    print(f"警告: qid={qid} 无法提取答案，content: {content[:200]}...")
                    failed_count += 1
                    answer = "?"

                results.append({
                    "qid": qid,
                    "answer": answer
                })

            except Exception as e:
                print(f"错误: 解析行失败 - {e}")
                print(f"原始行: {line[:200]}...")
                failed_count += 1

    # 按 qid 排序
    results.sort(key=lambda x: int(x["qid"]))

    # 写入 CSV
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("qid,answer\n")
        for result in results:
            f.write(f"{result['qid']},{result['answer']}\n")

    print("转换完成！")
    print(f"读取行数: {total_count}")
    print(f"成功结果数: {len(results)}")
    print(f"无法提取数: {failed_count}")
    print(f"输出文件: {output_file}")

    if len(results) < 8200:
        print(f"警告: 仅有 {len(results)} 条结果，预期 8200 条")


if __name__ == "__main__":
    result_file = "/data/lijia/PCBA_2026/result_sft/SFT_H100_data_5_28/qwen35_27B_ori_r16_lr5e5_ep2_cp1258_sample8_vote/result.jsonl"
    output_file = "/data/lijia/PCBA_2026/submits_5_28/GRPO_H100_pcba/qwen35_27B_ori_r16_lr5e5_ep2_cp1258_sample8_vote/submission.csv"

    convert_result_to_submission(result_file, output_file)