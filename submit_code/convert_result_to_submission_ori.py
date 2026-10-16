import json
import re
import os

def extract_answer_from_content(content):
    """
    从模型生成的content中提取答案
    支持多种格式：
    - "Explanation: xxx\nAnswer: A"
    - "Answer: A"
    - "<think>...</think>Answer: A"
    - "<think>...</think>\nAnswer: A"
    - "A: xxx"
    - 纯数字答案（如计数题）
    """
    # 移除think标签包裹的内容
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.strip()
    
    if not content:
        return None
    
    # 优先匹配 Answer: X 格式（字母）
    match = re.search(r'Answer:\s*([A-F])', content)
    if match:
        return match.group(1)
    
    # 匹配 Answer: 数字 格式（计数题）
    match = re.search(r'Answer:\s*(\d+)', content)
    if match:
        return match.group(1)
    
    # 匹配 A: xxx 格式
    match = re.search(r'^([A-F]):', content)
    if match:
        return match.group(1)
    
    # 匹配单独的大写字母（答案）
    match = re.search(r'^([A-F])$', content.strip())
    if match:
        return match.group(1)
    
    # 尝试匹配第一个大写字母
    match = re.search(r'\b([A-F])\b', content)
    if match:
        return match.group(1)
    
    # 尝试匹配纯数字（计数题）
    match = re.search(r'^(\d+)$', content.strip())
    if match:
        return match.group(1)
    
    return None

def convert_result_to_submission(result_file, output_file):
    """
    将推理结果转换为提交格式
    
    Args:
        result_file: 推理结果文件路径 (result.jsonl)
        output_file: 输出提交文件路径 (submission.csv)
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    failed_count = 0
    total_count = 0
    
    # 读取推理结果
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            try:
                result = json.loads(line)
                qid = result['qid']
                content = result['content']
                answer = extract_answer_from_content(content)
                
                if answer is None:
                    print(f"警告: qid={qid} 无法提取答案，content: {content[:200]}...")
                    failed_count += 1
                    answer = "?"  # 无法提取时使用占位符
                
                results.append({"qid": qid, "answer": answer})
            except Exception as e:
                print(f"错误: 解析行失败 - {e}")
                failed_count += 1
    
    # 按qid排序
    results.sort(key=lambda x: x['qid'])
    
    # 写入CSV文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("qid,answer\n")
        for result in results:
            f.write(f"{result['qid']},{result['answer']}\n")
    
    print(f"转换完成！")
    print(f"总结果数: {len(results)}")
    print(f"无法提取数: {failed_count}")
    print(f"输出文件: {output_file}")
    
    # 检查是否有遗漏
    if len(results) < 8200:
        print(f"警告: 仅有 {len(results)} 条结果，预期 8200 条")

if __name__ == '__main__':
    result_file = '/data/lijia/PCBA_2026/results_grpo/GRPO_H100_pcba/qwen35_27B_checkpoint3000_temp1/result.jsonl'
    output_file = '/data/lijia/PCBA_2026/submits_grpo/GRPO_H100_pcba/qwen35_27B_checkpoint3000_temp1/submission.csv'
    
    convert_result_to_submission(result_file, output_file)