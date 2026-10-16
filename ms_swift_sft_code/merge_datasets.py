import json

def merge_jsonl_files(input_files, output_file):
    """
    合并多个 JSONL 文件到一个文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
    """
    total_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            print(f"正在处理: {input_file}")
            count = 0
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + '\n')
                        count += 1
            print(f"  - 添加了 {count} 条数据")
            total_count += count
    
    print(f"\n合并完成！共 {total_count} 条数据")
    print(f"输出文件: {output_file}")

if __name__ == '__main__':
    # 配置文件路径
    input_files = [
        '/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_realworld.jsonl',
        '/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_standard.jsonl',
        "/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_pcbbench.jsonl"
    ]
    output_file = '/data/lijia/PCBA_2026/sft_dataset_system_answer_only/train_fix_data.jsonl'
    
    # 执行合并
    merge_jsonl_files(input_files, output_file)