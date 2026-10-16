import os
import json

# 多卡设置：指定使用的GPU编号（0,1 表示2张卡）
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

# 输入输出路径
input_file = '/data/lijia/PCBA_2026/sft_dataset_5_22_ori/test.jsonl'
output_file = "/data/lijia/PCBA_2026/results_grpo/GRPO_H100_pcba/qwen35_27B_checkpoint3000_temp1/result.jsonl"

# 批量推理大小（多卡时建议增大）
BATCH_SIZE = 16

# 多卡配置
USE_MULTI_GPU = True
DEVICE_MAP = "auto"  # "auto" 自动分配，或使用 {"": 0} 手动指定到第一张卡

def main():
    adapter_dir = '/data/lijia/PCBA_2026/models/GRPO_H100_pcba/qwen35_27B_fix_g8_bs2_c256_len4096_server/v1-20260526-120142/checkpoint-3000'
    enable_thinking = False
    
    # 加载模型和处理器
    print("正在加载模型...")
    
    if USE_MULTI_GPU:
        print(f"使用多卡推理，CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        import torch
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3)}GB" for i in range(torch.cuda.device_count())}
        print(f"每卡最大内存: {max_memory}")
        
        model, processor = get_model_processor(
            '/data/lijia/PCBA_2026/models_res/Qwen35_27B_PCBA_fixdata_SFT',
            device_map=DEVICE_MAP,
            max_memory=max_memory
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        template = get_template(processor, enable_thinking=enable_thinking)
        engine = TransformersEngine(model, template=template, max_batch_size=BATCH_SIZE)
    else:
        model, processor = get_model_processor('/data/lijia/PCBA_2026/models_res/Qwen35_27B_PCBA_fixdata_SFT')
        model = PeftModel.from_pretrained(model, adapter_dir)
        template = get_template(processor, enable_thinking=enable_thinking)
        engine = TransformersEngine(model, template=template, max_batch_size=BATCH_SIZE)
    
    # 配置推理参数
    request_config = RequestConfig(max_tokens=256, temperature=1.0)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查是否存在已有的结果文件
    completed_qids = set()
    if os.path.exists(output_file):
        print("检测到已存在的结果文件，正在读取已完成的qid...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        completed_qids.add(result['qid'])
                    except:
                        pass
        print(f"已完成 {len(completed_qids)} 条推理")
    
    # 读取测试数据
    print("正在读取测试数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"共 {len(test_samples)} 条测试数据")
    
    # 过滤未完成的样本
    pending_samples = [s for s in test_samples if s['qid'] not in completed_qids]
    print(f"待处理 {len(pending_samples)} 条数据")
    
    # 如果没有待处理数据，直接退出
    if len(pending_samples) == 0:
        print("所有数据已处理完成！")
        return
    
    # 批量推理并实时写入结果
    print(f"开始批量推理（batch_size={BATCH_SIZE}）...")
    
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for batch_start in range(0, len(pending_samples), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(pending_samples))
            batch_samples = pending_samples[batch_start:batch_end]
            
            # 构建批量推理请求
            infer_requests = []
            for sample in batch_samples:
                messages = sample['messages']
                images = sample.get('images', [])
                user_message = messages[0]['content']
                
                infer_request = InferRequest(
                    messages=[{
                        "role": "user",
                        "content": user_message
                    }],
                    images=images
                )
                infer_requests.append(infer_request)
            
            # 批量执行推理
            resp_list = engine.infer(infer_requests, request_config=request_config)
            
            # 处理批量结果
            for sample, resp in zip(batch_samples, resp_list):
                qid = sample['qid']
                response = resp.choices[0].message.content
                
                result = {
                    "qid": qid,
                    "content": response
                }
                
                # 实时写入结果
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_f.flush()
            
            # 打印进度
            processed_count = len(completed_qids) + batch_end
            total_count = len(test_samples)
            print(f"已处理 {processed_count}/{total_count}")
    
    print("推理完成！")
    print(f"结果已保存到: {output_file}")

if __name__ == '__main__':
    main()