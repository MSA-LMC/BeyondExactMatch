import os
import re
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

model, processor = get_model_processor('/data/lijia/PCBA_2026/models/qwen36_27B')
template = get_template(processor, enable_thinking=True)
engine = TransformersEngine(model, template=template)

image_path = '/data/lijia/PCBA_2026/datasets/Test/images/4674.jpg'

# 这里填你 GRPO 模型原来的预测结果
original_answer = "32"

prompt = f"""
<image>
You are a strict and careful verifier for a PCB visual question.

Question:
Count the pins on the primary component framed by the red box and provide the number.

Original model prediction:
{original_answer}

Your task:
1. Carefully inspect the primary component inside the red box.
2. Directly check whether the original model prediction is correct.
3. Verify the number of visible pins/leads/pads on the red-boxed component.
4. If the original prediction matches the visual evidence, keep it.
5. If there is clear visual evidence that the original prediction is wrong, you must correct it.
6. Do not blindly trust the original prediction.
7. The final verified_answer must be the number that best matches the image.

Decision rules:
- If the original prediction is correct, set changed to false.
- If the original prediction is incorrect, set changed to true and provide the corrected number.
- verified_answer must contain only the final number.
- Do not output anything outside JSON.

Output only JSON, no markdown, no extra text:
{{
  "original_answer": "{original_answer}",
  "verified_answer": "number only",
  "changed": true_or_false,
  "confidence": "high/medium/low",
  "reason": "brief visual evidence for the decision"
}}
"""

infer_request = InferRequest(
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    images=[image_path]
)

request_config = RequestConfig(
    # max_tokens=2048,
    temperature=0
)

resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content

print(response)

# 提取 JSON
clean = re.sub(r"<think>.*?</think>", "", response, flags=re.S).strip()

match = re.search(r"\{.*\}", clean, flags=re.S)
if match:
    result = json.loads(match.group())
    print("原答案:", result["original_answer"])
    print("审核后答案:", result["verified_answer"])
    print("是否修改:", result["changed"])
else:
    print("无法解析 JSON：")
    print(clean)