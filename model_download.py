from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

login("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")

model_name = "meta-llama/Llama-2-7b-hf"  # 7B 모델 (Mac M3에서는 7B 이하 추천)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="mps", torch_dtype=torch.float16
)

print("✅ 모델 로드 완료!")
