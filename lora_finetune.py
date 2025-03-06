from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("Abirate/english_quotes")  # 예제 데이터셋

print(dataset["train"][0])

model_name = "meta-llama/Llama-2-7b-hf"  # 7B 모델 (Mac M3에서는 7B 이하 추천)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="mps", torch_dtype=torch.float16
)

print("✅ 모델 로드 완료!")


lora_config = LoraConfig(
    r=8,  # Rank 값 (메모리 사용량 조절)
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()
