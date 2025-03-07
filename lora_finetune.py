from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("Abirate/english_quotes")  # 예제 데이터셋
device = "cuda" if torch.cuda.is_available() else "cpu"

print(dataset["train"][0])

model_name = "meta-llama/Llama-2-7b-hf"  # 7B 모델 (Mac M3에서는 7B 이하 추천)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.float16
)
# 모델에도 패딩 토큰 설정
model.config.pad_token_id = tokenizer.pad_token_id

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

# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=2,
#     num_train_epochs=3,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=10,
#     fp16=True,
#     optim="adamw_torch",
#     report_to="none",
# )

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # 배치 사이즈를 2에서 1로 줄임
    gradient_accumulation_steps=4,   # 그래디언트 누적 사용
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    # 메모리 최적화를 위한 추가 설정
    gradient_checkpointing=True,    # 메모리 절약을 위한 그래디언트 체크포인팅
    max_grad_norm=0.3,             # 그래디언트 클리핑
)


# 데이터 전처리 함수
def preprocess_function(examples):
    # 프롬프트 형식 지정
    prompts = [f"Quote: {quote}\nAuthor: {author}" for quote, author in zip(examples['quote'], examples['author'])]
    
    # 토크나이즈
    model_inputs = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # labels도 input_ids와 동일하게 설정 (for causal language modeling)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

# 데이터셋 전처리 적용
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
)
trainer.train()

# 학습 완료 후 모델 저장
output_dir = "./finetuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
