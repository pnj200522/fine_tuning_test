# fine_tuning_test

### * MAC
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

### 필수 패키지 설치
# cmake: 크로스 플랫폼 빌드 자동화 도구로, C/C++ 프로젝트 빌드에 사용됩니다.
# 많은 머신러닝 라이브러리들이 C++ 기반이므로 컴파일 시 필요합니다.
brew install cmake git wget

brew install pytorch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### Mac os 에서는 conda 환경 권장.
brew install miniforge  
conda init zsh

conda create -n llm-finetune python=3.10
conda activate llm-finetune

### 파이썬 패키지 설치
pip install torch torchvision torchaudio

### PyTorch가 Metal 가속을 지원하는지 확인
check_gpu.py

### 모델 다운로드( 테스트 모델 : Llama 2)

### 허깅페이스 패키지 설치
pip install transformers datasets accelerate bitsandbytes peft

### 모델 다운로드
model_download.py

### 데이터 셋 준비.
dataset_load.py

### LoRA 기반 파인튜닝 (RAM 절약)
lora_finetune.py

---

### * Ubuntu
# fine_tuning_test

### Ubuntu 필수 패키지 설치
# cmake: 크로스 플랫폼 빌드 자동화 도구로, C/C++ 프로젝트 빌드에 사용됩니다.
# 많은 머신러닝 라이브러리들이 C++ 기반이므로 컴파일 시 필요합니다.
sudo apt-get update
sudo apt-get install wget git cmake

### Miniconda 설치 (Linux/Ubuntu)
# Miniconda 설치 스크립트 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 스크립트 실행
bash Miniconda3-latest-Linux-x86_64.sh

# 설치 스크립트 삭제 (선택사항)
rm Miniconda3-latest-Linux-x86_64.sh

### Conda 초기화
# conda 명령어 활성화 (설치 경로에 따라 다를 수 있음)
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
# 또는
eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"

# conda 초기화
conda init bash

# 터미널 재시작 효과
source ~/.bashrc

# (선택사항) conda base 환경 자동 활성화 비활성화
conda config --set auto_activate_base false

### 가상환경 생성 및 활성화
# Python 3.10 버전으로 새로운 환경 생성
conda create -n llm-finetune python=3.10

# 가상환경 활성화
conda activate llm-finetune

### 파이썬 패키지 설치
pip install torch torchvision torchaudio

### PyTorch가 Metal 가속을 지원하는지 확인
check_gpu.py

### 모델 다운로드( 테스트 모델 : Llama 2)

### 허깅페이스 패키지 설치
pip install transformers datasets accelerate bitsandbytes peft

### 모델 다운로드
model_download.py

### 데이터 셋 준비.
dataset_load.py

### LoRA 기반 파인튜닝 (RAM 절약)
lora_finetune.py

save_pretrained 명령어로 모델을 파인튜닝 후 로컬에 PyTorch 형식으로 저장.

(llm-finetune) root@b68e9f3b5f12:/workspaces/fine_tuning_test# python lora_finetune.py 
{'quote': '“Be yourself; everyone else is already taken.”', 'author': 'Oscar Wilde', 'tags': ['be-yourself', 'gilbert-perreira', 'honesty', 'inspirational', 'misattributed-oscar-wilde', 'quote-investigator']}
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.51s/it]
✅ 모델 로드 완료!
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
  0%|                                                                                                                                                                                                              | 0/1881 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/torch/utils/checkpoint.py:87: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
Traceback (most recent call last):
  File "/workspaces/fine_tuning_test/lora_finetune.py", line 97, in <module>
    trainer.train()
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/transformers/trainer.py", line 2241, in train
    return inner_training_loop(
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/transformers/trainer.py", line 2548, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/transformers/trainer.py", line 3740, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/accelerate/accelerator.py", line 2325, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/root/miniconda3/envs/llm-finetune/lib/python3.10/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
  0%|          | 0/1881 [00:01<?, ?it/s]                          

  위 에러 해결중.