# fine_tuning_test

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

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
