## mac os
# import torch

# print(torch.backends.mps.is_available())  # True이면 성공

## ubuntu
import torch

# CUDA 사용 가능 여부 체크
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("사용 가능한 GPU:", torch.cuda.get_device_name(0))
    print("현재 GPU 개수:", torch.cuda.device_count())
else:
    print("CUDA를 사용할 수 없습니다. 다음을 확인해주세요:")
    print("1. NVIDIA GPU가 설치되어 있는지")
    print("2. NVIDIA 드라이버가 설치되어 있는지")
    print("3. CUDA toolkit이 설치되어 있는지")