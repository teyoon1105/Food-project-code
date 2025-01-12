import torch

# torch 버전 및 gpu를 사용할 수 있는 지 확인
# 공통적으로 PyTorch 버전 출력
print("PyTorch 버전:", torch.__version__)

Mac_bool = True  # Mac 환경이면 True, Window면 False

if Mac_bool:
    # Mac 환경에서 MPS 확인
    print("MPS 사용 가능 여부:", torch.backends.mps.is_available())
    print("MPS 빌드 여부:", torch.backends.mps.is_built())
else:
    # Windows 환경에서 CUDA 확인
    print("CUDA 사용 가능 여부:", torch.cuda.is_available())
    print("CUDA 버전:", torch.version.cuda)
    print("CuDNN 사용 여부:", torch.backends.cudnn.enabled)