import torch
print(torch.__version__)  # PyTorch 버전 출력
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인
print(torch.version.cuda)  # PyTorch가 인식하는 CUDA 버전
print(torch.backends.cudnn.enabled)  # True면 CuDNN 활성화
