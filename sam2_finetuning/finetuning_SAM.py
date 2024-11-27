import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path 

class SimpleSAM:

    def modify_sam_for_640(model): 
        """SAM 모델을 640x640 입력에 맞게 수정""" 
        # 이미지 인코더의 포지셔널 임베딩 수정 
        old_pe = model.image_encoder.pos_embed 
        new_pe = torch.nn.Parameter(
                 torch.nn.functional.interpolate(
                         old_pe.unsqueeze(0), 
                         size=(40, 40),  # 1024->640에 맞는 크기 
                         mode='bicubic', 
                         align_corners=False ).squeeze(0) )
        model.image_encoder.pos_embed = new_pe 
        return model

    def __init__(self, model_path):
        """
        SAM 모델 초기화

        Args:
            model_path (str): SAM2 모델 가중치 파일 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 또는 CPU 사용 설정
        self.model = self.load_sam_model(model_path) # SAM 모델 로드
        self.model = self.modify_sam_for_640(self.model)
        self.model.to(self.device) # 모델을 device로 이동


    def load_sam_model(self, model_path):
        """
        SAM 모델 로드

        Args:
            model_path (str): SAM2 모델 가중치 파일 경로, 예: sam2_b.pt 

        Returns:
            sam_model: 로드된 SAM 모델
        """
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            # sam1 기준
            # # 모델 타입 결정 (vit_s, vit_b, vit_l, vit_h)
            # if 's' in model_path.lower():
            #     model_type = 'vit_s'
            # elif 'b' in model_path.lower():
            #     model_type = 'vit_b'
            # elif 'l' in model_path.lower():
            #     model_type = 'vit_l'
            # else:
            #     model_type = 'vit_h'

            # model type은 sam2_b 이렇게 .pt를 제외한 값이 들어가야 함
            # checkpoint는 modelpath에 있는 sam2_b.pt 전부 들어가야 함
            model_type = 'vit_b'
            sam = sam_model_registry[model_type](checkpoint=model_path) # SAM 모델 생성
            return sam

        else:
            raise ValueError("지원하지 않는 모델 형식입니다. .pt 또는 .pth 파일을 사용하세요.")


    def train(self, data, epochs=500, batch=8, name='sam_model'):
        """
        SAM 모델 학습

        Args:
            data (str): 데이터 디렉토리 경로 (이미지와 라벨 포함)
            epochs (int): 학습 에포크 수 (기본값: 500)
            batch (int): 배치 크기 (기본값: 8)
            name (str): 모델 저장 이름 (기본값: 'sam_model')
        """
        # 데이터셋 및 데이터로더 설정
        dataset = self._setup_dataset(data)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

        # 옵티마이저 설정
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001) # Adam 옵티마이저 사용

        # 손실 함수 설정
        criterion = torch.nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss 사용

        # 학습 루프
        print(f"{epochs} 에포크 동안 학습을 시작합니다...")
        for epoch in range(epochs):
            total_loss = 0
            for batch_data in dataloader:
                images = batch_data['image'].to(self.device)
                masks = batch_data['mask'].to(self.device)

                image_embeddings = self.model.image_encoder(images)
                mask_predictions = self.model.mask_decoder(image_embeddings=image_embeddings, multimask_output=False,)
                

                loss = criterion(mask_predictions, masks)  # 손실 계산

                # 역전파
                optimizer.zero_grad() # 그래디언트 초기화
                loss.backward() # 역전파
                optimizer.step() # 가중치 업데이트

                total_loss += loss.item() # 손실 누적

            # 진행 상황 출력
            avg_loss = total_loss / len(dataloader) # 평균 손실 계산
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}") # 에포크별 평균 손실 출력

            # 50 에포크마다 체크포인트 저장
            if (epoch + 1) % 50 == 0:
                self.save_model(f"{name}_epoch_{epoch+1}.pt") # 모델 저장


    def _setup_dataset(self, data_path):
        """
        데이터셋 설정

        Args:
            data_path (str): 데이터 디렉토리 경로

        Returns:
            SAMDataset: 생성된 데이터셋 객체
        """
        class SAMDataset(Dataset):
            def __init__(self, data_path, image_size=640):
                """
                데이터셋 클래스 초기화

                Args:
                    data_path (str): 데이터 디렉토리 경로
                    image_size (int): 이미지 크기 (기본값: 640)
                """
                self.data_path = data_path
                self.image_size = image_size
                self.image_dir = os.path.join(self.data_path, 'images') # 이미지 디렉토리 경로
                self.label_dir = os.path.join(self.data_path, 'labels') # 라벨 디렉토리 경로
                self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]) # 이미지 파일 목록
                self.transforms = transforms.Compose([ # 이미지 변환 설정
                        # transforms.Resize((self.image_size, self.image_size)), # 크기 조정은 이미 한 상태
                        transforms.ToTensor(), # 텐서로 변환
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
                    ])


            def __len__(self):
                """데이터셋 크기 반환"""
                return len(self.image_files)


            def __getitem__(self, idx):
                """
                데이터셋에서 idx번째 아이템 반환

                Args:
                    idx (int): 인덱스

                Returns:
                    dict: 이미지와 마스크를 포함하는 딕셔너리
                """
                image_name = self.image_files[idx] # 이미지 파일 이름
                image_path = os.path.join(self.image_dir, image_name) # 이미지 파일 경로
                label_path = os.path.join(self.label_dir, Path(image_name).stem + '.json') # 라벨 파일 경로

                # 이미지 로드 및 전처리
                image = Image.open(image_path).convert('RGB') # 이미지 로드 (RGB 모드)
                image = self.transforms(image) # 이미지 변환 적용

                # 폴리곤 좌표 로드 및 마스크 생성
                mask = self.create_mask_from_polygon_json(label_path, self.image_size, self.image_size) # 마스크 생성 함수 호출

                return {'image': image, 'mask': mask} # 이미지와 마스크 반환


            #def _create_mask_from_polygon(self, label_path, width, height):
            def create_mask_from_polygon_json(self, label_path, width, height):
                """
                JSON 다각형 좌표 파일을 마스크로 변환

                Args:
                    label_path (str): 라벨 파일 경로 (JSON)
                    width (int): 이미지 너비
                    height (int): 이미지 높이

                Returns:
                    torch.Tensor: 생성된 마스크 텐서
                """

                mask = np.zeros((height, width), dtype=np.uint8)

                with open(label_path, 'r') as f:
                    data = json.load(f)
                    shapes = data['shapes']
                    json_width = data['imageWidth']
                    json_height = data['imageHeight']

                    # 불일치 확인
                    if json_width != width or json_height != height:
                        print(f"경고: JSON의 이미지 크기 ({json_width}, {json_height})가 제공된 크기 ({width}, {height})와 일치하지 않습니다. JSON 크기를 사용합니다.")
                        width = json_width
                        height = json_height

                    for shape in shapes:
                        class_id_json = shape['label'] # 클래스 ID는 문자열입니다.
                        try:
                            class_id = int(class_id_json) # 정수로 변환 시도
                        except ValueError:
                            print(f"경고: 클래스 ID '{class_id}'을 정수로 변환할 수 없습니다. 이 도형은 건너뜁니다.")
                            continue
                        polygon_points = np.array(shape['points']).astype(np.float32)
                        polygon_points[:, 0] *= width  # 너비 스케일링
                        polygon_points[:, 1] *= height  # 높이 스케일링
                        polygon_points = polygon_points.astype(np.int32)  # 정수형 변환
                        cv2.fillPoly(mask, [polygon_points], 255)

                mask = torch.from_numpy(mask).float() / 255.0
                mask = mask.unsqueeze(0)
                return mask


        return SAMDataset(data_path)


    def _calculate_loss(self, outputs, targets):
        """
        손실 계산

        2

        Args:
            outputs (torch.Tensor): 모델 출력
            targets (torch.Tensor): 정답 마스크

        Returns:
            torch.Tensor: 계산된 손실 값
        """
        return torch.nn.BCEWithLogitsLoss()(outputs, targets) # BCEWithLogitsLoss 사용


    def save_model(self, path):
        """
        모델 저장

        Args:
            path (str): 모델 저장 경로
        """
        torch.save(self.model.state_dict(), path)  # 모델 가중치 저장
        print(f"모델이 {path}에 저장되었습니다.")



# 사용 예시
if __name__ == "__main__":
    model_path = "C:/Users/SBA/finetuning/segment-anything/segment_anything/sam_vit_b_01ec64.pth" # SAM 모델 가중치 파일 경로
    data_path = "D:/data_set"  # 데이터 디렉토리 경로 (이미지와 라벨 포함)

    # 모델 객체 생성
    model = SimpleSAM(model_path)

    # 모델 학습
    model.train(data=data_path, epochs=500, batch=8, name='test_1')