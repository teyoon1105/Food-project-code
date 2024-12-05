import json
import os

def transform_polygon_coordinates_in_folder(json_folder, transform_func):
    """
    폴더 안의 모든 JSON 파일에서 폴리곤 좌표를 변환하고, 같은 파일에 덮어씌움.

    Args:
        json_folder (str): JSON 파일들이 저장된 폴더 경로.
        transform_func (function): 좌표를 변환하는 함수. (예: lambda x, y: (x*2, y*2))
    """
    # 폴더 내 모든 JSON 파일 처리
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):  # JSON 파일만 처리
            json_path = os.path.join(json_folder, file_name)

            # JSON 파일 읽기
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 폴리곤 좌표 변환
            for shape in data.get("shapes", []):
                if shape["shape_type"] == "polygon":  # 폴리곤인 경우만 처리
                    shape["points"] = [
                        transform_func(x, y) for x, y in shape["points"]
                    ]

            # 변환된 데이터를 같은 파일에 덮어쓰기
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Transformed and saved: {json_path}")


# 좌표 변환 공식
def custom_transform(x, y):
    new_x = 0.5 * x + 320
    new_y = 0.6 * y + 256
    return new_x, new_y


# 사용 예시
json_folder_path = "C:/Users/SBA/Desktop/scale/scale_0411005/json_label"  # JSON 파일들이 저장된 폴더 경로

# 함수 실행
transform_polygon_coordinates_in_folder(
    json_folder=json_folder_path,
    transform_func=custom_transform
)
