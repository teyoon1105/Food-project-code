import os  # 파일 및 경로 작업

Now_path = os.getcwd()
model_folder = os.path.join(Now_path, 'model')
model_list = os.listdir(model_folder)
print(model_list)