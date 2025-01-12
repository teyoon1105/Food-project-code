import cv2
import os
import numpy as np
import glob
import csv
import re

class FaceRegistration:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_extractor(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        return [img[y:y + h, x:x + w] for (x, y, w, h) in faces]

    def generate_folder(self, name):
        i = 1
        while True:
            folder_name = f"sesac{i:02d}"
            folder_path = os.path.join("train", folder_name)
            if not os.path.exists(folder_path):
                return folder_name, folder_path
            i += 1

    def save_csv(self, user_info, filename="user_info.csv"):
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "아이디", "이름", "생년월일", "성별", "연락처", "주소", "키", "몸무게", "운동량"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(user_info)

    def train_model(self, train_path):
        Training_Data, Labels = [], []
        image_paths = glob.glob(os.path.join(train_path, "*.jpg"))

        for i, image_path in enumerate(image_paths):
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)

        Labels = np.asarray(Labels, dtype=np.int32)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.asarray(Training_Data), np.asarray(Labels))

        model_file_path = os.path.join(train_path, "trained_model.yml")
        model.save(model_file_path)
        print("Model training completed.")

    def run(self, training_folder):
        if not training_folder or not os.path.exists(training_folder):
            print("Invalid training folder.")
            return

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            faces = self.face_extractor(frame)

            if faces:
                for i, face in enumerate(faces):
                    count += 1
                    face = cv2.resize(face, (200, 200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    file_name_path = os.path.join(training_folder, f"image_{count}.jpg")
                    cv2.imwrite(file_name_path, face)

                    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('FACE TRAIN', face)
            else:
                print("No face detected.")

            if cv2.waitKey(1) == 27 or count >= 200:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {count} images in {training_folder}.")

        # Train model
        self.train_model(training_folder)
