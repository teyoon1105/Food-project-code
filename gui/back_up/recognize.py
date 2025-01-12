import os
import cv2
import csv
import numpy as np
import glob
import playsound

class FaceRecognition:
    def __init__(self, csv_filepath="user_info.csv"):
        self.csv_filepath = csv_filepath
        self.model = None
        self.label_mapping = {}
        self.correct_uid = ''

    def create_unified_model(self):
        """Creates and trains a unified face recognition model from CSV data."""
        if self.model is not None:
            print("Model already created. Skipping re-creation.")
            return
        
        
        Training_Data, Labels = [], []
        label_count = 0

        try:
            with open(self.csv_filepath, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    user_id = row["아이디"]
                    user_name = row["이름"]
                    train_path = os.path.join("train", user_id)

                    if user_id not in self.label_mapping:
                        self.label_mapping[user_id] = {"label": label_count, "name": user_name}
                        label_count += 1

                    image_paths = glob.glob(os.path.join(train_path, "*.jpg"))
                    for image_path in image_paths:
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        Training_Data.append(np.asarray(image, dtype=np.uint8))
                        Labels.append(self.label_mapping[user_id]["label"])

            Labels = np.asarray(Labels, dtype=np.int32)
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model.train(np.asarray(Training_Data), np.asarray(Labels))
            self.model.save("unified_trained_model.yml")
            print("Unified model created and saved.")
        except FileNotFoundError:
            print("CSV file not found for training.")
        except Exception as e:
            print(f"Error during model creation: {e}")

    def face_detector(self, img, size=0.5):
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is None or len(faces) == 0:
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    def face_recognizer(self):
        """Recognizes faces using the unified model."""
        if self.model is None:
            print("Model not loaded. Please create or load a model first.")
            return

        width = 640
        height = 480

        cv2.namedWindow('FACE RECOGNITION', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FACE RECOGNITION', width, height)

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            image, face = self.face_detector(frame)

            try:
                if face is not None and len(face) > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    result = self.model.predict(face)
                    user_id = list(self.label_mapping.keys())[result[0]]
                    user_name = self.label_mapping[user_id]["name"]

                    if result[1] < 500:
                        confidence = int(100 * (1 - (result[1]) / 300))
                        display_string = f"{confidence}% Confidence: {user_name} ({user_id})"

                    cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

                    if confidence > 80:
                        cv2.putText(image, "Enjoy your meal", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('FACE RECOGNITION', image)

                        self.correct_uid = user_id
                        print(display_string)
                        playsound.playsound('./sound/01_user_login.mp3')
                    else:
                        cv2.putText(image, "You are not our member", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('FACE RECOGNITION', image)

            except Exception as e:
                cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('FACE RECOGNITION', image)
                print(f"Error during recognition: {e}")

            if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognition = FaceRecognition()
    face_recognition.create_unified_model()
    face_recognition.face_recognizer()
