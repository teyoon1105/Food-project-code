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
                    user_id = row["id"]
                    user_name = row["name"]
                    train_path = os.path.join("train", user_id)

                    if user_id not in self.label_mapping:
                        self.label_mapping[user_id] = {"label": label_count, "name": user_name}
                        label_count += 1

                    image_paths = glob.glob(os.path.join(train_path, "*.jpg"))
                    for image_path in image_paths:
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            print(f"Failed to load image: {image_path}")
                        else:
                            Training_Data.append(np.asarray(image, dtype=np.uint8))
                            Labels.append(self.label_mapping[user_id]["label"])

        

            if len(Training_Data) < 2:
                print("Not enough training data. At least 2 images are required.")
                return

            Labels = np.asarray(Labels, dtype=np.int32)
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model.train(np.asarray(Training_Data), np.asarray(Labels))
            self.model.save("unified_trained_model.yml")
            print("Unified model created and saved.")
        except FileNotFoundError:
            print("CSV file not found for training.")
        except Exception as e:
            print(f"Error during model creation: {e}")

    def detect_and_recognize(self, img):
        """Detects faces and recognizes user ID if a face is found."""
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is None or len(faces) == 0:
            print("No faces detected.")
            return img, None

        print("Running face detection and recognition...")
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))

            if self.model is not None:
                label, confidence = self.model.predict(roi)
                user_id = list(self.label_mapping.keys())[label]
                user_name = self.label_mapping[user_id]["name"]
                confidence_percent = int(100 * (1 - (confidence / 300)))

                print(f"Detected {user_name} ({user_id}) with {confidence_percent}% confidence.")
                if confidence_percent > 80:
                    self.correct_uid = user_id
                    try:
                        playsound.playsound("C:/Github/50_project/move_ver1/sound/01_user_login.mp3")
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                    return img, user_id
                else:
                    print("Confidence too low to identify user.")
            else:
                print("ㅈ됨")
        return img, None

    @property
    def get_correct_uid(self):
        """Returns the last recognized user ID."""
        return self.correct_uid