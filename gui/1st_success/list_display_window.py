from PyQt5 import QtCore, QtWidgets, QtGui
import threading
import cv2
from volume_mask_with_gpu import DepthVolumeCalculator

class ListDisplayWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.volume_calculator = DepthVolumeCalculator(
            model_path="model/large_epoch200.pt",
            roi_points=[(130, 15), (1020, 665)],
            cls_name_color={
    '01011001': ('Rice', (255, 0, 255), 0),  # Steamed Rice
    '04017001': ('Soybean Soup', (0, 255, 255), 1),  # Soybean Paste Stew
    '06012004': ('Tteokgalbi', (0, 255, 0), 2),  # Grilled Short Rib Patties (Tteokgalbi)
    '07014001': ('Egg Roll', (0, 0, 255), 3),  # Rolled Omelette
    '11013007': ('Spinach Namul', (255, 255, 0), 4),  # Spinach Namul
    '12011008': ('Kimchi', (100, 100, 100), 5),  # Napa Cabbage Kimchi
    '01012006': ('Black Rice', (255, 0, 255), 6),  # Black Rice
    '04011005': ('Seaweed Soup', (0, 255, 255), 7),  # Seaweed Soup
    '04011007': ('Beef Stew', (0, 255, 255), 8),  # Beef Radish Soup
    '06012008': ('Beef Bulgogi', (0, 255, 0), 9),  # Beef Bulgogi
    '08011003': ('Stir-fried Anchovies', (0, 0, 255), 10),  # Stir-fried Anchovies
    '10012001': ('Chicken Gangjeong', (0, 0, 255), 11),  # Sweet and Spicy Fried Chicken
    '11013002': ('Fernbrake Namul', (255, 255, 0), 12),  # Fernbrake Namul
    '12011003': ('Radish Kimchi', (100, 100, 100), 13),  # Radish Kimchi (Kkakdugi)
    '01012002': ('Bean Rice', (255, 0, 255), 14),  # Soybean Rice
    '04011011': ('Fish Cake Soup', (0, 255, 255), 15),  # Fish Cake Soup
    '07013003': ('Kimchi Pancake', (0, 0, 255), 16),  # Kimchi Pancake
    '11013010': ('Bean Sprouts Namul', (255, 255, 0), 17),  # Bean Sprout Namul
    '03011011': ('Pumpkin Soup', (255, 0, 255), 18),  # Pumpkin Porridge
    '08012001': ('Stir-fried Potatoes', (255, 255, 0), 19)  # Stir-fried Potatoes
}  # 클래스 매핑
        )
        self.is_running = False

    def setup_ui(self):
        self.setObjectName("window2")
        self.resize(1200, 700)

        self.camera_frame = QtWidgets.QLabel(self)
        self.camera_frame.setGeometry(QtCore.QRect(20, 40, 880, 640))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)

        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setGeometry(QtCore.QRect(920, 40, 250, 521))

        self.btn_complete = QtWidgets.QPushButton("Complete", self)
        self.btn_complete.setGeometry(QtCore.QRect(920, 580, 250, 100))

    def showEvent(self, event):
        super().showEvent(event)
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.run_main_loop, daemon=True).start()

    def run_main_loop(self):
        for blend_image, current_frame_objects in self.volume_calculator.main_loop():
            # OpenCV 이미지 → QPixmap 변환
            rgb_image = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.camera_frame.setPixmap(pixmap)

            # QListWidget에 객체 정보 추가
            self.list_widget.clear()
            for obj_id, (obj_name, bbox, region, volume) in current_frame_objects.items():
                self.list_widget.addItem(f"{obj_name}: {volume:.1f}cm³ in region {region}")

    def closeEvent(self, event):
        self.is_running = False
        super().closeEvent(event)

