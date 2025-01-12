from PyQt5 import QtCore, QtWidgets, QtGui
import threading
import cv2
from camera import DepthVolumeCalculator
from get_weight import GetWeight
from food_processor import FoodProcessor
import traceback
from summary import NutritionSummary
from customer import CustomerManager


class ListDisplayWindow(QtWidgets.QWidget):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app  # MainApp 참조 저장
        self.customer_id = None  # Customer ID를 저장하는 속성 추가
        self.setup_ui()

        # Camera and Weight Components
        self.volume_calculator = DepthVolumeCalculator(
            model_path="model/large_epoch200.pt",
            roi_points=[(130, 15), (1020, 665)],
            cls_name_color={
                '01011001': ('Rice', (255, 0, 255), 0),
                '04017001': ('Soybean Soup', (0, 255, 255), 1),
                '06012004': ('Tteokgalbi', (0, 255, 0), 2),
                '07014001': ('Egg Roll', (0, 0, 255), 3),
                '11013007': ('Spinach Namul', (255, 255, 0), 4),
                '12011008': ('Kimchi', (100, 100, 100), 5),
                '01012006': ('Black Rice', (255, 0, 255), 6),
                '04011005': ('Seaweed Soup', (0, 255, 255), 7),
                '04011007': ('Beef Stew', (0, 255, 255), 8),
                '06012008': ('Beef Bulgogi', (0, 255, 0), 9),
                '08011003': ('Stir-fried Anchovies', (0, 0, 255), 10),
                '10012001': ('Chicken Gangjeong', (0, 0, 255), 11),
                '11013002': ('Fernbrake Namul', (255, 255, 0), 12),
                '12011003': ('Radish Kimchi', (100, 100, 100), 13),
                '01012002': ('Bean Rice', (255, 0, 255), 14),
                '04011011': ('Fish Cake Soup', (0, 255, 255), 15),
                '07013003': ('Kimchi Pancake', (0, 0, 255), 16),
                '11013010': ('Bean Sprouts Namul', (255, 255, 0), 17),
                '03011011': ('Pumpkin Soup', (255, 0, 255), 18),
                '08012001': ('Stir-fried Potatoes', (255, 255, 0), 19)
            },
            food_processor=FoodProcessor(
                food_data_path='FOOD_DB/food_project_food_info.csv',
                real_time_csv_path='FOOD_DB/real_time_food_info.csv',
                customer_diet_csv_path='FOOD_DB/customer_diet_detail.csv',
                min_max_table='FOOD_DB/quantity_min_max.csv'
            ),
            total_calories=0
        )

        self.is_running = False

    def setup_ui(self):
        self.setObjectName("window2")
        self.resize(1200, 700)

        self.camera_frame = QtWidgets.QLabel(self)
        self.camera_frame.setGeometry(QtCore.QRect(20, 40, 880, 640))
        self.camera_frame.setAlignment(QtCore.Qt.AlignCenter)

        # self.list_widget = QtWidgets.QListWidget(self)
        # self.list_widget.setGeometry(QtCore.QRect(920, 40, 250, 521))
        
        # 상단 리스트 위젯
        self.list_widget_top = QtWidgets.QListWidget(self)
        self.list_widget_top.setGeometry(QtCore.QRect(920, 40, 250, 400))  # 상단 리스트의 위치와 크기
        self.list_widget_top.setStyleSheet("font-size: 10pt;background-color: #2C2F33;  /* 진한 회색 */")

        # 하단 리스트 위젯
        self.list_widget_bottom = QtWidgets.QListWidget(self)
        self.list_widget_bottom.setGeometry(QtCore.QRect(920, 450, 250, 111))  # 하단 리스트의 위치와 크기
        self.list_widget_bottom.setStyleSheet("font-size: 12pt;background-color: #2C2F33;  /* 진한 회색 */") 
    
        self.btn_complete = QtWidgets.QPushButton("배식완료", self)
        self.btn_complete.setGeometry(QtCore.QRect(920, 580, 250, 50))
        self.btn_complete.setStyleSheet("font-family: 'Arial'; font-size: 14pt; font-weight: bold;")
        self.btn_complete.clicked.connect(self.save_results)

        # "대시보드 보러가기" 버튼 추가
        self.btn_go_dashboard = QtWidgets.QPushButton("대시보드 보러가기", self)
        self.btn_go_dashboard.setGeometry(QtCore.QRect(920, 630, 250, 50))
        self.btn_go_dashboard.setStyleSheet("font-family: 'Arial'; font-size: 14pt; font-weight: bold;")
        self.btn_go_dashboard.clicked.connect(self.go_to_dashboard)
        
        self.label_customer_id = QtWidgets.QLabel("Customer ID: Not Set", self)
        self.label_customer_id.setGeometry(QtCore.QRect(20, 10, 400, 30))

    def set_customer_id(self, user_id):
        """Customer ID를 설정하고 UI를 업데이트합니다."""
        self.customer_id = user_id
        self.label_customer_id.setText(f"Customer ID: {self.customer_id} ")
        self.label_customer_id.setStyleSheet("font-weight: bold; font-size: 12pt;")
        print(f"[DEBUG] Customer ID set in ListDisplayWindow: {self.customer_id}")  # 디버깅 로그


    def showEvent(self, event):
        super().showEvent(event)
        print(f"[DEBUG] ListDisplayWindow displayed with Customer ID: {self.customer_id}")  # 디버깅 로그
        if not self.customer_id:
            self.label_customer_id.setText("Customer ID: Not Set")
            print("[ERROR] Customer ID not set before showing ListDisplayWindow.")
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.run_main_loop, daemon=True).start()
            
    
    # current frame 대신 confirmed_object를 가져와서 실시간 업데이트 
    def run_main_loop(self):
        if not self.customer_id:
            print("Error: Customer ID is not set. Exiting main loop.")
            return
        # obj_id와 한글명을 매핑하는 딕셔너리
        id_to_korean_name = {
            "01011001": "쌀밥",
            "04017001": "된장찌개",
            "06012004": "떡갈비",
            "07014001": "달걀말이",
            "11013007": "시금치",
            "12011008": "배추김치",
            "01012006": "흑미밥",
            "04011005": "미역국",
            "04011007": "소고기무국",
            "06012008": "소불고기",
            "08011003": "멸치볶음",
            "10012001": "닭강정",
            "11013002": "고사리",
            "12011003": "깍두기",
            "01012002": "콩밥",
            "04011011": "어묵국",
            "07013003": "김치전",
            "11013010": "콩나물",
            "03011011": "호박죽",
            "08012001": "감자볶음"
        }
        
        id_to_calories_per_gram = {
            "01011001": 1.5942857142857143,  # 쌀밥
            "04017001": 0.36765,             # 된장찌개
            "06012004": 3.05196,             # 떡갈비
            "07014001": 1.7224,              # 달걀말이
            "11013007": 0.7502,              # 시금치나물
            "12011008": 0.3686,              # 배추김치
            "01012006": 1.59,                # 흑미밥
            "04011005": 0.10034,             # 미역국
            "04011007": 0.31305,             # 소고기무국
            "06012008": 0.87365,             # 소불고기
            "08011003": 3.4615,              # 멸치볶음
            "10012001": 3.2325,              # 닭강정
            "11013002": 0.8702,              # 고사리나물
            "12011003": 0.3598,              # 깍두기
            "01012002": 1.6145,              # 콩밥
            "04011011": 0.42015,             # 어묵국
            "07013003": 1.9048666666666667,  # 김치전
            "11013010": 0.4826,              # 콩나물
            "03011011": 0.7172166666666667,  # 호박죽
            "08012001": 1.156                # 감자볶음
        }

        try:
            
            # CustomerManager 객체 생성
            customer_manager = CustomerManager(user_csv_file_path="user_info.csv")
            customer_info = customer_manager.get_customer_info(self.customer_id)
            # 고객의 BMR 값을 계산
            # customer_id = self.customer_id
            # 필요한 데이터 추출
            weight = customer_info.get("weight")
            height = customer_info.get("height")
            birth_date = customer_info.get("birth")
            birth_date = str(birth_date)
            gender = customer_info.get("gender")
            exercise_score = customer_info.get("exercise", 'C')  # 기본값 'C'

            # 나이 계산
            age = customer_manager.calculate_age(birth_date)

            # BMR 계산
            bmr = customer_manager.calculate_bmr(weight, height, age, gender, exercise_score)

            # 한 끼 권장 칼로리 계산
            recommended_calories = bmr / 3

            for blend_image, _ in self.volume_calculator.main_loop(self.customer_id):
                
                
                # Display blend image on QLabel
                rgb_image = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.camera_frame.setPixmap(pixmap)

                # Update Top List Widget (Confirmed Objects)
                self.list_widget_top.clear()
                self.list_widget_bottom.clear()
                
                # 객체별 개별 칼로리 추적
                total_calories_in_current_loop = 0  # 현재 루프에서 누적 칼로리
                # 이부분괜찮을지 id 필요없을지
                for obj_id, obj_data in self.volume_calculator.confirmed_objects.items():
                    # previous_calories = 0  # 이전 객체들의 총 칼로리를 추적
                    obj_name = obj_data.get("obj_name")
                    korean_name = id_to_korean_name.get(obj_id, obj_name)  # 매핑되지 않은 경우 원래 이름 사용
                    total_weight = obj_data.get("weight", 0)
                    
                    calories_per_gram = id_to_calories_per_gram.get(obj_id, 0)
                    current_calories = total_weight * calories_per_gram
                    total_calories_in_current_loop += current_calories  # 루프 내에서 누적

                    # Debugging: 값 확인
                    # print(f"[DEBUG] Current Object ID: {obj_id}")
                    # print(f"[DEBUG] Current Calories: {current_calories}")
                    # print(f"[DEBUG] Total Calories in Loop (so far): {total_calories_in_current_loop}")


                    # Add to widget
                    self.list_widget_top.addItem(f"{korean_name} : {current_calories:.2f}kcal, {total_weight}g")
                # self.list_widget_bottom.addItem(f"total calories : {round(self.volume_calculator.total_calories,2)}")
                self.list_widget_bottom.addItem("누적칼로리/권장칼로리")
                self.list_widget_bottom.addItem(f"{round(self.volume_calculator.total_calories,2)}kcal/{round(recommended_calories, 2)}kcal")

        except Exception as e:
            print(f"Error in volume_calculator.main_loop: {e}")
            print(traceback.format_exc())
        
    
    def save_results(self):
        """Complete 버튼 클릭 시 저장 호출"""
        try:
            success = self.volume_calculator.save_results()
            print(f"[DEBUG] save_results returned: {success}")  # 반환값 로그 출력

            self.is_running = False  # 메인 루프 종료 플래그 설정

            # 메시지 박스 생성
            msg_box = QtWidgets.QMessageBox(self)

            if success:
                msg_box.setWindowTitle("Complete")
                msg_box.setText("Results saved successfully!")
                msg_box.setIcon(QtWidgets.QMessageBox.Information)
            else:
                msg_box.setWindowTitle("Incomplete")
                msg_box.setText("No results were saved.")
                msg_box.setIcon(QtWidgets.QMessageBox.Warning)

            # 스타일시트 적용
            msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2C2F33;  /* 배경색 진한 회색 */
                color: white;              /* 텍스트 흰색 */
            }
            QPushButton {
                background-color: #4F545C; /* 버튼 배경 회색 */
                color: white;              /* 버튼 텍스트 흰색 */
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #6C717C; /* 버튼 호버 색상 */
            }
            """)

            msg_box.exec_()
        except Exception as e:
            print(f"Error in save_results: {e}")
            # 오류 메시지 박스 생성
            error_box = QtWidgets.QMessageBox(self)
            error_box.setWindowTitle("Error")
            error_box.setText(f"Failed to save results: {e}")
            error_box.setIcon(QtWidgets.QMessageBox.Critical)

            # 스타일시트 적용
            error_box.setStyleSheet("""
            QMessageBox {
                background-color: #2C2F33;  /* 배경색 진한 회색 */
                color: white;              /* 텍스트 흰색 */
            }
            QPushButton {
                background-color: #4F545C; /* 버튼 배경 회색 */
                color: white;              /* 버튼 텍스트 흰색 */
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #6C717C; /* 버튼 호버 색상 */
            }
            """)

            error_box.exec_()
    
    def go_to_dashboard(self):
        """대시보드 화면으로 이동"""
        try:
            # 새 NutritionSummary 객체 생성
            nutrition_summary = NutritionSummary(
                "user_info.csv",  # 유저 정보 파일 경로
                "FOOD_DB/customer_diet_detail.csv",  # 새로운 데이터가 반영된 파일 경로
                self.customer_id  # 현재 고객 ID
            )

            # 대시보드 업데이트
            self.main_app.nutrition_dashboard.update_dashboard(nutrition_summary)

            # 화면 전환
            self.main_app.switch_window(3)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load updated data: {e}")
            
    def closeEvent(self, event):
        self.is_running = False
        super().closeEvent(event)


