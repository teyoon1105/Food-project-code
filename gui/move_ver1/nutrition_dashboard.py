import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class NutritionDashboard(QMainWindow):
    go_home_signal = pyqtSignal()  # 홈 버튼 클릭 시 발생하는 신호

    def __init__(self):
        super().__init__()
        self.nutrition_summary = None  # 초기화 시 NutritionSummary가 없음
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Nutrition Dashboard")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #282a36; color: #f8f8f2;")

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # 고객 이름 라벨 초기화
        self.name_label = QLabel("고객 정보를 입력해주세요.")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("""
            color: #f8f8f2;
            font-family: '맑은 고딕';
            font-size: 20px;
            font-weight: bold;
        """)
        self.layout.addWidget(self.name_label)

        # 차트 영역 초기화
        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout, stretch=4)

        self.calorie_canvas = FigureCanvas(plt.figure())
        self.top_layout.addWidget(self.calorie_canvas)

        self.nutrition_canvas = FigureCanvas(plt.figure())
        self.top_layout.addWidget(self.nutrition_canvas)

        # 영양 상태 타이틀
        bottom_title = QLabel("Nutrient Intake Status")
        bottom_title.setAlignment(Qt.AlignCenter)
        bottom_title.setStyleSheet("color: #f8f8f2; font-size: 16px; font-weight: 1500;")
        self.layout.addWidget(bottom_title)

        # 영양소 진행 바 레이아웃
        self.bottom_layout = QHBoxLayout()
        self.layout.addLayout(self.bottom_layout, stretch=2)

        # 홈 버튼 추가
        self.add_home_button()

    def update_dashboard(self, nutrition_summary):
        self.nutrition_summary = nutrition_summary

        # 고객 이름 라벨 업데이트
        customer_name = self.nutrition_summary.get_customer_name()
        self.name_label.setText(f"{customer_name}님의 섭취 칼로리 및 영양정보")

        # 차트 업데이트
        self.calorie_canvas.figure = self.create_bar_chart()
        self.calorie_canvas.draw()

        self.nutrition_canvas.figure = self.create_pie_chart()
        self.nutrition_canvas.draw()

        # 영양소 진행 바 업데이트
        for i in reversed(range(self.bottom_layout.count())):
            widget = self.bottom_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.add_nutrient_progress_bars()

    def add_home_button(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addSpacerItem(spacer)

        home_button = QPushButton("Home")
        home_button.setStyleSheet("""
            background-color: #6272a4;
            color: #f8f8f2;
            font-size: 14px;
            padding: 10px;
            border-radius: 5px;
        """)
        home_button.clicked.connect(self.emit_go_home_signal)  # 신호 발생 연결
        self.layout.addWidget(home_button)

    def emit_go_home_signal(self):
        self.go_home_signal.emit()  # 홈 화면 전환 신호 발생

    def add_nutrient_progress_bars(self):
        if not self.nutrition_summary:
            return  # NutritionSummary가 없으면 진행 바를 추가하지 않음

        nutrient_percentages = self.nutrition_summary.get_nutrient_percentages()

        colors = {
            'ca': QColor(139, 233, 253),
            'k': QColor(189, 147, 249),
            'fe': QColor(255, 121, 198),
            'zn': QColor(85, 85, 100),
            'mg': QColor(240, 240, 240)
        }

        for nutrient, percentage in nutrient_percentages.items():
            label = self.get_nutrient_label(nutrient)
            color = colors.get(nutrient, QColor(200, 200, 200))
            self.add_nutrient_progress(label, percentage, color)

    def get_nutrient_label(self, nutrient_key):
        labels = {
            'ca': "Calcium",
            'k': "Potassium",
            'fe': "Iron",
            'zn': "Zinc",
            'mg': "Magnesium"
        }
        return labels.get(nutrient_key, "Unknown")

    def add_nutrient_progress(self, label, value, color):
        progress = QWidget()
        progress.setMinimumSize(150, 150)
        progress.paintEvent = lambda event: self.draw_circular_progress(event, progress, value, label, color)
        self.bottom_layout.addWidget(progress)

    def draw_circular_progress(self, event, widget, value, label, color):
        painter = QPainter(widget)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = widget.rect()
        rect_adjusted = rect.adjusted(20, 20, -20, -20)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(50, 50, 60))
        painter.drawEllipse(rect)

        pen = QPen(color, 10)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(
            rect_adjusted,
            90 * 16,
            int(value * 3.6 * 16),
        )

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 15, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{value}%\n{label}")

    def create_bar_chart(self):
        if not self.nutrition_summary:
            return plt.figure()  # NutritionSummary가 없으면 빈 차트 반환

        consumed_calories, _, one_meal_recommend_calories = self.nutrition_summary.get_calories_data()
        fig, ax = plt.subplots()
        categories = ['Calories', 'Recommended Calories']
        values = [consumed_calories, one_meal_recommend_calories]
        ax.bar(categories, values, color=['#6272a4', '#bd93f9'])
        ax.set_title("Calorie Comparison", fontdict={'color': '#f8f8f2', 'weight': 'bold', 'size': 12})
        ax.tick_params(colors='#f8f8f2')
        fig.patch.set_facecolor('#282a36')
        ax.set_facecolor('#282a36')
        return fig

    def create_pie_chart(self):
        if not self.nutrition_summary:
            return plt.figure()  # NutritionSummary가 없으면 빈 차트 반환

        consumed_ratios = self.nutrition_summary.get_consumed_nutrient_ratio()
        recommended_ratios = self.nutrition_summary.get_recommended_nutrient_ratio()

        current = [
            consumed_ratios['proteins'],
            consumed_ratios['carbohydrates'],
            consumed_ratios['fats']
        ]
        recommended = recommended_ratios

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#282a36')

        axs[0].pie(current, labels=['Protein', 'Carb', 'Fat'], autopct='%1.1f%%',
                   colors=['#6272a4', '#bd93f9', '#ff79c6'],
                   textprops={'color': '#f8f8f2'})
        axs[0].set_title("Current Ratio", fontdict={'color': '#f8f8f2', 'weight': 'bold', 'size': 12})

        axs[1].pie(recommended, labels=['Protein', 'Carb', 'Fat'], autopct='%1.1f%%',
                   colors=['#6272a4', '#bd93f9', '#ff79c6'],
                   textprops={'color': '#f8f8f2'})
        axs[1].set_title("Recommended Ratio", fontdict={'color': '#f8f8f2', 'weight': 'bold', 'size': 12})

        for ax in axs:
            ax.set_facecolor('#282a36')
        return fig

# if __name__ == "__main__":
#     user_csv_file_path = "FOOD_DB/user_info.csv"
#     diet_csv_file_path = "FOOD_DB/customer_diet_detail.csv"

#     nutrition_summary = NutritionSummary(user_csv_file_path, diet_csv_file_path)
#     app = QApplication(sys.argv)
#     window = NutritionDashboard(nutrition_summary)
#     window.show()
#     sys.exit(app.exec_())
