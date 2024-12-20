import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

# QmainWindow를 상속 받아 앱의 Main Window를 커스텀 합시다!
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press Me! for god sake.")

        self.setFixedSize(QSize(1280, 720))

        # 윈도우 중앙에 위치할 Widget 설정
        self.setCentralWidget(button)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()