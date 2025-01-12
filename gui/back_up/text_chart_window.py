from PyQt5 import QtCore, QtWidgets

class TextChartWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("window3")
        self.resize(1200, 700)

        self.text_box = QtWidgets.QTextEdit(self)
        self.text_box.setGeometry(QtCore.QRect(50, 130, 1100, 300))

        self.user_name = QtWidgets.QLabel("User Name", self)
        self.user_name.setGeometry(QtCore.QRect(50, 20, 200, 100))

        self.btn_home = QtWidgets.QPushButton("Home", self)
        self.btn_home.setGeometry(QtCore.QRect(960, 20, 200, 100))

        self.chart_1 = QtWidgets.QLabel("Chart 1", self)
        self.chart_1.setGeometry(QtCore.QRect(50, 460, 350, 250))
        self.chart_1.setAlignment(QtCore.Qt.AlignCenter)

        self.chart_2 = QtWidgets.QLabel("Chart 2", self)
        self.chart_2.setGeometry(QtCore.QRect(420, 460, 350, 250))
        self.chart_2.setAlignment(QtCore.Qt.AlignCenter)

        self.chart_3 = QtWidgets.QLabel("Chart 3", self)
        self.chart_3.setGeometry(QtCore.QRect(790, 460, 350, 250))
        self.chart_3.setAlignment(QtCore.Qt.AlignCenter)