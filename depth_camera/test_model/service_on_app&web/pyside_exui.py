# -*- coding: utf-8 -*-
################################################################################
## UI 파일 'pyside_exui.ui'로부터 생성된 폼
##
## Qt User Interface Compiler 버전 6.8.1로 생성됨
##
## 경고! UI 파일을 다시 컴파일하면 이 파일의 모든 변경사항이 손실됩니다!
################################################################################

# PySide6의 핵심 기능들을 임포트
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
   QMetaObject, QObject, QPoint, QRect, 
   QSize, QTime, QUrl, Qt)

# PySide6의 GUI 관련 클래스들을 임포트 (그래픽, 폰트, 색상 등)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
   QFont, QFontDatabase, QGradient, QIcon,
   QImage, QKeySequence, QLinearGradient, QPainter,
   QPalette, QPixmap, QRadialGradient, QTransform)

# PySide6의 위젯 관련 클래스들을 임포트
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QListWidget,
   QListWidgetItem, QPushButton, QSizePolicy, QWidget)

class Ui_Form(object):
   def setupUi(self, Form):
       # Form 객체의 이름이 설정되어 있지 않으면 "Form"으로 설정
       if not Form.objectName():
           Form.setObjectName(u"Form")
       # 폼의 크기를 1280x720으로 설정
       Form.resize(1280, 720)

       # QLabel을 사용하여 프레임 생성
       self.frame = QLabel(Form)
       self.frame.setObjectName(u"frame")
       # 프레임의 위치(x:40, y:40)와 크기(880x640) 설정
       self.frame.setGeometry(QRect(40, 40, 880, 640))
       # 프레임 스타일 설정 (테두리 모양과 그림자)
       self.frame.setFrameShape(QFrame.Shape.StyledPanel)
       self.frame.setFrameShadow(QFrame.Shadow.Raised)

       # Complete 버튼 생성
       self.Complete = QPushButton(Form)
       self.Complete.setObjectName(u"Complete")
       # 버튼의 위치(x:960, y:560)와 크기(280x120) 설정
       self.Complete.setGeometry(QRect(960, 560, 280, 120))

       # 리스트 위젯 생성
       self.listWidget = QListWidget(Form)
       self.listWidget.setObjectName(u"listWidget")
       # 리스트 위젯의 위치(x:960, y:40)와 크기(280x480) 설정
       self.listWidget.setGeometry(QRect(960, 40, 280, 480))

       # UI 요소들의 텍스트를 설정하는 함수 호출
       self.retranslateUi(Form)
       # 시그널/슬롯 연결
       QMetaObject.connectSlotsByName(Form)

   def retranslateUi(self, Form):
       # 윈도우 제목 설정
       Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
       # Complete 버튼의 텍스트를 "PushButton"으로 설정
       self.Complete.setText(QCoreApplication.translate("Form", u"PushButton", None))