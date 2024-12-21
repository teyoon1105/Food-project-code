# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pyside_exui.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1280, 720)
        self.frame = QLabel(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(40, 40, 880, 640))
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.Complete = QPushButton(Form)
        self.Complete.setObjectName(u"Complete")
        self.Complete.setGeometry(QRect(960, 560, 280, 120))
        self.listWidget = QListWidget(Form)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setGeometry(QRect(960, 40, 280, 480))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.Complete.setText(QCoreApplication.translate("Form", u"PushButton", None))
    # retranslateUi

