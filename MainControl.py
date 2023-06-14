import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSlider
from untitled import *


class MyClass(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent)
        self.setupUi(self)
        self.actionImages.triggered.connect(self.openimage)
        self.Reconstruction.clicked.connect(self.showmodel)
        self.SHLS.setMinimum(1878)  # 设置最小值
        self.SHLS.setMaximum(3878)  # 设置最大值
        self.SHLS.valueChanged.connect(lambda: self.on_change_func(self.SHLS))
        self.SSLS.setMinimum(1503)  # 设置最小值
        self.SSLS.setMaximum(3001)  # 设置最大值
        self.SSLS.valueChanged.connect(lambda: self.on_change_func(self.SSLS))
        self.CWS.setMinimum(1503)  # 设置最小值
        self.CWS.setMaximum(3001)  # 设置最大值
        self.CWS.valueChanged.connect(lambda: self.on_change_func(self.CWS))
        self.HWS.setMinimum(1503)  # 设置最小值
        self.HWS.setMaximum(3001)  # 设置最大值
        self.HWS.valueChanged.connect(lambda: self.on_change_func(self.HWS))
        self.WWS.setMinimum(1503)  # 设置最小值
        self.WWS.setMaximum(3001)  # 设置最大值
        self.WWS.valueChanged.connect(lambda: self.on_change_func(self.WWS))
        self.SPLS.setMinimum(5724)  # 设置最小值
        self.SPLS.setMaximum(7724)  # 设置最大值
        self.SPLS.valueChanged.connect(lambda: self.on_change_func(self.SPLS))
        self.timer = QTimer(self)
        self.timer.start(10)  # 单位为毫秒
        self.n = 1
        self.cap = cv2.VideoCapture(str('/home/baoyoust/桌面/show/089608.mp4'))
        self.cap1 = cv2.VideoCapture(str('/home/baoyoust/桌面/show/sleeve.mp4'))
        self.cap2 = cv2.VideoCapture(str('/home/baoyoust/桌面/show/length.mp4'))
        self.setWindowTitle("3D Reconstruction")

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")

        jpg = QtGui.QPixmap(imgName).scaled(self.InputImage.width(), self.InputImage.height())
        self.InputImage.setPixmap(jpg)

    def showmodel(self):
        self.timer.timeout.connect(self.show_single)

    def show_single(self):
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频帧数
        frame_count1 = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频帧数
        frame_count2 = self.cap2.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频帧数
        # print(frame_count)
        self.n += 1
        if self.n > frame_count:
            self.n = 1
            self.cap = cv2.VideoCapture(str('/home/baoyoust/桌面/show/089608.mp4'))
            self.cap1 = cv2.VideoCapture(str('/home/baoyoust/桌面/show/sleeve.mp4'))
            self.cap2 = cv2.VideoCapture(str('/home/baoyoust/桌面/show/length.mp4'))
        # 视频txt帧播放
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (h, w) = frame.shape[:2]  # 10
        center = (w // 2, h // 2)  # 11
        M = cv2.getRotationMatrix2D(center, 180, 1.0)  # 12
        frame = cv2.warpAffine(frame, M, (w, h))  # 13
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).scaled(self.OutputVideo.width(), self.OutputVideo.height())
        self.pm = QPixmap.fromImage(img)
        self.OutputVideo.setPixmap(self.pm)

        ret, frame = self.cap1.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (h, w) = frame.shape[:2]  # 10
        center = (w // 2, h // 2)  # 11
        M = cv2.getRotationMatrix2D(center, 180, 1.0)  # 12
        frame = cv2.warpAffine(frame, M, (w, h))  # 13
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).scaled(self.OutputUpper.width(), self.OutputUpper.height())
        self.pm1 = QPixmap.fromImage(img)
        self.OutputUpper.setPixmap(self.pm1)

        ret, frame = self.cap2.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (h, w) = frame.shape[:2]  # 10
        center = (w // 2, h // 2)  # 11
        M = cv2.getRotationMatrix2D(center, 180, 1.0)  # 12
        frame = cv2.warpAffine(frame, M, (w, h))  # 13
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).scaled(self.OutputBottom.width(), self.OutputBottom.height())
        self.pm2 = QPixmap.fromImage(img)
        self.OutputBottom.setPixmap(self.pm2)

    def on_change_func(self, slider):
        if slider == self.SHLS:
            self.SHL.setText(str(self.SHLS.value()/10000))
        elif slider == self.SSLS:
            self.SLL.setText(str(self.SSLS.value()/10000))
        elif slider == self.CWS:
            self.CW.setText(str(self.CWS.value()/10000))
        elif slider == self.HWS:
            self.HW.setText(str(self.HWS.value()/10000))
        elif slider == self.WWS:
            self.WW.setText(str(self.WWS.value()/10000))
        else:
            self.SPL.setText(str(self.SPLS.value()/10000))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyClass()
    myWin.show()
    sys.exit(app.exec_())


