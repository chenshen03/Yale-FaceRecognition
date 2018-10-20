import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QDir

from sklearn.externals import joblib
from skimage.feature import hog

class Windows(QWidget):

    def __init__(self):
        super().__init__()

        self.model = joblib.load('model/noise3_hog_softmax_yale.h5')
        [self.X_all, self.X_noise] = self.loadHogData()

        self.initUI()

    def initUI(self):
        self.createImgLayout()
        self.createSetupLayout()

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.imgGroupBox)
        mainLayout.addWidget(self.gridGroupBox)
        self.setLayout(mainLayout)

        self.resize(1000, 550)
        self.center()
        self.setWindowTitle('Face Recognition')
        self.show()

    # 图片区域的布局
    def createImgLayout(self):
        self.imgGroupBox = QGroupBox("测试图片")
        layout = QVBoxLayout()

        fileName = "./imgs/s1.bmp"
        img = QPixmap(fileName)
        self.loadImg(fileName)
        self.lbl_img = QLabel()
        self.lbl_img.setScaledContents(True)
        self.lbl_img.setPixmap(img)
        self.lbl_img.setFixedSize(450, 450)

        self.lbl_imgName = QLabel('s1.bmp')
        btn_import = QPushButton("从文件导入图片")
        btn_import.clicked.connect(self.loadImgDialog)

        layout.addWidget(self.lbl_img)
        layout.addWidget(self.lbl_imgName, stretch=0, alignment=Qt.AlignCenter)
        layout.addWidget(btn_import)

        self.imgGroupBox.setLayout(layout)

    # 预测区域的布局
    def createSetupLayout(self):
        self.gridGroupBox = QGroupBox("测试")
        layout = QGridLayout()

        # 预测相关参数设置
        lbl_dataset = QLabel('数据集：')
        lbl_dataName = QLabel('Yale')
        lbl_trainClass = QLabel('数据集类别：')
        lbl_noiseClass = QLabel('噪声集类别：')
        lbl_trainClassNum = QLabel('12类（第1-12类）')
        lbl_noiseClassNum = QLabel(' 3类（第13-15类）')

        btn_pred_train = QPushButton('预测（数据集）')
        btn_pred_train.clicked.connect(self.predict_all)
        btn_pred_noise = QPushButton('预测（噪声集）')
        btn_pred_noise.clicked.connect(self.predict_noise)
        btn_pred = QPushButton('预测')
        btn_pred.clicked.connect(self.predict_single)
        self.qte_result = QTextEdit()
        self.qte_result.setFont(QFont('SansSerif', 15))

        layout.addWidget(lbl_dataset, 0, 0)
        layout.addWidget(lbl_dataName, 0, 1)
        layout.addWidget(lbl_trainClass, 1, 0)
        layout.addWidget(lbl_trainClassNum, 1, 1)
        layout.addWidget(lbl_noiseClass, 2, 0)
        layout.addWidget(lbl_noiseClassNum, 2, 1)
        layout.addWidget(btn_pred_train, 3, 0, 1, 2)
        layout.addWidget(btn_pred_noise, 4, 0, 1, 2)
        layout.addWidget(btn_pred, 5, 0, 1, 2)
        layout.addWidget(self.qte_result, 6, 0, 1, 2)

        self.gridGroupBox.setLayout(layout)

    # 对图片进行预测
    def predict(self, X):
        threshold = 0.33
        result = ''
        if (np.size(X, 0)  == 1 or np.size(X, 0) == 132): ti = 1
        else: ti = 133
        for i, x in enumerate(X):
            x = x[np.newaxis, :]
            pred_prob = self.model.predict_proba(x)
            if (np.max(pred_prob) < threshold):
                result = result + 'Img {} is NOT IN the train dataset\n'.format(i+ti)
            else:
                y = np.argmax(pred_prob)

                result = result + 'Img {} is person: {}\n'.format(i+ti, y + 1)
        self.qte_result.setText(result)

    # 对单张图片进行预测
    def predict_single(self):
        self.predict(self.x)

    # 对所有训练图片进行预测
    def predict_all(self):
        self.predict(self.X_all)

    # 对所有噪声图片进行预测
    def predict_noise(self):
        self.predict(self.X_noise)

    # 从文件导入图片
    def loadImgDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open file', QDir.currentPath(), 'bmp(*.bmp)')

        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Face Recognition",
                        "Cannot load %s." % fileName)
                return

            imgName = fileName.split('/')[-1]
            self.lbl_imgName.setText(imgName)
            self.lbl_img.setPixmap(QPixmap.fromImage(image))
            self.loadImg(fileName)

    # 将图片转换为可供模型预测的格式
    def loadImg(self, imgPath):
        img = cv2.imread(imgPath)
        img_resize = cv2.resize(img, (100, 100))
        # img_hog = self.getHog(img_resize)
        img_hog = self.getHogTemp(imgPath)
        self.x = img_hog[np.newaxis, :]

    # 临时辅助读取图像hog特征
    def getHogTemp(self, imgPath):
        imgName = imgPath.split('/')[-1]
        i = int(imgName.lstrip('s').rstrip('.bmp'))
        if i > 132:
            img_hog = self.X_noise[i%132]
        else:
            img_hog = self.X_all[i]
        return img_hog

    # 加载所有数据的HOG特征
    def loadHogData(self):
        data = np.load('data/yale_hog.npy')
        X_all = data[:12*11, :]
        X_noise = data[12*11:, :]
        print('X_all shape：{}\n'.format(X_all.shape))
        print('X_noise shape：{}\n'.format(X_noise.shape))
        return X_all, X_noise

    # 得到数据的hog特征
    def getHog(self, img):
        hog_func = lambda x:hog(x, orientations=8, pixels_per_cell=(16, 16), block_norm='L2-Hys',
                            cells_per_block=(1, 1), visualise=True)
        x = np.array(hog_func(img)[0])
        return x

    # 窗口居中
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "确定退出?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Windows()
    sys.exit(app.exec_())