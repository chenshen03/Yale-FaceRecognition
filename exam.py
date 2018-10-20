import cv2
import os
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from skimage.feature import hog


# 从图像数据集中读取数据
def read_data(imgDir = 'imgs/'):
    imgs = []
    labels = []
    for i in range(1, 166):
        fileName = 's' + str(i) + '.bmp'
        img = cv2.imread(imgDir + fileName)
        imgs.append(img)
        labels.append((i - 1) // 11)
    print("image shape:")
    print(imgs[0].shape)
    return imgs, labels


# 提取数据的HOG特征
def get_hog_feature(imgs):
    if os.path.exists('data/yale_hog.npy'):
        X_hog = np.load('data/yale_hog.npy')
    else:
        hog_func = lambda x:hog(x, orientations=8, pixels_per_cell=(16, 16), block_norm='L2-Hys',
                            cells_per_block=(1, 1), visualise=True)
        X_hog = np.array([hog_func(img)[0] for img in imgs])
        np.save('data/yale_hog.npy', X_hog)
    print("hog feature shape:")
    print(X_hog.shape)
    return X_hog


# 将数据集划分为训练集、噪声集
def split_data(X_hog, y):
    X_all = X_hog[:12*11, :]
    y_all = y[:12*11]
    X_noise = X_hog[12*11:, :]
    y_noise = y[12*11:]

    # X_all = X_hog[3*11:, :]
    # y_all = y[3*11:]
    # X_noise = X_hog[:3*11:, :]
    # y_noise = y[:3*11:]
    print("train data shape:")
    print(X_all.shape)
    print("noise data shape:")
    print(X_noise.shape)
    return X_all, y_all, X_noise, y_noise


class Classfilter:
    def __init__(self):
        self.clf = self.__model()

    # 分类器定义
    def __model(self):
        return LogisticRegression(random_state=1, solver='lbfgs', multi_class='multinomial')

    # K折交叉训练
    def train(self, X, y):
        if os.path.exists('model/noise3_hog_softmax_yale.h5'):
            self.clf = joblib.load('model/noise3_hog_softmax_yale.h5')
            print('load model successful!')
        else:
            print('traing model...')
            rkf = RepeatedKFold(n_splits=4, n_repeats=6, random_state=0)
            for i_train, i_test in rkf.split(X):
                self.clf.fit(X[i_train], y[i_train])
            joblib.dump(clf, "model/noise3_hog_softmax_yale.h5")
            print('train model successful!')
        print('\n')

    # 模型性能测试
    def model_test(self, X, y, X_noise):
        scores = self.clf.score(X, y)
        print("scores：{}".format(scores))

        pro_min = np.min(np.max(self.clf.predict_proba(X), axis=1))
        print("The minimum probability of all data corresponding to groundtruth：")
        print(pro_min)

        pred_noise = self.clf.predict_proba(X_noise)
        print("The maximum probability of noise data：")
        print(np.max(pred_noise))

    # 预测输出，输入数据为hog特征
    def predict(self, X, threshold=0.33):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        result = ''
        if (np.size(X, 0)  == 1 or np.size(X, 0) == 132): ti = 1
        else: ti = 133
        for i, x in enumerate(X):
            x = x[np.newaxis, :]
            pred_prob = self.clf.predict_proba(x)
            if (np.max(pred_prob) < threshold):
                result = result + 'Img {} is NOT IN the train dataset\n'.format(i+ti)
            else:
                y = np.argmax(pred_prob)
                result = result + 'Img {} is person: {}\n'.format(i+ti, y+1)
        print(result)

    # 预测输出，输入数据为图像
    def img_predict(self, img):
        if img.ndim == 3:
            img = img[np.newaxis, :]
        hog_func = lambda x: hog(x, orientations=8, pixels_per_cell=(16, 16), block_norm='L2-Hys',
                                 cells_per_block=(1, 1), visualise=True)
        X = hog_func(img)
        self.predict(X)

if __name__ == '__main__':
    imgs, labels = read_data()
    X_hog = get_hog_feature(imgs)
    y = np.array(labels)
    X_all, y_all, X_noise, y_noise = split_data(X_hog, y)

    clf = Classfilter()
    clf.train(X_all, y_all)
    # clf.model_test(X_all, y_all, X_noise)
    print("Train data predicting...")
    clf.predict(X_all)
    print("Noise data predicting...")
    clf.predict(X_noise)