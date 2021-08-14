from GUI import *
from PyQt5 import QtCore, QtGui,QtWidgets
import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray
import cv2 as cv2

colors = [(255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,255), (255,69,0), (255,0,255), (0,255,255)]


class GUI(Ui_MainWindow):
    def __init__(self,MainWindow):
        super(GUI,self).setupUi(MainWindow)
        
        filename = 'cat.jpg'
        srcImg1 = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        grayImg1 = rgb2gray(srcImg1)
        features1 = self.harris(grayImg1)
        result_image = srcImg1

        srcImg2 = cv2.imread("cat22.jpg", cv2.COLOR_BGR2RGB)
        grayImg2 = rgb2gray(srcImg2)
        features2 = self.harris(grayImg2)
        result_image2 = srcImg2

        for match in features1:
            result_image = cv2.circle(result_image, (match[1], match[0]), radius=0, color=(0, 0, 255), thickness=-1)
        for match in features2:
            result_image2 = cv2.circle(result_image2, (match[1], match[0]), radius=0, color=(0, 0, 255), thickness=-1)
        imageName = 'result.jpg'   
        # cv2.imwrite('reult2.jpg', result_image2)
        # cv2.imwrite(imageName, result_image)
        
        match_img1 =srcImg1
        match_img2 = srcImg2
        for i, match in enumerate(features1):
            match_img1 = cv2.circle(match_img1, (match[1], match[0]), radius=0, color=colors[i%8], thickness=-1)
        for j, match in enumerate(features2):
            match_img2 = cv2.circle(match_img2, (match[1], match[0]), radius=0, color=colors[j%8], thickness=-1)
        cv2.imwrite('match1.jpg', match_img1)
        cv2.imwrite('match2.jpg', match_img2)

        # self.harrisInput1.setPixmap(QtGui.QPixmap('match1.jpg'))
        # self.harrisOutput1.setPixmap(QtGui.QPixmap('match2.jpg'))
    def gradient_x(self,grayImg):
        ##Sobel operator kernels.
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        return sig.convolve2d(grayImg, kernel_x, mode='same')


    def gradient_y(self,grayImg):
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return sig.convolve2d(grayImg, kernel_y, mode='same')


    def harris(self,grayImg):
        Ix = self.gradient_x(grayImg)
        Iy = self.gradient_y(grayImg)

        Ixx = ndi.gaussian_filter(Ix**2, sigma=1)
        Ixy = ndi.gaussian_filter(Iy*Ix, sigma=1)
        Iyy = ndi.gaussian_filter(Iy**2, sigma=1)

        k = 0.05

        # determinant
        detA = Ixx * Iyy - Ixy ** 2
        # trace
        traceA = Ixx + Iyy
            
        R = detA - k * traceA ** 2

        zeroArray = np.zeros((grayImg.shape[0],grayImg.shape[1]))
        zeroArray[R>0.001*R.max()] = True
        x = np.where(zeroArray == True)
        features =np.asarray(x).T.tolist()

        return features

    def matching(self, gray1, gray2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1,None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        matches = [[] for i in range(2)]
        for idx, ele2  in enumerate(des2):
            points_dis = []
            for ele1 in des1:
                nnc = np.mean(np.multiply((ele1-np.mean(ele1)),(ele2-np.mean(ele2))))/(np.std(ele1)*np.std(ele2))
                points_dis.append(nnc)
            # index = np.unravel_index(np.argmin(points_dis), len(points_dis))
            index = points_dis.index(max(points_dis))
            matches[0].append(kp1[index])
            matches[1].append(kp2[idx])
        return matches


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = GUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())  