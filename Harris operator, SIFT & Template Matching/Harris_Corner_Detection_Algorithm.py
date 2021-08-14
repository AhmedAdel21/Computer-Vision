import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import cv2 as cv2






def gradient_x(grayImg):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(grayImg, kernel_x, mode='same')


def gradient_y(grayImg):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(grayImg, kernel_y, mode='same')


def harris(grayImg):
    Ix = gradient_x(grayImg)
    Iy = gradient_y(grayImg)

    Ixx = ndi.gaussian_filter(Ix**2, sigma=1)
    Ixy = ndi.gaussian_filter(Iy*Ix, sigma=1)
    Iyy = ndi.gaussian_filter(Iy**2, sigma=1)


    k = 0.05

    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
        
    R = detA - k * traceA ** 2




    dola = np.zeros((grayImg.shape[0],grayImg.shape[1]))
    dola[R>0.1*R.max()] = True
    x = np.where(dola == True)
    features =np.asarray(x).T.tolist()
    return features


# srcImg[R>0.01*R.max()]=[0,0,255]
# filename = 'cat.jpg'
# srcImg = cv2.imread(filename,cv2.COLOR_BGR2RGB)
# grayImg = rgb2gray(srcImg)
# features = dola(grayImg)
# result_image = srcImg
# for match in features:
#     result_image = cv2.circle(result_image, (match[1], match[0]), radius=0, color=(0, 0, 255), thickness=-1)
# cv2.imshow("result", result_image)
# cv2.waitKey(0)
