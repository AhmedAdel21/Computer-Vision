import numpy as np
import cv2 
from otsu import otsu_threshold
from optimal import optimal_threshold
from spectral import spectral_threshold


def Global_threshold(image , thresh_typ = "Optimal"):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_img = np.zeros(image.shape)
    if thresh_typ == "Otsu":
        threshold = otsu_threshold(image)
        thresh_img = np.uint8(np.where(image > threshold, 255, 0))
    elif thresh_typ == "Optimal":
        threshold = optimal_threshold(image)
        thresh_img = np.uint8(np.where(image > threshold, 255, 0))
    else:
        threshold1, threshold2 = spectral_threshold(image)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row, col] > threshold2[0]:
                    thresh_img[row, col] = 255
                elif image[row, col] < threshold1[0]:
                    thresh_img[row, col] = 0
                else:
                    thresh_img[row, col] = 128   
    return thresh_img

def Local_threshold(image, block_size , thresh_typ = "Optimal"):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_img = np.copy(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            mask = image[row:min(row+block_size,image.shape[0]),col:min(col+block_size,image.shape[1])]
            thresh_img[row:min(row+block_size,image.shape[0]),col:min(col+block_size,image.shape[1])] = Global_threshold(mask, thresh_typ)
    return thresh_img




# source_image = cv2.imread("lena.jpg")
# optimal = Local_threshold(source_image, 100,  "Optimal")
# otsu = Local_threshold(source_image, 100, "Otsu")
# spectral = Local_threshold(source_image, 100, "Spectral")
# # print(otsu)
# cv2.imshow('Original image', source_image)
# cv2.imshow('optimal thresholding', optimal)
# cv2.imshow('otsu thresholding', otsu)
# cv2.imshow('spectral thresholding', spectral)
# cv2.waitKey(0)
# cv2.destroyAllWindows()