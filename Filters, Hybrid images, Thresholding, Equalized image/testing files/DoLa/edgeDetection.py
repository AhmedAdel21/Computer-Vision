from cv2 import cv2
import numpy as np

pic = cv2.imread('test.jpg')
pic = cv2.resize(pic, (800,800)) #resizing the image 

picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow('original',picGray)

# filter = np.array([(1,1,1),(1,1,1),(1,1,1)]) * (1/9) # filter 3*3
# Sobel Mask
sobal = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]]) #  Sobel Mask
prewitt = np.array([[-1, 0, 1],
                 [-1, 0, 1],
                 [-1, 0, 1]]) #  prewitt Mask     
robert = np.array([[1, 0],
                 [ 0, -1]])   #  robert Mask     .

# canny = np.array([[2,4,5,4,2],
#                  [4,9,12,9,4],
#                  [5,12,15,12,5],
#                  [4,9,12,9,4],
#                  [2,4,5,4,2]]) * (1/159) #Canny mask
# picShape = pic.shape
# filterShape = canny.shape
# inputPicRow = picShape[0] + filterShape[0] - 1
# inputPicColumn = picShape[1] + filterShape[1] - 1
# zeros = np.zeros((inputPicRow,inputPicColumn))
# for i in range(picShape[0]):
#     for j in range(picShape[1]):
#         zeros[i+np.int((filterShape[0]-1)/2),j+np.int((filterShape[1]-1)/2)] = picGray[i,j]

# for i in range(picShape[0]):
#     for j in range(picShape[1]):
#         targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]
#         result = np.sum(targetWindow*canny)
#         picGray[i,j] = result


maskVertical =  sobal
maskHorizontal = maskVertical.T
picShape = pic.shape
filterShape = maskVertical.shape


inputPicRow = picShape[0] + filterShape[0] - 1
inputPicColumn = picShape[1] + filterShape[1] - 1
zeros = np.zeros((inputPicRow,inputPicColumn))

for i in range(picShape[0]):
    for j in range(picShape[1]):
        zeros[i+np.int((filterShape[0]-1)/2),j+np.int((filterShape[1]-1)/2)] = picGray[i,j]

for i in range(picShape[0]):
    for j in range(picShape[1]):
        targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]

        verticalResult = (targetWindow*maskVertical)
        verticalScore = verticalResult.sum() / 4

        horizontalResult = (targetWindow*maskHorizontal)
        horizontalScore = horizontalResult.sum() / 4

        result = (verticalScore**2 + horizontalScore**2)**.5
        picGray[i,j] = result*3

cv2.imshow('final image',picGray)

cv2.waitKey(0)