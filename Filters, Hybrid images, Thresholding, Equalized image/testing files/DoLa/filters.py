from cv2 import cv2
import numpy as np

pic = cv2.imread('./Dola/test.jpg')
pic = cv2.resize(pic, (800,800)) #resizing the image 

# filter = np.array([(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1)]) * (1/25) # filter 5*5
# filter = np.array([(1,1,1),(1,1,1),(1,1,1)]) * (1/9) # filter 3*3
# filter = np.array([(0,0,0),(0,0,0),(0,0,0)]) # for median filter 3*3
filter = np.array([(1,2,1),(2,4,2),(1,2,1)]) * (1/16) #  Gaussian filter 3*3

picShape = pic.shape
filterShape = filter.shape

picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow('original',picGray)

inputPicRow = picShape[0] + filterShape[0] - 1
inputPicColumn = picShape[1] + filterShape[1] - 1
zeros = np.zeros((inputPicRow,inputPicColumn))

for i in range(picShape[0]):
    for j in range(picShape[1]):
        zeros[i+np.int((filterShape[0]-1)/2),j+np.int((filterShape[1]-1)/2)] = picGray[i,j]

for i in range(picShape[0]):
    for j in range(picShape[1]):
        targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]
        result = np.median(targetWindow)
        # result = np.sum(targetWindow*filter)
        picGray[i,j] = result

cv2.imshow('final image',picGray)
cv2.waitKey(0)