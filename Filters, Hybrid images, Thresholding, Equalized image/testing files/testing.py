import cv2 as cv 
import numpy as np


image=cv.imread('D:\CV\Task1\\test.jpg',0)
noisy=image+np.random.normal(0,50**.5,image.shape)
noisy=np.array(noisy).astype(np.uint8)
print("img",image)
print('noise',noisy)
cv.imshow("orig",image)
cv.imshow("noisy",noisy)
cv.waitKey(0)
