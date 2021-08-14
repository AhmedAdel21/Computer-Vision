from cv2 import cv2
import numpy as np

pic = cv2.imread('Flower.jpg')
pic = cv2.resize(pic, (800,800)) #resizing the image 


picShape = pic.shape

picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow('original',picGray)
####################

# f = cv2.dft(picGray.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
# f_shifted = np.fft.fftshift(f)
# f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]


# f_abs = np.abs(f_complex) +1
# f_bounded = 20* np.log(f_abs)
# f_img = 255* f_bounded / np.max(f_bounded)
# f_img =f_img.astype(np.uint8)

# mag = 20 * np.log(cv2.magnitude(freqPic[:, :, 0] ,freqPic[:, :, 1]) )
# cv2.imshow('freqPic',f_img)

#######################################

# def idealFilterLP(D0,imgShape):
#     base = np.zeros(imgShape[:2])
#     rows, cols = imgShape[:2]
#     center = (rows/2,cols/2)
#     for x in range(cols):
#         for y in range(rows):
#             if distance((y,x),center) < D0:
#                 base[y,x] = 1
#     #print(base)
#     return base

# def idealFilterHP(D0,imgShape):
#     base = np.ones(imgShape[:2])
#     rows, cols = imgShape[:2]
#     center = (rows/2,cols/2)
#     for x in range(cols):
#         for y in range(rows):
#             if distance((y,x),center) < D0:
#                 base[y,x] = 0
#     return base
# original = np.fft.fft2(img)
# plt.imshow(np.log(1+np.abs(original)), "gray")
# center = np.fft.fftshift(original)
# plt.imshow(np.log(1+np.abs(center)), "gray")
# LowPassCenter = center * idealFilterLP(50,img.shape)
# plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray")
# LowPass = np.fft.ifftshift(LowPassCenter)
# inverse_LowPass = np.fft.ifft2(LowPass)
# plt.imshow(np.abs(inverse_LowPass), "gray")

# #HPF Mask
# rows = picShape[0]
# cols = picShape[1]
# crow, ccol = int(rows / 2), int(cols / 2)

# mask = np.ones((rows, cols, 2), np.uint8)
# r = 200
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
# mask[mask_area] = 0


# fshift = f_shifted * mask

# fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
# cv2.imshow('filter',fshift_mask_mag)

# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# cv2.imshow('img_back',img_back)

###########################3
# # LPF Mask
# rows = picShape[0]
# cols = picShape[1]
# crow, ccol = int(rows / 2), int(cols / 2)
# mask = np.zeros((rows, cols, 2), np.uint8)
# r = 100
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
# mask[mask_area] = 1

# fshift = f_shifted * mask

# fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# cv2.imshow('img_back',img_back)
#########################################


# inputPicRow = picShape[0] + filterShape[0] - 1
# inputPicColumn = picShape[1] + filterShape[1] - 1
# zeros = np.zeros((inputPicRow,inputPicColumn))

# for i in range(picShape[0]):
#     for j in range(picShape[1]):
#         zeros[i+np.int((filterShape[0]-1)/2),j+np.int((filterShape[1]-1)/2)] = picGray[i,j]

# for i in range(picShape[0]):
#     for j in range(picShape[1]):
#         targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]
#         result = np.median(targetWindow)
#         # result = np.sum(targetWindow*filter)
#         picGray[i,j] = result

# cv2.imshow('final image',picGray)
cv2.waitKey(0)