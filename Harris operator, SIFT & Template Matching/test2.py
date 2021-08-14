import numpy as np
from scipy import signal

img1 = [[1, 2, 3, 4], [5, 6, 7, 8]]
image3 = [[1, 2, 3, 4]]
img2 = [[1, 2, 3, 4], [5, 6, 7, 8]]#[[-116, -488, -202, -91], [784, 994, 102, 325]]
result = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
# result_idx = np.unravel_index(np.argmax(result), result.shape)
# print(result_idx)
print(result)