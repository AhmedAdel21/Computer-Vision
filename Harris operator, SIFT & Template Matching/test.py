import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from Harris_Corner_Detection_Algorithm import harris
from sift3 import main_orientation, local_descriptors



face1 = cv2.imread("lena.jpg")
face2 = face1[150:250, 200:300]
# face2 = cv2.imread("cat22.jpg")
# face2 = cv2.imread("lena_temp.jpg")
face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(face1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(face2, cv2.COLOR_RGB2GRAY)

# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(gray1,None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

kp1 = harris(gray1)
kp2 = harris(gray2)
des1 = main_orientation(gray1, kp1)
des1 = local_descriptors(gray1, des1)
des2 = main_orientation(gray2, kp2)
des2 = local_descriptors(gray2, des2)





def ssd_match(des1, des2):
    matches = [[] for i in range(2)]
    sort_list = []
    for idx, ele2  in enumerate(des2):
        points_dis = []
        for ele1 in des1:
            diff = ele1[3]-ele2[3]
            ssd = np.sum(np.square(diff))
            # print(ssd)
            points_dis.append(ssd)
        # index = np.unravel_index(np.argmin(points_dis), len(points_dis))
        min_value = min(points_dis)
        index = points_dis.index(min_value)
        sort_list.append((kp1[index], kp2[idx], min_value))
    sorted_list = sorted(sort_list, key=lambda x: x[2])
    for element in sorted_list:
        matches[0].append(element[0])
        matches[1].append(element[1])
        # matches = [matches[0][:50], matches[1][:50]]
    return matches
    
def normalized_match(des1, des2):
    sort_list = []
    matches = [[] for i in range(2)]
    for idx, ele2  in enumerate(des2):
        points_dis = []
        for ele1 in des1:
            nnc = np.mean(np.multiply((ele1[3]-np.mean(ele1[3])),(ele2[3]-np.mean(ele2[3]))))/(np.std(ele1[3])*np.std(ele2[3]))
            points_dis.append(nnc)
        # index = np.unravel_index(np.argmin(points_dis), len(points_dis))
        min_value = max(points_dis)
        index = points_dis.index(min_value)
        sort_list.append((kp1[index], kp2[idx], min_value))
    sorted_list = sorted(sort_list, key=lambda x: x[2])
    for element in sorted_list:
        matches[0].append(element[0])
        matches[1].append(element[1])
    return matches
# matches2 = cv2.BFMatcher().knnMatch(des1, des2, k=2)
# matches = ssd_match(des1, des2)
matches = normalized_match(des1,des2)
# print(matches[0])
# print(matches[0][0].pt)
# print(type(matches2[0][0]))
# img=cv2.drawKeypoints(gray1, matches[0], face1)
# cv2.imwrite('testing.jpg',img)
# img2=cv2.drawKeypoints(gray2,matches[1],face2)
# cv2.imwrite('testing2.jpg',img2)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(face1)
ax2.imshow(face2)

colors = ["red", "blue", "green", "black", "orange", "white", "cyan", "magenta"]
# for i in range(0, len(matches[0])):
#     con = ConnectionPatch(xyA=(matches[0][i].pt[0], matches[0][i].pt[1]), xyB=(matches[1][i].pt[0], matches[1][i].pt[1]), coordsA="data",
#                         axesA=ax1, axesB=ax2, color=colors[i%8])
#     ax2.add_artist(con)
#     # print("hello")
# plt.show()


for i in range(0, 50):
    con = ConnectionPatch(xyA=(matches[0][i][0], matches[0][i][1]), xyB=(matches[1][i][0], matches[1][i][1]), coordsA="data",
                        axesA=ax1, axesB=ax2, color=colors[i%8])
    ax2.add_artist(con)
    # print("hello")
plt.show()


# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# Match descriptors.
# matches = bf.match(des1, des2)
# print(type(matches[0]))
# matches = bf.knnMatch(des1,des2, k=2)
# Sort them in the order of their distance.

# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# print(good)
# matches = sorted(matches, key=lambda x: x.distance)
# print(good[0].distance)
# Draw first 10 matches.
# img3 = cv2.drawMatches(face1, kp1, face2, kp2, matches[:25],None, flags=2)
# cv2.imshow("result", img3)
# cv2.waitKey(0)