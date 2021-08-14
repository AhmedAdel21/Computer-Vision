import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 as cv2
import matplotlib.image as mpimg

def calc_distance(X1, X2):
    return(sum((X1 - X2)**2))**0.5

def findClosestCentroids(ic, X):
    assigned_centroid = []
    for i in X:
        distance=[]
        for j in ic:
            distance.append(calc_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid


def calc_centroids(clusters, X):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])],
                      axis=1)
    for c in set(new_df['cluster']):
        current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

im1 = cv2.imread('screenshots/temp.jpg')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

# print(im.shape)
im = (im1/255).reshape(im1.shape[0]*im1.shape[1], 3)
# print(im.shape)
random_index = random.sample(range(0, len(im)), 2)

centroids = []
for i in random_index:
    centroids.append(im[i])
centroids = np.array(centroids)

for i in range(2):
    get_centroids = findClosestCentroids(centroids, im)
    centroids = calc_centroids(get_centroids, im)


im_recovered = im.copy()
for i in range(len(im)):
    im_recovered[i] = centroids[get_centroids[i]]



im_recovered = im_recovered.reshape(im1.shape[0],im1.shape[1], 3)

fig,ax = plt.subplots(1,2)
ax[0].imshow(im1)
ax[1].imshow(im_recovered)
plt.show()