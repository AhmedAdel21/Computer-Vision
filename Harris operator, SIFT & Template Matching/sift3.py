from re import T
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from Harris_Corner_Detection_Algorithm import harris 


#get gradient of a point in image
def get_grad(img,x,y):
    '''
    params: 
        img: gray scale image
        x,y: coordinate of pixel
    return:
        magnitude and the angle of the gradient oriantation 
    '''
    dx=int(img[x+1,y])-int(img[x-1,y])
    dy=int(img[x,y+1])-int(img[x,y-1])
    magnitude=np.sqrt(np.power(dx,2)+np.power(dy,2))
    theta=np.degrees(np.arctan2([dy] ,[dx]))[0]
    if np.isnan(theta):
        theta=90
    if theta<0:
        theta+=360
    if theta >360:
        theta -= 360
    if theta == 360:
        theta-=1    
    return magnitude,int(theta)



def main_orientation(img,key_points):
    '''
    params:
        img: gray scale image
        key_points: list of oe key points each point is list [x,y]
    return:
        new_key_points : new key points each points is list [x,y,oriantations]
    '''

    #define bins number =36
    BINS_NUMBER=36
    BIN_WIDTH=int(360/BINS_NUMBER)

    # define kernal size around the key point
    KERNAL_SIZE=(16,16)
    HALF_KERNAL_SIZE=int(KERNAL_SIZE[0]/2)

    # define guassian kernal with mean and sigma 
    SIGMA=1
    MEAN=0
    guasian_kernal=np.random.normal(MEAN,SIGMA,KERNAL_SIZE)
    
    
    #intialize histogram and the new key points
    hist=[0]*BINS_NUMBER
    new_key_points=[]

    #loop over the key points
    for point in key_points:
        x=point[0]
        y=point[1]
        if(x >= img.shape[0]-8 or y >= img.shape[1]-8):
            continue
    # take window around the point 16 * 16
        window=img[x-HALF_KERNAL_SIZE:x+HALF_KERNAL_SIZE,y-HALF_KERNAL_SIZE:y+HALF_KERNAL_SIZE]
    
    # for each point of the window compute its oriantation(magnitude and theta) 
        for i in range(-HALF_KERNAL_SIZE,HALF_KERNAL_SIZE):
            for j in range(-HALF_KERNAL_SIZE,HALF_KERNAL_SIZE):
                wx=x+i
                wy=y+j
                # get the gradient magnitude and theta 
                magnitude,theta=get_grad(img,wx,wy)
                
                # multiply the magnitude by the guassian kernal value
                magnitude=magnitude*guasian_kernal[i+HALF_KERNAL_SIZE,j+HALF_KERNAL_SIZE]
                
                # assign each point to a bin
                bin=int(theta/BIN_WIDTH)
                # add the magnitude of this pin to the histogram
                hist[bin]+=magnitude
        # the maximum oriantation is the max bin in histogram
        max_value=np.max(hist)
        max_bin=(np.argmax(hist)+1)*BIN_WIDTH
        new_key_points.append([x,y,max_bin])
        
        #uncomment this later
        #loop over the histogram if bin value >= .8 of the maximum value
        #add it to the key points
        # for bin_no in range(len(hist)):
        #     if bin_no==max_bin:continue
        #     if hist[bin_no]>=.8*max_value:
        #        new_key_points.append([x,y,bin_no*BIN_WIDTH]) 

    # return the new key point with its orientations
    return new_key_points

def local_descriptors(img,key_points):
    '''
    params:
        img: gray scale image
        key_points: list of oe key points each point is list [x,y,oriatation]
    return:
        new key points :add descriptors for each key with size =128
    '''
    new_key_points=[]
    # number of bins =8
    BINS_NUMBER=8

    # bin width = 360/8
    BIN_WIDTH=int(360/BINS_NUMBER)

    # kernal of size 16*16
    KERNAL_SIZE=(16,16)
    HALF_KERNAL_SIZE=int(KERNAL_SIZE[0]/2)
     # define guassian kernal with mean and sigma 
    SIGMA=1
    MEAN=0
    guasian_kernal=np.random.normal(MEAN,SIGMA,KERNAL_SIZE)
    #loop over eack key point 
    for point in key_points:
        x=point[0]
        y=point[1]
        oriantation=point[2]

        #take kernal around he point
        window=img[x-HALF_KERNAL_SIZE:x+HALF_KERNAL_SIZE,y-HALF_KERNAL_SIZE:y+HALF_KERNAL_SIZE]
        window_gradients=np.zeros(KERNAL_SIZE)
        window_thetas=np.zeros(KERNAL_SIZE)
        
        # for each point of the window compute its oriantation(magnitude and theta) 
        for i in range(-HALF_KERNAL_SIZE,HALF_KERNAL_SIZE):
            for j in range(-HALF_KERNAL_SIZE,HALF_KERNAL_SIZE):
                wx=x+i
                wy=y+j
                # get the gradient magnitude and theta 
                magnitude,theta=get_grad(img,wx,wy)
                
                # multiply the magnitude by the guassian kernal value
                magnitude=magnitude*guasian_kernal[i+HALF_KERNAL_SIZE,j+HALF_KERNAL_SIZE]

                #subtract key point oriatation from theta 
                theta =np.abs(theta - oriantation)
                window_gradients[i+HALF_KERNAL_SIZE,j+HALF_KERNAL_SIZE]=magnitude
                window_thetas[i+HALF_KERNAL_SIZE,j+HALF_KERNAL_SIZE]=theta

        # divide the window by 4 = 4 * 4
        window_sub_regions=[]
        window_sub_gradients=[]
        window_sub_thetas=[]
        for i in range(0,16,4):
            for j in range (0,16,4):
                window_sub_regions.append(window[i:i+4,j:j+4])
                window_sub_gradients.append(window_gradients[i:i+4,j:j+4]) 
                window_sub_thetas.append(window_thetas[i:i+4,j:j+4])
        
        #define the descriptor vector 
        descriptor=[]

        #loop over the window regions
        for region_idx in range(16):
            #define histogram for each region
            hist=[0]*BINS_NUMBER
            #loop over the region elements
            for i in range(4):
                for j in range(4):
                    theta_p=window_sub_thetas[region_idx][i,j]
                    magnitude_p=window_sub_gradients[region_idx][i,j]
                    if theta_p>=360:
                        theta_p -=360
                    bin=int(theta_p/BIN_WIDTH)
                    hist[bin]+=magnitude_p
            descriptor+=hist
        
        # normalize the descriptor vector hint can be replaced by vectorize version later
        descriptor_sum=sum(descriptor)
        if descriptor_sum>0:
            for d in range(128):
                descriptor[d] /= descriptor_sum
                # remove and descriptor above .2
                if descriptor[d]>.2:
                    descriptor[d]=.2
        
        # renormalize it
        descriptor_sum=sum(descriptor)
        if descriptor_sum>0:
            for d in range(128):
                descriptor[d] /= descriptor_sum
        
        new_key_points.append([x,y,oriantation,np.asarray(descriptor)])

    return new_key_points





# filename = 'cat.jpg'
# img = cv.imread(filename)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# key_points = harris(gray)
# #
# new_key_points = main_orientation(gray,key_points)
# new_key_points = local_descriptors(gray,new_key_points)
# result1 = np.array(new_key_points)
# #
# image2 = cv.imread("cat22.jpg", cv.IMREAD_GRAYSCALE)
# key_points2 = harris(image2)
# new_key_points2 = main_orientation(image2, key_points2)
# new_key_points2 = local_descriptors(image2, new_key_points2)
# result2 = np.array(new_key_points)
# #
# matches1 = []
# matches2 = []
# for idx, ele2  in enumerate(result2):
#     points_dis = []
#     for ele1 in result1:
#         diff = ele1[3]-ele2[3]
#         ssd = np.sum(np.square(diff))
#         points_dis.append(ssd)
#     # index = np.unravel_index(np.argmin(points_dis), len(points_dis))
#     index = points_dis.index(min(points_dis))
#     coordinates = (result1[index][0], result1[index][1])
#     matches1.append(coordinates)
#     matches2.append(result2[idx])
# # result_image = img
# # for match in matches1:
# #     result_image = cv.circle(result_image, (match[0], match[1]), radius=0, color=(0, 0, 255), thickness=-1)
# # cv.imshow("result", result_image)
# # cv.waitKey(0)


# fig = plt.figure(figsize=(10,5))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)

# ax1.imshow(gray)
# ax2.imshow(image2)


# for i in range(0, len(matches1), 15):
#     con = ConnectionPatch(xyA=(matches1[i][0], matches1[i][1]), xyB=(matches2[i][0], matches2[i][1]), coordsA="data", coordsB="data",
#                         axesA=ax1, axesB=ax2, color="red")
#     ax2.add_artist(con)
#     # print("hello")
# plt.show()




