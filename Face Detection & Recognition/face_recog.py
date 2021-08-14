import os, glob
from sklearn import preprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_image(images, titles, h,w,n_row,n_col):
    plt.figure(figsize=(2.2*n_col,2.2*n_row))
    plt.subplots_adjust(bottom=0,left=.01,right=.99,top=.90,hspace=.20)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def face_recog(testimage):
    dataset_path = 'DataSet'
    tot_images = 0
    shape = None
    # print(glob.glob(dataset_path))
    for images in glob.glob(dataset_path+'\*\*' , recursive=False):#Loop through all the images in the folder
    #     print(images)
        if images[-3:] == 'pgm' or images[-3:] == 'jpg':
            tot_images += 1

    shape = (112,92)#height of the image is 112 and width is 92
    all_img = np.zeros((tot_images,shape[0],shape[1]),dtype='float64')#Creating 0 matrix with 112 rows and 92 columns of zeros for 400 images
    names = list()
    i=0
    for folder in glob.glob(dataset_path + '\*'):#Loop through folders
        for _ in range(10):
            names.append(folder[-3:].replace('/',''))
        for image in glob.glob(folder +'/*'):#Loop through images
            read_image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            resized_image =cv2.resize(read_image,(shape[1],shape[0]))#cv2.resize resizes an image into (# column x # height)
            all_img[i]=np.array(resized_image)
            i+=1

    A = np.resize(all_img,(tot_images,shape[0]*shape[1]))#Creating a matrix of n^2 x m
    mean_vector = np.sum(A,axis=0,dtype='float64')/tot_images
    mean_matrix = np.tile(mean_vector,(tot_images,1))#Calculating mean for all the 400 images
    A_tilde = A - mean_matrix# Matrix A - the mean value of all the images
    L =(A_tilde.dot(A_tilde.T))/tot_images#Creating a m x m symmetric matrix
    eigenvalues,eigenvectors = np.linalg.eig(L)# Calculating the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]#sort eigenvalues in descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]#Sort the eigenvectors according to the highest eigenvalues
    eigenvector_C = A_tilde.T @ eigenvectors
    eigenfaces = preprocessing.normalize(eigenvector_C.T)
    eigenface_labels = [x for x in range(eigenfaces.shape[0])]
    test_img = cv2.imread(testimage, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img,(shape[1],shape[0]))#resize the test image to 92 x 112
    mean_sub_testimg=np.reshape(test_img,(test_img.shape[0]*test_img.shape[1]))-mean_vector#Subtract test image with

    q=350 # 350 eigenvectors is chosen
    E = eigenfaces[:q].dot(mean_sub_testimg)
    reconstruction = eigenfaces[:q].T.dot(E)

    thres_1 = 3000 # Chosen threshold to detect face
    projected_new_img_vect=eigenfaces[:q].T @ E#Perform Linear combination for the new face space
    diff = mean_sub_testimg-projected_new_img_vect
    beta = math.sqrt(diff.dot(diff))#Find the difference between the projected test image vector and the mean vector of the images
    if beta<thres_1:
        print("Face Detected in the image!")
    else:
        print("No face Detected in the image!")

    #Classify the image belongs to which class
    thres_2 = 10000
    smallest_value =None # to keep track of the smallest value
    index = None #to keep track of the class that produces the smallest value
    for z in range(tot_images):#Loop through all the image vectors
        E_z=eigenfaces[:q].dot(A_tilde[z])#Calculate and represent the vectors of the image in the dataset
        diff = E-E_z
        epsilon_z = math.sqrt(diff.dot(diff))
        if smallest_value==None:
            smallest_value=epsilon_z
            index = z
        if smallest_value>epsilon_z:
            smallest_value=epsilon_z
            index=z
    if smallest_value<thres_2:
        print("The image matches the dataset of face in folder: ",names[index])
        # plt.imshow(all_img[index],cmap='gray')
        # plt.show()
        return names[index], all_img[index]
    else:
        print("unknown Face")
        return []


# image = face_recog('test.pgm')
# print(image)