# from UI import * 
from numpy.core.fromnumeric import shape
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui,QtWidgets
from cv2 import cv2 as cv
from math import sqrt
import numpy as np
from PIL import Image
import matplotlib as plt
import random
from UI import Ui_MainWindow
from lines_hough import hough_lines
import snake as sn
import canny
# from collections import Counter # Replaced


class GUI(Ui_MainWindow):
    def __init__(self,MainWindow):
        super(GUI,self).setupUi(MainWindow) 

        self.images=[self.cannyInputImage,self.cannyOutputImage,
                    self.activeContoursInputImage,self.activeContoursOutputImage]   

        #removing unwanted options from the image display widget
        for i in range(len(self.images)):
            self.images[i].ui.histogram.hide()
            self.images[i].ui.roiPlot.hide()
            self.images[i].ui.roiBtn.hide()
            self.images[i].ui.menuBtn.hide()
            self.images[i].view.setContentsMargins(0,0,0,0)
            self.images[i].view.setAspectLocked(False)
            self.images[i].view.setRange(xRange=[0,100],yRange=[0,100], padding=0)

        #retrieve the original image data
<<<<<<< HEAD
        hough_lines("linesInput.jpg")
=======
>>>>>>> f84900ccb35d78754c2205ba7f275868319481b6

        #   Active contour
        self.snakeContour()
        
        
    
    
        
######################################################################################################
#       DoLa
    def snakeContour(self):
        img = np.load('./img.npy')
        t = np.arange(0, 2*np.pi, 0.1)
        x = 120+50*np.cos(t)
        y = 140+60*np.sin(t)

        alpha = 0.001
        beta = 0.4
        gamma = 100
        iterations = 50

        # fx and fy are callable functions
        fx, fy = sn.create_external_edge_force_gradients_from_img( img )

        snakes = sn.iterate_snake(
            x = x,
            y = y,
            a = alpha,
            b = beta,
            fx = fx,
            fy = fy,
            gamma = gamma,
            n_iters = iterations,
            return_all = True
        )
        self.activeContoursInputImage.setImage(img,xvals=np.linspace(1., 3., img.shape[0]))
        # self.activeContoursOutputImage.setImage(img,xvals=np.linspace(1., 3., img.shape[0]))

        fig = plt.pyplot.figure()
        ax  = fig.add_subplot()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0,img.shape[1])
        ax.set_ylim(img.shape[0],0)
        ax.plot(np.r_[x,x[0]], np.r_[y,y[0]], c=(0,1,0), lw=2)

        for i, snake in enumerate(snakes):
            if i % 10 == 0:
                ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0,0,1), lw=2)

        # Plot the last one a different color.
        ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)

        plt.pyplot.savefig('snake.jpg')
        outImg = cv.imread('./snake.jpg')
        self.activeContoursOutputImage.setImage(outImg)
        cny_img_in = cv.imread('CannyInput.jpg')
        self.cannyInputImage.setImage(cny_img_in.T)
        cny_img_out = canny.canny_apply("CannyInput.jpg")
        # print(type(np.asarray(cny_img_out)))
        self.cannyOutputImage.setImage(np.asarray(cny_img_out).T)
######################################################################################################


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = GUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())  