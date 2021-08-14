from UI import Ui_MainWindow
from UI import *
from PyQt5 import QtCore, QtGui,QtWidgets
import numpy as np
import cv2 as cv2
class Task4 (Ui_MainWindow):
    def __init__(self,MainWindow):
        super(Task4,self).setupUi(MainWindow)
        self.segm_input.setPixmap(QtGui.QPixmap("screenshots/temp.jpg")) #set segmentation input image
        self.segm_output.setPixmap(QtGui.QPixmap("screenshots/kmean.png"))
        self.select_segm_algo.currentTextChanged.connect(self.display_segm)
        self.select_thresh_algo.currentTextChanged.connect(self.display_thresh) #calling a function when user select an segmentation algorithm
    def display_thresh(self, value):
        if(value == "Optimal(G)"):
            self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/optimal_global.png"))
        elif(value == "Otsu(G)"):
            self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/otsu_global.png"))
        elif(value == "Spectral(G)"):
            self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/spectral_global.png"))
        elif(value == "Optimal(L)"):
            self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/optimal_local.png"))
        elif(value == "Otsu(L)"):
            self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/otsu_local.png"))
        elif(value == "Spectral(L)"):
            self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/spectral_local.png"))

    
    def display_segm(self,value):
        if (value == "Mean Shift"):
            self.segm_input.setPixmap(QtGui.QPixmap("screenshots/seg3.png"))
            self.segm_output.setPixmap(QtGui.QPixmap("screenshots/mean_shift_result.jpg"))  #set mean shift segmentation output image
        if (value == "Kmeans"):
           self.segm_input.setPixmap(QtGui.QPixmap("screenshots/temp.jpg"))
           self.segm_output.setPixmap(QtGui.QPixmap("screenshots/kmean.png"))
        if(value == "Agglomerative"):
            self.segm_input.setPixmap(QtGui.QPixmap("screenshots/rat.jpg"))
            self.segm_output.setPixmap(QtGui.QPixmap("screenshots/agglomerative.png"))
        if(value == "Region Growing"):
            self.segm_input.setPixmap(QtGui.QPixmap("screenshots/forg.png"))
            image = cv2.imread('screenshots/forg.png', 0)
            ret, img = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
            seed= (56, 32)
            out = self.region_growing(img,seed)

    def get8n(self,x, y, shape):
        out = []
        maxx = shape[1]-1
        maxy = shape[0]-1
        
        #top left
        outx = min(max(x-1,0),maxx)
        outy = min(max(y-1,0),maxy)
        out.append((outx,outy))
        
        #top center
        outx = x
        outy = min(max(y-1,0),maxy)
        out.append((outx,outy))
        
        #top right
        outx = min(max(x+1,0),maxx)
        outy = min(max(y-1,0),maxy)
        out.append((outx,outy))
        
        #left
        outx = min(max(x-1,0),maxx)
        outy = y
        out.append((outx,outy))
        
        #right
        outx = min(max(x+1,0),maxx)
        outy = y
        out.append((outx,outy))
        
        #bottom left
        outx = min(max(x-1,0),maxx)
        outy = min(max(y+1,0),maxy)
        out.append((outx,outy))
        
        #bottom center
        outx = x
        outy = min(max(y+1,0),maxy)
        out.append((outx,outy))
        
        #bottom right
        outx = min(max(x+1,0),maxx)
        outy = min(max(y+1,0),maxy)
        out.append((outx,outy))
        
        return out

    def region_growing(self,img, seed):
        seed_points = []
        outimg = np.zeros_like(img)
        print(outimg.shape)
        seed_points.append((seed[0], seed[1]))
        processed = []
        while(len(seed_points) > 0):
            pix = seed_points[0]
            outimg[pix[0], pix[1]] = 255
            for coord in self.get8n(pix[0], pix[1], img.shape):
                if img[coord[0], coord[1]] != 0:
                    outimg[coord[0], coord[1]] = 255
                    if not coord in processed:
                        seed_points.append(coord)
                    processed.append(coord)
            seed_points.pop(0)
            cv2.imwrite("screenshots/forgRunning.png",outimg)
            self.segm_output.setPixmap(QtGui.QPixmap("screenshots/forgRunning.png"))
            cv2.waitKey(2)
        return outimg

if __name__ =="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    MainWindow=QtWidgets.QMainWindow()
    ui=Task4(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())