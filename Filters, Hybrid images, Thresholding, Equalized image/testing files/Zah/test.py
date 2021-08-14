# Requiered:
# - Draw histogram and distribution curve [Done]
# - Equalize the image [Done]
# - Normalize the image
# from collections import Counter # Replaced
from cv2 import cv2 as cv
from PyQt5 import QtCore, QtGui,QtWidgets
import numpy as np
from testUI import Ui_MainWindow
import pyqtgraph as pg

class GUI(Ui_MainWindow):
    def __init__(self,MainWindow):
        super(GUI,self).setupUi(MainWindow) 

        self.pushButton.clicked.connect(self.giveIt)

    def giveIt(self):
        img = cv.cvtColor(cv.imread('../test.jpg'), cv.COLOR_BGR2GRAY)
        # histogramValues = np.array(list(Counter(img.flatten()).items())) ### basic method for counting...

        # row, col = img.shape[:2]

        # def df(img):
        #     values = [0]*256
        #     for i in range(img.shape[0]):
        #         for j in range(img.shape[1]):
        #             values[img[i,j]]+=1
        #     return values

        ### This part for Histogram Graph ###
        # x = np.linspace(0, 255, num=256)
        # y = df(img)
        # bg = pg.BarGraphItem(x=x, height=y, width=1, brush='r')
        # self.plotItem.addItem(bg) # P.S. PlotItem type is: PlotWidget

####################################################################################################

        ### This part for Equalized Image ###
        # def cdf(hist):
        #     cdf = [0] * len(hist)
        #     cdf[0] = hist[0]
        #     for i in range(1, len(hist)):
        #         cdf[i]= cdf[i-1]+hist[i]
        #     cdf = [ele*255/cdf[-1] for ele in cdf]
        #     return cdf
        # def equalize_image(image):
        #     my_cdf = cdf(df(img))
        #     image_equalized = np.interp(image, range(0,256), my_cdf)
        #     return image_equalized
        # eq = equalize_image(img)
        # self.plotItem.setImage(eq.T) # P.S. PlotItem type is: ImageView

####################################################################################################

        ### This part for Normalized Image ###
        # def normalize_image(img):
        #     minValue = 0
        #     maxValue = max(img.flatten())
        #     values = np.zeros(img.shape)
        #     for i in range(img.shape[0]):
        #         for j in range(img.shape[1]):
        #             values[i,j] = (img[i,j] - minValue)/(maxValue - minValue) * 255.0
        #     return values
        # nr = normalize_image(img)
        # self.plotItem.setImage(nr.T) # P.S. PlotItem type is: ImageView




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = GUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())  