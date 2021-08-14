from face_recog import face_recog
from UI import Ui_MainWindow
from PyQt5 import QtCore, QtGui,QtWidgets
import numpy as np
import cv2 
from face_detection import face_detection
from face_recog import face_recog
class Task5 (Ui_MainWindow):
    def __init__(self,MainWindow):
        super(Task5,self).setupUi(MainWindow)
        self.path1 = 'clark22.jpg'
        self.path2 = 'test2.jpg'
        self.label_3.setPixmap(QtGui.QPixmap(self.path1))
        self.label_8.setPixmap(QtGui.QPixmap(self.path2))
    # self.thesh_output.setPixmap(QtGui.QPixmap("screenshots/optimal_global.png"))
        _, det_img = face_detection(self.path1)
        place, rec_img = face_recog(self.path2)
        cv2.imwrite("cla_det.jpg", det_img)
        cv2.imwrite("rec.jpg", rec_img)
        self.label_5.setPixmap(QtGui.QPixmap("cla_det.jpg"))
        self.label_10.setPixmap(QtGui.QPixmap("rec.jpg"))
        self.label_9.setText("Output: the image matches face in file: " + place)
        # rec_img = face_recog(self.path2)

if __name__ =="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    MainWindow=QtWidgets.QMainWindow()
    ui=Task5(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())