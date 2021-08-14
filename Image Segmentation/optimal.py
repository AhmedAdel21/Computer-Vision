import numpy as np
import cv2 
import math

def optimal_threshold(image):
    Corners = [image[0,0], image[0,-1], image[-1, 0], image[-1, -1]]
    BMean = np.mean(Corners)
    FMean = np.mean(image) - BMean
    threshold = (BMean + FMean) / float(2)
    flag = True
    while flag:
        old_thresh = threshold

        ForeHalf = np.extract(image > threshold, image)
        BackHalf = np.extract(image < threshold, image)

        if ForeHalf.size:
            FMean = np.mean(ForeHalf)
        else:
            FMean = 0

        if BackHalf.size:
            BMean = np.mean(BackHalf)
        else:
            BMean = 0

        threshold = (BMean + FMean) / float(2)
        if old_thresh == threshold:
            flag = False

    return threshold