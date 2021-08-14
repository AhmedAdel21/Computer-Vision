import numpy as np
import cv2 
import math

def otsu_threshold(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    BetweenClassVarsList = []
    for bar, _ in enumerate(hist):
        Foregroundlevels = [] #np.extract(hist > bar, hist)
        BackgroundLevels = [] #np.extract(hist < bar, hist)
        ForegroundHist = []
        BackgroundHist = []
        for level, value in enumerate(hist):
            if level < bar:
                BackgroundLevels.append(level)
                BackgroundHist.append(value)
            else:
                Foregroundlevels.append(level)
                ForegroundHist.append(value)
        
        FWeights = np.sum(ForegroundHist) / float(np.sum(hist))
        BWeights = np.sum(BackgroundHist) / float(np.sum(hist))
        FMean = np.sum(np.multiply(ForegroundHist, Foregroundlevels)) / float(np.sum(ForegroundHist))
        BMean = np.sum(np.multiply(BackgroundHist, BackgroundLevels)) / float(np.sum(BackgroundHist))
        BetClsVar = FWeights * BWeights * np.square(BMean - FMean)
        BetweenClassVarsList.append(BetClsVar)

    BetweenClassVarsList = list(np.nan_to_num(BetweenClassVarsList))
    return BetweenClassVarsList.index(np.max(BetweenClassVarsList))

