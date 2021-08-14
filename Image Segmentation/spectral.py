import numpy as np
import cv2 
import math

def spectral_threshold(image):
    blur = cv2.GaussianBlur(image,(5,5),0)
    hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    hist /= float(np.sum(hist)) 
    BetweenClassVarsList = np.zeros((256, 256))
    for bar1 in range(len(hist)):
        for bar2 in range(bar1, len(hist)):
            ForegroundLevels = []
            BackgroundLevels = []
            MidgroundLevels = []
            ForegroundHist = []
            BackgroundHist = []
            MidgroundHist = []
            for level, value in enumerate(hist):
                if level < bar1:
                    BackgroundLevels.append(level)
                    BackgroundHist.append(value)
                elif level > bar1 and level < bar2:
                    MidgroundLevels.append(level)
                    MidgroundHist.append(value)
                else:
                    ForegroundLevels.append(level)
                    ForegroundHist.append(value)
            
            FWeights = np.sum(ForegroundHist) / float(np.sum(hist))
            BWeights = np.sum(BackgroundHist) / float(np.sum(hist))
            MWeights = np.sum(MidgroundHist) / float(np.sum(hist))
            FMean = np.sum(np.multiply(ForegroundHist, ForegroundLevels)) / float(np.sum(ForegroundHist))
            BMean = np.sum(np.multiply(BackgroundHist, BackgroundLevels)) / float(np.sum(BackgroundHist))
            MMean = np.sum(np.multiply(MidgroundHist, MidgroundLevels)) / float(np.sum(MidgroundHist))
            BetClsVar = FWeights * BWeights * np.square(BMean - FMean) + \
                                                FWeights * MWeights * np.square(FMean - MMean) + \
                                                    BWeights * MWeights * np.square(BMean - MMean)
            BetweenClassVarsList[bar1, bar2] = BetClsVar
        max_value = np.nanmax(BetweenClassVarsList)
    threshold = np.where(BetweenClassVarsList == max_value)
    return threshold







