# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:10:59 2023

@author: rayce
"""

import cv2
import numpy as np

img = cv2.imread('Pushstart White@200x.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

for j in range(img.shape[0]):
    for i in range(img.shape[1]):
        if np.sum(img[j,i,:-1])==0:
            img[j,i,3] = 0
            
                    

cv2.imshow('img',img)


cv2.imwrite('Pushstart_White_Transparent.png', img)


