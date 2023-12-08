# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:19:03 2023

@author: rayce
"""

import cv2
import numpy as np
import sys

s = 1
if len(sys.argv) > 1:
    s = sys.argv[1]
    
sensitivity = 15    
source = cv2.VideoCapture(s)

window_name = 'Green Object'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: #Escape
    has_frame, frame = source.read()
    frame = cv2.flip(frame, 0)
    cv2.imshow('Original',frame)

source.release()
cv2.destroyAllWindows() 