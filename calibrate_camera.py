# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:34:44 2023

@author: rayce
"""

import cv2
import numpy as np
import sys
import keyboard
import time
import os

current_directory = os.getcwd()
new_directory = os.path.join(current_directory, r'calibration_images')
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
    
i = 1
source = cv2.VideoCapture(s)

while cv2.waitKey(1) != 27: #Escape to exit
    has_frame, frame = source.read()
    frame = cv2.flip(frame, 0)
        
    if keyboard.is_pressed('s'): # wait for 's' key to save images
        cv2.imwrite(str(new_directory) + '/calibration_img_' + str(i) + '.png',frame)
        i += 1
        time.sleep(.1) #debounce key press

    
    cv2.imshow('Camera', frame)

source.release()
cv2.destroyAllWindows() 