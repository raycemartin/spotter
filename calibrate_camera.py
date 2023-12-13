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
import glob

take_images = False
size_of_target = (7*5, 3)

if take_images: # are new calibration images needed?
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
            right_img = frame[:,:int(frame.shape[1]/2),:]
            left_img = frame[:,int(frame.shape[1]/2):,:]
            cv2.imwrite(str(new_directory) + '/cal_right_img_' + str(i) + '.png',right_img)
            cv2.imwrite(str(new_directory) + '/cal_left_img_' + str(i) + '.png',left_img)
            
            i += 1
            time.sleep(.1) #debounce key press
    
        
        cv2.imshow('Camera', frame)
    source.release()
    cv2.destroyAllWindows() 
    
### Calibration ###    
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (2,0,0) ..., (6,5,0)
objp = np.zeros(size_of_target, np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# arrays to store object points and image points from all the images
objpoints = [] # 3d points in real space
imgpoints = [] # 2d points in real space

images = glob.glob('*/*.png')

for fname in images:
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,5), None)
    
    # if found, add object points, image poinrts (after refining them)
    if ret == True:
        print(fname)
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # draw and display corners
        cv2.drawChessboardCorners(img, (7,5), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()