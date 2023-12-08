# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:19:03 2023

@author: rayce
"""

import cv2
import numpy as np
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
    
sensitivity = 15    
source = cv2.VideoCapture(s)

window_name = 'Green Object'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: #Escape
    has_frame, frame = source.read()
    
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_limit = np.array([60 - sensitivity, 50, 50])
    upper_limit = np.array([60 + sensitivity,255,255])
    
    green_mask = cv2.inRange(into_hsv, lower_limit, upper_limit)
    green = cv2.bitwise_and(frame,frame,mask=green_mask)
    
    contours, hierarchies = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    
    #for i in contours:
    #    areas.append((contours.index(i),cv2.contourArea(i)))
    #areas.sort()
    #areas = areas[-2:]
    pnts = []
    for i in cnts[:2]:
        area = cv2.contourArea(i)
        if area > 100:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                pnts.append((cx,cy))
    if not has_frame:
        break
    if len(pnts) == 2:
        try: 
            angle = np.round(
                np.arctan(
                    (pnts[1][1]-pnts[0][1])/(pnts[1][0]-pnts[0][0]))*180/np.pi, 1)
        except:
            print('Divide by zero')
        cv2.putText(
            img = frame, 
            text = str(angle), 
            org = (50,50),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.50,
            color = (125, 246, 55),
            thickness = 1)
        cv2.line(frame, pnts[0], pnts[1], (0, 0, 255))
    cv2.imshow('Original',frame)
    #cv2.imshow('Green',green)

source.release()
cv2.destroyAllWindows() 