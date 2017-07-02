#sudo apt-get install libopencv-dev python-opencv

import datetime
import logging
import json
import cv2
import os


imagePath = "./faces2.jpg"

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier("./opencv-face.xml").detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(20, 20),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print(str(faces))