#sudo apt-get install libopencv-dev python-opencv
import datetime
import logging

import cv2
import os
from app.utils.constants import ROOT_PATH



class FaceDetect:


    def detect(self, imagePath):


        before = datetime.datetime.now()

        # Read the image
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = cv2.CascadeClassifier(os.path.join(ROOT_PATH, "app/utils/opencv-face.xml")).detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        rez = {
            "number_faces":  len(faces),
            "face_boxes": (faces),
            "time_elapsed": datetime.datetime.now() - before
        }



        return rez

