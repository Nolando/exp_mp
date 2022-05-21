#!/usr/bin/env python3
# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import cv2 as cv
#import matplotlib.pyplot as plt
#import cvlib as cv

def faceDetect(currentImage):
    # Get the current image frame and make a greyscale copy
    #currentImage = cv.imread("artists.jpg")
    currentImageGrey = cv.cvtColor(currentImage, cv.COLOR_BGR2GRAY)

    # Get the face cascade classifier
    cascadePath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

    # Detect faces in the image as a bounding box
    faces = faceCascade.detectMultiScale(
        currentImageGrey,
        scaleFactor=1.1,        # Attempting to tune this was originally: 1.1
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )

    return faces




    # Detect face
    # faces, confidences = cv.detect_face(currentImage)

    # # Add a bounding box around each face
    # for c in confidences:
    #     print("Confidence = ", c)



# Send face data to the neural network

# Determine if the person is a target

# Calculate the coordinates of the person's face