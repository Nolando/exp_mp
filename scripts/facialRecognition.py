#!/usr/bin/env python3
# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt
#import cvlib as cv

# # Initialise the face cascade classifier (done outside of function)
# cascadePath = 'haarcascade_frontalface_default.xml'
# faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

def faceDetect(img, faceCascade):

    # Get the current image frame and make a greyscale copy
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Decrease the resolution of the image slightly for faster speed
    percentage_decrease = 50
    width = int(img_gray.shape[1] * percentage_decrease / 100)
    height = int(img_gray.shape[0] * percentage_decrease / 100)
    dim = (width, height)
    img_gray = cv.resize(img_gray, dim, interpolation=cv.INTER_AREA)

    # # Get the face cascade classifier
    # cascadePath = 'haarcascade_frontalface_default.xml'
    # faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

    # Detect faces in the image as a bounding box
    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.12,           # Attempting to tune this was originally 1.1
        minNeighbors=5,             # Can adjust to fix latency also, originally was 5
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )

    # Test if face was detected in the frame
    if faces is not tuple():

        # Convert box coordiantes back to original frame resolution
        faces = faces * (100 / percentage_decrease)
        faces = np.array(faces, int)

    # Return the face bounding box
    return faces




    # Detect face
    # faces, confidences = cv.detect_face(img)

    # # Add a bounding box around each face
    # for c in confidences:
    #     print("Confidence = ", c)



# Send face data to the neural network

# Determine if the person is a target

# Calculate the coordinates of the person's face