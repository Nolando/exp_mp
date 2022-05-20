#!/usr/bin/env python3
# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import cv2
#import matplotlib.pyplot as plt
#import cvlib as cv

def faceDetect(currentImage):
    # Get the current image frame and make a greyscale copy
    #currentImage = cv2.imread("artists.jpg")
    currentImageGrey = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

    # Get the face cascade classifier
    # cascadePath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # faceCascade = cv2.CascadeClassifier(cascadePath)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        currentImageGrey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    #print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(currentImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("Faces found", currentImage)
    # cv2.waitKey(0)
    return currentImage




    # Detect face
    # faces, confidences = cv.detect_face(currentImage)

    # # Add a bounding box around each face
    # for c in confidences:
    #     print("Confidence = ", c)



# Send face data to the neural network

# Determine if the person is a target

# Calculate the coordinates of the person's face