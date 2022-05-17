# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import cv2
import matplotlib.pyplot as plt
#import cvlib as cv

# Get the current image frame
currentImage = cv2.imread("stopSign.jpg")

# Detect face
faces, confidences = cv2.detect_face(currentImage)

# Add a bounding box around each face
for c in confidences:
    print("Confidence = ", c)



# Send face data to the neural network

# Determine if the person is a target

# Calculate the coordinates of the person's face