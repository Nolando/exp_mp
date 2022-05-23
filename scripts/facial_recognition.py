#!/usr/bin/env python3
# This script detects human faces and matches them with a trained
# neural network to determine if they are a target. The coordinate
# of the person's face is also calculated.

# Packages
import rospy
import cv2 as cv
import numpy as np
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import CompressedImage
import camera_functions

# Create the publisher
face_box_pub = rospy.Publisher("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), queue_size=1)

# Initialise the face cascade classifier (done outside of function to try improve latency)
# + LATENCY fixed by setting camera subscriber queue size to 1
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

#################################################################################
def camera_callback(frame):

    # log some info about the image topic
    # rospy.loginfo('facial_recognition\tCAMERA FRAME RECEIVED')

    # Convert the ROS image to OpenCV compatible image
    converted_frame = camera_functions.camera_bridge_convert(frame)

    # Get the face bounding box
    face_bounding_box = face_detect(converted_frame)      # np.ndarray if detected, tuple if empty

    # Test if the returned variable contains a detected face bounding box
    if face_bounding_box is not tuple():

        # Log message
        # rospy.loginfo('facial_recognition\tDETECTED FACE WITH BOX ' + np.array2string(face_bounding_box))

        # PUBLISH THE BOUNDING BOX
        face_box_pub.publish(face_bounding_box)

        # Test for checking box is correct
        # for (x, y, w, h) in face_bounding_box:
        #     cv.rectangle(converted_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # camera_functions.show_image("face_detection node", converted_frame)

#################################################################################
# Uses the Haar feature based cascade classifiers in OpenCV to detect faces
def face_detect(img):

    # Get the current image frame and make a greyscale copy
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Adjust the resolution percentage decrease - FOR FASTER DETECTION
    percentage_decrease = 50

    # Get the width and height dimensions
    width = int(img_gray.shape[1] * percentage_decrease / 100)
    height = int(img_gray.shape[0] * percentage_decrease / 100)
    dim = (width, height)

    # Resize the image to smaller resolution
    img_gray = cv.resize(img_gray, dim, interpolation=cv.INTER_AREA)

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

#################################################################################
def recognise_face():

    # Initilaise the node and display message 
    rospy.init_node('facial_recognition', anonymous=True)
    rospy.loginfo('facial_recognition\tNODE INIT')

    # Set the ROS spin rate: 1Hz ~ 1 second
    rate = rospy.Rate(1)        ############ Can make this an argument in launch and streamline rates##############

    # Subscriber callbacks
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, camera_callback, queue_size=1)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        recognise_face()
    except rospy.ROSInterruptException:
        pass