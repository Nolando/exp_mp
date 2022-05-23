#!/usr/bin/env python3
# Machine Unlearning

# RUN: roslaunch exp_mp realsense.launch

# FOR REALSENSE INSTALL: sudo apt-get install ros-melodic-realsense2-camera
# FOR REALSENSE INFO: https://github.com/IntelRealSense/realsense-ros

import os
import rospy
import cv2 as cv
import numpy as np
import copy
from std_msgs.msg import String
from exp_mp.msg import bounding_box
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from itertools import product

import facialRecognition
import pub_bounding_box

# Initialize the CvBridge class
bridge = CvBridge()

# Set the limits for the colours (in HSV)
red_low_limit_lower = np.array([0, 100, 90])           # Red
red_high_limit_lower = np.array([5, 255, 255])
red_low_limit_upper = np.array([150, 100, 90])           # Red
red_high_limit_upper = np.array([180, 255, 255])
green_low_limit = np.array([140, 30, 0])          # Green
green_high_limit = np.array([179, 255, 255])

# Initialise the face cascade classifier (done outside of function to try improve latency)
# + LATENCY fixed by setting camera subscriber queue size to 1
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)

#################################################################################
def camera_callback(frame):

    # log some info about the image topic
    rospy.loginfo("CAMERA FRAME RECEIVED")

    # Initially empty bounding box
    whole_bounding_box_R = []
    whole_bounding_box_G = []

    # Convert the frame from a ROS Image message to a CV2 Image
    try:
        bridged_frame = bridge.compressed_imgmsg_to_cv2(frame)

        # Flip the camera feed along y axis for correct orientation
        cv_frame = cv.flip(bridged_frame, 1)

        # Convert RGB into HSV
        hsv_frame = cv.cvtColor(cv_frame, cv.COLOR_BGR2HSV)
    
    # Error message if failed
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Get the face bounding box
    face_bounding_box = facialRecognition.faceDetect(hsv_frame, faceCascade)      # np.ndarray if detected, tuple if empty

    # Get the red and green colour masks
    mask_R = red_mask(hsv_frame)
    mask_G = green_mask(hsv_frame)

    # Colour segment to get the bounding box for shirts for red and green
    shirt_bounding_box_R = colour_segmentation(mask_R)          # np.ndarray
    shirt_bounding_box_G = colour_segmentation(mask_G)

    # Test if face was detected and either red or green shirt was found
    if (shirt_bounding_box_R.size is not 0 or shirt_bounding_box_G.size is not 0) and face_bounding_box is not tuple():       

        # Log message
        rospy.loginfo('Face bounding box is: ' + np.array2string(face_bounding_box))
        rospy.loginfo('RED shirt bounding box is: ' + np.array2string(shirt_bounding_box_R))
        rospy.loginfo('GREEN shirt bounding box is: ' + np.array2string(shirt_bounding_box_G))
        
        # Loop through to test the detected bounding boxes
        for (xf, yf, wf, hf) in face_bounding_box:
            for (xs, ys, ws, hs) in shirt_bounding_box_R:

                # If the x dimensions of the face are inside the box, then can assume that is a person (shirt with a face)
                if xf >= xs and xf + wf <= xs + ws and yf + hf < ys:
                    
                    # Get the total bounding box of the torso
                    whole_bounding_box_R.append([xs, yf, ws, hf+hs])

            for (xs, ys, ws, hs) in shirt_bounding_box_G:
                if xf >= xs and xf + wf <= xs + ws and yf + hf < ys:
                    whole_bounding_box_G.append([xs, yf, ws, hf+hs])

        # Convert to numpy
        whole_bounding_box_R = np.array(whole_bounding_box_R)
        whole_bounding_box_G = np.array(whole_bounding_box_G)

        # Add bounding box to the frame
        for (x, y, w, h) in whole_bounding_box_R:
            cv.rectangle(cv_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in whole_bounding_box_G:
            cv.rectangle(cv_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the detected features with bounding boxes
    show_image("Converted Image", cv_frame)




    # Get the bounding box - Superceded but keep in for reporting and referencing
    # hog = cv.HOGDescriptor()
    # pub_bounding_box.HOG_features(cv_frame, hog)

#################################################################################
# Define a function to show the image in an OpenCV Window
def show_image(window_name, frame):

    # Display the image frame
    cv.imshow(window_name, frame)
    cv.waitKey(3)

#################################################################################
# The red colour masks
def red_mask(frame):

    # Threshold the image for red (HSV colour space)
    mask_lower = cv.inRange(frame, red_low_limit_lower, red_high_limit_lower)
    mask_upper = cv.inRange(frame, red_low_limit_upper, red_low_limit_upper)

    # Combine the masks and return to caller
    mask = cv.bitwise_or(mask_lower, mask_upper)
    return mask

# Green colour mask
def green_mask(frame):

    # Threshold the image with preset low and high HSV limits
    mask = cv.inRange(frame, green_low_limit, green_high_limit)
    return mask

#################################################################################
# Script segments the colours based on t-shirt colour: using RED and GREEN
def colour_segmentation(mask):

    # Initially empty bounding box
    shirt_box = []

    # Morphological close to help detections with large kernel size - filters out noise
    kernel = np.ones((30, 30), np.uint8)
    frame_bw = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Display the red shirt detection
    # show_image("Segmented", frame_bw)

    # Find the valid contours in the bw image for the regions detected
    contours = cv.findContours(frame_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

     # Loop through the detected contours
    for c in contours:

        # Calculate the area of the region
        area = cv.contourArea(c)

        # Filter out any small colour detections
        if area > 2000:

            # Show detection message
            rospy.loginfo("Detected red shirt")

            # Get the bounding box for the shirt
            x,y,w,h = cv.boundingRect(c)
            current_box = [x, y, w, h]

            # Append to the list of detected shirt boxes
            shirt_box.append(current_box)
            
    # Return the bounding box as a numpy array
    shirt_box = np.array(shirt_box)
    return shirt_box    

#################################################################################
def bound_callback(box):

    rospy.loginfo("Bounding box received")

    print(box)

#################################################################################
# /camera/color/image_raw/compressed

def listener():

    # Initilaise the node
    rospy.init_node('listener', anonymous=True)
    print('Initiating listener')

    # Set the ROS spin rate: 1Hz ~ 1 second
    rate = rospy.Rate(1)

    # Subscriber callbacks
    # rospy.Subscriber("/camera/color/image_raw/compressed_slow", CompressedImage, camera_callback, queue_size=1)
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, camera_callback, queue_size=1)
    rospy.Subscriber("bounding_box", bounding_box, bound_callback)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass