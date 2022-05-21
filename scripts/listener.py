#!/usr/bin/env python3
# Machine Unlearning

# RUN: roslaunch exp_mp realsense.launch

# FOR REALSENSE INSTALL: sudo apt-get install ros-melodic-realsense2-camera
# FOR REALSENSE INFO: https://github.com/IntelRealSense/realsense-ros


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
# green_low_limit = np.array([140, 30, 0])          # Green
# green_high_limit = np.array([179, 255, 255])


#################################################################################
def camera_callback(frame):

    # log some info about the image topic
    rospy.loginfo("CAMERA FRAME RECEIVED")

    # Convert the frame from a ROS Image message to a CV2 Image
    try:
        cv_frame = bridge.compressed_imgmsg_to_cv2(frame)
    
    # Error message if failed
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Get the face bounding box
    face_bounding_box = facialRecognition.faceDetect(cv_frame)      # np.ndarray if detected, tuple if empty

    # Test if face was detected in the frame
    if face_bounding_box is not tuple():
        rospy.loginfo('Face bounding box is: ' + np.array2string(face_bounding_box))

        # Add the bounding box to the current frame
        for (x, y, w, h) in face_bounding_box:
            cv.rectangle(cv_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            show_image("Converted Image", cv_frame)

    # Colour segmentation
    shirt_bounding_box = red_colour_segmentation(cv_frame)          # np.ndarray if detected, tuple if empty

    if shirt_bounding_box is not tuple():       
        rospy.loginfo('Shirt bounding box is: ' + np.array2string(shirt_bounding_box))

        # Add bounding box to image frame
        cv.rectangle(cv_frame, (shirt_bounding_box[0], shirt_bounding_box[1]), 
                               (shirt_bounding_box[2], shirt_bounding_box[3]), (0, 0, 255), 2)
    
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
# Script segments the colours based on t-shirt colour: using RED and GREEN
def red_colour_segmentation(frame):

    # Initially empty bounding box
    shirt_box = tuple()

    # Convert RGB into HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold the image for red (HSV colour space)
    mask_lower = cv.inRange(hsv, red_low_limit_lower, red_high_limit_lower)
    mask_upper = cv.inRange(hsv, red_low_limit_upper, red_low_limit_upper)

    # Combine the masks
    mask = cv.bitwise_or(mask_lower, mask_upper)

    # Morphological close to help detections with large kernel size - filters out noise
    kernel = np.ones((30, 30), np.uint8)
    frame_bw = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Display the red shirt detection
    # show_image("Red segmented", frame_bw)

    # Things to consider: area check in case of other objects

    # Find the valid contours in the bw image for the regions detected
    contours = cv.findContours(frame_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

     # Loop through the detected contours
    for c in contours:

        # Calculate the area of the region
        area = cv.contourArea(c)

        # Filter out any small colour detections
        if area > 5000:

            # Show detection message
            rospy.loginfo("Detected red shirt")
            
            # Perimeter shows the outline of the detected area
            # perimeter = cv.drawContours(frame, [c], 0, (0,255,0), 3)

            # Get the bounding box for the shirt
            x,y,w,h = cv.boundingRect(c)
            shirt_box = np.array([x, y, x+w, y+h])

    # Return the bounding box
    return shirt_box

    # To do:
    # - figure out if red (differentiate from green/ other colours)
    # - do bounding box over all the tshirt - assume tshirt height is fixed
    # - try fill in the binary image

    

#################################################################################
def bound_callback(box):

    rospy.loginfo("Bounding box received")

    print(box)

#################################################################################
# /camera/color/image_raw/compressed

def listener():

    print('Initiating listener')

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, camera_callback)
    # rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.Subscriber("bounding_box", bounding_box, bound_callback)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass