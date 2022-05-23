#!/usr/bin/env python3
# Machine Unlearning

import rospy
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# import numpy as np
# from sensor_msgs.msg import CompressedImage, Image

# Initialize the CvBridge class
bridge = CvBridge()

#################################################################################
# Converts the input ROS image message to CV2 image
def camera_bridge_convert(frame):

    # Convert the frame from a ROS Image message to a CV2 Image
    try:
        bridged_frame = bridge.compressed_imgmsg_to_cv2(frame)

        # Flip the camera feed along y axis for correct orientation
        cv_frame = cv.flip(bridged_frame, 1)
    
    # Error message if failed
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Return the CV2 image
    return cv_frame
    
#################################################################################
# Define a function to show the image in an OpenCV Window
def show_image(window_name, frame):

    # Display the image frame
    cv.imshow(window_name, frame)
    cv.waitKey(3)