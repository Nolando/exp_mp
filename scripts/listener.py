#!/usr/bin/env python
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
def image_callback(img_msg):

    # log some info about the image topic
    rospy.loginfo("Image received")

    # Try to convert the ROS Image message to a CV2 Image
    try:
        # cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_image = bridge.compressed_imgmsg_to_cv2(img_msg)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Show the converted image
    # show_image("Converted Image", cv_image)

    face_recognised = facialRecognition.faceDetect(cv_image)
    show_image("Face Image", face_recognised)

    # Get the bounding box
    # hog = cv.HOGDescriptor()
    # pub_bounding_box.HOG_features(cv_image, hog)

    # Colour segmentation
    # red_colour_segmentaion(cv_image)

#################################################################################
# Define a function to show the image in an OpenCV Window
def show_image(window_name, img):

    # Display the image
    cv.imshow(window_name, img)
    cv.waitKey(3)

#################################################################################
# Script segments the colours based on t-shirt colour: using RED and GREEN
def red_colour_segmentaion(img):

    # Convert RGB into HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Threshold the image for red (HSV colour space)
    mask_lower = cv.inRange(hsv, red_low_limit_lower, red_high_limit_lower)
    mask_upper = cv.inRange(hsv, red_low_limit_upper, red_low_limit_upper)

    # Combine the masks
    mask = cv.bitwise_or(mask_lower, mask_upper)
    # show_image(mask)

    # Morphological close to help detections with large kernel size - filters out noise
    kernel = np.ones((30, 30), np.uint8)
    img_bw = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Display the red shirt detection
    # show_image("Red segmented", img_bw)

    # Things to consider: area check in case of other objects

    # Find the valid contours in the bw image for the regions detected
    contours = cv.findContours(img_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

     # Loop through the detected contours
    for c in contours:

        # Calculate the area of the region
        area = cv.contourArea(c)

        # Filter out any small colour detections
        if area > 5000:
            # print(area)
            perimeter = cv.drawContours(img, [c], 0, (0,255,0), 3)
            show_image("Red outline", perimeter)


    
    # Get the maximum and minimum detected points in x and y direction
    # - With camera flat: y(0) at top of frame, x(0) on left side from show_image
    result = np.where(img_bw == np.amax(img_bw))
    y_start = np.min(result[0])
    y_end = np.max(result[0])
    x_start = np.min(result[1])
    x_end = np.max(result[1])
    
    # print(x_start, x_end, y_start, y_end)

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

    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, image_callback)
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