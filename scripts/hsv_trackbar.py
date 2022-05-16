#!/usr/bin/env python
# Machine Unlearning
# TEMPORARY SCRIPT TO TEST THE HSV THRESHOLDS - ADJUST ACCORDINGLY


# RUN THIS FIRST FOR EXECUTABLE: chmod +x src/exp_mp/scripts/listener.py


import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage

# Initialize the CvBridge class
bridge = CvBridge()

#################################################################################
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

# TEMPORARY USING THIS TO GET COLOUR THRESHOLD LIMITS
cap = cv.VideoCapture(0)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

#################################################################################
def image_callback(img_msg):
    
    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(img_msg)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # USE THIS FOR TESTING THE HSV THRESHOLD LIMITS
    temp_thresh_fn(cv_image)

#################################################################################
# Define a function to show the image in an OpenCV Window
def show_image(window_name, img):

    # Display the image
    cv.imshow(window_name, img)
    cv.waitKey(3)

#################################################################################
# Temp function for testing the threshold limits
def temp_thresh_fn(img):

    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

    ret, frame = cap.read()
    frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    show_image(window_capture_name, img)
    show_image(window_detection_name, frame_threshold)

    print("\n\n{}\t{}\n{}\t{}".format("Low H value:", low_H, "High H value:", high_H))
    print("{}\t{}\n{}\t{}".format("Low S value:", low_S, "High S value:", high_S))
    print("{}\t{}\n{}\t{}".format("Low V value:", low_V, "High V value:", high_V))

#################################################################################
def hsv_trackbar():
    print('Initiating listener')
    rospy.init_node('hsv_trackbar', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, image_callback)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        hsv_trackbar()
    except rospy.ROSInterruptException:
        pass