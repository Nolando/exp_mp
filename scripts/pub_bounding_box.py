#!/usr/bin/env python
# license removed for brevity

import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from exp_mp.msg import bounding_box

#################################################################################
def detect_human():

    # Temporary using image from webcam
    img = cv.imread("testimage.jpg")
    cv.imshow("showing image", img)

    # Initialise HOG descriptor
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

#################################################################################
def box():

    pub = rospy.Publisher('bounding_box', bounding_box, queue_size=10)
    rospy.init_node('box_publisher', anonymous=True)
    rate = rospy.Rate(0.25) # 1hz

    print('Launching bounding box publisher')

    # Detect the humans from the camera feed
    detect_human()

    box = [0,1,2,3]

    while not rospy.is_shutdown():
        rospy.loginfo(box)
        pub.publish(box)
        rate.sleep()

if __name__ == '__main__':
    try:
        box()
    except rospy.ROSInterruptException:
        pass