#!/usr/bin/env python
# license removed for brevity

import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from exp_mp.msg import bounding_box

#################################################################################
# https://chrisjmccormick.wordpress.com/2013/05/09/hog-person-detector-tutorial/
# https://learnopencv.com/histogram-of-oriented-gradients/
# https://thedatafrog.com/en/articles/human-detection-video/
# Useful links ^

# HOG features with SVM
def detect_human():

    # Temporary using image from webcam
    img = cv.imread("testimage004.jpg")
    
    # Check the dimensions of the image/ frame
    # h, w, c = img.shape
    # print('width: ', w)
    # print('height: ', h)
    # print('channel: ', c)

    # Initialise HOG descriptor
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    # Convert to grayscale for faster detection
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Detect people in the image and get bounding boxes
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
    
    # Change the x, y, width and height values into bounding box coodinates
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    print(boxes)

    # Display bounding boxes overlayed onto the image/ frame
    for (xA, yA, xB, yB) in boxes:
        cv.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv.imshow("showing image", img)
    cv.waitKey(1)

#################################################################################
def box():

    pub = rospy.Publisher('bounding_box', bounding_box, queue_size=10)
    rospy.init_node('box_publisher', anonymous=True)

    # rate = rospy.Rate(0.25) # 1hz
    rate = rospy.Rate(5)     # faster spin rate

    print('Launching bounding box publisher')

    box = [0,1,2,3]

    while not rospy.is_shutdown():

        # Detect the humans from the camera feed
        detect_human()

        # rospy.loginfo(box)
        pub.publish(box)
        rate.sleep()

    # ADD IF STATEMENT IF NO PERSON IN FRAME - DISPLAY A MESSAGE SAYING MOVE INTO FRAME OR SOMETHING
    # THIS IS BECAUSE IT DETECTS FULL BODY SO WILL NEED WHOLE TORSO IN SHOT OF CAMERA

if __name__ == '__main__':
    try:
        box()
    except rospy.ROSInterruptException:
        pass