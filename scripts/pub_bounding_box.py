#!/usr/bin/env python
# license removed for brevity

import rospy
import cv2 as cv
import numpy as np
from sklearn import svm
from std_msgs.msg import String
from exp_mp.msg import bounding_box

#################################################################################
# HOG Info and theory
# - https://learnopencv.com/histogram-of-oriented-gradients/
# - https://thedatafrog.com/en/articles/human-detection-video/
# - https://arxiv.org/pdf/2205.02689v1.pdf - HOG and SVM classifier
# SVM Theory
# - https://www.youtube.com/watch?v=efR1C6CvhmE
# HOG and SVM OpenCV Applications
# - https://chrisjmccormick.wordpress.com/2013/05/09/hog-person-detector-tutorial/
# - https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
# - https://data-flair.training/blogs/python-project-real-time-human-detection-counting/
# 
# USING SVM > CNN: https://iopscience.iop.org/article/10.1088/1755-1315/357/1/012035/pdf#:~:text=Classification%20Accuracy%20of%20SVM%20and,to%20the%20big%20data%20hyperspectral


# TO DO:
# - get a dataset of people - realsense footage, security footage, training data
# - start SVM
# - use TUT 7 for help - layout

#################################################################################
# HOG features
# WILL BE USING HOG FEATURES WITH SVM TO TRAIN ALGORITHM
def HOG_features(frame, hog):

    # Initialise HOG descriptor with inbuilt SVM from OpenCV
    # hog = cv.HOGDescriptor()

    # Calls pretrained model for human detection
    # Choosing this over getPeopleDetector64x128() since accounts for all window sizes
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    # Process the image before detecting output is a grayscale image
    gray = image_processing(frame)

    # Detect people in the image and get bounding boxes
    # - changes in winStride
    # - changes in padding
    # - changes in scale
    bounding_boxes, weights = hog.detectMultiScale(gray, winStride=(4,4), padding=(8,8), scale=1.5)

    # Count for the detected people
    detected_people = 0
    
    # Change the x, y, width and height values into bounding box coodinates
    bounding_box_coords = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bounding_boxes])
    # print(bounding_box_coords)

    # Display bounding boxes overlayed onto the image/ frame
    for xA, yA, xB, yB in bounding_box_coords:
        cv.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
        cv.putText(frame, ('Person ' + str(detected_people)), (xA, yA), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        detected_people += 1

    # Display detected people data
    cv.putText(frame, ('Total People: ' + str(detected_people)), (30, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    # Display the image with bounding boxes
    cv.imshow("showing image", frame)
    cv.waitKey(1)
    print('Total people: ', detected_people)

#################################################################################
def image_processing(img):

    # Reshape may be requires to improve speed and accuracy?
    # h, w, c = img.shape
    # img = cv.resize(img, (int(w*0.9), int(h*0.9)))

    # Convert to grayscale for faster detection
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Enhance the image to improve contrast at edges
    gray = cv.equalizeHist(gray)
    return gray

#################################################################################
def box():

    pub = rospy.Publisher('bounding_box', bounding_box, queue_size=10)
    rospy.init_node('box_publisher', anonymous=True)

    # rate = rospy.Rate(0.25) # 1hz
    rate = rospy.Rate(5)     # faster spin rate

    print('Launching bounding box publisher')

    box = [0,1,2,3]

    while not rospy.is_shutdown():

        # Temporary using image from webcam
        img = cv.imread("testimage001.jpg")

        # Detect the humans from the camera feed
        HOG_features(img)

        # rospy.loginfo(box)
        pub.publish(box)
        rate.sleep()

    # ADD IF STATEMENT IF NO PERSON IN FRAME - DISPLAY A MESSAGE SAYING MOVE INTO FRAME OR SOMETHING
    # THIS IS BECAUSE IT DETECTS FULL BODY SO WILL NEED WHOLE TORSO IN SHOT OF CAMERA

# if __name__ == '__main__':
#     try:
#         box()
#     except rospy.ROSInterruptException:
#         pass