#!/usr/bin/env python3
# Machine Unlearning

# Packages
import rospy
import cv2 as cv
import numpy as np
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import CompressedImage
import camera_functions

# Create the publisher
# face_box_pub = rospy.Publisher("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), queue_size=1)

# Some links to help/ sus out:
# https://github.com/IntelRealSense/realsense-ros/issues/1524
# https://github.com/IntelRealSense/realsense-ros/issues/436
# https://github.com/IntelRealSense/realsense-ros/issues/714
# https://www.optisolbusiness.com/insight/depth-perception-using-intel-realsense
# https://docs.openvino.ai/latest/omz_models_model_mobilenet_ssd.html


#################################################################################
def depth_callback(frame):

    # log some info about the image topic
    rospy.loginfo('motion_prediciton\tDEPTH DATA POINTS RECEIVED')

    # Convert to CV2 image
    cv_frame = camera_functions.camera_bridge_convert(frame)

    # print(cv_frame)

    # PLAY AROUND WITH THE DIFFERENT TOPICS? ALSO WILL HAVE TO TAKE OFF TAPE FOR IT TO WORK
    camera_functions.show_image("Depth", cv_frame)


#################################################################################
def predict_motion():

    # Initilaise the node and display message 
    rospy.init_node('motion_prediciton', anonymous=True)
    rospy.loginfo('motion_prediciton\tNODE INIT')

    # Set the ROS spin rate: 1Hz ~ 1 second
    rate = rospy.Rate(1)        ############ Can make this an argument in launch and streamline rates##############

    # Subscriber callbacks
    rospy.Subscriber("/camera/depth/image_rect_raw/compressed", CompressedImage, depth_callback, queue_size=1)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        predict_motion()
    except rospy.ROSInterruptException:
        pass