#!/usr/bin/env python3
# Machine Unlearning

# RUN: roslaunch exp_mp realsense.launch

# FOR REALSENSE INSTALL: sudo apt-get install ros-melodic-realsense2-camera
# FOR REALSENSE INFO: https://github.com/IntelRealSense/realsense-ros

import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from exp_mp.msg import bounding_box
from sensor_msgs.msg import CompressedImage, Image
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


#################################################################################
# CHANGE TO LOGIC OF HAVING BOTH BOUNDING BOXES AND GETTING WHOLE BOX (TORSO)
def camera_callback(frame):

    # log some info about the image topic
    rospy.loginfo("CAMERA FRAME RECEIVED")

    # CHANGE TO LOGIC OF BOTH BOXES....
    
    # # Initially empty bounding box
    # whole_bounding_box_R = []
    # whole_bounding_box_G = []

    # # Test if face was detected and either red or green shirt was found
    # if (shirt_bounding_box_R.size is not 0 or shirt_bounding_box_G.size is not 0) and face_bounding_box is not tuple():       
        
    #     # Loop through to test the detected bounding boxes
    #     for (xf, yf, wf, hf) in face_bounding_box:
    #         for (xs, ys, ws, hs) in shirt_bounding_box_R:

    #             # If the x dimensions of the face are inside the box, then can assume that is a person (shirt with a face)
    #             if xf >= xs and xf + wf <= xs + ws and yf + hf < ys:
                    
    #                 # Get the total bounding box of the torso
    #                 whole_bounding_box_R.append([xs, yf, ws, hf+hs])

    #         for (xs, ys, ws, hs) in shirt_bounding_box_G:
    #             if xf >= xs and xf + wf <= xs + ws and yf + hf < ys:
    #                 whole_bounding_box_G.append([xs, yf, ws, hf+hs])

    #     # Convert to numpy
    #     whole_bounding_box_R = np.array(whole_bounding_box_R)
    #     whole_bounding_box_G = np.array(whole_bounding_box_G)

    #     # Add bounding box to the frame
    #     for (x, y, w, h) in whole_bounding_box_R:
    #         cv.rectangle(cv_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     for (x, y, w, h) in whole_bounding_box_G:
    #         cv.rectangle(cv_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # # Show the detected features with bounding boxes
    # show_image("Converted Image", cv_frame)

#################################################################################
def see_face(box):
    rospy.loginfo("listener\tFACE RECOGNISED")

#################################################################################
def enemy_sighted(box):
    rospy.loginfo("listener\tENEMY DETECTED")

#################################################################################
def homies_sighted(box):
    rospy.loginfo("listener\tHOMIE SIGHTED")

#################################################################################
# /camera/color/image_raw/compressed

def listener():

    # Initilaise the node
    rospy.init_node('listener', anonymous=True)
    rospy.loginfo('listener\tNODE INIT')

    # Set the ROS spin rate: 1Hz ~ 1 second
    rate = rospy.Rate(1)        ############ Can make this an argument in launch and streamline rates##############

    # Subscriber callbacks
    rospy.Subscriber("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), see_face, queue_size=1)
    # rospy.Subscriber("/django/eagle_eye/bounding_box_enemy", numpy_msg(Floats), enemy_sighted, queue_size=1)
    # rospy.Subscriber("/django/eagle_eye/bounding_box_homies", numpy_msg(Floats), homies_sighted, queue_size=1)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass