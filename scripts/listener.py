#!/usr/bin/env python3
# Machine Unlearning

# RUN: roslaunch exp_mp realsense.launch

# FOR REALSENSE INSTALL: sudo apt-get install ros-melodic-realsense2-camera
# FOR REALSENSE INFO: https://github.com/IntelRealSense/realsense-ros

import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import Int64, Int64MultiArray
from exp_mp.msg import bounding_box
from sensor_msgs.msg import CompressedImage
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

import camera_functions


#################################################################################
# CHANGE TO LOGIC OF HAVING BOTH BOUNDING BOXES AND GETTING WHOLE BOX (TORSO)
# def django_eyes(frame):

#     # log some info about the image topic
#     rospy.loginfo("listener\tEYES ON")

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

# Global variables
# global converted_frame


#################################################################################
# Class definition
class listener:

    #############################################################################
    # Initialisation constructor function creates the subscribers
    def __init__(self):

        # Variable initialisation
        self.converted_frame = np.empty((2,2))
        self.face_boxes = np.empty((2,2))
        self.enemy_boxes = np.empty((2,2))
        self.homie_boxes = np.empty((2,2))
        

        # ROS node initialisation
        rospy.init_node('listener', anonymous=True)
        rospy.loginfo('listener\tNODE INIT')

        # ROS subscriber callbacks
        # self.eyes_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.django_eyes, queue_size=1)
        # self.face_sub = rospy.Subscriber("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), self.see_face, queue_size=1)
        # self.enemy_sub = rospy.Subscriber("/django/eagle_eye/bounding_box_enemy", numpy_msg(Floats), self.enemy_sighted, queue_size=1)
        # self.homie_sub = rospy.Subscriber("/django/eagle_eye/bounding_box_homies", numpy_msg(Floats), self.homies_sighted, queue_size=1)
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.django_eyes, queue_size=1)
        rospy.Subscriber("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), self.see_face, queue_size=1)
        # rospy.Subscriber("/django/eagle_eye/bounding_box_enemy", numpy_msg(Int64), self.enemy_sighted, queue_size=1)
        # rospy.Subscriber("/django/eagle_eye/bounding_box_homies", numpy_msg(Int64), self.homies_sighted, queue_size=1)

        # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
        while not rospy.is_shutdown():
            rospy.spin()

    #############################################################################
    # Camera callback
    def django_eyes(self, frame):

        # global converted_frame - WORKING
        self.converted_frame = camera_functions.camera_bridge_convert(frame)

        # log some info about the image topic
        rospy.loginfo("listener\tEYES ON")

    #############################################################################
    def see_face(self, b_box):

        # Save bounding box in self scope
        # self.face_boxes = b_box.data

        # log some info about the image topic
        rospy.loginfo("listener\tFACE")

        print(b_box.data)
        # print(self.face_boxes)
        
        # show boubding box on frame
        # camera_functions.bounding_box_to_frame(self.face_boxes, self.converted_frame)

    #############################################################################
    def enemy_sighted(self, b_box):

        # Save bounding box in self scope
        self.enemy_boxes = b_box

        # log some info about the image topic
        rospy.loginfo("listener\tENEMY DETECTED")

    #############################################################################
    def homies_sighted(self, b_box):

        # Save bounding box in self scope
        self.homie_boxes = b_box

        # log some info about the image topic
        rospy.loginfo("listener\tHOMIE SIGHTED")

#################################################################################
def main():

    # Initialise the listener class and functions
    listener()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass