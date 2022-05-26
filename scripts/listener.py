#!/usr/bin/env python3
# Machine Unlearning

# RUN: roslaunch exp_mp realsense.launch

# FOR REALSENSE INSTALL: sudo apt-get install ros-melodic-realsense2-camera
# FOR REALSENSE INFO: https://github.com/IntelRealSense/realsense-ros

import rospy
import cv2 as cv
import numpy as np
from exp_mp.msg import bounding_box
from sensor_msgs.msg import CompressedImage
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

import camera_functions


#################################################################################
# CHANGE TO LOGIC OF HAVING BOTH BOUNDING BOXES AND GETTING WHOLE BOX (TORSO)
# def django_eyes(frame):


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
# Class definition
class listener:

    #############################################################################
    # Initialisation constructor function creates the subscribers
    def __init__(self):

        # Variable initialisation
        self.converted_frame = np.empty((2,2))      # CV2 compatible current camera feed
        self.face_boxes = np.empty((1,4))
        self.enemy_boxes = np.empty((1,4))
        self.homie_boxes = np.empty((1,4))
        
        # ROS node initialisation
        rospy.init_node('listener', anonymous=True)
        rospy.loginfo('listener\tNODE INIT')

        # Booleans for the bounding box availability
        self.faced_detected = False
        self.enemy_seen = False
        self.homie_seen = False
        self.eyes_open = False

        # ROS subscriber callbacks
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.django_eyes, queue_size=1)
        rospy.Subscriber("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), self.see_face, queue_size=1)
        rospy.Subscriber("/django/eagle_eye/bounding_box_enemy", numpy_msg(Floats), self.enemy_sighted, queue_size=1)
        rospy.Subscriber("/django/eagle_eye/bounding_box_homies", numpy_msg(Floats), self.homies_sighted, queue_size=1)

    #############################################################################
    def test(self):

        # ROS rate of the listener needs to be slower than the shirt colour and facial publishers
        rate = rospy.Rate(25)   

        while not rospy.is_shutdown():

            if self.faced_detected:
                print("face detected")
                self.faced_detected = False
            else:
                print("no face")

            if self.enemy_seen:
                print("\t\tenemy FOUND FUCK")
                self.enemy_seen = False
            else:
                print("\t\tsafe boys")

            if self.homie_seen:
                print("\t\t\t\thomie seen")
                self.homie_seen = False
            else:
                print("\t\t\t\tno homedogs rip")

            rate.sleep()

    #############################################################################
    def create_whole_box(self):

        # ROS rate
        rate = rospy.Rate(50)

        # Initially empty bounding box
        self.whole_bounding_box_R = []
        self.whole_bounding_box_G = []

        # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
        while not rospy.is_shutdown():

            print(self.faced_detected)

            if self.eyes_open:

                # Test if face was detected and either red or green shirt was found
                if (self.enemy_seen or self.homie_seen) and self.faced_detected:

                    # Loop through to test the detected bounding boxes
                    for (xf, yf, wf, hf) in self.face_boxes:

                        # Enemy with red shirt
                        for (xs, ys, ws, hs) in self.enemy_boxes:

                            # If the x dimensions of the face are inside the box, then can assume that is a person (shirt with a face)
                            if xf >= xs and xf + wf <= xs + ws and yf + hf < ys:
                                
                                # Get the total bounding box of the torso
                                self.whole_bounding_box_R.append([xs, yf, ws, hf+hs])

                        # Homie with green shirt
                        # for (xs, ys, ws, hs) in self.homie_boxes:
                        #     if xf >= xs and xf + wf <= xs + ws and yf + hf < ys:
                        #         self.whole_bounding_box_G.append([xs, yf, ws, hf+hs])

                    # Convert to numpy
                    self.whole_bounding_box_R = np.array(self.whole_bounding_box_R)
                    # self.whole_bounding_box_G = np.array(self.whole_bounding_box_G)

                    # Add bounding box to the frame
                    for (x, y, w, h) in self.whole_bounding_box_R:
                        cv.rectangle(self.converted_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # for (x, y, w, h) in self.whole_bounding_box_G:
                    #     cv.rectangle(self.converted_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    self.faced_detected = False
                    self.enemy_seen = False
                    self.homie_seen = False

                # Show the detected features with bounding boxes
                camera_functions.show_image("Converted Image", self.converted_frame)

                self.eyes_open = False
                
            # rospy.spin()
            rate.sleep()

    #############################################################################
    # Camera callback
    def django_eyes(self, frame):

        # global converted_frame - WORKING
        global converted_frame
        converted_frame = camera_functions.camera_bridge_convert(frame)

        # Boolean to track camera subscription
        self.eyes_open = True

        # Log some info about the image topic
        # rospy.loginfo("listener\tEYES ON")

    #############################################################################
    # Face bounding box subscriber at a rate of 50Hz = 20ms
    def see_face(self, b_box):

        # Log some info about the image topic
        # rospy.loginfo("listener\tFACE")

        # Save bounding box data into a CV2 suitable format - CHANGE TO THIS IN SUBMITTED CODE
        # self.face_boxes = listener.convert_box_format(self, b_box)

        # Save bounding box as a converted 2D numpy array for iteration
        self.face_boxes = np.array([b_box.data], int)
        
        # Test if multiple faces detected
        if self.face_boxes.size > 4:

            # Reshape into 2D
            self.face_boxes = np.reshape(self.face_boxes, (-1, 4))

        # Boolean to track detection
        self.faced_detected = True

        # Show bounding box on frame
        frame = camera_functions.bounding_box_to_frame(self.face_boxes, converted_frame, (255, 0, 0))
        camera_functions.show_image("Testing facial bounding box", frame)

    #############################################################################
    def enemy_sighted(self, b_box):

        # Log some info about the image topic
        # rospy.loginfo("listener\tENEMY DETECTED")

        # Save bounding box data into a CV2 suitable format - CHANGE TO THIS IN SUBMITTED CODE
        # self.enemy_boxes = listener.convert_box_format(self, b_box)

        # Save bounding box as a converted 2D numpy array for iteration
        self.enemy_boxes = np.array([b_box.data], int)
        
        # Test if multiple faces detected
        if self.enemy_boxes.size > 4:

            # Reshape into 2D
            self.enemy_boxes = np.reshape(self.enemy_boxes, (-1, 4))

        # Boolean to track detection
        self.enemy_seen = True

        # Show red bounding box on frame
        # frame = camera_functions.bounding_box_to_frame(self.enemy_boxes, self.converted_frame, (0, 0, 255))
        # self.frame = camera_functions.bounding_box_to_frame(self.enemy_boxes, self.frame, (0, 0, 255))
        # camera_functions.show_image("Testing RED SHIRT ENEMY bounding box", self.frame)

    #############################################################################
    def homies_sighted(self, b_box):

        # log some info about the image topic
        # rospy.loginfo("listener\tHOMIE SIGHTED")

        # Save bounding box data into a CV2 suitable format - CHANGE TO THIS IN SUBMITTED CODE
        # self.homie_boxes = listener.convert_box_format(self, b_box)

        # Save bounding box as a converted 2D numpy array for iteration
        self.homie_boxes = np.array([b_box.data], int)
        
        # Test if multiple faces detected
        if self.homie_boxes.size > 4:

            # Reshape into 2D
            self.homie_boxes = np.reshape(self.homie_boxes, (-1, 4))

        # Boolean to track detection
        self.homie_seen = True

        # Show red bounding box on frame
        # self.frame = camera_functions.bounding_box_to_frame(self.homie_boxes, self.converted_frame, (0, 255, 0))
        # camera_functions.show_image("Testing GREEN SHIRT homedog bounding box", self.frame)

    #############################################################################
    def convert_box_format(self, subscribed_box):

        # Converts the ROS subscribed to numpy float to compatible with CV
        formated_box = np.array([subscribed_box.data], int)

        # Test if multiple boxes detected
        if formated_box.size > 4:

            # Reshape into 2D
            formated_box = np.reshape(formated_box, (-1, 4))
            # print(formated_box)
        
        # Return the formatted box
        return formated_box

#################################################################################
def main():

    # Initialise the listener class and functions
    listener_node = listener()
    # listener()

    # Get the whole bounding box of detected people
    listener_node.test()
    # listener_node.create_whole_box()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass