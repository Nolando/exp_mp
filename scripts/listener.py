#!/usr/bin/env python3
# Machine Unlearning

# RUN: roslaunch exp_mp realsense.launch

# FOR REALSENSE INSTALL: sudo apt-get install ros-melodic-realsense2-camera
# FOR REALSENSE INFO: https://github.com/IntelRealSense/realsense-ros

import rospy
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import CompressedImage
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import camera_functions


# Create the publisher
gui_pub = rospy.Publisher("/django/gui_scene/compressed", CompressedImage, queue_size=1)
target_pub = rospy.Publisher("/django/target_heart", numpy_msg(Floats), queue_size=1)

# Initialize the CvBridge class
bridge = CvBridge()

# global converted_frame
publish_frame = CompressedImage()

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
    def create_whole_box(self):

        # ROS rate of the listener needs to be slower than the shirt colour and facial publishers
        rate = rospy.Rate(15)   

        # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
        while not rospy.is_shutdown():

            # Test if enemy face detected
            if self.faced_detected and self.enemy_seen:

                self.faced_detected = False
                self.enemy_seen = False

                # Loop through to test the face bounding boxes
                for (xf, yf, wf, hf) in self.face_boxes:
                    
                    # Loop through the enemy shirt boxes
                    for (xe, ye, we, he) in self.enemy_boxes:

                        # If the x dimensions of the face are inside the box, then can assume that is a person (shirt with a face)
                        # Using hf/2 in case the shirt covers bottoms of face height
                        if xe <= xf and xf + wf <= xe + we and yf + hf/2 < ye:

                            # print("Confirmed enemy spotted")

                            # Add the bounding box to the frame
                            self.converted_frame = camera_functions.bounding_box_to_frame([[xe, yf, we, hf+he]], self.converted_frame, (0, 0, 255))
                            # camera_functions.show_image("Bounding Box Over Enemy", self.converted_frame)

                            # Publish the heart location
                            to_publish = np.array([xe+we/2], dtype=np.float32)
                            target_pub.publish(to_publish)

            # Test green shirt and face detected
            if self.faced_detected and self.homie_seen:

                self.faced_detected = False
                self.homie_seen = False

                # Similarly loop through to test if eligible person detected
                for (xf, yf, wf, hf) in self.face_boxes:
                    for (xe, ye, we, he) in self.homie_boxes:

                        # Get the whole bounding box over the person
                        if xe <= xf and xf + wf <= xe + we and yf + hf/2 < ye:
                            print("Confirmed homedog spotted")
                            self.converted_frame = camera_functions.bounding_box_to_frame([[xe, yf, we, hf+he]], self.converted_frame, (0, 255, 0))
                            # camera_functions.show_image("Bounding Box Over HOMIE", self.converted_frame)

            else:
                print("\t\t\tnope")

            # Convert from CV2 image to ROS compressed image
            publish_frame.data = bridge.cv2_to_compressed_imgmsg(self.converted_frame.tobytes(), dst_format='jpg')

            # Publish the frame with bounding boxes
            gui_pub.publish(publish_frame)

            camera_functions.show_image("DJANGO VISION", self.converted_frame)

            rate.sleep()

    #############################################################################
    # Camera callback
    def django_eyes(self, frame):

        # global converted_frame
        self.converted_frame = camera_functions.camera_bridge_convert(frame)

        # Boolean to track camera subscription
        self.eyes_open = True

        # camera_functions.show_image("Test", self.converted_frame)

        # Log some info about the image topic
        # rospy.loginfo("listener\tEYES ON")

    #############################################################################
    # Face bounding box subscriber at a rate of 50Hz = 20ms
    def see_face(self, b_box):

        # Log some info about the image topic
        # rospy.loginfo("listener\tFACE")

        # Save bounding box data into a CV2 suitable format
        self.face_boxes = listener.convert_box_format(self, b_box)

        # Save bounding box as a converted 2D numpy array for iteration
        self.face_boxes = np.array([b_box.data], int)
        
        # Test if multiple faces detected
        if self.face_boxes.size > 4:

            # Reshape into 2D
            self.face_boxes = np.reshape(self.face_boxes, (-1, 4))

        # Boolean to track detection
        self.faced_detected = True

        # Show bounding box on frame
        # frame = camera_functions.bounding_box_to_frame(self.face_boxes, converted_frame, (255, 0, 0))
        # camera_functions.show_image("Testing facial bounding box", frame)

    #############################################################################
    def enemy_sighted(self, b_box):

        # Log some info about the image topic
        # rospy.loginfo("listener\tENEMY DETECTED")

        # Save bounding box data into a CV2 suitable format
        self.enemy_boxes = listener.convert_box_format(self, b_box)

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
        # camera_functions.show_image("Testing RED SHIRT ENEMY bounding box", frame)

    #############################################################################
    def homies_sighted(self, b_box):

        # log some info about the image topic
        # rospy.loginfo("listener\tHOMIE SIGHTED")

        # Save bounding box data into a CV2 suitable format
        self.homie_boxes = listener.convert_box_format(self, b_box)

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
    listener_node.create_whole_box()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass