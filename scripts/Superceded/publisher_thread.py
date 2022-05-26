#!/usr/bin/env python3
# Machine Unlearning


from sniffio import current_async_library
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from exp_mp.msg import bounding_box
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage

# Class definiiton
class republish_camera:

    #################################################################################
    # Initialisation function creates publisher and subscriber
    def __init__(self):

        # ROS node initialisation
        rospy.init_node('camera_republisher', anonymous=True)

        # Publisher creation
        self.camera_repub = rospy.Publisher("/camera/color/image_raw/compressed_slow", CompressedImage, queue_size=1)
        
        # Camera subscriber callback
        self.sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.camera_callback, queue_size=1)

        rospy.spin()

    #################################################################################
    # Camera callback
    def camera_callback(self, img_msg):

        # # Set the ROS spin rate: 1Hz ~ 1 second
        # rate = rospy.Rate(1)

        # # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
        # while not rospy.is_shutdown():

        # Log republishing
        rospy.loginfo("Republishing image at slower rate")

        # Publish and sleep for excess spin time
        self.camera_repub.publish(img_msg)
            # rate.sleep()

#################################################################################
def main():

    # Initialise the republish class and functions
    republish_camera()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass