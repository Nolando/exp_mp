#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

# Initialize the CvBridge class
bridge = CvBridge()    

# Define a function to show the image in an OpenCV Window
def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

def image_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo("Image received")

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Show the converted image
    show_image(cv_image)

def listener():

    print('Initiating listener')

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("webcam", Image, image_callback)
    rospy.Subscriber("detect_people", )

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass