#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError

def dummy_image():

    print('Launching image publisher')

    pub = rospy.Publisher('webcam', Image, queue_size=10)
    rospy.init_node('dummy_image', anonymous=True)
    rate = rospy.Rate(0.25) # 1hz
    img = cv2.imread('test_im.png')

    bridge = CvBridge()
    imgMsg = bridge.cv2_to_imgmsg(img, "bgr8")

    while not rospy.is_shutdown():
        rospy.loginfo("Published img %s" % rospy.get_time())
        pub.publish(imgMsg)
        rate.sleep()

if __name__ == '__main__':
    try:
        dummy_image()
    except rospy.ROSInterruptException:
        pass