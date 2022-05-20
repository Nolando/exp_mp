#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String,Empty

def serial_test():

    print('Launching serial publisher')

    pub = rospy.Publisher('toggle_led', String, queue_size=10)
    rospy.init_node('toggle_led', anonymous=True)
    rate = rospy.Rate(1) # 1hz

    string = "noot"

    # empty = []   

    while not rospy.is_shutdown():
        rospy.loginfo("Published msg %s at %s" % (string, rospy.get_time()))
        pub.publish(string)
        rate.sleep()

if __name__ == '__main__':
    try:
        serial_test()
    except rospy.ROSInterruptException:
        pass