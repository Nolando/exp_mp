#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


def talker():
    pub = rospy.Publisher("/django/target_heart", numpy_msg(Floats), queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.5) # 10hz
    while not rospy.is_shutdown():

        out = np.array([200, 300], dtype=np.float32)

        rospy.loginfo(out)
        pub.publish(out)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
