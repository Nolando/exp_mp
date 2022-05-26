#!/usr/bin/env python
# license removed for brevity
import serial
import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


class to_arduino():

    def __init__(self):

        print('Launching serial output')

        # ROS setup
        rospy.Subscriber("/django/target_heart", numpy_msg(Floats), self.engage_enemy, queue_size=1)


        # Serial setup
        # self.arduino = serial.Serial('/dev/ttyACM0', 9600)
        # print(arduino.name)

        # Camera specs
        self.cam_fov_half = 69.0/2         # degrees
        self.cam_width_half = 640.0/2     # px

        # Logic setup
        self.angle_servo = 90            # Central angle for servo
        self.gear_ratio = 2.1            # Ratio of 2.1:1 (gun:servo)


    def engage_enemy(self, centroids):

        converted_centroids = np.array(centroids.data, dtype=np.float32)

        print('Enemies identified at', converted_centroids)

        # Distance from centre to target (px) (range = +-320 px)
        target_locations = [x - self.cam_width_half for x in converted_centroids]
        print('target_locations', target_locations)

        # Choose closest target
        target_final_idx = np.argmin(np.abs(target_locations))      # Index
        target_final = target_locations[target_final_idx]           # Value
        print('target_final', target_final)

        # Angular adjustment to target from centre (degrees to 1 dp) (range = +- 34.5 deg)
        adjustment_gun = (target_final/self.cam_width_half)*self.cam_fov_half
        adjustment_servo = round(adjustment_gun*self.gear_ratio, 1)
        print('adjustment_gun', adjustment_gun, 'adjustment_servo', adjustment_servo)

        angle_out = self.angle_servo - adjustment_servo
        print('angle_out', angle_out)

        # self.arduino.write(angle_out)



def main():

    output = to_arduino()
    rospy.init_node('to_arduino', anonymous=True)

    rospy.spin()


if __name__ == '__main__':
    main()
