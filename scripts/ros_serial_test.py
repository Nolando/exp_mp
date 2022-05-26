#!/usr/bin/env python
# license removed for brevity
import serial
import sys
import time
import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Int8, Bool
import struct


class to_arduino():

    def __init__(self):

        print('Launching serial output')

        # ROS setup
        rospy.Subscriber("/django/mode", Int8, self.set_mode, queue_size=1)
        rospy.Subscriber("/django/fire_gun", Bool, self.set_fire, queue_size=1)
        rospy.Subscriber("/django/target_heart", numpy_msg(Floats), self.engage_enemy, queue_size=1)


        # Serial setup
        self.arduino = serial.Serial('/dev/ttyACM1', 9600)
        print(self.arduino.name)

        # Camera specs
        self.mode = 1                    # 1 = auto, 0 = manual
        self.fire = False
        self.cam_fov_half = 69.0/2       # degrees
        self.cam_width_half = 640.0/2    # px

        # Logic setup
        self.angle_servo = 90            # Central angle for servo
        self.gear_ratio = 2.1            # Ratio of 2.1:1 (gun:servo)

    def set_mode(self, new_mode):

        self.mode = Int8(new_mode.data)
        print('Swtiching modes', new_mode)


    def set_fire(self, new_fire):

        self.fire = bool(new_fire)
        print('Firing requested', new_fire)


    def engage_enemy(self, centroids):

        arduino = serial.Serial('/dev/ttyACM1', 9600)

        converted_centroids = np.array(centroids.data, dtype=np.float32)

        print('Enemies identified at', converted_centroids)

        # Distance from centre to target (px) (range = +-320 px)
        target_locations = [x - self.cam_width_half for x in converted_centroids]
        # print('target_locations', target_locations)

        # Choose closest target
        target_final_idx = np.argmin(np.abs(target_locations))      # Index
        target_final = target_locations[target_final_idx]           # Value
        # print('target_final', target_final)

        # Angular adjustment to target from centre (degrees to 1 dp) (range = +- 34.5 deg)
        adjustment_gun = (target_final/self.cam_width_half)*self.cam_fov_half
        adjustment_servo = round(adjustment_gun*self.gear_ratio, 1)
        # print('adjustment_gun', adjustment_gun, 'adjustment_servo', adjustment_servo)

        angle_out = int(self.angle_servo - adjustment_servo)

        # If user has to requested fire, shoot then set to false again
        if self.fire == True:   
            self.fire = False
            print('Firing!')

        # If in manual mode and user hasn't requested to fire, don't fire
        elif self.mode == 0:
            angle_out += 1000
            print('Holding fire')

        if self.mode == 1:
            time.sleep(3)
            print('Auto mode')

        print(self.mode, self.fire)
        print(type(angle_out), angle_out)

        # arduino.write(b'hello')

        
        string = b''
        string += struct.pack('!B',angle_out)


        self.arduino.write(string)
        self.arduino.flush()
        time.sleep(2)


def main():

    output = to_arduino()
    rospy.init_node('to_arduino', anonymous=True)

    rospy.spin()


if __name__ == '__main__':
    main()
