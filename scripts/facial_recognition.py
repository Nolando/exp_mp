#!/usr/bin/env python3
# Machine Unlearning

# Packages
import os

import rospy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import CompressedImage
import camera_functions
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


# Create the publisher
face_box_pub = rospy.Publisher("/django/eagle_eye/bounding_box_face", numpy_msg(Floats), queue_size=1)

# Initialise the face cascade classifier (done outside of function to try improve latency)
# + LATENCY fixed by setting camera subscriber queue size to 1
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + cascadePath)


#################################################################################
class recognise_face:

    #############################################################################
    def __init__(self):

        # Initialise bounding box variable
        self.box = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # ROS node initialisation
        rospy.init_node('facial_recognition', anonymous=True)
        rospy.loginfo('facial_recognition\tNODE INIT')

    #############################################################################
    def subscribe_to_cam(self):

        # Subscriber callbacks
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.camera_callback, queue_size=1)
        
    #############################################################################
    def publish_box(self):

        # Set the ROS rate for publishing
        rate = rospy.Rate(120)

        # # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
        while not rospy.is_shutdown():

            # Check that there was a face detected (tuple returned if no)
            if self.box is not tuple():
                
                # Publish the bounding box
                face_box_pub.publish(self.box)

            # Sleep for the rest of the ROS rate
            rate.sleep()
            # rospy.spin()

    #############################################################################
    def camera_callback(self, frame):

        # log some info about the image topic
        # rospy.loginfo('facial_recognition\tCAMERA FRAME RECEIVED')

        # Convert the ROS image to OpenCV compatible image
        converted_frame = camera_functions.camera_bridge_convert(frame)

        # Get the face bounding box
        self.box = recognise_face.face_detect(self, converted_frame)      # np.ndarray if detected, tuple if empty

        # Test for checking box is correct
        # for (x, y, w, h) in self.box:
        #     cv.rectangle(converted_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # camera_functions.show_image("face_detection node", converted_frame)

    #################################################################################
    # Uses the Haar feature based cascade classifiers in OpenCV to detect faces
    def face_detect(self, img):

        # Get the current image frame and make a greyscale copy
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Adjust the resolution percentage decrease - FOR FASTER DETECTION
        # 100% resolution range is around 5m
        # 70% resolution range is around 2.5-3m
        percentage_decrease = 80

        # Get the width and height dimensions
        width = int(img_gray.shape[1] * percentage_decrease / 100)
        height = int(img_gray.shape[0] * percentage_decrease / 100)
        dim = (width, height)

        # Resize the image to smaller resolution
        img_gray = cv.resize(img_gray, dim, interpolation=cv.INTER_AREA)

        # Detect faces in the image as a bounding box
        faces = faceCascade.detectMultiScale(
            img_gray,
            scaleFactor=1.12,           # Attempting to tune this was originally 1.1
            minNeighbors=5,             # Can adjust to fix latency also, originally was 5
            minSize=(30, 30),
            flags = cv.CASCADE_SCALE_IMAGE
        )

        # Test if face was detected in the frame
        if faces is not tuple():

            # Convert box coordiantes back to original frame resolution
            faces = faces * (100 / percentage_decrease)                 # np array float64
            faces = np.array(faces, dtype=np.float32)                   # np array float32

            # Convert to 1D array from 2D **KEY
            faces = faces.flatten()

        # Return the face bounding box
        return faces

#################################################################################
def main():

    # Create the class instance
    facial = recognise_face()

    # Subscribe to the camera topic and find faces
    facial.subscribe_to_cam()

    # Publish the face bounding boxes
    facial.publish_box()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
