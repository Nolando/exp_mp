#!/usr/bin/env python3
# This script identifies the colour of the human's shirt to determine
# if a target or non-threat

# Packages
import rospy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import CompressedImage
import camera_functions
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# Create the publishers
shirt_box_pub_R = rospy.Publisher("/django/eagle_eye/bounding_box_enemy", numpy_msg(Floats), queue_size=1)
shirt_box_pub_G = rospy.Publisher("/django/eagle_eye/bounding_box_homies", numpy_msg(Floats), queue_size=1)

# Set the limits for the colours (in HSV)
red_low_limit_lower = np.array([0, 100, 90])           # Red
red_high_limit_lower = np.array([5, 255, 255])
red_low_limit_upper = np.array([160, 88, 90])           # Red
red_high_limit_upper = np.array([180, 255, 255])

green_low_limit = np.array([58, 35, 0])          # Green
green_high_limit = np.array([86, 255, 255])

#################################################################################
def camera_callback(frame):

    # log some info about the image topic
    # rospy.loginfo('shirt_detection\tCAMERA FRAME RECEIVED')

    # Convert the ROS image to OpenCV compatible image
    converted_frame = camera_functions.camera_bridge_convert(frame)

    # Convert RGB into HSV
    hsv_frame = cv.cvtColor(converted_frame, cv.COLOR_BGR2HSV)

    # Get the red and green colour masks using HSV thresholds
    mask_R = red_mask(hsv_frame)
    mask_G = green_mask(hsv_frame)

    # RGB
    # rgb_mask = cv.inRange(converted_frame, np.array([0, 0, 50]), np.array([52, 52, 255]))  

    # L*a*b* with Otsu threshold on A-channel
    # lab_frame = cv.cvtColor(converted_frame, cv.COLOR_BGR2LAB)
    # th = cv.threshold(lab_frame[:,:,1], 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # Colour segment to get the bounding box for shirts for red and green
    shirt_bounding_box_R = colour_segmentation(mask_R)          # np.ndarray float32
    # shirt_bounding_box_G = colour_segmentation(mask_G)
    
    # Convert to 1D array from 2D
    shirt_bounding_box_R = shirt_bounding_box_R.flatten()
    # shirt_bounding_box_G = shirt_bounding_box_G.flatten()

    # Test if the returned variable contains a detected face bounding box
    if shirt_bounding_box_R.size is not 0:

        # rospy.loginfo('shirt_detection\tDETECTED RED SHIRT ' + np.array2string(shirt_bounding_box_R))

        # Publish the bounding box
        shirt_box_pub_R.publish(shirt_bounding_box_R)

    # Disabling for now until green shirt bought
    # if shirt_bounding_box_G.size is not 0:

        # rospy.loginfo('shirt_detection\tDETECTED GREEN SHIRT ' + np.array2string(shirt_bounding_box_G))

        # Publish bounding box
        # shirt_box_pub_G.publish(shirt_bounding_box_G)

#################################################################################
# The red colour masks
def red_mask(frame):

    # Threshold the image for red (HSV colour space)
    mask_lower = cv.inRange(frame, red_low_limit_lower, red_high_limit_lower)
    mask_upper = cv.inRange(frame, red_low_limit_upper, red_high_limit_upper)

    # Combine the masks and return to caller
    mask = cv.bitwise_or(mask_lower, mask_upper)
    return mask

# Green colour mask
def green_mask(frame):

    # Threshold the image with preset low and high HSV limits
    mask = cv.inRange(frame, green_low_limit, green_high_limit)
    # camera_functions.show_image("Mask", mask)
    return mask

#################################################################################
# Script segments the colours based on t-shirt colour: using RED and GREEN
def colour_segmentation(mask):

    # Initially empty bounding box
    shirt_box = []
    
    # Apply median filter to remove salt pepper noise
    # mask_gauss = cv.GaussianBlur(mask, (5,5), 0)
    mask_filtered = cv.medianBlur(mask, 9)

    # Morphological close to help detections with large kernel size - filters out noise
    kernel = np.ones((25, 25), np.uint8)
    frame_bw = cv.morphologyEx(mask_filtered, cv.MORPH_CLOSE, kernel)

    # Display the red shirt detection
    # camera_functions.show_image("Post Morph Op", frame_bw)

    # Find the valid contours in the bw image for the regions detected
    contours = cv.findContours(frame_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

     # Loop through the detected contours
    for c in contours:

        # Calculate the area of the region
        area = cv.contourArea(c)

        # Filter out any small colour detections
        if area > 2000:

            # Get the bounding box for the shirt
            x,y,w,h = cv.boundingRect(c)
            current_box = [x, y, w, h]

            # Append to the list of detected shirt boxes
            shirt_box.append(current_box)
            
    # Return the bounding box as a numpy array of float32 for publishing
    shirt_box = np.array(shirt_box, dtype=np.float32)
    return shirt_box    



#################################################################################
def detect_shirt_colour():

    # Initilaise the node and display message 
    rospy.init_node('shirt_detection', anonymous=True)
    rospy.loginfo('shirt_detection\tNODE INIT')

    # Set the ROS spin rate: 1Hz ~ 1 second
    rate = rospy.Rate(120)        ############ Can make this an argument in launch and streamline rates##############

    # Subscriber callbacks
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, camera_callback, queue_size=1)

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        # rospy.spin()
        rate.sleep()

if __name__ == '__main__':
    try:
        detect_shirt_colour()
    except rospy.ROSInterruptException:
        pass