# This script takes in webcam images and identifies the colour of 
# people's shirts

# Packages
import cv2
import numpy as np

# Read the original image and convert to HSV format
currentImage = cv2.imread("stopSign.jpg")
currentImageHSV = cv2.cvtColor(currentImage, cv2.COLOR_BGR2HSV)

# Get the dimensions of the image
rowSize = currentImage.shape[0]
columnSize = currentImage.shape[1]

# Image threshold
# Create two separate masks for each region of red on the HSV spectrum 
mask1 = cv2.inRange(currentImageHSV, (0,220,0), (5,255,255))
mask2 = cv2.inRange(currentImageHSV, (160,220,20), (180,255,255))

# Combine both masks to make one
mask = cv2.bitwise_or(mask1, mask2)
thresholdImage = cv2.bitwise_and(currentImage, currentImage, mask=mask)

# Generate bounding boxes around red elements (red shirts)

# Calculate the center coordinate of the bounding boxes (target coordinates)

# Determine if there is a suitable target
# Loop through image pixels
# for row in range(rowSize - 1):
#     for column in range(columnSize - 1):

#         print("Current Pixel = ", currentImage[row, column])

        # Check if the current pixel is the target colour

# Calculate the center coordinate of the theshold area


# Display the threshold image
#cv2.imshow("Red threshold", thresholdImage)
#cv2.waitKey()

#############################################################################
# Minimum percentage of pixels of same hue to consider dominant colour
MIN_PIXEL_CNT_PCT = (1.0/20.0)

#image = cv2.imread('colourblobs.png')
# if image is None:
#     print("Failed to load iamge.")
#     exit(-1)

image_hsv = cv2.cvtColor(currentImage, cv2.COLOR_BGR2HSV)
# We're only interested in the hue
h,_,_ = cv2.split(image_hsv)
# Let's count the number of occurrences of each hue
bins = np.bincount(h.flatten())
# And then find the dominant hues
peaks = np.where(bins > (h.size * MIN_PIXEL_CNT_PCT))[0]

# Now let's find the shape matching each dominant hue
for i, peak in enumerate(peaks):
    # First we create a mask selecting all the pixels of this hue
    mask = cv2.inRange(h, peak, peak)
    # And use it to extract the corresponding part of the original colour image
    blob = cv2.bitwise_and(currentImage, currentImage, mask=mask)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for j, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, j, 255, -1)

        #print "Found hue %d in region %s." % (peak, bbox)
        # Extract and save the area of the contour
        region = blob.copy()[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_mask = contour_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_masked = cv2.bitwise_and(region, region, mask=region_mask)
        file_name_section = "colourblobs-%d-hue_%03d-region_%d-section.png" % (i, peak, j)
        cv2.imwrite(file_name_section, region_masked)
        #print " * wrote '%s'" % file_name_section

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
        file_name_bbox = "colourblobs-%d-hue_%03d-region_%d-bbox.png" % (i, peak, j)
        cv2.imwrite(file_name_bbox, result)
        #print " * wrote '%s'" % file_name_bbox

