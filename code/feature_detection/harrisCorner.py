from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

def cornerHarris_demo(val):
    thresh = val
    # Detector parameters
    blockSize = 5
    apertureSize = 7
    k = 0.5
    # Detecting corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)
# Load source image and convert it to gray


parser = argparse.ArgumentParser(description='Code for Harris corner detector tutorial.')
parser.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(parser.parse_args())
src = cv.imread(args["image"])

src = cv.medianBlur(src, 3)

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)


alpha = 1.5
beta = 0

new_image = np.zeros(src.shape, src.dtype)
for y in range(src.shape[0]):
    for x in range(src.shape[1]):
        for c in range(src.shape[2]):
            new_image[y,x,c] = np.clip(alpha*src[y,x,c] + beta, 0, 255)

new_image = cv.medianBlur(new_image, 3)


gray = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(gray,3,7,.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
new_image[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',new_image)
cv.waitKey()


'''src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Create a window and a trackbar
cv.namedWindow(source_window)
thresh = 150 # initial threshold
cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, src)
cornerHarris_demo(thresh)
cv.waitKey()
'''

