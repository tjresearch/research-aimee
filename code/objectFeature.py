from __future__ import print_function
import sys, os
sys.path.append("/usr/local/lib/python3.7/site-packages")


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse, math

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

numSegments = 400


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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
src = image
#src = cv.medianBlur(image, 3)

img = cv.imread(args["image"])



if image.shape[1] > 600 or image.shape[0] > 600:
    image = cv.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
    img = cv.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    src = image


# load the image and convert it to a floating point data type
#shadowImage = img_as_float(io.imread(args["image"]))
shadowImage = img_as_float(image)
segments = slic(shadowImage, n_segments=numSegments, sigma=5)

def build_filters(angle):
    filters = []
    ksize = 25
    for theta in np.arange(0, np.pi, np.pi / 30):
        #cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        kern = cv.getGaborKernel((ksize, ksize), 4.0, angle, 9.3, 1.0, 50, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


#segmentation
imgLAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
imgL, imgA, imgB = cv.split(imgLAB)

min = 10000
max = 0

shadPix = set()
imgSegs = []
for x in range(500):
    imgSegs.append([])

for r in range(segments.shape[0]):
    for c in range(segments.shape[1]):
        imgSegs[segments.item(r,c)].append((r,c))


# Shadow detection from LAB color model
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        if imgL.item(r,c) < 110 and imgB.item(r,c) < 135 and imgA.item(r,c) < 135:
            #img[r][c] = (0,0,0)
            shadPix.add((r,c))
        #else:
            #img[r][c] = (255, 255, 255)


black = set()
counter = 0
while not len(imgSegs[counter]) == 0:
    pixcounter = 0
    for thing in range(len(imgSegs[counter])):
        if imgSegs[counter][thing] in shadPix:
            pixcounter += 1
    if pixcounter > 3 * len(imgSegs[counter])/4:
        black.add(counter)
    counter += 1

for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        if segments.item(r,c) in black:
            img[r][c] = (0,0,0)
        else:
            img[r][c] = (255,255,255)

cv.imshow("shadow?", img)

#####




filters = build_filters(0)
filters2 = build_filters(90)


alpha = .75
beta = 30

new_image = np.zeros(image.shape, image.dtype)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

image = cv.medianBlur(new_image, 9)

res1 = process(image, filters)
res2 = process(image, filters2)

cv.imshow("res1", res1)

print(res1.shape)
for r in range(res1.shape[0]):
    for c in range(res1.shape[1]):
        tot = int(res1[r][c][0]) + int(res1[r][c][1]) + int(res1[r][c][2])
        tot2 = int(res2[r][c][0]) + int(res2[r][c][1]) + int(res2[r][c][2])
        if tot > 255*3-100 or tot2 > 255*3-100:
            res1[r][c] = (0,0,0)
        else:
            res1[r][c] = (255, 255, 255)

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
#new_image[dst>0.01*dst.max()]=[0,0,255]
#print(dst.shape)

for r in range(new_image.shape[0]):
    for c in range(new_image.shape[1]):
        if dst[r][c] > 0.01 * dst.max():
            for smallr in range(r-5, r+5):
                for smallc in range(c-5, c+5):
                    if smallr > 0 and smallr < new_image.shape[0] and smallc > 0 and smallc < new_image.shape[1]:
                        if abs(smallr - r) <= 2 and abs(smallc - c) <= 2 and res1[smallr][smallc].all() == 0:
                            new_image[r][c] = [0,0,255]
                            break
                        if img[r][c].all() == 0:
                            new_image[r][c] = [255,0,0]
                            break

cv.imshow('dst',new_image)
cv.imshow("result", res1)
cv.waitKey(0)
cv.destroyAllWindows()