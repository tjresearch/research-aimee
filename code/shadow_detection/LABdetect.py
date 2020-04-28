import sys, os
sys.path.append("/usr/local/lib/python3.7/site-packages")

import cv2 as cv
import argparse, math
import numpy as np

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

numSegments = 400

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
#img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))


# load the image and convert it to a floating point data type
image = img_as_float(img)

# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments=numSegments, sigma=5)

print(segments)

min = 10000
max = 0

shadPix = set()
imgSegs = []
for x in range(500):
    imgSegs.append([])

for r in range(segments.shape[0]):
    for c in range(segments.shape[1]):
        imgSegs[segments.item(r,c)].append((r,c))

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

imgLAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
imgL, imgA, imgB = cv.split(imgLAB)

totalL = 0
totalA = 0
totalB = 0
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        totalL += imgL.item(r,c)
        totalA += imgA.item(r,c)
        totalB += imgB.item(r,c)

totalPix = img.shape[0] * img.shape[1]
print("Averages: ", totalL/totalPix, totalA/totalPix, totalB/totalPix)

if totalL/totalPix > 175:
    alpha = 1.7
    beta = -175

    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

    img = new_image

if totalL/totalPix < 75:
    alpha = 2.5
    beta = 0

    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)

    img = new_image

imgLAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
imgL, imgA, imgB = cv.split(imgLAB)

cv.imshow("contrast", img)

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

num = input("output name: ")

cv.imwrite('../../images/outputs/'+num, img)

cv.imshow("shadow?", img)

# show the plots
plt.show()


cv.waitKey(0)


