import sys, os
sys.path.append("/usr/local/lib/python3.7/site-packages")

import cv2 as cv
import argparse, math

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
#img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))


# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))

# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments=numSegments, sigma=5)

print(segments)

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

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        if imgL.item(r,c) < 100 and imgB.item(r,c) < 135 and imgA.item(r,c) < 135:
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

# show the plots
plt.show()


cv.waitKey(0)


