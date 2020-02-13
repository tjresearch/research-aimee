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
import numpy as np

def build_filters(angle):
    filters = []
    ksize = 31
    '''
    for theta in np.arange(0, np.pi, np.pi / 60):
        #cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 9.3, 1.0, 65, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters
        
    '''
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

def logistic(L):
    return 255/(1+math.exp(-0.05*(L-127.5)))


numSegments = 600

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
image = cv.imread(args["image"])
#img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

cv.imshow("original image", image)


# load the image and convert it to a floating point data type
#image = img_as_float(io.imread(args["image"]))

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
for x in range(700):
    imgSegs.append([])

for r in range(segments.shape[0]):
    for c in range(segments.shape[1]):
        imgSegs[segments.item(r,c)].append((r,c))

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

maxShade = 0
minShade = 1000
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        shadValL = logistic(imgL.item(r, c))
        if shadValL > maxShade:
            maxShade = shadValL
        if shadValL < minShade:
            minShade = shadValL
        if imgL.item(r,c) < 105 and imgB.item(r,c) < 160 and imgA.item(r,c) < 150:
            if shadValL > .5 and shadValL < 30:
                shadPix.add((r,c))
                img[r][c] = (0,0,0)
            else:
                img[r][c] = (255,255,255)
        else:
            img[r][c] = (255,255,255)

print(minShade, maxShade)

black = set()
white = set()
counter = 0
while not len(imgSegs[counter]) == 0:
    pixcounter = 0
    for thing in range(len(imgSegs[counter])):
        if imgSegs[counter][thing] in shadPix:
            pixcounter += 1
    if pixcounter > .55 * len(imgSegs[counter]):
        black.add(counter)
    elif pixcounter < .25 * len(imgSegs[counter]):
        white.add(counter)
    counter += 1

for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        if segments.item(r,c) in black:
            img[r][c] = (0,0,0)
        elif segments.item(r,c) in white:
            img[r][c] = (255,255,255)

#img = cv.resize(img, (600, 600))

cv.imshow("shadow?", img)


filters = build_filters(90)
filters2 = build_filters(0)

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
#res2 = res1

cv.imshow("res1", res1)
cv.imshow("res2", res2)

for r in range(res1.shape[0]):
    for c in range(res1.shape[1]):
        tot = int(res1[r][c][0]) + int(res1[r][c][1]) + int(res1[r][c][2])
        tot2 = int(res2[r][c][0]) + int(res2[r][c][1]) + int(res2[r][c][2])
        if tot > 255*3-100 or tot2 > 255*3-100:
            res1[r][c] = (0,0,0)
        else:
            res1[r][c] = (255, 255, 255)

cv.imshow("result", img)

# show the plots
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()



