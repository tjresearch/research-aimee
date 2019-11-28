# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

numSegments = 500

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))

# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments=numSegments, sigma=5)

segValsA = [0]*700  #L
segValsB = [0]*700  #A
segValsC = [0]*700  #B
segSize = [0]*700

centers = [(0,0)]*700

print(segments)
numSegs = 0

for r in range(image.shape[0]):
    for c in range(image.shape[1]):
        segValsA[segments.item(r, c)] += image.item(r, c, 0)
        segValsB[segments.item(r, c)] += image.item(r, c, 1)
        segValsC[segments.item(r, c)] += image.item(r, c, 2)
        segSize[segments.item(r, c)] += 1

        if centers[segments.item(r,c)] == (0,0):
            centers[segments.item(r,c)] = (r+50, c+50)
            numSegs += 1

print("Segments:", numSegs)

imgSeg = np.zeros(image.shape)
for seg in range(numSegs):
    if segSize[seg] != 0:
        segValsA[seg] = segValsA[seg] / segSize[seg]
        segValsB[seg] = segValsB[seg] / segSize[seg]
        segValsC[seg] = segValsC[seg] / segSize[seg]


for r in range(segments.shape[0]):
    for c in range(segments.shape[1]):
        imgSeg[r, c, 0] = segValsA[segments.item(r, c)]
        imgSeg[r, c, 1] = segValsB[segments.item(r, c)]
        imgSeg[r, c, 2] = segValsC[segments.item(r, c)]


avgImg = np.zeros((numSegs, 10))
for color in range(numSegs):
    avgImg[color][0] = (segValsA[color] + segValsB[color] + segValsC[color])/3
    avgImg[color][1] = centers[numSegs][0]**2 + centers[numSegs][1]**2

est = estimate_bandwidth(avgImg, quantile=0.25)

print(est)

clustering = MeanShift(bandwidth=est).fit(avgImg)

print(clustering.labels_)

for r in range(segments.shape[0]):
    for c in range(segments.shape[1]):
        segments[r, c] = clustering.labels_[segments[r,c]]

print(segments)

# show the output of SLIC
#fig = plt.figure("Superpixels -- %d segments" % (numSegments))
#ax = fig.add_subplot(1, 1, 1)
#ax.imshow(mark_boundaries(image, segments))
#plt.axis("off")

fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

# show the plots
plt.show()