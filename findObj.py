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


numSegments = 600

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
#img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))


# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))




cv.imshow("shadow?", img)

# show the plots
plt.show()


cv.waitKey(0)


