#source: https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys, argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 30):
        #cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        kern = cv.getGaborKernel((ksize, ksize), 5.0, 0, 9.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

filters = build_filters()

res1 = process(image, filters)
cv.imshow("result", res1)
cv.waitKey(0)
cv.destroyAllWindows()