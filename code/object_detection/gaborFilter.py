#source: https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys, argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])

def build_filters(theta):
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 30):
        #cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 9.3, 1.0, 50, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

filters = build_filters(0)

res1 = process(image, filters)
#res2 = res1

cv.imshow("res1", res1)

edges = cv.Canny(image, 260, 300)
cv.imshow("canny", edges)

print(res1.shape)
for r in range(res1.shape[0]):
    for c in range(res1.shape[1]):
        tot = int(res1[r][c][0]) + int(res1[r][c][1]) + int(res1[r][c][2])
        '''if res1[r][c][0] == 255 and res1[r][c][1] == 255 and res1[r][c][2] == 255:
            res1[r][c] = (0,0,0)
        else:
            res1[r][c] = (255,255,255)'''
        if tot > 255*3-100:
            res1[r][c] = (0,0,0)
        else:
            res1[r][c] = (255, 255, 255)

cv.imshow("result", res1)
cv.waitKey(0)
cv.destroyAllWindows()