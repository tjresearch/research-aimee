import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('004.jpg')

mask = np.zeros(image.shape[:2], np.uint8)

backgroundModel = np.zeros((1,65), np.float64)
foregroundModel = np.zeros((1,65), np.float64)

rectangle = (20,20, image.shape[0]-5,image.shape[1]-5)
cv.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 2, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

image = image*mask2[:,:,np.newaxis]
plt.imshow(image)
plt.colorbar()
plt.show()