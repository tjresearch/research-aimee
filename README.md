# Locating Light Sources from Images Using Shadow Detection Techniques

## Overview
Shadows exist naturally in any illuminated image, and can convey a lot of information. I combine various published shadow detection techniques to improve and optimize shadow detection in images, then associate these shadows with their respective objects in images. Following a projection of the shadow to its object will create a ray along which the light source will lie. In images with multiple objects, the intersection of the rays will be identified to pinpoint the location of the light source.

## Required Libraries
OpenCV 4.0

Skimage

Matplotlib

Numpy

CVLib

## Installation Instructions
Download LABLog.py, the 'images' folder for test images, and gaborFilter.py from Object_detection into the same directory. This is the current most up-to-date set of files. 

## Run Instructions
Run LABLog.py first with the command: 

    python3.7 LABLog.py --image ./images/[image_name]

This will output the orginial image in black and white with shadow pixels marked in black. It will also output the original image segemented with k-means.

Run detectCVLib.py with the command:

    python3.7 gaborFilter.py --image ./images/[image_name]

This will output an image specified in the code with the texture and object boundaries labeled in white.

## Sample Output

![](/images/004.jpg)

![](/images/Shadow004.jpg)

![](/images/Kmeans004.jpg)
