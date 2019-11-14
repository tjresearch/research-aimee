import sys, os
sys.path.append("/usr/local/lib/python3.7/site-packages")

import cv2
import argparse, math



K = 300
KH = 30
KV = 10

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)))

imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
imgL, imgA, imgB = cv2.split(imgLAB)

S = int(math.sqrt(img.shape[0] * img.shape[1] / K))

centers = {}
Kwidth = img.shape[1]/KH
Klength = img.shape[0]/KV

for r in range(KV):
    for c in range(KH):
        X = int(c*Kwidth+Kwidth/2)
        Y = int(r*Klength+Klength/2)
        centers[(X, Y, imgL.item((Y, X)), imgA.item((Y, X)), imgB.item((Y,X)))] = []  # (X, Y, L, A, B)

def dist(x1, y1, l1, a1, b1, x2, y2, l2, a2, b2):
    m = 10 #experiment change
    dlab = math.sqrt(int((l1-l2)**2 + (a1-a2)**2 + (b1-b2)**2))
    dxy = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dlab + (m/S)*dxy

toChange = {}

for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        toChange[(c, r)] = []
        cents = []
        for centcoords in centers:
            if abs(centcoords[0] - c) < 2 * S and abs(centcoords[1] - r) < 2 * S:
                cents.append(centcoords)  # (X, Y)
        minDis = 800
        minCoords = None
        for centcoords in cents:
            tempDist = dist(c, r, imgL.item(r, c), imgA.item(r, c), imgB.item(r, c),
                            centcoords[0], centcoords[1], centcoords[2], centcoords[3], centcoords[4])
            if tempDist < minDis:
                minCoords = centcoords
                minDis = tempDist

        centers[minCoords].append((c, r))  # (X, Y)
newCenters = {}
sigma = 0
for centcoords in centers:
    totL = 0
    totA = 0
    totB = 0
    totX = 0
    totY = 0
    for pts in centers[centcoords]:
        totX += pts[0]
        totY += pts[1]
        totL += imgL.item(pts[1], pts[0])
        totA += imgA.item(pts[1], pts[0])
        totB += imgB.item(pts[1], pts[0])
    tempCent = (
    int(totX / len(centers[centcoords])), int(totY / len(centers[centcoords])), int(totL / len(centers[centcoords])),
    int(totA / len(centers[centcoords])), int(totB / len(centers[centcoords])))
    newCenters[tempCent] = []
    sigma += math.sqrt((centcoords[0] - tempCent[0]) ** 2 + (centcoords[1] - tempCent[1]) ** 2)
print("Sigma: ", sigma)
centers = newCenters

while sigma > 5:
    for c, r in toChange:
        cents = []
        for centcoords in centers:
            if abs(centcoords[0]-c) < 2*S and abs(centcoords[1]-r) < 2*S:
                cents.append(centcoords)    # (X, Y)
        minDis = 800
        minCoords = None
        for centcoords in cents:
            tempDist = dist(c, r, imgL.item(r, c), imgA.item(r, c), imgB.item(r, c),
                            centcoords[0], centcoords[1], centcoords[2], centcoords[3], centcoords[4])
            if tempDist < minDis:
                minCoords = centcoords
                minDis = tempDist

        centers[minCoords].append((c, r))  # (X, Y)
    newCenters = {}
    sigma = 0
    for centcoords in centers:
        totL = 0
        totA = 0
        totB = 0
        totX = 0
        totY = 0
        for pts in centers[centcoords]:
            totX += pts[0]
            totY += pts[1]
            totL += imgL.item(pts[1], pts[0])
            totA += imgA.item(pts[1], pts[0])
            totB += imgB.item(pts[1], pts[0])
        tempCent = (int(totX/len(centers[centcoords])), int(totY/len(centers[centcoords])), int(totL/len(centers[centcoords])),
                    int(totA/len(centers[centcoords])), int(totB/len(centers[centcoords])))
        newCenters[tempCent] = []
        sigma += math.sqrt((centcoords[0]-tempCent[0])**2 + (centcoords[1]-tempCent[1])**2)
    print("Sigma: ", sigma)
    centers = newCenters







cv2.imshow("test", img)
cv2.imshow("LABtest", imgLAB)
cv2.waitKey()