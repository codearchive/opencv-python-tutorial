import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("../../data/blox.jpg", 0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img, None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location, not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
cv.imwrite("output-files/orb-result-0.png", img2)
cv.imshow("Result", img2)
k = cv.waitKey(0)
while True:
    if k == 27:
        cv.destroyAllWindows()
        break
