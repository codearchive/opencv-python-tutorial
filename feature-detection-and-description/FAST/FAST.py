import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("../../data/blox.jpg", 0)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()), '\n'
      "nonmaxSuppression: {}".format(fast.getNonmaxSuppression()), '\n'
      "neighborhood: {}".format(fast.getType()), '\n'
      "Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

cv.imwrite("output-files/fast_true.png", img2)
cv.imshow("With Suppression", img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(False)
kp = fast.detect(img, None)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imwrite("output-files/fast_false.png", img3)
cv.imshow("Without Suppression", img3)
k = cv.waitKey(0)
while True:
    if k == 27:
        cv.destroyAllWindows()
        break
