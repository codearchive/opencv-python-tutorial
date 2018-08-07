import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("../../data/blox.jpg", 0)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))

s = ("Threshold: {}".format(fast.getThreshold()) + "\n" +
     "nonmaxSuppression: {}".format(fast.getNonmaxSuppression()) + "\n" +
     "neighborhood: {}".format(fast.getType()))
s_with = (s + "\n" +
          "Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

# Disable nonmaxSuppression
fast.setNonmaxSuppression(False)
kp = fast.detect(img, None)

s_without = (s + "\n" +
             "Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))

plt.subplot(211), plt.imshow(img2), plt.title("with nonmaxSuppression")
plt.text(300, 255, s_with, fontsize=10)
plt.subplot(212), plt.imshow(img3), plt.title("without nonmaxSuppression")
plt.text(300, 255, s_without, fontsize=10)
plt.show()
