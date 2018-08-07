import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# image for Shi-Tomasi Corner Detector
img_scd = cv.imread('input-files/blox.jpg')
gray = cv.cvtColor(img_scd, cv.COLOR_BGR2GRAY)
# result of Harris Corner Detector
img_hcd = cv.imread('../harris-corner-detection/output-files/corners-blox.png')

corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(img_scd, (x, y), 3, 255, -1)

cv.imwrite('output-files/scd-blox.png', img_scd)

plt.figure("Corner Detector")
plt.suptitle("Comparison of the results of Harrison Corner Detector and "
             "Shi-Tomasi Corner Detector", fontsize=16)
plt.subplot(121), plt.imshow(img_hcd), plt.title("Harrison Corner Detector")
plt.subplot(122), plt.imshow(img_scd), plt.title("Shi-Tomasi Corner Detector")
plt.show()
