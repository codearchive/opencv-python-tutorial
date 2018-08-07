import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

filename = 'input-files/chessboard.png'
img = cv.imread(filename)

img = cv.resize(img, dsize=(500, 500), interpolation=cv.INTER_LINEAR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0, 0, 255]

cv.namedWindow('dst', flags=cv.WINDOW_NORMAL)

cv.imshow('dst', img)
cv.imwrite('output-files/corners-blox.png', img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
