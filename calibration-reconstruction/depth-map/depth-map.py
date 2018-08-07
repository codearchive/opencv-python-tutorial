import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread("../../data/tsukuba_l.png", 0)
imgR = cv.imread("../../data/tsukuba_r.png", 0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)

# cv.imwrite("output-files/depth-map-res.png", disparity)
plt.subplot(121), plt.imshow(imgL), plt.title("Image 1 - Original image")
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(disparity), plt.title("Image 2 - Disparity map")
plt.xticks([]), plt.yticks([])
plt.subplots_adjust(left=0.01, right=0.99, wspace=0.02)
plt.show()
