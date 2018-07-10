import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_original = cv.imread('input-files/messi.jpg')
mask = np.zeros(img_original.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
rect = (50, 50, 450, 290)
cv.grabCut(img_original, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)

# newmask is the mask image I manually labelled
newmask = cv.imread('input-files/newmask.jpg', 0)

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img_original, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img_original*mask[:, :, np.newaxis]

title = 'Foreground Extraction'
plt.figure(title), plt.suptitle(title, fontsize=16)
plt.subplot(221), plt.imshow(img_original), plt.title('Image 1 - Original Image')
plt.subplot(222), plt.imshow(mask), plt.title('Image 2 - Mask')
plt.subplot(223), plt.imshow(newmask), plt.title('Image 3 - Newmask')
plt.subplot(224), plt.imshow(img), plt.title('Image 4 - Result')
plt.show()

cv.imwrite('output-files/messi-result-2.jpg', img)
