import cv2 as cv

img = cv.imread("../../data/butterfly-0.jpg", 0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img, None)

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)

# Check upright flag, if it False, set it to True
if surf.getUpright() == False:
    surf.setUpright(True)

# Recompute the feature points and draw it
kp = surf.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)

# cv.imwrite("output-files/surf-result-6.png", img2)
cv.imshow("Result", img2)

# Find size of descriptor
print(surf.descriptorSize())  # 64

# That means flag, "extended" is False.
print(surf.getExtended())  # False

# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())  # 128
print(des.shape)  # (48, 128)

k = cv.waitKey(0)
while True:
    if k == 27:
        cv.destroyAllWindows()
        break
