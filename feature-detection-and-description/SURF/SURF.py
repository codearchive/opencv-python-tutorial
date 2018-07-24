import cv2 as cv

img = cv.imread("../../data/butterfly-0.jpg", 0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img, None)
print(len(kp))

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(5000)

# Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img, None)
print(len(kp))

img2 = cv.drawKeypoints(img, kp, None, (0, 0, 255), 4)
# cv.imwrite("output-files/surf-result-0.png", img2)
cv.imshow("Result", img2)
k = cv.waitKey(0)
while(1):
    if k == 27:
        cv.destroyAllWindows()
        break
