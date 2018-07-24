import cv2 as cv

img = cv.imread("../../data/blox.jpg", 0)
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img, None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

img2 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255), flags=0)
cv.imwrite("output-files/brief-result.png", img2)
cv.imshow("Result", img2)

print(brief.descriptorSize())
print(des.shape)

k = cv.waitKey(0)
while True:
    if k == 27:
        cv.destroyAllWindows()
        break
