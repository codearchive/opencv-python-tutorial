import numpy as np
import cv2 as cv
import glob


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # Draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

# Load previously saved data
mtx = np.load("../../data/calib-data-mtx.npy")
dist = np.load("../../data/calib-data-dist.npy")

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

for fname in glob.glob("../../data/left*.jpg"):
    print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret is True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        cv.imshow("Image - {}".format(str(fname[11:])), img)
        k = cv.waitKey(5000) & 0xFF
        if k == 27:  # Press "Esc" to exit
            break
        elif k == ord('n'):  # Press 'n' to open next image
            cv.destroyWindow("Image - {}".format(str(fname[11:])))
        elif k == ord('s'):  # Press 's' to save current image
            cv.destroyWindow("Image - {}".format(str(fname[11:])))
            cv.imwrite("output-files/" + "cube-" + str(fname[11:17]) +
                       ".png", img)
cv.destroyAllWindows()

