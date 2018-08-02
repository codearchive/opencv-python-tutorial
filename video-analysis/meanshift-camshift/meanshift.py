import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../../data/slow.mp4")
# take first frame of the video
ret, frame = cap.read()

# setup initial location of window
r, h, c, w = 190, 20, 330, 60  # simply hardcoded the values
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), \
                  np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
outVideo = cv.VideoWriter("output-files/meanshift-res.avi", \
                          fourcc, 25.0, (width, height))

# saved frame number
frame_number = 0

while True:
    ret, frame = cap.read()
    if ret is True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # draw it on image
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

        # save the image and show it
        outVideo.write(img2)
        cv.imshow("img2", img2)
        k = cv.waitKey(60) & 0xFF
        if k == 27:  # press "esc" to exit
            break
        elif k == 0x73:  # press "s" to save the current frame
            cv.imwrite("output-files/" + "meanshift-res-" + \
                       str(frame_number) + ".png", img2)
            frame_number += 1
    else:
        break
cv.destroyAllWindows()
cap.release()
