import cv2 as cv
import numpy as np

cap = cv.VideoCapture("../../data/vtest.avi")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
outVideo = cv.VideoWriter("output-files/dense-optical-flow-res.avi", \
                          fourcc, 25.0, (width, height))

# saved frame number
frame_number = 0

while True:
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, \
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # save the image and show it
    outVideo.write(bgr)
    cv.imshow("frame", frame2)
    cv.imshow("flow", bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:  # press "esc" to exit
        break
    elif k == ord('s'):  # press "s" to save current frame and result for it
        cv.imwrite("output-files/" + "dense-optical-flow-src-" + \
                   str(frame_number) + ".png", frame2)
        cv.imwrite("output-files/" + "dense-optical-flow-res-" + \
                   str(frame_number) + ".png", bgr)
        frame_number += 1
    prvs = next
cap.release()
cv.destroyAllWindows()
