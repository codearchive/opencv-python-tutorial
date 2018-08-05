import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../../data/vtest.avi")

# Create BackgroundSubtractor object
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
outVideo = cv.VideoWriter("output-files/MOG-res.avi",
                          fourcc, 25.0, (width, height), False)

# Saved frame number
frame_number = 0

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)  # Get the foreground mask

    # Save the image and show it
    outVideo.write(fgmask)
    cv.imshow("frame", frame)
    cv.imshow("foreground", fgmask)
    k = cv.waitKey(60) & 0xff
    if k == 27:  # Press "esc" to exit
        break
    elif k == ord('s'):  # Press "s" to save current frame and result for it
        cv.imwrite("output-files/" + "MOG-src-" +
                   str(frame_number) + ".png", frame)
        cv.imwrite("output-files/" + "MOG-res-" +
                   str(frame_number) + ".png", fgmask)
        frame_number += 1
cap.release()
outVideo.release()
cv.destroyAllWindows()
