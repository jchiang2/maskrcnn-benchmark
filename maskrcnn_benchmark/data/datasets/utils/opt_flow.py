import numpy as np
import cv2
import os

paths = sorted(os.listdir("dataApr2/Frames"))
filepath = os.path.join("dataApr2/Frames", paths[0])
frame = cv2.imread(filepath)
prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for path in paths:
    filepath = os.path.join('dataApr2/Frames', path)
    frame = cv2.imread(filepath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    np.save('dataApr2/OptFlow/{}'.format(path[:-4]), flow)

    print(path)



