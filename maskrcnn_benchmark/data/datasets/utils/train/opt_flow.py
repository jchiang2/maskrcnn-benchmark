import numpy as np
import cv2
import os
from natsort import natsorted

paths = natsorted(os.listdir('Frames'))
prev = cv2.imread('Frames/' + paths[0])
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
for path in paths:
    filepath = os.path.join('Frames', path)
    frame = cv2.imread(filepath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    np.save('opt_flow_np/{}'.format(path), flow)

    print(path)

# img = np.load('opt_flow_np/1_Frame0.jpg.npy')
# image = img[:,:,0]
# image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# image = Image.fromarray(image)
# cv2.imshow('Frame',img[:,:,0])
# cv2.waitKey(0)
