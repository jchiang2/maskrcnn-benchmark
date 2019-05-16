import os
import cv2
import argparse

import numpy as np

def main():
    """
    Main function to run preprocessing on input video.
    Computes optical flow for each subsequent frame.
    Saves optical flow (numpy array) and frame (.jpg) every 20 frames
    """

    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument("--video", help="path to input video", type=str)
    parser.add_argument("--out", help="path to output", type=str)
    parser.add_argument("--sampling_rate", type=int, default=20)
    args = parser.parse_args()

    samplingRate = args.sampling_rate

    outDir = args.out
    imgsDir = os.path.join(outDir, "Frames")
    bboxesDir = os.path.join(outDir, "Bboxes")
    optFlowDir = os.path.join(outDir, "OptFlow")
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    if not os.path.exists(imgsDir):
        os.makedirs(imgsDir)
    if not os.path.exists(bboxesDir):
        os.makedirs(bboxesDir)
    if not os.path.exists(optFlowDir):
        os.makedirs(optFlowDir)

    videos = os.listdir("/home/jchiang2/github/data_maskrcnn/dormont_data/Videos/2013")
    print(videos)
    # Begin frame capturing
    for i, video in enumerate(videos):
        if ".mp4" not in video:
            continue
        print(video)
        video = os.path.join("/home/jchiang2/github/data_maskrcnn/dormont_data/Videos/2013", video)
        cam = cv2.VideoCapture(video)
        frame_cnt = 0

        ret, prev = cam.read()
        if not ret:
            break
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        while True:
            ret, img = cam.read()
            if not ret:
                break
            frameID = "{}_Frame_{}".format(str(14).zfill(2), str(frame_cnt).zfill(8))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if frame_cnt % samplingRate == 0:
                imgFile = os.path.join(imgsDir, frameID + ".jpg")
                cv2.imwrite(imgFile, img)

                print("Saving frame {}".format(frameID))

            # Calculate and save optical flow
            if frame_cnt % samplingRate <= 2:
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0) 
                optFlowVol = np.linalg.norm(flow, axis=2)
                optFlowVol[optFlowVol > 50] = 0
                optFlowVol = cv2.GaussianBlur(optFlowVol,(51,51),10)
                cv2.normalize(optFlowVol, optFlowVol, 0, 255, cv2.NORM_MINMAX)
                # print("{} mean:".format(frameID), np.mean(optFlowVol))
                flowFile = os.path.join(optFlowDir, frameID + ".jpg")
                cv2.imwrite(flowFile, optFlowVol)

                print("Saving optical flow {}".format(frameID))

                if frame_cnt % samplingRate == 2:
                    print("-----------------------------")
            
            

            prevgray = gray

            # print("Saving frame and optical flow {}".format(frameID))

            frame_cnt += 1

if __name__ == "__main__":
    main()