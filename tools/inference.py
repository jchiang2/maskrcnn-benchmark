import numpy as np
import cv2
import os
import argparse
import pandas as pd 
import torch
from tqdm import tqdm
from maskrcnn_benchmark.data.datasets.mydataset import myDataset
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.predictor_car import COCODemo_car
from collections import defaultdict
from maskrcnn_benchmark.data.transforms import build_transforms

class Visualize(object):

    def __init__(self, cfg, root):

        self.dataPath = root
        self.transforms = build_transforms(cfg, True)
        self.dataloader = myDataset(self.dataPath,
                                    transforms=self.transforms)
        self.root = root
        self.cfg = cfg


    def build_predictor(self, confidence_threshold):
        """
        Build predictor class to visualize predictions
        """
        self.pred = COCODemo_car(
                self.cfg,
                min_image_size=1080,
                confidence_threshold=confidence_threshold,)

    def plot_pred(self, image, **kwargs):
        """
        Run through model and plot boxes on image
        """
        img, boxlist = self.pred.run_on_opencv_image(image, **kwargs)
        return img, boxlist
    
    def visualize_boxes(self, save_dir):
        """
        Visualize and save predictions
        """
        predicted_imgs = os.path.join(save_dir, "Frames_test")
        predicted_boxes = os.path.join(save_dir, "Bboxes_test")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(predicted_imgs):
            os.makedirs(predicted_imgs)
        if not os.path.exists(predicted_boxes):
            os.makedirs(predicted_boxes)

        for (image, optFlowVol, _, idx) in self.dataloader:

            imgFile = self.dataloader.frames[idx]
            rgb_image = cv2.imread(imgFile)

            kwargs = {'optFlowVol': optFlowVol, 'rgb_image': rgb_image}
            # Predict bounding boxes and plot on image
            img, boxes = self.plot_pred(image, **kwargs)

            fileID = os.path.basename(imgFile)
            save_path = os.path.join(predicted_imgs, fileID)
            cv2.imwrite(save_path, img)

            # Save bounding box data as csv files
            labels = np.array([boxes.get_field('labels').numpy()]).T
            bbox = boxes.bbox.numpy()
            if len(labels) > 0:
                results = np.concatenate((labels,bbox), axis=1)

                csvFile = os.path.splitext(fileID)[0] + ".csv"
                filePath = os.path.join(predicted_boxes, csvFile)
                pd.DataFrame(results).to_csv(filePath , header=None)

            print("Visualizing bounding boxes for {}...".format(fileID))

def main():
    parser = argparse.ArgumentParser(description="Visualize Predictions")
    parser.add_argument(
        "--config",
        default="configs/pretrained_optFlow.yaml",
        help="path to your config file",
        type=str,
    )
    parser.add_argument("--root", default="", 
                                  help="path to your dataset", 
                                  type=str
    )
    parser.add_argument("--confidence_threshold", default=0.7, type=float)
    args = parser.parse_args()

    # Setup
    config_file = args.config
    cfg.merge_from_file(config_file)
    root = args.root
    confidence_threshold = args.confidence_threshold

    print("Using cfg {}".format(config_file))
    print("Confidence threshold {}".format(confidence_threshold))

    Vis = Visualize(cfg, root)

    saveDir = os.path.join(root, "predictions")
    print("Saving to {}".format(saveDir))

    Vis.build_predictor(confidence_threshold)
    Vis.visualize_boxes(saveDir)


if __name__ == "__main__":
    main()
