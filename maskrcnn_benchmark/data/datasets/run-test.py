import numpy as np
import cv2
import os
import argparse
import pandas as pd 
import torch
from mydataset import myDataset
from natsort import natsorted
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from predictor_car import COCODemo_car
from collections import defaultdict
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.datasets.evaluation import evaluate

class Visualize(object):
    """
    Initialize dataloader
    """
    def __init__(self, cfg, root):
        self.dataPath = os.path.join("data_maskrcnn", root)
        self.frames = os.path.join(self.dataPath, 'Frames')
        self.bboxes = os.path.join(self.dataPath, 'Bboxes')
        self.optFlow = os.path.join(self.dataPath, 'OptFlow')
        self.transforms = build_transforms(cfg, True)
        self.dataloader = myDataset(self.dataPath,
                                    transforms=self.transforms)
        self.cfg = cfg


    def build_predictor(self, confidence_threshold):
        pred = COCODemo_car(
            self.cfg,
            min_image_size=1080,
            confidence_threshold=confidence_threshold,)

        self.pred = pred

    def getPrecRec(self):
        self.pred.getPrecRec()

    def plot_pred(self, image, **kwargs):
        img, boxlist, last_img = self.pred.run_on_opencv_image(image, **kwargs)
        return img, boxlist, last_img
    
    def visualize_boxes(self, save_dir):
        """
        Visualize and save predictions
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "Frames"))
            os.makedirs(os.path.join(save_dir, "Bboxes"))

        for (image, optFlowVol, _, idx) in self.dataloader:

            file = self.dataloader.frames[idx]
            rgb_image = cv2.imread(file)

            if idx % 20 == 0:
                kwargs = {'optFlowVol': optFlowVol, 'rgb_image': rgb_image}
                img, boxes, last_img = self.plot_pred(image, **kwargs)
            else:
                boxes = self.pred.track_objects(last_img, rgb_image, boxes)
                last_img = rgb_image

            file = os.path.basename(file)
            save_path = os.path.join(save_dir, "Frames", file)
            cv2.imwrite(save_path, img)

            print("{} saved".format(save_path))

            labels = np.array([boxes.get_field('labels').numpy()]).T
            bbox = boxes.bbox.numpy()
            
            if len(labels) > 0:
                results = np.concatenate((labels,bbox), axis=1)

                fileBase = os.path.splitext(file)[0]
                csvFile = fileBase + ".csv"
                filePath = os.path.join(save_dir, "Bboxes", csvFile)
                pd.DataFrame(results).to_csv(filePath , header=None)

                print("{} saved".format(filePath))
            else:
                print("No cars found in image! Skipping...")


def main():
    parser = argparse.ArgumentParser(description="Visualize Predictions")
    parser.add_argument(
        "--config",
        default="predictor.yaml",
        type=str,
    )
    parser.add_argument("--input", default="", type=str)
    parser.add_argument("--confidence_threshold", default=0.9, type=float)
    args = parser.parse_args()

    # Setup
    config_file = args.config
    cfg.merge_from_file(config_file)
    root = args.input
    confidence_threshold = args.confidence_threshold

    print("Using cfg {}".format(config_file))
    print("Dataset: {}".format(root))
    print("Confidence threshold: {}".format(confidence_threshold))

    save_dir = os.path.join("data_maskrcnn", root, "predictions")

    Vis = Visualize(cfg, root)


    Vis.build_predictor(confidence_threshold)
    # Vis.getPrecRec()
    Vis.visualize_boxes(save_dir)



if __name__ == "__main__":
    main()
