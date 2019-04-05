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

class Visualize(object):
    """
    Initialize dataloader
    """
    def __init__(self, cfg, mode, opt_flow_on=False):
        self.dataPath = "Movie_Frames_{}".format(mode)
        self.imgPath = os.path.join(self.dataPath, 'Frames')
        self.csvPath = os.path.join(self.dataPath, 'Bboxes_mydataset')
        self.npyPath = os.path.join(self.dataPath, 'opt_flow_np')
        self.transforms = build_transforms(cfg, True)
        self.dataloader = myDataset(self.csvPath, 
                                    self.imgPath, 
                                    transforms=self.transforms)
        self.mode = mode
        self.cfg = cfg


    def build_predictor(self, confidence_threshold, car):
        if car:
            pred = COCODemo_car(
                self.cfg,
                min_image_size=1080,
                confidence_threshold=confidence_threshold,)
        else:
            pred = COCODemo(
                self.cfg,
                min_image_size=1080,
                confidence_threshold=confidence_threshold,)
        self.pred = pred

    def plot_pred(self, image, optFlowVol, rgb_image):
        img, boxlist, count = self.pred.run_on_opencv_image(image, optFlowVol, rgb_image)
        return img, boxlist, count

    def plot_gt(self, image, target):
        self.pred.plot_gt(image, target)
        return image

    """
    Save bboxes to csv files
    """
    def get_bboxes(self):
        for (image, optFlowVol, targets, idx) in self.dataloader:
            file = self.dataloader.imgFiles[idx]

            img, boxes, count = self.plot_pred(input)

            labels = np.array([boxes.get_field('labels').numpy()]).T
            where = np.argwhere(labels!=3)[:,0]
            bbox = boxes.bbox.numpy()
            results = np.concatenate((labels,bbox), axis=1)
            if where.size>0:
                results = np.delete(results, (where), axis = 0)
            if not os.path.exists(self.csvPath):
                    os.makedirs(self.csvPath)
            if results.shape[0] > 0:
                filePath = os.path.join(self.csvPath, "{}.csv".format(file[:-4]))
                pd.DataFrame(results).to_csv(filePath , header=None)
                print("{}.csv saved".format(file[-4]))
    """
    Visualize and save predictions
    """
    def visualize_boxes(self, save_dir, dataloader=True, vis_gt=False):
        if dataloader:
            for (image, optFlowVol, targets, idx) in self.dataloader:

            	file = self.dataloader.imgFiles[idx]
            	filePath = os.path.join(self.imgPath, file)
            	rgb_image = cv2.imread(filePath)

            	img, boxes, count = self.plot_pred(image, optFlowVol, rgb_image)
            	if vis_gt:
            		img = self.plot_gt(img, targets)

            	if not os.path.exists(save_dir):
            		os.makedirs(save_dir)

            	save_path = os.path.join(save_dir, file)
            	cv2.imwrite(save_path, img)

            	print("{} saved".format(save_path))

        else:
            images = natsorted(os.listdir(self.imgPath))
            for idx, file in enumerate(images):
                img = os.path.join(self.imgPath, file)
                img = cv2.imread(img)
                img, boxes, count = self.plot_pred(img)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, img)
                print("{} saved".format(save_path))

    def remove_empty(self):
        csv_files = natsorted(os.listdir(self.csvPath))
        img_files = natsorted(os.listdir(self.imgPath))
        npy_files = natsorted(os.listdir(self.npyPath))
        # Remove empty csv files
        for file in csv_files:
            file = os.path.join(self.csvPath, file)
            bboxes = np.genfromtxt(file, delimiter = ',')
            if len(bboxes)==0:
                os.remove(file)
                print("Deleted empty file {}".format(file))
        # Build dict of csv filenames for checking later
        csv_dict = defaultdict(int)
        for file in csv_files:
            csv_dict[file[:-4]] = 1
        # Remove image files if name not in csv dict
        for file in img_files:
            if csv_dict[file[:-4]] == 0:
                file = os.path.join(self.imgPath,file)
                os.remove(file)
                print("Deleted empty image {}".format(file))
        # Remove optFlow files if name not in csv dict
        for file in npy_files:
            if csv_dict[file[:-4]] == 0:
                file = os.path.join(self.npyPath,file)
                os.remove(file)
                print("Deleted empty optFlow {}".format(file))


def main():
    parser = argparse.ArgumentParser(description="Visualize Predictions")
    parser.add_argument(
        "--config",
        default="e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        type=str,
    )
    parser.add_argument("--mode", default="", type=str)
    parser.add_argument("--use-car", action='store_true')
    parser.add_argument("--save", default="predictions", type=str)
    args = parser.parse_args()

    # Setup
    config_file = args.config
    cfg.merge_from_file(config_file)
    mode = args.mode
    use_car = args.use_car
    confidence_threshold = 0.7

    print("Using cfg {}".format(config_file))
    print("Mode:{}".format(mode))
    print("Use_car: {}".format(use_car))

    Vis = Visualize(cfg, mode)

    save_dir = args.save
    save_dir = os.path.join(Vis.dataPath, save_dir)
    print("Predictions dir: {}".format(save_dir))

    Vis.build_predictor(confidence_threshold, use_car)
    Vis.visualize_boxes(save_dir, dataloader=True)
    # Vis.remove_empty()


if __name__ == "__main__":
    main()
