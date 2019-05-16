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
    def __init__(self, cfg, root, opt_flow_on=False):
        self.dataPath = "data_maskrcnn/{}".format(root)
        self.frames = os.path.join(self.dataPath, 'Frames')
        self.bboxes = os.path.join(self.dataPath, 'Bboxes')
        self.optFlow = os.path.join(self.dataPath, 'OptFlow')
        self.transforms = build_transforms(cfg, True)
        self.dataloader = myDataset(self.dataPath,
                                    transforms=self.transforms)
        self.root = root
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

    def plot_pred(self, image, **kwargs):
        img, boxlist = self.pred.run_on_opencv_image(image, **kwargs)
        return img, boxlist

    def plot_gt(self, image, target):
        self.pred.plot_gt(image, target)
        return image

    def get_bboxes(self):
        """
        Save bboxes to csv files
        """
        
        images = sorted(os.listdir(self.frames))

        for idx, file in enumerate(images):

            img = os.path.join(self.frames, file)
            img = cv2.imread(img)
            img, boxes = self.plot_pred(img)

            labels = np.array([boxes.get_field('labels').numpy()]).T
            where = np.argwhere(labels!=3)[:,0]
            bbox = boxes.bbox.numpy()
            results = np.concatenate((labels,bbox), axis=1)
            if where.size>0:
                results = np.delete(results, (where), axis = 0)
            if not os.path.exists(self.bboxes):
                    os.makedirs(self.bboxes)
            if results.shape[0] > 0:
                filePath = os.path.join(self.bboxes, file[:-4]) + ".csv"
                pd.DataFrame(results).to_csv(filePath , header=None)

                print("{} saved".format(filePath))
    
    def visualize_boxes(self, save_dir, dataloader=True, vis_gt=False):
        """
        Visualize and save predictions
        """

        if dataloader:
            for (image, optFlowVol, targets, idx) in self.dataloader:

                file = self.dataloader.frames[idx]
                rgb_image = cv2.imread(file)

                kwargs = {'optFlowVol': optFlowVol, 'rgb_image': rgb_image}
                img, boxes = self.plot_pred(image, **kwargs)

                if vis_gt:
                    img = self.plot_gt(img, targets)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                file = os.path.basename(file)
                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, img)

                print("{} saved".format(save_path))


                labels = np.array([boxes.get_field('labels').numpy()]).T
                bbox = boxes.bbox.numpy()
                
                if len(labels) > 0:
                    results = np.concatenate((labels,bbox), axis=1)

                    if not os.path.exists(self.dataPath + "/predictions/Bboxes"):
                            os.makedirs(self.dataPath + "/predictions/Bboxes")
                    if results.shape[0] > 0:
                        filePath = os.path.join(self.dataPath + "/predictions/Bboxes", file[:-4]) + ".csv"
                        pd.DataFrame(results).to_csv(filePath , header=None)

                        print("{} saved".format(filePath))



        else:
            images = sorted(os.listdir(self.frames))

            for idx, file in enumerate(images):
                img = os.path.join(self.frames, file)
                img = cv2.imread(img)
                img, boxes = self.plot_pred(img)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, img)
                print("{} saved".format(save_path))

    def remove_empty(self):
        bboxes = sorted(os.listdir(self.bboxes))
        frames = sorted(os.listdir(self.frames))
        optFlows = sorted(os.listdir(self.optFlow))

        # Remove empty csv files
        for file in bboxes:
            file = os.path.join(self.bboxes, file)
            boxes = np.genfromtxt(file, delimiter = ',')
            if len(boxes)==0:
                os.remove(file)
                print("Deleted empty file {}".format(file))
        # Build dict of csv filenames for checking later
        bbox_dict = defaultdict(int)
        for file in bboxes:
            bbox_dict[file[:-4]] = 1
        # Remove image files if name not in bbox_dict
        for file in frames:
            if bbox_dict[file[:-4]] == 0:
                file = os.path.join(self.frames, file)
                os.remove(file)
                print("Deleted empty image {}".format(file))
        # Remove optFlow files if name not in bbox_dict
        for file in optFlows:
            if bbox_dict[file[:-4]] == 0:
                file = os.path.join(self.optFlow, file)
                os.remove(file)
                print("Deleted empty optFlow {}".format(file))


def main():
    parser = argparse.ArgumentParser(description="Visualize Predictions")
    parser.add_argument(
        "--config",
        default="e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        type=str,
    )
    parser.add_argument("--root", default="", type=str)
    parser.add_argument("--use-car", action='store_true')
    parser.add_argument("--save", default="predictions", type=str)
    args = parser.parse_args()

    # Setup
    config_file = args.config
    cfg.merge_from_file(config_file)
    root = args.root
    use_car = args.use_car
    confidence_threshold = 0.9

    print("Using cfg {}".format(config_file))
    print("Dataset: {}".format(root))
    print("Use_car: {}".format(use_car))

    Vis = Visualize(cfg, root)

    save_dir = args.save
    save_dir = os.path.join(Vis.dataPath, save_dir)
    print("Predictions dir: {}".format(save_dir))

    # Vis.build_predictor(confidence_threshold, use_car)
    # Vis.get_bboxes()
    # Vis.visualize_boxes(save_dir, dataloader=True)
    Vis.remove_empty()


if __name__ == "__main__":
    main()
