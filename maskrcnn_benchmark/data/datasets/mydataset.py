from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
import csv
import numpy as np
import cv2
import torch
import torch.utils.data
import sys
from natsort import natsorted   
from PIL import Image
from torchvision.transforms import functional as F

from maskrcnn_benchmark.config import cfg


class myDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "moving_car",
        "parked_car",
    )
    '''
    CLASSES = (
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",                                                                                                                             
    )
    '''

    def __init__(self, ann_file, root, transforms=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join('./maskrcnn_benchmark/data/',root)
        self.ann_file = os.path.join('./maskrcnn_benchmark/data/',ann_file)

        if 'Movie_Frames_train' in root:
            self.np_path = os.path.join(dir_path, "Movie_Frames_train", "opt_flow_np")
        else:
            self.np_path = os.path.join(dir_path, "Movie_Frames_test", "split_train", "opt_flow_np")

        if not os.path.exists(self.root):
            self.root = os.path.join(dir_path, root)
            self.ann_file = os.path.join(dir_path, ann_file)
        print(self.root)
        print(self.ann_file)
        self.transforms = transforms
        self.imgFiles = natsorted(os.listdir(self.root))
        self.infoFiles = natsorted(os.listdir(self.ann_file))
        self.opt_flow_files = natsorted(os.listdir(self.np_path))



    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        """
        optFlowPath = os.path.join(self.np_path, self.opt_flow_files[idx])
        optFlowVol = np.load(optflowPath)
        optFlowVol_mean = np.mean(optFlowVol, axis=(0,1))
        optFlowVol = optFlowVol - optFlowVol_mean
        """
        """
        imgFile = self.imgFiles[idx]
        imgPath = os.path.join(self.root, imgFile)
        image = Image.open(imgPath)
        """
        optFlowVol = self.get_optFlow(idx)
        image = self.get_img(idx)
        labels, boxes = self.get_labels(idx)
        """
        infoFile = self.infoFiles[idx]
        infoPath = os.path.join(self.ann_file, infoFile)
        infoArray = np.genfromtxt(infoPath, delimiter = ',')
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        infoArray = np.reshape(infoArray, (-1,6))
        if infoArray.shape[1] > 2:
            boxes = infoArray[:,2:]

        # Labels should be torch.long() type
        labels = infoArray[:,1]
        labels[labels==3.] = 1.0
        labels = torch.from_numpy(labels)
        labels = labels.long()
        """
        # and labels
        # labels = torch.ones(infoArray[:,1].size, dtype=torch.long) 
        difficult = torch.zeros(boxes.shape[0], dtype=torch.long)
        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficult)

        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)
            
        # image = torch.cat((image,optFlowVol), 0)


        # return the image, the boxlist and the idx in your dataset
        return image, optFlowVol, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        imgFile = self.imgFiles[idx]
        imgPath = os.path.join(self.root, imgFile)
        image = cv2.imread(imgPath)

        img_height = image.shape[0]
        img_width = image.shape[1]

        return {"height": img_height, "width": img_width}


    def get_groundtruth(self, idx):
        """
        imgFile = self.imgFiles[idx]
        imgPath = os.path.join(self.root, imgFile)
        image = Image.open(imgPath)
        """
        image = self.get_img(idx)
        labels, boxes = self.get_labels(idx)
        """
        infoFile = self.infoFiles[idx]
        infoPath = os.path.join(self.ann_file, infoFile)
        infoArray = np.genfromtxt(infoPath, delimiter = ',')
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        infoArray = np.reshape(infoArray, (-1,6))
        if infoArray.shape[1] > 2:
            boxes = infoArray[:,2:]
        
        # Labels should be torch.long() type
        labels = infoArray[:,1]
        labels[labels==3.] = 1.0
        labels = torch.from_numpy(labels)
        labels = labels.long()
        """
        # labels = torch.ones(infoArray[:,1].size, dtype=torch.long)
        difficult = torch.zeros(boxes.shape[0], dtype=torch.long)
        # create a BoxList from the boxes
        target = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)
        target.add_field("difficult", difficult)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return target

    def map_class_id_to_class_name(self, class_id):
        return myDataset.CLASSES[class_id]

    def get_img(self, idx):

        imgFile = self.imgFiles[idx]
        imgPath = os.path.join(self.root, imgFile)
        image = Image.open(imgPath)  

        return image

    def get_optFlow(self, idx):

        optFlowPath = os.path.join(self.np_path, self.opt_flow_files[idx])
        optFlowVol = np.load(optFlowPath)

        optFlowVol_mean = np.mean(optFlowVol, axis=(0,1))
        optFlowVol = optFlowVol - optFlowVol_mean

        optFlowVol = F.to_tensor(optFlowVol)

        return optFlowVol

    def get_labels(self, idx):

        infoFile = self.infoFiles[idx]
        infoPath = os.path.join(self.ann_file, infoFile)
        infoArray = np.genfromtxt(infoPath, delimiter = ',')
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        infoArray = np.reshape(infoArray, (-1,6))
        if infoArray.shape[1] > 2:
            boxes = infoArray[:,2:]

        # Labels should be torch.long() type
        labels = infoArray[:,1]
        labels[labels==3.] = 1.0
        labels = torch.from_numpy(labels)
        labels = labels.long()

        return labels, boxes
