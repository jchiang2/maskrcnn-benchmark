import os
import csv
import numpy as np
import cv2
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList

class myDataset(torch.utils.data.Dataset):
    """
    Define our custom dataset
    """

    CLASSES = (
        "__background__ ",
        "moving_car",
        "parked_car",
    )

    def __init__(self, root, transforms=None):
        """
        Args:
          root: str path to dataset
          transforms: image transforms
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))

        framesPath = os.path.join(dir_path, root, "Frames")
        bboxesPath = os.path.join(dir_path, root, "Bboxes")
        optFlowPath = os.path.join(dir_path, root, "OptFlow")
        # Lists of files
        self.frames = sorted([os.path.join(framesPath, file)
                                for file in os.listdir(framesPath)])
        self.bboxes = sorted([os.path.join(bboxesPath, file)
                                for file in os.listdir(bboxesPath)])
        self.optFlow = sorted([os.path.join(optFlowPath, file)
                                for file in os.listdir(optFlowPath)])

        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """
        returns:
            image: transformed image
            optFlowVol: tensor of dim (C x H x W)
            boxlist: Boxlist container of labels a bbox coordinates
        """

        optFlowVol = self.get_optFlow(idx)
        image = self.get_img(idx)
        labels, boxes = self.get_labels(idx)

        # Add bboxes and labels to boxlist
        if labels is not None:
            boxlist = BoxList(boxes, image.size, mode="xyxy")
            boxlist.add_field("labels", labels)
        else:
            boxlist = None

        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)
        
        
        return image, optFlowVol, boxlist, idx

    def get_img_info(self, idx):
        """
        get img_height and img_width. This is used if
        we want to split the batches according to the aspect ratio
        of the image, as it can be more efficient than loading the
        image from disk
        """

        image = self.get_img(idx)
        img_height = image.size[0]
        img_width = image.size[1]

        return {"height": img_height, "width": img_width}

    def get_groundtruth(self, idx):
        """
        returns:
            target: ground truth boxlist
        """
        image = self.get_img(idx)
        labels, boxes = self.get_labels(idx)
        
        target = BoxList(boxes, image.size, mode="xyxy")
        target.add_field("labels", labels)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return target

    def map_class_id_to_class_name(self, class_id):
        return myDataset.CLASSES[class_id]

    def get_img(self, idx):
        """
        returns:
            image: PIL image
        """

        imgPath = self.frames[idx]
        image = Image.open(imgPath)

        return image

    def get_optFlow(self, idx):
        """
        returns:
            optFlowVol: optical flow tensor of dim (C x H x W)
        """

        optFlowPath = self.optFlow[idx]

        optFlowVol = cv2.imread(optFlowPath)

        # optFlowVol = np.load(optFlowPath)

        # optFlowVol = np.linalg.norm(optFlowVol, axis=2)
        # optFlowVol[optFlowVol > 50] = 0
        # optFlowVol = cv2.GaussianBlur(optFlowVol,(51,51),10)

        # optFlowVol = np.expand_dims(optFlowVol, 2)

        optFlowVol = F.to_tensor(optFlowVol)

        return optFlowVol

    def get_labels(self, idx):
        """
        returns:
            labels: ground truth labels of each bbox
            boxes: bounding box coordinates
        """

        # Load csv files
        if self.bboxes == []:
            return None, None

        infoPath = self.bboxes[idx]
        infoArray = np.genfromtxt(infoPath, delimiter = ',')

        infoArray = np.reshape(infoArray, (-1,6))
        if infoArray.shape[1] > 2:
            boxes = infoArray[:,2:]

        # Labels should be torch.long() type
        labels = infoArray[:,1]
        labels = torch.from_numpy(labels).long()

        return labels, boxes
