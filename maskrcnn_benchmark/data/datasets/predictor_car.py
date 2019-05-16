# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import cv2
import torch
from torchvision import transforms as T


from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.engine.trainer import *
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.engine.inference import inference


class COCODemo_car(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "moving_car",
        "parked_car",
        
    ]

    def __init__(
        self,
        cfg=None,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1], dtype=torch.long)

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim
        self.cuda = torch.device(cfg.MODEL.DEVICE)

    def getPrecRec(self):
        cfg = self.cfg
        distributed = False
        model = self.model
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
            synchronize()


    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def track_objects(self, curr, next, predictions):
        multiTracker = cv2.MultiTracker_create()
        bboxes = predictions.bbox
        for bbox in bboxes:
            if bbox[0] < 0 or bbox[1] < 0:
                continue
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            tracker = cv2.TrackerKCF_create()
            multiTracker.add(tracker, curr, bbox)

        ok, boxes = multiTracker.update(next)
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        predictions.bbox = boxes

        return predictions


    def run_on_opencv_image(self, image, optFlowVol, rgb_image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image, optFlowVol)
        top_predictions = self.select_top_predictions(predictions)

        if len(top_predictions.get_field("labels")) > 0:
            self.remove_overlapped(top_predictions)

        result = rgb_image.copy()
        '''
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        '''
        result = self.overlay_boxes(result, top_predictions,[0,255,0])
        '''
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        '''
        result = self.overlay_class_names(result, top_predictions, False)

        count = len(top_predictions)

        return result, top_predictions, rgb_image

    def plot_gt(self, image, target):
        result = image.copy()
        result = self.overlay_boxes(result, target, [0,0,255])
        result = self.overlay_class_names(result, target, True)

        return result

    def compute_prediction(self, image, optFlowVol):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        '''
        image = self.transforms(original_image)
        '''
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        
        optFlowVol = to_image_list(optFlowVol, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        optFlowVol = optFlowVol.to(self.device)
        
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, optFlowVol)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size

        #height, width = original_image.shape[:-1]
        #prediction = prediction.resize((width, height))
        '''
        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the boundin
        labels = predictions.get_field("labels")
        boxes = predictions.bboxg boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        '''
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions, color):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        # colors = self.compute_colors_for_labels(labels).tolist()
        g = [0,255,0]
        r = [0,0,255]

        for box, label in zip(boxes, labels):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

            if label==1:
                c = g
            else:
                c = r
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), c, 2
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions, predict):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        g = [0,255,0]
        r = [0,0,255]
        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x, y = box[:2]
            if predict==True:
                s = template.format(label+"(ground_truth)")
                c = (0,0,255)
            else:
                s = template.format(label, score)

            if label=='moving_car':
                c = g
            else:
                c = r
            # cv2.putText(
            #     image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, c, 2
            # )
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, c, 2
            )

        return image

    def remove_overlapped(self, predictions):
        """
        Remove box with lower score in overlapping boxes
        Args:
            predictions (boxlist): boxlist containing top predictions
        returns:
            predictions (boxlist): boxlist with overlapped boxes removed
        """
        iou = boxlist_iou(
                predictions,
                predictions,
            ).numpy()
        overlaps = np.argwhere(iou > 0.6)
        
        boxes = predictions.bbox.tolist()
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()

        remain = np.arange(len(scores)).tolist()
        for i in range(len(overlaps)):
            pair = overlaps[i]
            if pair[0] < pair[1]:
                score = scores[pair[0]]
                other_score = scores[pair[1]]
                if score <= other_score:
                    remain.remove(pair[0])
                else:
                    remain.remove(pair[1])

        new_boxes = []
        new_scores = []
        new_labels = []
        for idx in remain:
            new_boxes.append(boxes[idx])
            new_scores.append(scores[idx])
            new_labels.append(labels[idx])


        predictions.extra_fields['scores'] = torch.FloatTensor(new_scores)
        predictions.extra_fields['labels'] = torch.LongTensor(new_labels)
        predictions.bbox = torch.tensor(new_boxes)

        return predictions
                    


