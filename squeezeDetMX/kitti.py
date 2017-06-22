"""KITTI deserialization and read utilities"""

import cv2

from typing import Tuple
from typing import List
import os
import numpy as np

from .constants import CLASS_TO_INDEX
from .constants import IMAGE_WIDTH
from .constants import IMAGE_HEIGHT
from .utils import bbox_transform_inv


def grab_images_labels(
        data_root: str, dataset: str, shuffle: bool=True) -> Tuple[List, List]:
    """Grab all images and labels from the specified dataset."""
    assert dataset in ('train', 'trainval', 'val')
    with open(os.path.join(data_root, 'ImageSets/%s.txt' % dataset)) as f:
        ids = f.read().splitlines()

    image_data, image_labels = [], []
    for i, _id in enumerate(ids):
        if i % 1000 == 0 and i > 0:
            print(' * Loaded', i, 'images.')
        image_path = os.path.join(data_root, 'training/image_2/%s.png' % _id)
        image = cv2.imread(image_path)
        scale_x = image.shape[0] / IMAGE_WIDTH
        scale_y = image.shape[1] / IMAGE_HEIGHT
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image_data.append(image)
        label_path = os.path.join(data_root, 'training/label_2/%s.txt' % _id)
        with open(label_path) as f:
            label = read_bboxes(f.read().splitlines(), scale_x, scale_y)
            image_labels.append(label)
    if shuffle:
        groups = [group for group in zip(image_data, image_labels)]
        np.random.shuffle(groups)
        return zip(*groups)
    return image_data, image_labels


def read_bboxes(objects: List[str], scale_x: float=1.0, scale_y: float=1.0) -> List[List[float]]:
    """Read bounding boxes from provided serialized data."""
    bboxes = []
    for object_string in objects:
        object_data = object_string.strip().split(' ')
        category_index = object_data[0].lower()
        if category_index not in CLASS_TO_INDEX:
            continue
        category = CLASS_TO_INDEX[category_index]
        x, y, w, h = bbox_transform_inv(*map(float, object_data[4:8]))
        bboxes.append([x / scale_x, y / scale_y, w, h, category])
    return bboxes
