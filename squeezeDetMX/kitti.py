"""KITTI deserialization and read utilities"""

import cv2

from typing import Tuple
from typing import List
import os
import numpy as np

from .constants import CLASS_TO_INDEX
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
        image_data.append(cv2.imread(image_path))
        label_path = os.path.join(data_root, 'training/label_2/%s.txt' % _id)
        with open(label_path) as f:
            image_labels.append(read_bboxes(f.read().splitlines()))
    if shuffle:
        groups = [group for group in zip(image_data, image_labels)]
        np.random.shuffle(groups)
        return zip(*groups)
    return image_data, image_labels


def read_bboxes(objects: List[str]) -> List[List[float]]:
    """Read bounding boxes from provided serialized data."""
    bboxes = []
    for object_string in objects:
        object_data = object_string.strip().split(' ')
        category_index = object_data[0].lower()
        if category_index not in CLASS_TO_INDEX:
            continue
        category = CLASS_TO_INDEX[category_index]
        x, y, w, h = bbox_transform_inv(*map(float, object_data[4:8]))
        bboxes.append([x, y, w, h, category])
    return bboxes
