"""Reads raw KITTI data and preprocesses data for RecordIO.

Usage:
    kitti.py [options]

Options:
    --data=<path>       Root folder containing all data. [default: ../data/KITTI]
"""

import cv2
import docopt
import mxnet as mx
from mxnet import io

from typing import Tuple
from typing import List
import struct
import os

from util import bbox_transform_inv


CLASS_TO_INDEX = {
    'car': 0,
    'pedestrian': 1,
    'cyclist': 2
}


def main():
    """Translating KITTI data into RecordIO"""
    arguments = docopt.docopt(__doc__)
    data_root = arguments['--data']

    X_train, Y_train = grab_images_labels(data_root, 'train')
    X_val, Y_val = grab_images_labels(data_root, 'trainval')

    train_writer = KITTIWriter('train.brick')
    train_writer.write(X_train, Y_train)
    train_writer.close()
    print(' * Finished writing train.')

    val_writer = KITTIWriter('trainval.brick')
    val_writer.write(X_val, Y_val)
    val_writer.close()
    print(' * Finished writing trainval.')


def grab_images_labels(data_root: str, dataset: str) -> Tuple[List, List]:
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
    return image_data, image_labels


def read_bboxes(objects: List[str]) -> List[List]:
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


class KITTIWriter:
    """Designed for writing KITTI data as RecordIO objects"""

    def __init__(self, filename: str):
        self.filename = filename
        self.record = mx.recordio.MXRecordIO(filename, 'w')

    def byteIter(self, images: List, labels: List, struct_fmt: str='ffffi'):
        """Provide generator for images and labels as byte objects."""
        struct_size = bytes([struct.calcsize(struct_fmt)])
        for image, label in zip(images, labels):
            image_bytes = bytearray(cv2.imencode('.jpg', image)[1])
            yield b''.join([
                len(image_bytes).to_bytes(15, 'little'),
                image_bytes,
                struct_size,
                struct.pack(struct_fmt, *label[0])])

    def write(self, images: List, labels: List):
        """Write set of images and labels to the provided file."""
        for byte_data in self.byteIter(images, labels):
            self.record.write(byte_data)

    def close(self):
        self.record.close()


class KITTIIter(io.DataIter):
    """Iterator designed for reading KITTI data, compatible with MXNet"""
    pass


if __name__ == '__main__':
    main()
