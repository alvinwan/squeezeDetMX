"""Reads raw KITTI data and preprocesses data for RecordIO.

Usage:
    kitti.py [options]
Options:
    --data=<path>       Root folder containing all data. [default: ../data/KITTI]
"""

import cv2
import docopt
import mxnet as mx
import numpy as np
from mxnet import io
from mxnet import ndarray as nd
from mxnet._ndarray_internal import _cvimresize as imresize

from typing import Tuple
from typing import List
import struct
import os

from utils import bbox_transform_inv


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

    def byteIter(self, images: List, labels: List, label_fmt: str='ffffi'):
        """Provide generator for images and labels as byte objects."""
        struct_size = struct.calcsize(label_fmt).to_bytes(5, 'little')
        for i, (image, label) in enumerate(zip(images, labels)):
            if i % 1000 == 0 and i > 0:
                print(' * Saved', i, 'images.')
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
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

    def __init__(
            self,
            filename: str,
            label_fmt: str='ffffi',
            img_shape: Tuple=(3, 1242, 375),
            batch_size=20):
        self.filename = filename
        self.record = mx.recordio.MXRecordIO(filename, 'r')
        self.bytedata = self.record.read()
        self.label_fmt = label_fmt
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.provide_data = [('image', (batch_size, *img_shape))]
        self.provide_label = []

    def __iter__(self):
        return self

    def __next__(self):
        """Alias for next(object)."""
        return self.next()

    def next(self):
        """Yield the next datum for MXNet to run."""
        batch_image = nd.empty((self.batch_size, *self.img_shape))
        batch_label = nd.empty((self.batch_size, 4))
        try:
            for i in range(self.batch_size):
                batch_image[i][:] = self.__read_image()
                batch_label[i][:] = self.__read_label()
            return io.DataBatch([batch_image], [batch_label], batch_size-1-i)
        except StopIteration:
            self.record.close()
            raise StopIteration

    def __read_image(self):
        """Read image from the byte buffer."""
        image_size = int.from_bytes(self.step(15), 'little')
        image = mx.image.imdecode(
            (self.step(image_size)).astype(np.float32) - 127.) / 127.
        image = imresize(image, *self.img_shape)
        return nd.transpose(image, axes=(2, 0, 1))

    def __read_label(self):
        """Read label from the byte buffer."""
        label_size = int.from_bytes(self.step(5), 'little')
        label = struct.unpack(self.label_fmt, self.step(label_size))
        # TODO(Alvin): finish label conversion
        return

    def step(self, steps):
        """Step forward by `steps` in the byte buffer."""
        if not self.bytedata:
            raise StopIteration
        if steps > len(self.bytedata):
            print(' * Warning: Failed to read expected data from byte buffer.')
            raise StopIteration
        rv, self.bytedata = self.bytedata[:steps], self.bytedata[steps:]
        return rv


if __name__ == '__main__':
    main()
