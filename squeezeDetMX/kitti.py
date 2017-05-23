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

from utils import batch_iou
from utils import bbox_transform_inv
from utils import image_to_jpeg_bytes
from utils import jpeg_bytes_to_image

from utils import ANCHORS_PER_GRID
from utils import NUM_OUT_CHANNELS
from utils import GRID_WIDTH
from utils import GRID_HEIGHT
from utils import IMAGE_WIDTH
from utils import IMAGE_HEIGHT
from utils import RANDOM_WIDTHS_HEIGHTS


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


def create_anchors(
        num_x: int=GRID_WIDTH,
        num_y: int=GRID_HEIGHT,
        whs: List[List[int]]=RANDOM_WIDTHS_HEIGHTS):
    """Generates a list of [x, y, w, h], where centers are spread uniformly."""
    xs = np.linspace(0, IMAGE_WIDTH, num_x+2)[1:-1]  # exclude 0, IMAGE_WIDTH
    ys = np.linspace(0, IMAGE_HEIGHT, num_x+2)[1:-1]  # exclude 0, IMAGE_HEIGHT
    return np.vstack([(x, y, w, h) for x in xs for y in ys for w, h in whs])


class KITTIWriter:
    """Designed for writing KITTI data as RecordIO objects"""

    def __init__(self, filename: str):
        self.filename = filename
        self.record = mx.recordio.MXRecordIO(filename, 'w')

    def __enter__(self):
        return self

    @staticmethod
    def byteIter(images: List, labels: List, label_fmt: str='ffffi'):
        """Provide generator for images and labels as byte objects."""
        struct_size = struct.calcsize(label_fmt).to_bytes(5, 'little')
        for i, (image, label) in enumerate(zip(images, labels)):
            if i % 1000 == 0 and i > 0:
                print(' * Saved', i, 'images.')
            image_bytes = image_to_jpeg_bytes(image)
            yield b''.join([
                len(image_bytes).to_bytes(15, 'little'),
                image_bytes,
                struct_size,
                struct.pack(label_fmt, *label[0])])

    def write(self, images: List, labels: List):
        """Write set of images and labels to the provided file."""
        for byte_data in self.byteIter(images, labels):
            self.record.write(byte_data)

    def close(self):
        self.record.close()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class KITTIIter(io.DataIter):
    """Iterator designed for reading KITTI data, compatible with MXNet"""

    anchors = create_anchors()

    def __init__(
            self,
            filename: str=None,
            label_fmt: str='ffffi',
            img_shape: Tuple=(3, IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=20):
        self.filename = filename
        self.label_fmt = label_fmt
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.provide_data = [('image', (batch_size, *img_shape))]
        self.provide_label = [('label', (
            batch_size, NUM_OUT_CHANNELS, GRID_HEIGHT, GRID_WIDTH))]

        if filename is not None:
            self.record = mx.recordio.MXRecordIO(filename, 'r')
            self.bytedata = self.record.read()
        else:
            self.record = self.bytedata = None

    @classmethod
    def from_bytes(cls, bytedata, *args, **kwargs):
        obj = cls(*args, **kwargs)
        obj.bytedata = bytedata
        return obj

    def __iter__(self):
        return self

    def __next__(self):
        """Alias for next(object)."""
        return self.next()

    def __enter__(self):
        return self

    def next(self):
        """Yield the next datum for MXNet to run."""
        batch_image = nd.empty((self.batch_size, *self.img_shape))
        batch_label = nd.empty((self.batch_size, 4))
        try:
            for i in range(self.batch_size):
                batch_image[i][:] = self.image_to_mx(self.read_image())
                batch_label[i][:] = self.read_label()
            batch_label = self.batch_label_to_mx(batch_label)
            return io.DataBatch([batch_image], [batch_label], batch_size-1-i)
        except StopIteration:
            self.record.close()

    def read_image(self):
        """Read image from the byte buffer."""
        image_size = int.from_bytes(self.step(15), 'little')
        return jpeg_bytes_to_image(self.step(image_size))

    @staticmethod
    def image_to_mx(image: np.array) -> mx.nd.array:
        """Convert a standard numpy array into MXNet-ready arrays."""
        return nd.transpose(
            imresize(  # TODO(Alvin): imresize should not be needed!
                mx.nd.array(image), IMAGE_WIDTH, IMAGE_HEIGHT, interp=2),
                axes=(2, 0, 1))

    def read_label(self):
        """Read label from the byte buffer."""
        label_size = int.from_bytes(self.step(5), 'little')
        return np.array(struct.unpack(self.label_fmt, self.step(label_size)))[:4]

    @staticmethod
    def batch_label_to_mx(label: List[np.ndarray]) -> mx.nd.array:
        """Convert standard label into SqueezeDet-specific formats.

        Input is a list of bounding boxes, with x, y, width, and height.
        However, SqueezeDet expects a grid of data around 72 channels deep. The
        grid is 76 wide and 22 high, where each grid contains 9 anchors. For
        each anchor, the output should hold information for a bounding box.

        1. Compute distance. First, use IOU as a metric, and if all IOUs are 0,
        use Euclidean distance.
        2. Assign this bbox to the closest anchor index.
        3. Fill in the big matrix accordingly: Compute the grid that this anchor
        belongs to, and compute the relative position of the anchor w.r.t. the
        grid.
        """
        taken_anchor_indices = set()
        final_label = np.zeros((GRID_WIDTH, GRID_HEIGHT, NUM_OUT_CHANNELS))
        for bbox in label:
            # 1. Compute distance
            dists = batch_iou(KITTIIter.anchors, bbox)
            if max(metrics) == 0:
                dists = [np.linalg.norm(bbox, anchor) for anchor in KITTIITer.anchors]

            # 2. Assign to anchor
            anchor_index = np.argmax(metrics)
            if anchor_index in taken_anchor_indices:
                continue
            taken_anchor_indices.add(anchor_index)

            # 3. Place in grid
            anchor_x, anchor_y = self.anchors[anchor_index][:2]
            grid_x, grid_y = anchor_x // GRID_WIDTH, anchor_y // GRID_HEIGHT
            air = anchor_index % ANCHORS_PER_GRID
            final_label[grid_x][grid_y][air: air+4] = bbox
        return final_label

    def step(self, steps):
        """Step forward by `steps` in the byte buffer."""
        if not self.bytedata:
            raise StopIteration
        if steps > len(self.bytedata):
            print(' * Warning: Failed to read expected data from byte buffer.')
            raise StopIteration
        rv, self.bytedata = self.bytedata[:steps], self.bytedata[steps:]
        return rv

    def close(self):
        if self.record:
            self.record.close()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


if __name__ == '__main__':
    main()
