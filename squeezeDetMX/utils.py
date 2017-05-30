"""Utilities for squeezeDet's MXNet implementation."""

import cv2
import mxnet as mx
import numpy as np
from mxnet import io
from mxnet import ndarray as nd
from mxnet._ndarray_internal import _cvimresize as imresize

from typing import Tuple
from typing import List
import struct
import logging
import os
import os.path

from .constants import ANCHORS_PER_GRID
from .constants import NUM_OUT_CHANNELS
from .constants import NUM_BBOX_ATTRS
from .constants import NUM_CLASSES
from .constants import GRID_WIDTH
from .constants import GRID_HEIGHT
from .constants import IMAGE_WIDTH
from .constants import IMAGE_HEIGHT
from .constants import RANDOM_WIDTHS_HEIGHTS
from .constants import IMAGE_BYTES_SLOT
from .constants import BBOXES_BYTES_SLOT
from .constants import BBOX_FORMAT


def build_module(symbol, name, data_iter,
        inputs_need_grad=False,
        learning_rate=0.01,
        momentum=0.9,
        wd=0.0005,
        lr_scheduler=None,
        checkpoint=None,
        ctx=(mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3))):
    data_shapes = data_iter.provide_data
    label_shapes = data_iter.provide_label

    def get_names(shapes):
        if not shapes:
            return None
        return tuple(map(lambda shape: shape[0], shapes))

    module = mx.mod.Module(symbol=symbol,
        data_names=get_names(data_shapes),
        label_names=get_names(label_shapes),
        context=ctx)
    module.bind(
        data_shapes=data_shapes,
        label_shapes=label_shapes,
        inputs_need_grad=inputs_need_grad)
    module.init_params(initializer=mx.init.MSRAPrelu())
    module.init_optimizer(kvstore='device',
        optimizer=mx.optimizer.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            wd=wd,
            lr_scheduler=mx.lr_scheduler.FactorScheduler(60000, 0.20)
            if lr_scheduler is None else lr_scheduler,
        )
    )

    symbol.save('{}-symbol.json'.format(name))
    if checkpoint is not None:
        _, arg, aux = mx.model.load_checkpoint(name, checkpoint)
        module.set_params(arg, aux)

    return module


def bbox_transform_inv(xmin: int, ymin: int, xmax: int, ymax: int) -> List[int]:
    """Converts coordinates from corners to cx, cy, w, h."""
    return [
        (xmax + xmin) / 2,
        (ymax + ymin) / 2,
        xmax - xmin,
        ymax - ymin
    ]


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    return cv2.imencode('.jpg', image)[1].tobytes()


def jpeg_bytes_to_image(bytedata: bytes) -> np.array:
    return mx.image.imdecode(bytedata, to_rgb=False).asnumpy().astype(np.float32)


def batch_iou(boxes: np.array, box: np.array) -> np.array:
    """
    Args:
        b1: 2D array of [cx, cy, width, height].
        b2: a single array of [cx, cy, width, height]
    Returns:
        ious: array of a float number in range [0, 1].
    """
    return batches_iou(boxes, box.reshape((1, -1)))


def batches_iou(b1: np.array, b2: np.array) -> np.array:
    """
    Compute the Intersection-Over-Union of a batch of boxes with another
    batch of boxes.

    From original repository, written by Bichen Wu

    Args:
        b1: 2D array of [cx, cy, width, height].
        b2: 2D array of [cx, cy, width, height]
    Returns:
        ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(
        np.minimum(b1[:, 0] + 0.5 * b1[:, 2], b2[:, 0] + 0.5 * b2[:, 2]) - \
        np.maximum(b1[:, 0] - 0.5 * b1[:, 2], b2[:, 0] - 0.5 * b2[:, 2]),
        0
    )
    tb = np.maximum(
        np.minimum(b1[:, 1] + 0.5 * b1[:, 3], b2[:, 1] + 0.5 * b2[:, 3]) - \
        np.maximum(b1[:, 1] - 0.5 * b1[:, 3], b2[:, 1] - 0.5 * b2[:, 3]),
        0
    )
    inter = lr*tb
    union = b1[:, 2] * b1[:, 3] + b2[:, 2] * b2[:, 3] - inter
    return inter/union


def size_in_bytes(bytedata: bytes, slot_size: int) -> bytes:
    """Compute size of bytedata, in bytes."""
    return len(bytedata).to_bytes(slot_size, 'little')


def create_anchors(
        num_x: int=GRID_WIDTH,
        num_y: int=GRID_HEIGHT,
        whs: List[List[int]]=RANDOM_WIDTHS_HEIGHTS) -> np.array:
    """Generates a list of [x, y, w, h], where centers are spread uniformly."""
    xs = np.linspace(0, IMAGE_WIDTH, num_x+2)[1:-1]  # exclude 0, IMAGE_WIDTH
    ys = np.linspace(0, IMAGE_HEIGHT, num_y+2)[1:-1]  # exclude 0, IMAGE_HEIGHT
    return np.vstack([(x, y, w, h) for x in xs for y in ys for w, h in whs])


def setup_logger(path: str='./logs/model.log'):
    """Setup the logs are ./logs/model.log"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_formatter = logging.Formatter('%(asctime)s %(message)s', '%y%m%d %H:%M:%S')
    logger = logging.getLogger()
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)


class Writer:
    """Designed for writing images and labels as RecordIO objects"""

    def __init__(self, filename: str):
        self.filename = filename
        self.record = mx.recordio.MXRecordIO(filename, 'w')

    def __enter__(self):
        return self

    @staticmethod
    def byteIter(images: List, labels: List, bbox_fmt: str=BBOX_FORMAT):
        """Provide generator for images and labels as byte objects."""
        for i, (image, bboxes) in enumerate(zip(images, labels)):
            if i % 1000 == 0 and i > 0:
                print(' * Saved', i, 'images.')
            image_bytes = image_to_jpeg_bytes(image)
            bboxes_bytes = b''.join([
                struct.pack(bbox_fmt, *bbox) for bbox in bboxes])
            yield b''.join([
                size_in_bytes(image_bytes, IMAGE_BYTES_SLOT),
                image_bytes,
                size_in_bytes(bboxes_bytes, BBOXES_BYTES_SLOT),
                bboxes_bytes])

    def write(self, images: List, labels: List):
        """Write set of images and labels to the provided file."""
        for byte_data in self.byteIter(images, labels):
            self.record.write(byte_data)

    def close(self):
        self.record.close()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class Reader(io.DataIter):
    """Iterator designed for reading recordIO , compatible with MXNet"""

    anchors = create_anchors()

    def __init__(
            self,
            filename: str=None,
            label_fmt: str='ffffi',
            img_shape: Tuple=(3, IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size: int=20):
        super(Reader, self).__init__()
        self.filename = filename
        self.label_fmt = label_fmt
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.provide_data = [('image', (batch_size, *img_shape))]
        self.provide_label = [
            ('label_box', (
                batch_size, ANCHORS_PER_GRID * NUM_BBOX_ATTRS, GRID_HEIGHT, GRID_WIDTH)),
            ('label_score', (
                batch_size, ANCHORS_PER_GRID, GRID_HEIGHT, GRID_WIDTH))]

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
        batch_images = nd.empty((self.batch_size, *self.img_shape))
        batch_labels = []
        for i in range(self.batch_size):
            batch_images[i][:] = self.image_to_mx(self.read_image())
            batch_labels.append(self.read_label())
            if self.record:
                self.bytedata = self.record.read()
        batch_label_box, batch_label_class, batch_label_score = \
            self.batch_label_to_mx(batch_labels)
        return io.DataBatch(
            [batch_images],
            [batch_label_box, batch_label_score],
            self.batch_size-1-i)

    def read_image(self):
        """Read image from the byte buffer."""
        image_size = int.from_bytes(self.step(15), 'little')
        return jpeg_bytes_to_image(self.step(image_size))

    @staticmethod
    def image_to_mx(image: np.array) -> nd.array:
        """Convert a standard numpy array into MXNet-ready arrays."""
        return nd.transpose(
            imresize(  # TODO(Alvin): imresize should not be needed!
                nd.array(image), IMAGE_WIDTH, IMAGE_HEIGHT, interp=2),
                axes=(2, 0, 1))

    def read_label(self):
        """Read label from the byte buffer."""
        labels_size = int.from_bytes(self.step(BBOXES_BYTES_SLOT), 'little')
        label_size = struct.calcsize(BBOX_FORMAT)
        num_labels = labels_size / label_size
        assert num_labels % 1 == 0, 'Faulty formatting: Size per label does' \
                                    'not divide total space allocated to labels.'
        return np.array([
            struct.unpack(self.label_fmt, self.step(label_size))
            for _ in range(int(num_labels))])

    @staticmethod
    def batch_label_to_mx(labels: List[np.array]) -> nd.array:
        """Convert standard label into SqueezeDet-specific formats.

        Input is a list of bounding boxes, with x, y, width, and height.
        However, SqueezeDet expects a grid of data around 72 channels deep. The
        grid is 76 wide and 22 high, where each grid contains 9 anchors. For
        each anchor, the output should hold information for a bounding box:

            - placeholder for confidence score
            - bounding box attributes: x, y, w, h
            - one-hot class probabilities

        1. Compute distance. First, use IOU as a metric, and if all IOUs are 0,
        use Euclidean distance.
        2. Assign this bbox to the closest anchor index.
        3. Fill in the big matrix accordingly: Compute the grid that this anchor
        belongs to, and compute the relative position of the anchor w.r.t. the
        grid. For each grid cell, we fill:

            - first ANCHORS_PER_GRID * NUM_BBOX_ATTRS with bounding box attrs
            - next ANCHORS_PER_GRID * ONE with nothing
            - last ANCHORS_PER_GRID * NUM_CLASSES with one hot class labels
        """
        taken_anchor_indices = set()
        final_label = np.zeros((
            len(labels), NUM_OUT_CHANNELS, GRID_HEIGHT, GRID_WIDTH))
        one_hot_mapping = np.eye(NUM_CLASSES)
        i_box = ANCHORS_PER_GRID * NUM_BBOX_ATTRS
        for i, bboxes in enumerate(labels):
            for bbox in bboxes:
                # 1. Compute distance
                dists = batch_iou(Reader.anchors, bbox)
                if np.max(dists) == 0:
                    dists = [np.linalg.norm(bbox[:4] - anchor)
                             for anchor in Reader.anchors]

                # 2. Assign to anchor
                anchor_index = int(np.argmax(dists))
                if anchor_index in taken_anchor_indices:
                    continue
                taken_anchor_indices.add(anchor_index)

                # 3. Place in grid
                anchor_x, anchor_y = Reader.anchors[anchor_index][:2]
                grid_x = int(anchor_x // GRID_WIDTH)
                grid_y = int(anchor_y // GRID_HEIGHT)
                air = anchor_index % ANCHORS_PER_GRID

                st = air * NUM_BBOX_ATTRS
                final_label[i, st: st + NUM_BBOX_ATTRS, grid_x, grid_y] = \
                    bbox[:NUM_BBOX_ATTRS]

                st = i_box + (air * NUM_CLASSES)
                final_label[i, st: st + NUM_CLASSES, grid_x, grid_y] = \
                    one_hot_mapping[int(bbox[-1])]
        label_box = nd.array(final_label[:, :i_box])
        label_class = nd.array(
            final_label[:, i_box: i_box + (ANCHORS_PER_GRID * NUM_BBOX_ATTRS)])
        label_score = nd.array(final_label[:, -ANCHORS_PER_GRID:])
        return label_box, label_class, label_score

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

    def reset(self):
        if self.record:
            self.record.reset()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
