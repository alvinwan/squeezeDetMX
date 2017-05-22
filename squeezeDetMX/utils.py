"""Utilities for squeezeDet's MXNet implementation."""

from typing import List

import cv2
import mxnet as mx
import numpy as np


def build_module(symbol, name, data_iter,
        inputs_need_grad=False,
        learning_rate=0.01,
        momentum=0.9,
        wd=0.0005,
        lr_scheduler=None,
        checkpoint=None,
        ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]):
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
    module.bind(data_shapes=data_shapes, label_shapes=label_shapes, inputs_need_grad=inputs_need_grad)
    module.init_params(initializer=mx.init.MSRAPrelu())
    module.init_optimizer(kvstore='device',
        optimizer=mx.optimizer.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            wd=wd,
            lr_scheduler=mx.lr_scheduler.FactorScheduler(60000, 0.20) if lr_scheduler is None else lr_scheduler,
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


def image_to_jpeg_bytes(image) -> bytes:
    return cv2.imencode('.jpg', image)[1].tobytes()


def jpeg_bytes_to_image(bytedata: bytes) -> np.array:
    return mx.image.imdecode(bytedata, to_rgb=False).asnumpy().astype(np.float32)
