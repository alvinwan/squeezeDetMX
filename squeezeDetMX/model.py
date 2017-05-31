"""Specify the SqueezeDet architecture in MXNet"""

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.symbol as sym
import numpy as np
from .constants import NUM_OUT_CHANNELS
from .constants import ANCHORS_PER_GRID
from .constants import NUM_CLASSES
from .constants import NUM_BBOX_ATTRS
from .utils import batches_iou
from typing import List


class SqueezeDet:
    """Setup the original squeezeDet architecture"""

    def __init__(self):
        self.data = sym.Variable('image')
        self.label_box = sym.Variable('label_box')
        self.label_class = sym.Variable('label_class')
        self.label_score = sym.Variable('label_score')
        self.net = self.add_forward(self.data)
        self.error = self.add_loss(self.net)

    def add_forward(self, data: sym.Variable):
        """Add neural network model."""
        conv1 = sym.Convolution(
            data, name='conv1', num_filter=64, kernel=(3, 3), stride=(2, 2))
        relu1 = sym.Activation(conv1, act_type='relu')
        pool1 = sym.Pooling(relu1, pool_type='max', kernel=(3, 3), stride=(2, 2))
        fire2 = self._fire_layer('fire2', pool1, s1x1=16, e1x1=64, e3x3=64)
        fire3 = self._fire_layer('fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
        pool3 = sym.Pooling(fire3, name='pool3', kernel=(3, 3), stride=(2, 2), pool_type='max')
        fire4 = self._fire_layer('fire4', pool3, s1x1=32, e1x1=128, e3x3=128)
        fire5 = self._fire_layer('fire5', fire4, s1x1=32, e1x1=128, e3x3=128)
        pool5 = sym.Pooling(fire5, name='pool5', kernel=(3, 3), stride=(2, 2), pool_type='max')
        fire6 = self._fire_layer('fire6', pool5, s1x1=48, e1x1=192, e3x3=192)
        fire7 = self._fire_layer('fire7', fire6, s1x1=48, e1x1=192, e3x3=192)
        fire8 = self._fire_layer('fire8', fire7, s1x1=64, e1x1=256, e3x3=256)
        fire9 = self._fire_layer('fire9', fire8, s1x1=64, e1x1=256, e3x3=256)
        fire10 = self._fire_layer('fire10', fire9, s1x1=96, e1x1=384, e3x3=384)
        fire11 = self._fire_layer('fire11', fire10, s1x1=96, e1x1=384, e3x3=384)
        dropout11 = sym.Dropout(fire11, p=0.1, name='drop11')
        return sym.Convolution(
            dropout11, name='conv12', num_filter=NUM_OUT_CHANNELS,
            kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    def add_loss(self, net: sym.Variable):
        """Add loss functions.

        Below, we splice the network output accordingly to compute losses for
        the following:

            1. Bounding box attributes
            2. Class probabilities
            3. IOUS as "confidence scores"

        Below, the ugly splice functions are replacements for reshaping.
        Instead, split along a dimension into multiple chunks, and then
        restack the arrays in a consistent way.

        Due to quirk in MXNet, we create a placeholder label_score. However,
        we actually use pred_box and label_box to compute IOU (true labels),
        which are then compared with pred_score.
        """
        num_splits = int(NUM_OUT_CHANNELS / ANCHORS_PER_GRID)
        net_splits = list(sym.split(net, num_outputs=num_splits))

        # Compute loss for bounding box
        pred_box = sym.concat(*net_splits[:NUM_BBOX_ATTRS])
        loss_box = sym.LinearRegressionOutput(data=pred_box, label=self.label_box)

        # Compute loss for class probabilities
        cidx = NUM_BBOX_ATTRS + NUM_CLASSES
        reformat = lambda x: sym.transpose(sym.concat(*sym.split(
            x, num_outputs=ANCHORS_PER_GRID), dim=0), axes=(1, 0, 2, 3))
        pred_class = reformat(sym.concat(
            *net_splits[NUM_BBOX_ATTRS:cidx], dim=1))
        label_class = reformat(self.label_class)
        loss_class = sym.SoftmaxOutput(data=pred_class, label=label_class)

        # Compute loss for confidence scores.
        pred_score = net_splits[cidx]
        loss_iou = mx.symbol.Custom(
            data=pred_score,
            label=sym.concat(self.label_score, pred_box, self.label_box, dim=1),
            op_type='IOURegressionOutput')

        return mx.sym.Group([loss_box, loss_iou, loss_class])

    def _fire_layer(
            self,
            name: str,
            inputs: sym.Variable,
            s1x1: int,
            e1x1: int,
            e3x3: int):
        """Fire layer constructor. Written by Bichen Wu from UC Berkeley.

        Args:
          layer_name: layer name
          inputs: input tensor
          s1x1: number of 1x1 filters in squeeze layer.
          e1x1: number of 1x1 filters in expand layer.
          e3x3: number of 3x3 filters in expand layer.
          freeze: if true, do not train parameters in this layer.
        Returns:
          fire layer operation.
        """
        sq1x1 = sym.Convolution(
            inputs, name=name+'/s1x1', num_filter=s1x1, kernel=(1, 1), stride=(1, 1))
        relu1 = sym.Activation(sq1x1, act_type='relu')
        ex1x1 = sym.Convolution(
            relu1, name=name+'/e1x1', num_filter=e1x1, kernel=(1, 1), stride=(1, 1))
        relu2 = sym.Activation(ex1x1, act_type='relu')
        ex3x3 = sym.Convolution(
            relu1, name=name+'/e3x3', num_filter=e3x3, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        relu3 = sym.Activation(ex3x3, act_type='relu')
        return sym.Concat(relu2, relu3, dim=1, name=name+'/concat')


class IOURegressionOutput(mx.operator.CustomOp):
    """Custom operator for IOU regression."""

    def __init__(self, ctx):
        super(IOURegressionOutput, self).__init__()
        self.ctx = ctx

    def forward(self, is_train: bool, req, in_data: List, out_data: List, aux):
        pred_ious = np.ravel(self.reformat(in_data[0]))
        self.assign(out_data[0], req[0], pred_ious)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pred_ious = out_data[0].asnumpy()
        ious = self.ious(in_data[1])
        gradient = -2 * (ious - pred_ious)

        # TODO(Alvin): Is this last reshape consistent?
        self.assign(in_grad[0], req[0], gradient.reshape(in_data[0].shape))

    @staticmethod
    def reformat(x: nd.array) -> nd.array:
        """Reformat array to be (-1, 4)"""
        return nd.flatten(nd.transpose(nd.concat(*nd.split(
            x, num_outputs=9, axis=1), dim=0), axes=(1, 0, 2, 3))).T.asnumpy()

    @classmethod
    def ious(cls, label: np.array) -> np.array:
        """Convert the label provided to this layer to ious."""
        pred_idx = ANCHORS_PER_GRID * (1 + NUM_BBOX_ATTRS)
        pred_box = cls.reformat(nd.slice_axis(
            label, axis=1, begin=ANCHORS_PER_GRID, end=pred_idx))
        label_box = cls.reformat(nd.slice_axis(
            label, axis=1, begin=pred_idx, end=None))
        return batches_iou(pred_box, label_box)


@mx.operator.register("IOURegressionOutput")
class IOURegressionOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(IOURegressionOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        pred_shape = in_shape[0]
        label_shape = in_shape[0][:]
        label_shape[1] = ANCHORS_PER_GRID * (NUM_BBOX_ATTRS * 2 + 1)
        return [pred_shape, label_shape], [(np.prod(pred_shape),)], []

    def create_operator(self, ctx, shapes, dtypes):
        return IOURegressionOutput(ctx)
