"""Specify the SqueezeDet architecture in MXNet"""

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.symbol as sym
import numpy as np
from .constants import NUM_OUT_CHANNELS
from .constants import EPSILON
from .constants import ANCHORS_PER_GRID
from .constants import NUM_CLASSES
from .constants import NUM_BBOX_ATTRS
from .constants import NUM_CHANNELS
from .utils import nd_batch_iou
from .utils import mask_with
from typing import List
from typing import Tuple


class SqueezeDet:
    """Setup the original squeezeDet architecture"""

    def __init__(self):
        self.data = sym.Variable('image')
        self.label = sym.Variable('label')
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
        dropout11 = sym.Dropout(fire11, p=0.5, name='drop11')
        return sym.Convolution(
            dropout11, name='conv12', num_filter=NUM_OUT_CHANNELS,
            kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    def add_loss(self, pred: sym.Variable):
        """Add loss. To save trouble, all passed to one custom layer."""
        return mx.sym.Custom(
            data=pred,
            label=self.label,
            op_type='BigRegressionOutput')

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


################
# MXNET LAYERS #
################


class BigRegressionOutput(mx.operator.CustomOp):

    def __init__(self, ctx):
        super(BigRegressionOutput, self).__init__()
        self.ctx = ctx

    def forward(self, is_train: bool, req, in_data: List, out_data: List, aux):
        """Forward predicted values. Apply sigmoid to labels for class."""
        pred, label = in_data[0], in_data[1].as_in_context(in_data[0].context)
        pred_bbox, pred_class, pred_score = self.split_block(pred)
        _, _, mask = self.split_block(label)

        pred_bbox *= mask

        pred_merged = self.merge_block((pred_bbox, pred_class, pred_score))

        self.assign(out_data[0], req[0], pred_merged)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Evaluate gradient for mean-squared error."""
        pred, label = in_data[0], in_data[1].as_in_context(in_data[0].context)

        pred_bbox, pred_class, pred_score = self.split_block(pred)
        label_bbox, label_class, mask = self.split_block(label)

        grad_bbox = self.backward_bbox(pred_bbox, label_bbox, mask)
        grad_class = pred_class * 0
        grad_score = pred_score * 0
        gradient = self.merge_block((grad_bbox, grad_class, grad_score))

        self.assign(in_grad[0], req[0], gradient)

    @staticmethod
    def split_block(block: nd.array) -> Tuple[nd.array, nd.array, nd.array]:
        """Split up predicted block into bbox, class, and score chunks.

        1. Splits all anchors, even for grid cells.

        Specifically, converts from shape (b, NUM_OUT_CHANNELS...) to a list of
        (b, ANCHORS_PER_GRID, 1, ...). This is employed to keep blocks together,
        since MXNet does not support multi-dimensional slicing.

        2. Rejoins splits into predicted blocks for bbox, class, and score.
        """
        splits = nd.split(block, num_outputs=int(block.shape[1] / ANCHORS_PER_GRID))
        expanded_splits = [nd.expand_dims(split, axis=2) for split in splits]

        data_class = nd.concat(*expanded_splits[NUM_BBOX_ATTRS: NUM_BBOX_ATTRS + NUM_CLASSES], dim=2)
        data_bbox = nd.concat(*expanded_splits[:NUM_BBOX_ATTRS], dim=2)
        data_score = expanded_splits[-1]
        return data_bbox, data_class, data_score

    @staticmethod
    def merge_block(splits: Tuple[nd.array, nd.array, nd.array]) -> nd.array:
        """Merge splits from `split_block` back into one block."""
        shrunk_splits = []
        for split in splits:
            shrunk_split = nd.split(
                split, num_outputs=split.shape[2], axis=2, squeeze_axis=True)
            if split.shape[2] == 1:
                shrunk_split = [shrunk_split]
            shrunk_splits.extend(shrunk_split)
        return nd.concat(*shrunk_splits, dim=1)

    def backward_bbox(
            self,
            pred_bbox: nd.array,
            label_bbox: nd.array,
            mask: nd.array) -> nd.array:
        """Compute gradient for bbox values. Learning: 1e-7

        Mean-squared error between the bounding box values.
        """
        masked_pred_bbox = mask_with(pred_bbox, mask)
        return 2 * (masked_pred_bbox - label_bbox)


@mx.operator.register("BigRegressionOutput")
class BigRegressionOutputProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(BigRegressionOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        pred_shape = in_shape[0]
        label_shape = in_shape[1]
        return [pred_shape, label_shape], [pred_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return BigRegressionOutput(ctx)


################
# MXNET LOSSES #
################

def transform(x: nd.array) -> nd.array:
    return np.transpose(np.concatenate([
        np.expand_dims(split, 2)
        for split in np.split(x, x.shape[1] / ANCHORS_PER_GRID, 1)], 2), (2, 0, 1, 3, 4))


def bigMetric(label: nd.array, pred: nd.array) -> float:
    mask = transform(label[:, -ANCHORS_PER_GRID:, :, :])
    if np.sum(mask) == 0:
        return 0

    pred_bbox = transform(pred[:, :ANCHORS_PER_GRID * NUM_BBOX_ATTRS, :, :])
    label_bbox = transform(label[:, :ANCHORS_PER_GRID * NUM_BBOX_ATTRS, :, :])
    loss_bbox = ((pred_bbox - label_bbox) ** 2).sum() / mask.sum() * 1e2
    return loss_bbox