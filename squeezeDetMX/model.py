"""Specify the SqueezeDet architecture in MXNet"""

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.symbol as sym
import numpy as np
from sklearn import metrics
from .constants import NUM_OUT_CHANNELS
from .constants import EPSILON
from .constants import ANCHORS_PER_GRID
from .constants import NUM_CLASSES
from .constants import NUM_BBOX_ATTRS
from .utils import nd_batch_iou
from .utils import batches_iou
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
        net = sym.Convolution(
            data, name='conv1', num_filter=64, kernel=(3, 3), stride=(2, 2))
        net = sym.Activation(net, act_type='relu')
        net = sym.Pooling(net, pool_type='max', kernel=(3, 3), stride=(2, 2))
        net = self._fire_layer('fire2', net, s1x1=16, e1x1=64, e3x3=64)
        net = self._fire_layer('fire3', net, s1x1=16, e1x1=64, e3x3=64)
        net = sym.Pooling(net, name='pool3', kernel=(3, 3), stride=(2, 2), pool_type='max')
        net = self._fire_layer('fire4', net, s1x1=32, e1x1=128, e3x3=128)
        net = self._fire_layer('fire5', net, s1x1=32, e1x1=128, e3x3=128)
        net = sym.Pooling(net, name='pool5', kernel=(3, 3), stride=(2, 2), pool_type='max')
        net = self._fire_layer('fire6', net, s1x1=48, e1x1=192, e3x3=192)
        net = self._fire_layer('fire7', net, s1x1=48, e1x1=192, e3x3=192)
        net = self._fire_layer('fire8', net, s1x1=64, e1x1=256, e3x3=256)
        net = self._fire_layer('fire9', net, s1x1=64, e1x1=256, e3x3=256)
        net = self._fire_layer('fire10', net, s1x1=96, e1x1=384, e3x3=384)
        net = self._fire_layer('fire11', net, s1x1=96, e1x1=384, e3x3=384)
        net = sym.Dropout(net, p=0.5, name='drop11')
        return sym.Convolution(
            net, name='conv12', num_filter=NUM_OUT_CHANNELS,
            kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    def add_loss(self, pred: sym.Variable):
        """Add loss. To save trouble, all passed to one custom layer."""
        return mx.sym.Custom(
            data=pred,
            label=self.label,
            op_type='BigRegressionOutput')

        # pred_bbox, pred_class, pred_score = BigRegressionOutput.split_block(pred, sym, sizes={'image': (64, 3, 1242, 375)})
        # label_bbox, label_class, label_score = BigRegressionOutput.split_block(self.label, sym, sizes={'label': (64, 72, 22, 76)})
        #
        # label_class *= 0
        # pred_class = sym.LogisticRegressionOutput(data=pred_class, label=label_class)
        #
        # return pred_class


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

    # def forward(self, is_train: bool, req, in_data: List, out_data: List, aux):
    #     """Forward predicted values.
    #
    #     1. Apply sigmoid to labels for class.
    #     2. Apply input mask on all predicted values.
    #     """
    #     pred, label = in_data[0], in_data[1].as_in_context(in_data[0].context)
    #
    #     label_bbox, label_class, mask = self.split_block(label)
    #     pred_bbox, pred_class, pred_score = map(
    #         lambda p: p * mask, self.split_block(pred))
    #
    #     # WHY ARE THERE NANS?
    #     assert not np.isnan(pred.asnumpy()).any()
    #     assert not np.isnan(pred_bbox.asnumpy()).any()
    #     assert not np.isnan(pred_class.asnumpy()).any()
    #     assert not np.isnan(pred_score.asnumpy()).any()
    #
    #     # pred_bbox = nd.LinearRegressionOutput(data=pred_bbox, label=label_bbox)
    #
    #     # label_score = nd.transpose(nd_batch_iou(
    #     #     nd.transpose(pred_bbox, axes=(2, 0, 1, 3, 4)),
    #     #     nd.transpose(label_bbox, axes=(2, 0, 1, 3, 4))), axes=(1, 2, 0, 3, 4))
    #     # label_score *= 0
    #     # pred_score = nd.LinearRegressionOutput(data=pred_score, label=label_score)
    #
    #     num_nans = np.isnan(pred_class.asnumpy()).sum()
    #     if num_nans > 0:
    #         print(num_nans)
    #         import pdb
    #         pdb.set_trace()
    #
    #     label_class *= 0
    #     pred_class = nd.LogisticRegressionOutput(data=pred_class, label=label_class)
    #
    #     pred_merged = self.merge_block((pred_bbox, pred_class, pred_score))
    #     self.assign(out_data[0], req[0], pred_merged)
    #
    # def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    #     # pred, label = in_data[0], in_data[1].as_in_context(in_data[0].context)
    #     # , label_class, __ = self.split_block(label)
    #     #
    #     # import pdb
    #     # pdb.set_trace()
    #     # gradient =
    #     self.assign(in_grad[0], req[0], in_data[0])

    @staticmethod
    def split_block(block: nd.array, pkg=nd, sizes={}) -> Tuple[nd.array, nd.array, nd.array]:
        """Split up predicted block into bbox, class, and score chunks.

        1. Splits all anchors, even for grid cells.

        Specifically, converts from shape (b, NUM_OUT_CHANNELS...) to a list of
        (b, ANCHORS_PER_GRID, 1, ...). This is employed to keep blocks together,
        since MXNet does not support multi-dimensional slicing.

        2. Rejoins splits into predicted blocks for bbox, class, and score.
        """
        splits = pkg.split(block, num_outputs=int(block.shape[1] / ANCHORS_PER_GRID))
        # block_shape = block.infer_shape(**sizes)[-2][0]
        splits = pkg.split(block, num_outputs=int(block_shape[1] / ANCHORS_PER_GRID))
        expanded_splits = [pkg.expand_dims(split, axis=2) for split in splits]

        data_class = pkg.concat(*expanded_splits[NUM_BBOX_ATTRS: NUM_BBOX_ATTRS + NUM_CLASSES], dim=2)
        data_bbox = pkg.concat(*expanded_splits[:NUM_BBOX_ATTRS], dim=2)
        data_score = expanded_splits[-1]
        return data_bbox, data_class, data_score

    @staticmethod
    def merge_block(splits: Tuple[nd.array, nd.array, nd.array], pkg=nd) -> nd.array:
        """Merge splits from `split_block` back into one block."""
        shrunk_splits = []
        for split in splits:
            # if 'label' not in split.list_arguments():
            #     sizes = {'image': (64, 3, 1242, 375)}
            # else:
            #     sizes = {'image': (64, 3, 1242, 375), 'label': (64, 72, 22, 76)}
            # split_shape = split.infer_shape(**sizes)[-2][0]
            shrunk_split = pkg.split(
                split, num_outputs=split.shape[2], axis=2, squeeze_axis=True)
                # split, num_outputs=split_shape[2], axis=2, squeeze_axis=True)
            if split.shape[2] == 1:
                shrunk_split = [shrunk_split]
            shrunk_splits.extend(shrunk_split)
        return pkg.concat(*shrunk_splits, dim=1)


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
    """Include metrics for all three losses: bbox, class, and score"""
    mask = transform(label[:, -ANCHORS_PER_GRID:, :, :])
    if np.sum(mask) == 0:
        return 0

    # pred_bbox = transform(pred[:, :ANCHORS_PER_GRID * NUM_BBOX_ATTRS, :, :])
    # label_bbox = transform(label[:, :ANCHORS_PER_GRID * NUM_BBOX_ATTRS, :, :])
    # loss_bbox = ((pred_bbox - label_bbox) ** 2).sum() / mask.sum() * 1e2

    st = ANCHORS_PER_GRID * NUM_BBOX_ATTRS
    label_class = np.argmax(  # n x 1
        transform(label[:, st: st + ANCHORS_PER_GRID * NUM_CLASSES, :, :])
        .reshape((NUM_CLASSES, -1)), axis=0)
    pred_class = np.argmax((transform(  # n x 3
        pred[:, st: st + ANCHORS_PER_GRID * NUM_CLASSES, :, :]) * mask)\
        .reshape((NUM_CLASSES, -1)), axis=0)
    loss_class = (pred_class - label_class > 0).sum() / label_class.shape[0]
    #
    # pred_score = transform(pred[:, -ANCHORS_PER_GRID:, :, :])
    # # label_score = transform(batches_iou(pred_bbox, label_bbox))
    # # loss_score = ((pred_score - label_score) ** 2).sum() / mask.sum() * 1e3

    return loss_class * 1e5

    # pred_class = (np.transpose(pred, (2, 0, 1, 3, 4)) * mask).reshape((NUM_CLASSES, -1)).T
    # loss_class = metrics.log_loss(label_class, pred_class, labels=tuple(range(NUM_CLASSES)))