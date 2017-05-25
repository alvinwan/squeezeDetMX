"""Specify the SqueezeDet architecture in MXNet"""

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.symbol as sym
from .constants import NUM_OUT_CHANNELS
from .constants import ANCHORS_PER_GRID
from .constants import NUM_CLASSES
from .constants import NUM_BBOX_ATTRS
from .utils import Reader
from .utils import mean_squared_loss
from .utils import iou


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
        dropout11 = sym.Dropout(fire11, p=0.1, name='drop11')
        return sym.Convolution(
            dropout11, name='conv12', num_filter=NUM_OUT_CHANNELS,
            kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    def add_loss(self, net: sym.Variable):
        """Add loss functions.

        Below, we splice the network output accordingly to compute losses for
        the following:

            1. Class probabilities
            2. IOUS as "confidence scores"
            3. Bounding box attributes

        Below, the ugly splice functions are replacements for reshaping.
        Instead, split along a dimension into multiple chunks, and then
        restack the arrays in a consistent way.
        """
        losses = []
        num_splits = int(NUM_OUT_CHANNELS / ANCHORS_PER_GRID)
        net_splits = list(sym.split(net, num_outputs=num_splits))
        lbl_splits = list(sym.split(self.label, num_outputs=num_splits))

        # Compute loss for bounding box
        splice_bbox = lambda x: sym.concat(*sym.split(sym.concat(
            *x[:NUM_BBOX_ATTRS]), num_outputs=ANCHORS_PER_GRID), dim=0)
        pred_box, label_box = splice_bbox(net_splits), splice_bbox(lbl_splits)
        losses.append(sym.LinearRegressionOutput(data=pred_box, label=label_box))

        # Compute loss for class probabilities
        cidx = NUM_BBOX_ATTRS + NUM_CLASSES
        splice_class = lambda x: sym.concat(*sym.split(sym.concat(
            *x[NUM_BBOX_ATTRS:cidx], dim=1), num_outputs=ANCHORS_PER_GRID), dim=0)
        pred_class_probs = sym.softmax(splice_class(net_splits), axis=1)
        label_class_probs = splice_class(lbl_splits)
        losses.append(sym.LogisticRegressionOutput(
            data=pred_class_probs, label=label_class_probs))

        # Compute loss for confidence scores
        # TODO(Alvin): Fix IOU computation
        pred_score = net_splits[cidx]
        label_score = iou(pred_box, label_box)
        losses.append(mean_squared_loss(pred_score, label_score))

        # TODO(Alvin): Add BlockGrad
        return sym.add_n(*losses)

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
