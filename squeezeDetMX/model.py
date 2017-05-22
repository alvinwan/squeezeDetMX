"""Specify the SqueezeDet architecture in MXNet"""

import mxnet as mx

from utils import NUM_OUT_CHANNELS
from utils import GRID_WIDTH
from utils import GRID_HEIGHT


class SqueezeDet:
    """Setup the original squeezeDet architecture"""

    def __init__(self):
        self.data = mx.sym.Variable('image')
        self.label = mx.sym.Variable('label')
        self.net = self.add_forward(self.data)
        self.error = self.add_interpretation(self.net)

    def add_forward(self, data: mx.sym.Variable):
        """Add neural network model."""
        conv1 = mx.sym.Convolution(
            data, name='conv1', num_filter=64, kernel=(3, 3), stride=(2, 2))
        relu1 = mx.sym.Activation(conv1, act_type='relu')
        pool1 = mx.sym.Pooling(relu1, pool_type='max', kernel=(3, 3), stride=(2, 2))
        fire2 = self._fire_layer('fire2', pool1, s1x1=16, e1x1=64, e3x3=64)
        fire3 = self._fire_layer('fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
        pool3 = mx.sym.Pooling(fire3, name='pool3', kernel=(3, 3), stride=(2, 2), pool_type='max')
        fire4 = self._fire_layer('fire4', pool3, s1x1=32, e1x1=128, e3x3=128)
        fire5 = self._fire_layer('fire5', fire4, s1x1=32, e1x1=128, e3x3=128)
        pool5 = mx.sym.Pooling(fire5, name='pool5', kernel=(3, 3), stride=(2, 2), pool_type='max')
        fire6 = self._fire_layer('fire6', pool5, s1x1=48, e1x1=192, e3x3=192)
        fire7 = self._fire_layer('fire7', fire6, s1x1=48, e1x1=192, e3x3=192)
        fire8 = self._fire_layer('fire8', fire7, s1x1=64, e1x1=256, e3x3=256)
        fire9 = self._fire_layer('fire9', fire8, s1x1=64, e1x1=256, e3x3=256)
        fire10 = self._fire_layer('fire10', fire9, s1x1=96, e1x1=384, e3x3=384)
        fire11 = self._fire_layer('fire11', fire10, s1x1=96, e1x1=384, e3x3=384)
        dropout11 = mx.sym.Dropout(fire11, p=0.1, name='drop11')
        return mx.sym.Convolution(
            dropout11, name='conv12', num_filter=NUM_OUT_CHANNELS,
            kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    def add_interpretation(self, net: mx.sym.Variable):
        """Add loss functions."""
        return mx.symbol.LogisticRegressionOutput(
            data=net, label=self.label, name='loss')

    def _fire_layer(
            self,
            name: str,
            inputs: mx.sym.Variable,
            s1x1: int,
            e1x1: int,
            e3x3: int,
            freeze: bool=False):
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
        sq1x1 = mx.sym.Convolution(
            inputs, name=name+'/s1x1', num_filter=s1x1, kernel=(1, 1), stride=(1, 1))
        relu1 = mx.sym.Activation(sq1x1, act_type='relu')
        ex1x1 = mx.sym.Convolution(
            sq1x1, name=name+'/e1x1', num_filter=e1x1, kernel=(1, 1), stride=(1, 1))
        relu2 = mx.sym.Activation(ex1x1, act_type='relu')
        ex3x3 = mx.sym.Convolution(
            sq1x1, name=name+'/e3x3', num_filter=e3x3, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        relu3 = mx.sym.Activation(ex3x3, act_type='relu')
        return mx.sym.Concat(relu2, relu3, dim=1, name=name+'/concat')
