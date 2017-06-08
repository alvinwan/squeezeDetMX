"""Main training script for squeezeDet

Usage:
    train.py [options]

Options:
    --data=<path>           Path to root directory of data. [default: ./data/KITTI]
    --batch_size=<size>     Number of samples in a single batch [default: 20]
    --learning_rate=<lr>    Learning rate for neural net [default: 1e-7]
    --momentum=<mom>        Momentum for neural net [default: 0.9]
    --n_epochs=<n>          Number of epochs to run for [default: 50]
"""

import docopt
import mxnet as mx
import os.path
import numpy as np
import time

from mxnet import metric

from squeezeDetMX.model import SqueezeDet
from squeezeDetMX.utils import Reader
from squeezeDetMX.utils import build_module
from squeezeDetMX.utils import setup_logger
from squeezeDetMX.model import bigMetric


def main():
    setup_logger()
    arguments = docopt.docopt(__doc__)
    data_root = arguments['--data']
    batch_size = int(arguments['--batch_size'])
    learning_rate = float(arguments['--learning_rate'])
    momentum = float(arguments['--momentum'])
    num_epochs = int(arguments['--n_epochs'])

    train_path = os.path.join(data_root, 'train.brick')
    train_iter = Reader(train_path, batch_size=batch_size)

    val_path = os.path.join(data_root, 'val.brick')
    val_iter = Reader(val_path, batch_size=batch_size)
    pre_iter = mx.io.PrefetchingIter([train_iter])

    model = SqueezeDet()
    module = build_module(model.error, 'squeezeDetMX', train_iter,
                          learning_rate=learning_rate,
                          momentum=momentum,
                          ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)])

    np.seterr('raise')

    try:
        module.fit(
            train_data=pre_iter,
            eval_data=val_iter,
            num_epoch=num_epochs,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            eval_metric=mx.metric.create(bigMetric),
            epoch_end_callback=mx.callback.do_checkpoint('squeezeDetMX', 1))
    except KeyboardInterrupt:
        module.save_params('squeezeDet-{}-9999.params'.format(
            str(time.time())[-5:]))


if __name__ == '__main__':
    main()
