"""Main training script for squeezeDet

Usage:
    train.py [options]

Options:
    --data=<path>           Path to root directory of data. [default: ./data/KITTI]
    --batch_size=<size>     Number of samples in a single batch [default: 20]
"""

import docopt
import mxnet as mx
import os.path
import time

from squeezeDetMX.model import SqueezeDet
from squeezeDetMX.utils import Reader
from squeezeDetMX.utils import build_module
from squeezeDetMX.utils import setup_logger


def main():
    setup_logger()
    arguments = docopt.docopt(__doc__)
    data_root = arguments['--data']
    batch_size = int(arguments['--batch_size'])

    train_path = os.path.join(data_root, 'train.brick')
    train_iter = Reader(train_path, batch_size=batch_size)

    val_path = os.path.join(data_root, 'val.brick')
    val_iter = Reader(val_path, batch_size=batch_size)
    pre_iter = mx.io.PrefetchingIter([train_iter])

    model = SqueezeDet()
    module = build_module(model.error, 'squeezeDetMX', train_iter)

    try:
        module.fit(
            train_data=pre_iter,
            eval_data=val_iter,
            num_epoch=50,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            eval_metric='acc',
            epoch_end_callback=mx.callback.do_checkpoint('squeezeDetMX', 1))
    except KeyboardInterrupt:
        module.save_params('squeezeDet-{}-9999.params'.format(
            str(time.time())[-5:]))


if __name__ == '__main__':
    main()
