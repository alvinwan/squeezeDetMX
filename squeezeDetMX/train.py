"""Main training script for squeezeDet
Usage:
    train.py [options]

Options:
    --data=<path>           Path to root directory of data. [default: ../data/KITTI]
    --batch_size=<size>     Number of samples in a single batch [default: 20]
"""

import docopt
import mxnet as mx
import numpy as np
import os.path

from model import SqueezeDet
from kitti import KITTIIter
from utils import build_module

def main():
    arguments = docopt.docopt(__doc__)
    data_root = arguments['--data']
    batch_size = int(arguments['--batch_size'])

    train_iter = KITTIIter(os.path.join(data_root, 'train.brick'))
    val_iter = KITTIIter(os.path.join(data_root, 'trainval.brick'))
    pre_iter = mx.io.PrefetchingIter([train_iter])

    model = SqueezeDet()
    module = build_module(model.error, 'squeezeDetMX', train_iter)

    try:
        module.fit(train_data=pre_iter, eval_data=val_iter, num_epoch=50,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            eval_metric='acc',
            epoch_end_callback=mx.callback.do_checkpoint('squeezeDetMX', 1))
    except KeyboardInterrupt:
        module.save_params('{}-9999.params'.format(args.module_name))

if __name__ == '__main__':
    main()
