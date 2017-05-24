"""Reads raw KITTI data and preprocesses data for RecordIO.

Usage:
    convert.py [options]

Options:
    --data=<path>       Root folder containing all data. [default: ./data/KITTI]
    --out=<dir>         Root of directory containing all outputs [default: ./data/KITTI]
"""

import docopt
import os.path

from squeezeDetMX.kitti import grab_images_labels
from squeezeDetMX.utils import Writer


def main():
    """Translating KITTI data into RecordIO"""
    arguments = docopt.docopt(__doc__)
    data_root = arguments['--data']
    out_root = arguments['--out']

    X_train, Y_train = grab_images_labels(data_root, 'train')
    X_val, Y_val = grab_images_labels(data_root, 'trainval')

    train_writer = Writer(os.path.join(out_root, 'train.brick'))
    train_writer.write(X_train, Y_train)
    train_writer.close()
    print(' * Finished writing train.')

    val_writer = Writer(os.path.join(out_root, 'trainval.brick'))
    val_writer.write(X_val, Y_val)
    val_writer.close()
    print(' * Finished writing trainval.')


if __name__ == '__main__':
    main()
