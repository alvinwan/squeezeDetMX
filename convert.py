"""Reads raw KITTI data and preprocesses data for RecordIO.

Usage:
    convert.py [options]

Options:
    --data=<path>       Root folder containing all data. [default: ../data/KITTI]
"""

import docopt

from squeezeDetMX.kitti import grab_images_labels
from squeezeDetMX.kitti import KITTIWriter
from squeezeDetMX.kitti import KITTIIter


def main():
    """Translating KITTI data into RecordIO"""
    arguments = docopt.docopt(__doc__)
    data_root = arguments['--data']

    X_train, Y_train = grab_images_labels(data_root, 'train')
    X_val, Y_val = grab_images_labels(data_root, 'trainval')

    train_writer = KITTIWriter('train.brick')
    train_writer.write(X_train, Y_train)
    train_writer.close()
    print(' * Finished writing train.')

    val_writer = KITTIWriter('trainval.brick')
    val_writer.write(X_val, Y_val)
    val_writer.close()
    print(' * Finished writing trainval.')


if __name__ == '__main__':
    main()
