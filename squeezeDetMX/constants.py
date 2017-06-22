"""Constants for squeezeDetMX"""

CLASS_TO_INDEX = {
    'car': 0,
    'pedestrian': 1,
    'cyclist': 2
}

ANCHORS_PER_GRID = 9
NUM_CLASSES = len(list(CLASS_TO_INDEX.keys()))
NUM_BBOX_ATTRS = 4
NUM_CONF = 1
NUM_OUT_CHANNELS = ANCHORS_PER_GRID * (NUM_CLASSES + NUM_BBOX_ATTRS + NUM_CONF)
GRID_WIDTH = 76
GRID_HEIGHT = 22
IMAGE_WIDTH = 1248
IMAGE_HEIGHT = 384
NUM_ANCHORS = ANCHORS_PER_GRID * GRID_HEIGHT * GRID_WIDTH

# Custom binary format for KITTI dataset
IMAGE_BYTES_SLOT = 15
BBOXES_BYTES_SLOT = 11
BBOX_FORMAT = 'ffffi'

# Hardcoded numbers from original repository - will experiment with alternatives
RANDOM_WIDTHS_HEIGHTS = [
   [  36.,  37.], [ 366., 174.], [ 115.,  59.],
   [ 162.,  87.], [  38.,  90.], [ 258., 173.],
   [ 224., 108.], [  78., 170.], [  72.,  43.]]
