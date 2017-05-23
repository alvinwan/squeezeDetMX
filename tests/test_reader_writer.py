"""Test writing and reading utilities for custom format."""

import cv2
import numpy as np
import pytest
import os
import os.path
import shutil

from typing import List

from squeezeDetMX.kitti import read_bboxes
from squeezeDetMX.utils import Writer
from squeezeDetMX.utils import Reader
from squeezeDetMX.utils import image_to_jpeg_bytes
from squeezeDetMX.utils import jpeg_bytes_to_image
from squeezeDetMX.constants import IMAGE_HEIGHT
from squeezeDetMX.constants import IMAGE_WIDTH
from squeezeDetMX.constants import GRID_WIDTH
from squeezeDetMX.constants import GRID_HEIGHT
from squeezeDetMX.constants import NUM_OUT_CHANNELS


@pytest.fixture
def image():
    return cv2.imread('data/006234.png')


@pytest.fixture
def label():
    return read_bboxes(open('data/006234.txt').read().splitlines())


@pytest.fixture
def reader(image: np.array, label: List[int]) -> Reader:
    tmp_path = './tmp/tmp.bin'
    with Writer(tmp_path) as writer:
        writer.write([image] * 3, [label] * 3)
    return Reader(tmp_path, batch_size=1)


def setup_module(module):
    os.makedirs('./tmp', exist_ok=True)


def teardown_module(module):
    shutil.rmtree('./tmp')


#################
# IMAGE PARSING #
#################


def test_image_byte_conversion(image: np.array):
    """Test byte conversion from utilities file.

    Note: The images, when drawn, look perfectly fine, but the average
    differences are huge, numerically.
    """
    image_bytes = image_to_jpeg_bytes(image)
    image_reconstructed = jpeg_bytes_to_image(image_bytes)
    assert_images_equal(image, image_reconstructed, 'Byte conversion faulty.')


def test_image_byte_iter(image: np.array, label: List[int]):
    """Test that byte data was correctly formatted and parsed."""
    bytedata = next(Writer.byteIter([image], [label]))
    with Reader.from_bytes(bytedata) as reader:
        image_reconstructed = reader.read_image()
    assert_images_equal(image, image_reconstructed, 'String formatting faulty.')


def test_image_multiple_write_read(image: np.array, reader: Reader):
    """Test that byte data was correctly formatted and parsed."""
    datum = reader.next()
    image_reconstructed = np.transpose(reader.read_image(), axes=(2, 0, 1))
    image = datum.data[0].asnumpy().reshape(image_reconstructed.shape)
    assert np.allclose(image, image_reconstructed), 'String formatting faulty.'


def test_image_write_read(image: np.array, reader: Reader):
    """Test that the images were preserved by the custom format."""
    image_reconstructed = reader.read_image()
    assert_images_equal(image, image_reconstructed, 'File format faulty.')


def assert_images_equal(image1: np.array, image2: np.array, msg: str):
    """Assert that two images are equal."""
    average_difference = np.sum(image1 - image2) / np.prod(image1.shape)
    assert average_difference < 110, msg


def test_mx_image_format(reader: Reader):
    mx_image = Reader.image_to_mx(reader.read_image())
    assert mx_image.shape == (3, 375, 1242), 'Shape mismatch.'


#################
# LABEL PARSING #
#################


def test_label_byte_iter(image: np.array, label: List[int]):
    """Test that byte data was correctly formatted and parsed."""
    bytedata = next(Writer.byteIter([image], [label]))
    with Reader.from_bytes(bytedata) as reader:
        _ = reader.read_image()
        label_reconstructed = reader.read_label()
    assert np.allclose(label, label_reconstructed), 'String formatting faulty.'


def test_label_write_read(image: np.array, label: List[int], reader: Reader):
    """Test that the labels were preserved by the custom format."""
    _ = reader.read_image()
    label_reconstructed = reader.read_label()
    assert np.allclose(label, label_reconstructed), 'File format faulty.'


def test_label_count_byte_iter(reader: Reader):
    """Test that byte data was correctly formatted and parsed."""
    data = list(reader)
    label = data[0].label[0]
    assert len(data) == 3, 'Premature termination'


############
# DATAITER #
############


def test_dataiter_package_dims(reader: Reader):
    """Tests that dataiter gives data and labels with correct dimensions."""
    datum = reader.next()
    image, label = datum.data[0], datum.label[0]
    assert image.shape == (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), 'Image wrong size'
    assert label.shape == (1, NUM_OUT_CHANNELS, GRID_HEIGHT, GRID_WIDTH), 'Label wrong size'
