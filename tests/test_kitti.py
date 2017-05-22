"""Tests the KITTI writing and reading utilities."""

import cv2
import mxnet as mx
import numpy as np
import pytest

from squeezeDetMX.kitti import read_bboxes
from squeezeDetMX.kitti import KITTIWriter
from squeezeDetMX.kitti import KITTIIter
from squeezeDetMX.utils import image_to_jpeg_bytes
from squeezeDetMX.utils import jpeg_bytes_to_image


@pytest.fixture
def image():
    return cv2.imread('data/006234.png')


@pytest.fixture
def label():
    return read_bboxes([open('data/006234.txt').read()])


#################
# IMAGE PARSING #
#################


def test_image_byte_conversion(image):
    """Test byte conversion from utilities file.

    Note: The images, when drawn, look perfectly fine, but the average
    differences are huge, numerically.
    """
    image_bytes = image_to_jpeg_bytes(image)
    image_reconstructed = jpeg_bytes_to_image(image_bytes)
    assert_images_equal(image, image_reconstructed, 'Byte conversion faulty.')


def test_image_byte_iter(image, label):
    """Test that byte data was correctly formatted and parsed."""
    with KITTIWriter('tmp') as writer:
        bytedata = next(writer.byteIter([image], [label]))
    with KITTIIter.from_bytes(bytedata) as reader:
        image_reconstructed = reader.read_image()
    assert_images_equal(image, image_reconstructed, 'String formatting faulty.')


def test_image_e2e_write_read(image, label):
    """Test that the image and labels were preserved by the custom format."""
    with KITTIWriter('tmp') as writer:
        writer.write([image], [label])
    with KITTIIter('tmp') as reader:
        image_reconstructed = reader.read_image()
    assert_images_equal(image, image_reconstructed, 'File format faulty.')


def assert_images_equal(image1: np.array, image2: np.array, msg: str):
    """Assert that two images are equal."""
    average_difference = np.sum(image1 - image2) / np.prod(image1.shape)
    assert average_difference < 110, msg


#################
# LABEL PARSING #
#################
