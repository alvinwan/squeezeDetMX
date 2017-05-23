"""Tests the KITTI writing and reading utilities."""

import cv2
import mxnet as mx
import numpy as np
import pytest
import os
import os.path
import shutil

from typing import List

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


@pytest.fixture
def reader(image: np.array, label: List[int]) -> KITTIIter:
    tmp_path = './tmp/tmp.bin'
    with KITTIWriter(tmp_path) as writer:
        writer.write([image] * 3, [label] * 3)
    return KITTIIter(tmp_path, batch_size=1)


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
    bytedata = next(KITTIWriter.byteIter([image], [label]))
    with KITTIIter.from_bytes(bytedata) as reader:
        image_reconstructed = reader.read_image()
    assert_images_equal(image, image_reconstructed, 'String formatting faulty.')


def test_image_multiple_e2e(image: np.array, reader: KITTIIter):
    """Test that byte data was correctly formatted and parsed."""
    datum = reader.next()
    image_reconstructed = np.transpose(reader.read_image(), axes=(2, 0, 1))
    image = datum.data[0].asnumpy().reshape(image_reconstructed.shape)
    assert np.allclose(image, image_reconstructed), 'String formatting faulty.'


def test_image_e2e(image: np.array, reader: KITTIIter):
    """Test that the images were preserved by the custom format."""
    image_reconstructed = reader.read_image()
    assert_images_equal(image, image_reconstructed, 'File format faulty.')


def assert_images_equal(image1: np.array, image2: np.array, msg: str):
    """Assert that two images are equal."""
    average_difference = np.sum(image1 - image2) / np.prod(image1.shape)
    assert average_difference < 110, msg


def test_mx_image_format(reader: KITTIIter):
    mx_image = KITTIIter.image_to_mx(reader.read_image())
    assert mx_image.shape == (3, 375, 1242), 'Shape mismatch.'


#################
# LABEL PARSING #
#################


def test_label_byte_iter(image: np.array, label: List[int]):
    """Test that byte data was correctly formatted and parsed."""
    bytedata = next(KITTIWriter.byteIter([image], [label]))
    with KITTIIter.from_bytes(bytedata) as reader:
        _ = reader.read_image()
        label_reconstructed = reader.read_label()
    assert np.allclose(label, label_reconstructed), 'String formatting faulty.'


def test_label_e2e_write_read(image: np.array, label: List[int], reader: KITTIIter):
    """Test that the labels were preserved by the custom format."""
    _ = reader.read_image()
    label_reconstructed = reader.read_label()
    assert np.allclose(label, label_reconstructed), 'File format faulty.'


def test_label_count_byte_iter(reader: KITTIIter):
    """Test that byte data was correctly formatted and parsed."""
    assert len(list(reader)) == 3, 'Premature termination'
