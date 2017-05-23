"""Tests the KITTI writing and reading utilities."""

import pytest

from typing import List

from squeezeDetMX.kitti import read_bboxes


@pytest.fixture
def label():
    return read_bboxes(open('data/006234.txt').read().splitlines())


def test_read_bbox(label: List[List[int]]):
    """Tests that bboxes are read correcty."""
    assert len(label) == 2, 'Insufficient number of bboxes.'