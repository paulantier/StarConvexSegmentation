import torch
import pytest
from StarConvexSegmentation.config import NUM_COORDINATES, NUM_CLASSES, PATCH_SIZE


def test_star_polygon_initialization(star_polygon_model):
    assert star_polygon_model is not None
    assert star_polygon_model.polygon_model is not None


def test_star_polygon_pad_image(star_polygon_model, sample_image):
    padded_image = star_polygon_model._pad_image(sample_image)
    assert padded_image.shape[2] % PATCH_SIZE == 0
    assert padded_image.shape[3] % PATCH_SIZE == 0


def test_star_polygon_create_patches(star_polygon_model, sample_image):
    padded_image = star_polygon_model._pad_image(sample_image)
    patches = star_polygon_model._create_patches(padded_image)
    assert len(patches) > 0
    assert all(patch.shape == (1, 3, 128, 128) for patch in patches)


def test_star_polygon_assemble_patches(star_polygon_model, sample_image):
    padded_image = star_polygon_model._pad_image(sample_image)
    assembled_image = star_polygon_model._assemble_patches(padded_image)

    # Updated expected output shape: (batch_size, 2 + NUM_COORDINATES + 1 + NUM_CLASSES, height, width)
    # For NUM_COORDINATES=8 and NUM_CLASSES=1, the number of channels is 12
    assert assembled_image.shape == (1, 2 + NUM_COORDINATES + 1 + NUM_CLASSES, 128, 128)


def test_star_polygon_forward_pass(star_polygon_model, large_sample_image):
    output_image = star_polygon_model(large_sample_image)
    # Updated expected output shape: (batch_size, 2 + NUM_COORDINATES + 1 + NUM_CLASSES, height, width)
    # For NUM_COORDINATES=8 and NUM_CLASSES=1, the number of channels is 12
    assert output_image.shape == (5, 2 + NUM_COORDINATES + 1 + NUM_CLASSES, 3000, 3000)