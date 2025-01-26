import torch
import pytest


def test_star_polygon_initialization(star_polygon_model):
    assert star_polygon_model is not None
    assert star_polygon_model.polygon_model is not None


def test_star_polygon_pad_image(star_polygon_model, sample_image):
    padded_image = star_polygon_model._pad_image(sample_image)
    assert padded_image.shape[2] % star_polygon_model.patch_size == 0
    assert padded_image.shape[3] % star_polygon_model.patch_size == 0


def test_star_polygon_create_patches(star_polygon_model, sample_image):
    padded_image = star_polygon_model._pad_image(sample_image)
    patches = star_polygon_model._create_patches(padded_image)
    assert len(patches) > 0
    assert all(patch.shape == (1, 3, 128, 128) for patch in patches)


def test_star_polygon_assemble_patches(star_polygon_model, sample_image):
    padded_image = star_polygon_model._pad_image(sample_image)
    patches = star_polygon_model._create_patches(padded_image)

    # Process patches through the model to get the correct number of channels
    processed_patches = []
    for patch in patches:
        with torch.no_grad():
            processed_patch = star_polygon_model.polygon_model(
                patch.to(star_polygon_model.device))
            processed_patches.append(processed_patch.to("cpu"))

    assembled_image = star_polygon_model._assemble_patches(
        processed_patches, sample_image.shape)

    assert assembled_image.shape == (1, 10, 128, 128)


def test_star_polygon_forward_pass(star_polygon_model, large_sample_image):
    output_image = star_polygon_model(large_sample_image)
    assert output_image.shape == (5, 10, 3000, 3000)
