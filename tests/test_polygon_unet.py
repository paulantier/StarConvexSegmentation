import torch
import pytest


def test_polygon_unet_initialization(polygon_unet_model):
    assert polygon_unet_model is not None
    assert isinstance(polygon_unet_model, torch.nn.Module)


def test_polygon_unet_forward_pass(polygon_unet_model, sample_image, device):
    sample_image = sample_image.to(device)
    output = polygon_unet_model(sample_image)
    assert output is not None
    assert output.shape == (1, 10, 128, 128)  # 8 coordinates + 1 + 1 class


def test_polygon_unet_decoder_block(polygon_unet_model):
    decoder_block = polygon_unet_model._decoder_block(512, 256)
    assert isinstance(decoder_block, torch.nn.Sequential)
    assert len(
        decoder_block) == 7  # 2 conv layers, 2 batchnorm, 2 relu, 1 upsample


def test_polygon_unet_encoder_layers(polygon_unet_model):
    assert isinstance(polygon_unet_model.enc1, torch.nn.Sequential)
    assert isinstance(polygon_unet_model.enc2, torch.nn.Sequential)
    assert isinstance(polygon_unet_model.enc3, torch.nn.Sequential)
    assert isinstance(polygon_unet_model.enc4, torch.nn.Sequential)
    assert isinstance(polygon_unet_model.enc5, torch.nn.Sequential)
