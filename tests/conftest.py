import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def polygon_unet_model(device):
    from StarConvexSegmentation.PolygonUnet import PolygonUnet
    model = PolygonUnet(num_coordinates=8, num_classes=1, pretrained=True)
    return model.to(device)


@pytest.fixture
def star_polygon_model(device):
    from StarConvexSegmentation.StarPolygon import StarPolygon
    model = StarPolygon(patch_size=128,
                        num_coordinates=8,
                        num_classes=1,
                        pretrained=True,
                        device=device)
    return model


@pytest.fixture
def sample_image():
    return torch.randn(1, 3, 128, 128)


@pytest.fixture
def large_sample_image():
    return torch.randn(5, 3, 3000, 3000)
