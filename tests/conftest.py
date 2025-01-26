import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def polygon_unet_model(device):
    from StarConvexSegmentation.PolygonUnet import PolygonUnet
    model = PolygonUnet(pretrained=True)
    return model.to(device)


@pytest.fixture
def star_polygon_model(device):
    from StarConvexSegmentation.StarPolygon import StarPolygon
    model = StarPolygon(pretrained=True,
                        device=device)
    return model


@pytest.fixture
def sample_image():
    return torch.randn(1, 3, 128, 128)


@pytest.fixture
def large_sample_image():
    return torch.randn(5, 3, 3000, 3000)
