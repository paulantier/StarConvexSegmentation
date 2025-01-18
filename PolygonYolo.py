import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block with Conv2d, BatchNorm, and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        return x

class PolygonYolo(nn.Module):
    def __init__(self, in_channels, num_coordinates, num_classes):
        super(PolygonYolo, self).__init__()
        self.num_coordinates = num_coordinates
        self.num_classes = num_classes

        # Initial layers
        self.layer1 = ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)

        # Residual Block 1
        self.res1_conv1 = ConvBlock(64, 32, kernel_size=1, stride=1, padding=0)
        self.res1_conv2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)

        # Downscaling and skip connections
        self.layer3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)

        # Residual Block 2
        self.res2_conv1 = ConvBlock(128, 64, kernel_size=1, stride=1, padding=0)
        self.res2_conv2 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)

        # Skip Connection Layer
        self.skip1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        # Additional layers
        self.layer4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)

        # Skip Connection Layer
        self.skip2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)

        # Intermediate layer
        self.intermediate_layer = ConvBlock(256 + 128 + 64, 512, kernel_size=3, stride=1, padding=1)

        # Final prediction layers
        self.prediction_layer = nn.Conv2d(512, self.num_coordinates + 1 + self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Initial layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Residual Block 1
        res1 = self.res1_conv1(x)
        res1 = self.res1_conv2(res1)
        x = x + res1

        # Downscaling and skip connections
        x = self.layer3(x)

        # Residual Block 2
        res2 = self.res2_conv1(x)
        res2 = self.res2_conv2(res2)
        x = x + res2

        # First skip connection
        skip1_out = self.skip1(x)

        # Additional layers
        x = self.layer4(x)
        x = self.layer5(x)

        # Second skip connection
        skip2_out = self.skip2(x)

        # Concatenate skip connections and prediction
        x = torch.cat([x, skip2_out, skip1_out], dim=1)
        x = self.prediction_layer(x)

        return x

# Example usage
if __name__ == "__main__":
    num_classes = 1  # Example number of classes
    model = PolygonYolo(in_channels=3, num_classes=num_classes)
    input_tensor = torch.randn(1, 3, 416, 416)  # Example input
    output = model(input_tensor)
    print(output.shape)
