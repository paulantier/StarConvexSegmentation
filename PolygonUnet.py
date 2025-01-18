import torch
import torch.nn as nn
import torchvision.models as models

class PolygonUnet(nn.Module):
    def __init__(self, num_coordinates=1, num_classes=1, pretrained=True):
        super(PolygonUnet, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(pretrained=pretrained)

        # Encoder (ResNet-18 layers)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # Output: 64x64x64
        self.enc2 = resnet.layer1                                        # Output: 64x64x64
        self.enc3 = resnet.layer2                                        # Output: 128x32x32
        self.enc4 = resnet.layer3                                        # Output: 256x16x16
        self.enc5 = resnet.layer4                                        # Output: 512x8x8

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.dec4 = self._decoder_block(512, 256)  # Output: 256x16x16
        self.dec3 = self._decoder_block(256, 128)  # Output: 128x32x32
        self.dec2 = self._decoder_block(128, 64)   # Output: 64x64x64
        self.dec1 = self._decoder_block(64, 64)    # Output: 64x64x64

        # Stronger Convolution Block
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_coordinates + 1 + num_classes, kernel_size=1)  # Output: num_classesx128x128
        )


    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        # Encoder forward pass
        
        enc1 = self.enc1(x)
        print(f"enc1 shape: {enc1.shape}")  # 64x128x128
        enc2 = self.enc2(enc1)
        print(f"enc2 shape: {enc2.shape}")  # 64x64x64
        enc3 = self.enc3(enc2)
        print(f"enc3 shape: {enc3.shape}")  # 128x32x32
        enc4 = self.enc4(enc3)
        print(f"enc4 shape: {enc4.shape}")  # 256x16x16
        enc5 = self.enc5(enc4)
        print(f"enc5 shape: {enc5.shape}")  # 512x8x8

        # Decoder forward pass with skip connections
        dec4 = self.dec4(enc5) + enc4
        print(f"dec4 shape: {dec4.shape}")  # 256x16x16
        dec3 = self.dec3(dec4) + enc3
        print(f"dec3 shape: {dec3.shape}")  # 128x32x32
        dec2 = self.dec2(dec3) + enc2
        print(f"dec2 shape: {dec2.shape}")  # 64x64x64
        dec1 = self.dec1(dec2) + self.up(enc1)
        print(f"dec1 shape: {dec1.shape}")  # 64x128x128

        # Final convolution
        out = self.final_conv(dec1)
        print(f"out shape: {out.shape}")  # num_classesx128x128
        return out

# Example usage
if __name__ == "__main__":
    model = PolygonUnet(num_classes=1, pretrained=True)
    x = torch.randn(1, 3, 128, 128)  # Example input
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, num_classes, 128, 128]
