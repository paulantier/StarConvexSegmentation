
from PolygonUnet import PolygonUnet
import numpy as np
import torch
import torch.nn.functional as F

class StarPolygon():
    def __init__(self, num_coordinates=8, num_classes=1, pretrained=True):
        self.model = PolygonUnet(num_coordinates=num_coordinates, num_classes=num_classes, pretrained=pretrained)

    def pad_image(image, patch_size=128, overlap=64):
        _, _, h, w = image.shape
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return padded_image

    def create_patches(image, patch_size=128, overlap=64):
        _, _, h, w = image.shape
        patches = []
        for i in range(0, h - patch_size + 1, patch_size - overlap):
            for j in range(0, w - patch_size + 1, patch_size - overlap):
                patch = image[:, :, i:i + patch_size, j:j + patch_size]
                patches.append(patch)
        return patches

        
    def forward(self, x):
        
