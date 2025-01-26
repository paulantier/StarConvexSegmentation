"""Class module containing the StarPolygon pipeline definition."""

import torch
import torch.nn.functional as F
from StarConvexSegmentation.PolygonUnet import PolygonUnet


class StarPolygon:
    """Class containing the StarPolygon pipeline definition."""

    def __init__(
        self,
        patch_size: int = 128,
        num_coordinates: int = 8,
        num_classes: int = 1,
        pretrained: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.patch_size = patch_size
        self.device = device
        self.featuremap_depth = num_coordinates + 1 + num_classes
        # Number of output channels for the PolygonUnet model

        self.polygon_model = PolygonUnet(
            num_coordinates=num_coordinates,
            num_classes=num_classes,
            pretrained=pretrained,
        ).to(device)

    def _pad_image(
        self,
        _image,
    ):

        _, _, h, w = _image.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        padded_image = F.pad(_image, (0, pad_w, 0, pad_h),
                             mode="constant",
                             value=0)
        return padded_image

    def _create_patches(
        self,
        _image,
    ):

        overlap = self.patch_size // 2
        _, _, h, w = _image.shape
        patches = []
        for i in range(0, h - self.patch_size + 1, self.patch_size - overlap):
            for j in range(0, w - self.patch_size + 1,
                           self.patch_size - overlap):
                patch = _image[:, :, i:i + self.patch_size,
                              j:j + self.patch_size]
                patches.append(patch)
        return patches

    def _assemble_patches(self, patches, image_size: tuple):
        b, _, h, w = image_size

        overlap = self.patch_size // 2

        reconstructed_image = torch.zeros((b, self.featuremap_depth, h, w),
                                          dtype=patches[0].dtype, device=patches[0].device)

        patch_idx = 0
        for i in range(0, h - self.patch_size + 1, self.patch_size - overlap):
            for j in range(0, w - self.patch_size + 1,
                           self.patch_size - overlap):
                reconstructed_image[:, :, i:i + self.patch_size, j:j +
                                    self.patch_size] += patches[patch_idx]
                patch_idx += 1

        return reconstructed_image

    def __call__(self, full_image):
        padded_image = self._pad_image(full_image)
        patches = self._create_patches(padded_image)

        processed_patches = []
        for patch in patches:
            with torch.no_grad():
                processed_patch = self.polygon_model(patch.to(self.device))
                processed_patches.append(processed_patch.to("cpu"))

        reconstructed_image = self._assemble_patches(processed_patches,
                                                     full_image.shape)

        return reconstructed_image[:, :, :full_image.shape[2], :full_image.
                                   shape[3]]


if __name__ == "__main__":

    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.randn(5, 3, 3000, 3000, dtype=torch.float32)

    # Initialize the StarPolygon model
    star_polygon = StarPolygon(
        patch_size=256,
        num_coordinates=32,
        num_classes=1,
        pretrained=True,
        device=main_device,
    )

    # Run the forward pass
    output_image = star_polygon(image)

    print("entrée :", image.shape)
    print("sortie :", output_image.shape)
