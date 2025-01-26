"""Class module containing the StarPolygon pipeline definition."""

import torch
import torch.nn.functional as F
from StarConvexSegmentation.PolygonUnet import PolygonUnet
from StarConvexSegmentation.StarConvexObject import StarConvexObject
from StarConvexSegmentation.config import NUM_COORDINATES, NUM_CLASSES, PATCH_SIZE


class StarPolygon:
    """Class containing the StarPolygon pipeline definition."""

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.featuremap_depth = NUM_COORDINATES + 2 + 1 + NUM_CLASSES
        # Number of output channels for the PolygonUnet model :
        # NUM_COORDINATES + 2 is for center x and y and vertexes distance to center
        # The + 1 is for objectness score

        

        self.polygon_model = PolygonUnet(
            pretrained=pretrained,
        ).to(device)

    def _pad_image(
        self,
        _image,
    ):

        _, _, h, w = _image.shape
        pad_h = (PATCH_SIZE - (h % PATCH_SIZE)) % PATCH_SIZE
        pad_w = (PATCH_SIZE - (w % PATCH_SIZE)) % PATCH_SIZE
        padded_image = F.pad(_image, (0, pad_w, 0, pad_h),
                             mode="constant",
                             value=0)
        return padded_image

    def _create_patches(
        self,
        _image,
    ):

        overlap = PATCH_SIZE // 2
        _, _, h, w = _image.shape
        patches = []
        for i in range(0, h - PATCH_SIZE + 1, PATCH_SIZE - overlap):
            for j in range(0, w - PATCH_SIZE + 1,
                           PATCH_SIZE - overlap):
                patch = _image[:, :, i:i + PATCH_SIZE,
                               j:j + PATCH_SIZE]
                patches.append(patch)
        return patches

    def _extract_star_convex_objects(self, feature_map):
        """
        Extract star-convex objects from the full feature map.
        
        Args:
            feature_map (torch.Tensor): Full feature map of shape [1, num_channels, H, W].
        
        Returns:
            List[StarConvexObject]: List of star-convex objects.
        """
        objects = []
        
        # Assuming feature_map shape is [1, num_channels, H, W]
        _, num_channels, H, W = feature_map.shape
        
        # Extract all properties at once using tensor operations
        center_x = feature_map[0, 0, :, :]  # Shape: [H, W]
        center_y = feature_map[0, 1, :, :]  # Shape: [H, W]
        vertex_distances = feature_map[0, 2:2+NUM_COORDINATES, :, :]  # Shape: [NUM_COORDINATES, H, W]
        objectness_score = feature_map[0, 2+NUM_COORDINATES, :, :]  # Shape: [H, W]
        class_scores = feature_map[0, 2+NUM_COORDINATES+1:, :, :]  # Shape: [NUM_CLASSES, H, W]
        
        # Get class labels by taking argmax over class scores
        class_labels = torch.argmax(class_scores, dim=0)  # Shape: [H, W]
        
        # Iterate over all pixels in the feature map
        for y in range(H):
            for x in range(W):
                # Create a StarConvexObject for each pixel
                obj = StarConvexObject(
                    center_x=center_x[y, x].item() + x,  # Add x offset for global coordinates
                    center_y=center_y[y, x].item() + y,  # Add y offset for global coordinates
                    vertex_distances=vertex_distances[:, y, x].cpu().numpy(),  # Convert to numpy array
                    objectness_score=objectness_score[y, x].item(),
                    class_label=class_labels[y, x].item(),
                )
                objects.append(obj)
        
        return objects
    
    def _apply_nms(self, objects, image_size):
        """
        Apply Non-Maximum Suppression (NMS) to merge overlapping objects.
        
        Args:
            objects (List[StarConvexObject]): List of star-convex objects.
            image_size (tuple): Original image size (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Final segmentation map.
        """
        b, _, h, w = image_size
        final_segmentation_map = torch.zeros((b, NUM_CLASSES, h, w), dtype=torch.float32, device=self.device)

        # Sort objects by objectness score in descending order
        sorted_objects = sorted(objects, key=lambda x: x.objectness_score, reverse=True)

        # Iterate over objects and add them to the segmentation map
        for obj in sorted_objects:
            x, y = int(obj.center_x), int(obj.center_y)
            
            # Check if the object is within the image bounds
            if 0 <= x < w and 0 <= y < h:
                # Check if the pixel is already occupied by a higher-confidence object
                if final_segmentation_map[0, obj.class_label, y, x] == 0:
                    final_segmentation_map[0, obj.class_label, y, x] = 1
        
        return final_segmentation_map
    
    def _process_image_with_offset(self, image, offset_x, offset_y):
        """
        Process the image with a given offset and return the feature map.
        
        Args:
            image (torch.Tensor): Input image of shape [batch_size, channels, height, width].
            offset_x (int): Horizontal offset for patch extraction.
            offset_y (int): Vertical offset for patch extraction.
        
        Returns:
            torch.Tensor: Feature map of shape [batch_size, featuremap_depth, height, width].
        """
        b, _, h, w = image.shape
        feature_map = torch.zeros(
            (b, self.featuremap_depth, h, w), 
            dtype=image.dtype, 
            device=image.device,
        )

        # Extract patches with the given offset
        patches = []
        for i in range(offset_y, h - PATCH_SIZE + 1, PATCH_SIZE):
            for j in range(offset_x, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = image[:, :, i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                patches.append(patch)

        # Process patches through the model
        processed_patches = []
        for patch in patches:
            with torch.no_grad():
                processed_patch = self.polygon_model(patch.to(self.device))
                processed_patches.append(processed_patch.to("cpu"))

        # Assemble patches into the feature map
        patch_idx = 0
        for i in range(offset_y, h - PATCH_SIZE + 1, PATCH_SIZE):
            for j in range(offset_x, w - PATCH_SIZE + 1, PATCH_SIZE):
                feature_map[:, :, i:i+PATCH_SIZE, j:j+PATCH_SIZE] = processed_patches[patch_idx]
                patch_idx += 1

        return feature_map

    def _assemble_patches(self, image):
        """
        Assemble patches from the image by processing it with and without offsets.
        
        Args:
            image (torch.Tensor): Input image of shape [batch_size, channels, height, width].
        
        Returns:
            torch.Tensor: Final feature map of shape [batch_size, featuremap_depth, height, width].
        """
        overlap = PATCH_SIZE // 2

        # Process the image without any offset (first pass)
        feature_map_no_offset = self._process_image_with_offset(image, offset_x=0, offset_y=0)

        # Process the image with an offset of overlap size (second pass)
        feature_map_with_offset = self._process_image_with_offset(image, offset_x=overlap, offset_y=overlap)

        # Combine the two feature maps
        final_feature_map = torch.zeros_like(feature_map_no_offset)
        final_feature_map[:, :, :overlap, :] = feature_map_no_offset[:, :, :overlap, :]  # Top rows
        final_feature_map[:, :, :, :overlap] = feature_map_no_offset[:, :, :, :overlap]  # Left columns
        final_feature_map[:, :, overlap:, overlap:] = feature_map_with_offset[:, :, overlap:, overlap:]  # Bottom-right region

        return final_feature_map

    def __call__(self, full_image):
        padded_image = self._pad_image(full_image)
        final_feature_map = self._assemble_patches(padded_image)
        return final_feature_map[:, :, :full_image.shape[2], :full_image.shape[3]]


if __name__ == "__main__":

    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.randn(5, 3, 3000, 3000, dtype=torch.float32)

    # Initialize the StarPolygon model
    star_polygon = StarPolygon(
        pretrained=True,
        device=main_device,
    )

    # Run the forward pass
    output_image = star_polygon(image)

    print("entrée :", image.shape)
    print("sortie :", output_image.shape)
