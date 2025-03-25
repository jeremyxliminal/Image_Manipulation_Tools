import numpy as np
from PIL import Image
import torch
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Node Definition
class GrayscaleConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI expects IMAGE as input type
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_grayscale"
    CATEGORY = "Custom Nodes"

    # Grayscale Conversion Logic
    def convert_to_grayscale(self, image):
        # Convert Torch Tensor to NumPy Array
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # Convert RGB to Grayscale (weighted average for better accuracy)
        grayscale_image = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Stack grayscale image to match RGB format (ComfyUI requires 3-channel tensors)
        stacked_grayscale = np.stack([grayscale_image]*3, axis=-1)

        # Convert Back to Tensor for ComfyUI Compatibility
        output_tensor = torch.tensor(stacked_grayscale).unsqueeze(0) / 255.0

        return (output_tensor,)

# Register the Node
NODE_CLASS_MAPPINGS.update({
    "GrayscaleConverter": GrayscaleConverter
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "GrayscaleConverter": "🖤 Grayscale Converter"
})
