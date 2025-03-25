import numpy as np
from PIL import Image
import torch
from torch import nn
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # Correct import for custom nodes in ComfyUI

# Node Definition
class PixelIntensityModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),     
                "intensity": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 3.0 }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "modify_intensity"
    CATEGORY = "Custom Nodes"

    # Image Manipulation Logic
    def modify_intensity(self, image, intensity):
        # Convert Torch Tensor to NumPy Array
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # Apply Intensity Change
        modified_image = np.clip(image_np * intensity, 0, 255).astype(np.uint8)

        # Convert Back to Tensor for ComfyUI Compatibility
        modified_image = torch.tensor(modified_image).unsqueeze(0) / 255.0

        return (modified_image,)

# Register the Node
NODE_CLASS_MAPPINGS.update({
    "PixelIntensityModifier": PixelIntensityModifier
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "PixelIntensityModifier": "🖼️ Pixel Intensity Modifier"
})

