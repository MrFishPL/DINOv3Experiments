"""Utilities for visualization tasks."""

from PIL import Image


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """
    Resize an image to a target height while maintaining aspect ratio.
    
    Args:
        img: Input PIL Image.
        target_h: Target height in pixels.
        
    Returns:
        Resized PIL Image.
    """
    if img.height == target_h:
        return img
    scale = target_h / img.height
    new_w = max(1, int(round(img.width * scale)))
    return img.resize((new_w, target_h), Image.BICUBIC)


def patch_grid_from_inputs(inputs: dict, patch_size: int) -> tuple:
    """
    Extract patch grid dimensions from model inputs.
    
    Args:
        inputs: Dictionary containing preprocessed image tensors with 'pixel_values' key.
        patch_size: Patch size used by the model.
        
    Returns:
        Tuple of (H, W, Hp, Wp, N) where:
            - H, W: Original image height and width
            - Hp, Wp: Patch grid height and width
            - N: Total number of patches
    """
    _, _, H, W = inputs["pixel_values"].shape
    Hp, Wp = H // patch_size, W // patch_size
    N = Hp * Wp
    return H, W, Hp, Wp, N
