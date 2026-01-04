"""Utilities for image processing and visualization."""

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers.image_utils import load_image

from .model_utils import load_dinov3_processor


def process_image(
    image_path: str, device: str = "cuda", resize: bool = False, center_crop: bool = False
) -> Tuple[Image.Image, dict]:
    """
    Load and process an image for DINOv3 model input.
    
    Args:
        image_path: Path or URL to the image.
        device: Device to move processed tensors to.
        resize: Whether to resize the image.
        center_crop: Whether to center crop the image.
        
    Returns:
        Tuple of (original PIL Image, processed model inputs dictionary).
    """
    image = load_image(image_path)
    processor = load_dinov3_processor()
    inputs = processor(
        images=image,
        return_tensors="pt",
        do_resize=resize,
        do_center_crop=center_crop,
    ).to(device)
    return image, inputs


def upsample_to_original_size(
    tensor: Tensor, original_size: Tuple[int, int], mode: str = "nearest"
) -> Tensor:
    """
    Upsample a tensor to match the original image size.
    
    Args:
        tensor: Tensor to upsample. Should be 4D (B, C, H, W) or 3D (C, H, W).
        original_size: Target size as (width, height) tuple.
        mode: Interpolation mode. Defaults to "nearest".
        
    Returns:
        Upsampled tensor.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4:
        upsampled = torch.nn.functional.interpolate(
            tensor, size=(original_size[1], original_size[0]), mode=mode
        )
        return upsampled.squeeze(0) if tensor.shape[0] == 1 else upsampled
    raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
