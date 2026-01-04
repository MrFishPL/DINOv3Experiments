"""Utilities for loading and configuring DINOv3 models."""

from typing import Literal

from transformers import AutoImageProcessor, AutoModel

# Model configuration
DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
Device = Literal["cpu", "cuda", "mps"]


def load_dinov3_model(device: Device = "cuda") -> AutoModel:
    """
    Load and configure a DINOv3 model.
    
    Args:
        device: Device to load the model on. Defaults to "cuda".
        
    Returns:
        Configured DINOv3 model in eval mode, moved to the specified device.
    """
    model = AutoModel.from_pretrained(DINOV3_MODEL_NAME)
    model.set_attn_implementation("eager")
    model.to(device)
    model.eval()
    return model


def load_dinov3_processor() -> AutoImageProcessor:
    """
    Load the DINOv3 image processor.
    
    Returns:
        DINOv3 image processor.
    """
    return AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME)
