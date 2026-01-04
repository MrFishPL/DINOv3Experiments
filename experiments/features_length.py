"""Features Length Experiment for DINOv3."""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

from experiments.base_experiment import BaseExperiment

# Constants
EPSILON = 1e-8  # Small value to prevent division by zero


class FeaturesLengthExperiment(BaseExperiment):
    """
    Experiment to compute normalized feature length maps from DINOv3 patch tokens.
    
    This experiment extracts patch tokens from the DINOv3 model, computes their
    vector norms, and normalizes them to [0, 1] range for visualization.
    """

    @property
    def input_names(self) -> Sequence[str]:
        """Input names for the experiment."""
        return ("preprocessed_inputs",)

    @property
    def output_names(self) -> Sequence[str]:
        """Output names for the experiment."""
        return ("norm_feat_lens",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        """
        Compute normalized feature length map.
        
        Args:
            preprocessed_inputs: Dictionary containing preprocessed image tensors,
                typically from AutoImageProcessor with 'pixel_values' key.
        
        Returns:
            Tuple containing normalized feature length map tensor of shape (B, Hp, Wp),
            where Hp and Wp are the patch grid dimensions.
        """
        outputs = self.dinov3_model(**preprocessed_inputs, output_hidden_states=True)

        x = outputs.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        # Extract patch tokens (skip CLS token and register tokens)
        patch_tokens = x[:, 1 + R :, :]
        B, _, H, W = preprocessed_inputs["pixel_values"].shape
        ps = self.dinov3_model.config.patch_size
        Hp, Wp = H // ps, W // ps

        # Reshape to patch grid and compute vector norms
        patch_grid = patch_tokens.reshape(B, Hp, Wp, -1)
        norm_feat_lens = torch.linalg.vector_norm(patch_grid, dim=-1)

        # Normalize to [0, 1] range
        min_lens = norm_feat_lens.amin(dim=(-2, -1), keepdim=True)
        max_lens = norm_feat_lens.amax(dim=(-2, -1), keepdim=True)
        norm_feat_lens = (norm_feat_lens - min_lens) / (max_lens - min_lens + EPSILON)
        norm_feat_lens = torch.clamp(norm_feat_lens, min=0.0, max=1.0)

        return (norm_feat_lens,)