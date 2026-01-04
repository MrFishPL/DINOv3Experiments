"""Similarity Experiment for DINOv3 patch similarity visualization."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments.base_experiment import BaseExperiment

# Constants
CLS_TOKEN_IDX = -1  # Special index to use CLS token instead of patch token


class SimilarityExperiment(BaseExperiment):
    """
    Experiment to compute similarity maps between a selected patch and all patches.
    
    This experiment computes cosine similarity between a selected patch (or CLS token)
    and all other patches in the image, creating a similarity visualization map.
    """

    @property
    def input_names(self) -> Sequence[str]:
        """Input names for the experiment."""
        return ("preprocessed_inputs", "patch_idx")

    @property
    def output_names(self) -> Sequence[str]:
        """Output names for the experiment."""
        return ("similarity",)

    @torch.no_grad()
    def forward(
        self, preprocessed_inputs: Dict[str, Tensor], patch_idx: int
    ) -> Tuple[Tensor, ...]:
        """
        Compute similarity map for a selected patch.
        
        Args:
            preprocessed_inputs: Dictionary containing preprocessed image tensors,
                typically from AutoImageProcessor with 'pixel_values' key.
            patch_idx: Index of the patch to compare against. Use -1 to use CLS token.
        
        Returns:
            Tuple containing similarity map tensor of shape (B, Hp, Wp, 1) where
            Hp and Wp are patch grid dimensions.
        """
        outputs = self.dinov3_model(**preprocessed_inputs, output_hidden_states=True)

        x = outputs.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        # Extract patch tokens (skip CLS token and register tokens)
        patch_tokens = x[:, 1 + R :, :]
        B, _, H, W = preprocessed_inputs["pixel_values"].shape
        ps = self.dinov3_model.config.patch_size
        Hp, Wp = H // ps, W // ps

        # Select reference patch (CLS token if patch_idx == -1, otherwise patch token)
        if patch_idx == CLS_TOKEN_IDX:
            selected_patch = x[:, 0, :]
        else:
            selected_patch = patch_tokens[:, patch_idx, :]

        # Compute cosine similarity between selected patch and all patches
        cos_sim = F.cosine_similarity(patch_tokens, selected_patch.unsqueeze(1), dim=-1)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)

        # Reshape to patch grid
        patch_grid = cos_sim.reshape(B, Hp, Wp, -1)

        return (patch_grid,)