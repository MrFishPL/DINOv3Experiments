"""PCA Experiment for DINOv3 patch token dimensionality reduction."""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor
from torch_pca import PCA

from experiments.base_experiment import BaseExperiment

# Constants
PCA_COMPONENTS = 3  # Number of PCA components (RGB channels for visualization)
EPSILON = 1e-8  # Small value to prevent division by zero


class PCAExperiment(BaseExperiment):
    """
    Experiment to compute PCA visualization of DINOv3 patch tokens.
    
    This experiment extracts patch tokens, standardizes them, applies PCA with
    3 components, and normalizes the result to [0, 1] for RGB visualization.
    """

    @property
    def input_names(self) -> Sequence[str]:
        """Input names for the experiment."""
        return ("preprocessed_inputs",)

    @property
    def output_names(self) -> Sequence[str]:
        """Output names for the experiment."""
        return ("pca",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        """
        Compute PCA visualization of patch tokens.
        
        Args:
            preprocessed_inputs: Dictionary containing preprocessed image tensors,
                typically from AutoImageProcessor with 'pixel_values' key.
        
        Returns:
            Tuple containing PCA visualization tensor of shape (3, Hp, Wp) where
            3 represents RGB channels and Hp, Wp are patch grid dimensions.
        """
        outputs = self.dinov3_model(**preprocessed_inputs, output_hidden_states=True)

        x = outputs.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        # Extract patch tokens (skip CLS token and register tokens)
        patch_tokens = x[:, 1 + R :, :]
        B, _, H, W = preprocessed_inputs["pixel_values"].shape
        ps = self.dinov3_model.config.patch_size
        Hp, Wp = H // ps, W // ps

        # Standardize patch tokens
        standardized_patch_tokens = (patch_tokens - patch_tokens.mean(dim=1)) / (
            patch_tokens.std(dim=1) + EPSILON
        )

        # Apply PCA (only on first batch item)
        pca_model = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
        standardized_transformed = pca_model.fit_transform(standardized_patch_tokens[0])

        # Normalize to [0, 1] range for each component
        x_min = standardized_transformed.min(dim=0, keepdim=True).values
        x_max = standardized_transformed.max(dim=0, keepdim=True).values
        standardized_transformed = (standardized_transformed - x_min) / (x_max - x_min + EPSILON)

        # Reshape to (3, Hp, Wp) for RGB visualization
        patch_grid = standardized_transformed.T.reshape(PCA_COMPONENTS, Hp, Wp)

        return (patch_grid,)