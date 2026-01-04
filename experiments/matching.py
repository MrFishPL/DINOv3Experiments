"""Matching Experiment for DINOv3 patch matching between two images."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_linear_assignment import assignment_to_indices, batch_linear_assignment

from experiments.base_experiment import BaseExperiment


class MatchingExperiment(BaseExperiment):
    """
    Experiment to match patches between two images using DINOv3 features.
    
    This experiment computes optimal patch matching between two images using
    cosine similarity and linear assignment algorithm. The first image must have
    fewer or equal patches than the second image.
    """

    @property
    def input_names(self) -> Sequence[str]:
        """Input names for the experiment."""
        return ("preprocessed_inputs1", "preprocessed_inputs2")

    @property
    def output_names(self) -> Sequence[str]:
        """Output names for the experiment."""
        return ("matching", "scores")

    @torch.no_grad()
    def forward(
        self, preprocessed_inputs1: Dict[str, Tensor], preprocessed_inputs2: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute optimal patch matching between two images.
        
        Args:
            preprocessed_inputs1: Preprocessed inputs for the first image (smaller or equal size).
            preprocessed_inputs2: Preprocessed inputs for the second image (larger or equal size).
        
        Returns:
            Tuple of (matching, scores):
                - matching: Tensor of shape (B, N1) containing matched patch indices from image2.
                - scores: Tensor of shape (B, N1) containing cosine similarity scores for matches.
        
        Raises:
            ValueError: If image1 has more patches than image2.
        """
        outputs1 = self.dinov3_model(**preprocessed_inputs1, output_hidden_states=True)
        outputs2 = self.dinov3_model(**preprocessed_inputs2, output_hidden_states=True)

        x1 = outputs1.last_hidden_state
        x2 = outputs2.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        # Extract patch tokens (skip CLS token and register tokens)
        patch_tokens1 = x1[:, 1 + R :, :]
        patch_tokens2 = x2[:, 1 + R :, :]

        if patch_tokens1.shape[1] > patch_tokens2.shape[1]:
            raise ValueError(
                f"Expected image1 to have <= patches than image2, "
                f"got N1={patch_tokens1.shape[1]} > N2={patch_tokens2.shape[1]}."
            )

        # Normalize patch tokens
        p1 = F.normalize(patch_tokens1, p=2, dim=-1)
        p2 = F.normalize(patch_tokens2, p=2, dim=-1)

        # Compute cosine similarity and convert to cost matrix
        cos_sim = torch.matmul(p1, p2.transpose(-1, -2))
        cost = 1.0 - cos_sim

        # Linear assignment (Hungarian algorithm)
        # Note: batch_linear_assignment may require CPU for some backends
        dev_str = str(self.device)
        if dev_str.startswith("cuda"):
            assignment = batch_linear_assignment(cost)
        else:
            assignment = batch_linear_assignment(cost.cpu())

        row_ind, col_ind = assignment_to_indices(assignment)

        # Build mapping tensor
        B, N1 = patch_tokens1.shape[:2]
        mapping = torch.empty((B, N1), dtype=torch.long, device=row_ind.device)
        for b in range(B):
            mapping[b, row_ind[b]] = col_ind[b]

        # Extract scores for matched patches
        if cos_sim.device != mapping.device:
            cos_sim = cos_sim.to(mapping.device)

        scores = cos_sim.gather(2, mapping.unsqueeze(-1)).squeeze(-1)

        return (mapping, scores)