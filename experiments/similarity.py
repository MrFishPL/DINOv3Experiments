from experiments.base_experiment import BaseExperiment
from typing import Sequence, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

class SimilarityExperiment(BaseExperiment):
    @property
    def input_names(self) -> Sequence[str]:
        return ("preprocessed_inputs", "patch_idx")

    @property
    def output_names(self) -> Sequence[str]:
        return ("similarity",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs: Tensor, patch_idx: int) -> Tuple[Tensor]:
        outputs = self.dinov3_model(**preprocessed_inputs, output_hidden_states=True)
        
        x = outputs.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        patch_tokens = x[:, 1 + R:, :]
        B, _, H, W = preprocessed_inputs.pixel_values.shape
        ps = self.dinov3_model.config.patch_size
        Hp, Wp = H // ps, W // ps
        
        if patch_idx == -1:
            selected_patch = x[:, 0, :]
        else:
            selected_patch = patch_tokens[:, patch_idx, :]

        cos_sim = F.cosine_similarity(patch_tokens, selected_patch.unsqueeze(1), dim=-1)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)

        patch_grid = cos_sim.reshape(B, Hp, Wp, -1)
        
        return (patch_grid,)