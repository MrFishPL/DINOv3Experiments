from experiments.base_experiment import BaseExperiment
from typing import Sequence, Tuple
import torch
from torch import Tensor

class FeaturesLengthExperiment(BaseExperiment):
    @property
    def input_names(self) -> Sequence[str]:
        return ("preprocessed_inputs",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("norm_feat_lens",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs: Tensor) -> Tuple[Tensor]:
        outputs = self.dinov3_model(**preprocessed_inputs, output_hidden_states=True)
        
        x = outputs.last_hidden_state
        print(outputs.keys())
        R = self.dinov3_model.config.num_register_tokens

        patch_tokens = x[:, 1 + R:, :]
        B, _, H, W = preprocessed_inputs.pixel_values.shape
        ps = self.dinov3_model.config.patch_size
        Hp, Wp = H // ps, W // ps
        patch_grid = patch_tokens.reshape(B, Hp, Wp, -1)
        norm_feat_lens = torch.linalg.vector_norm(patch_grid, dim=-1)
        min_lens = norm_feat_lens.amin(dim=(-2, -1), keepdim=True)
        max_lens = norm_feat_lens.amax(dim=(-2, -1), keepdim=True)
        norm_feat_lens = (norm_feat_lens - min_lens) / (max_lens - min_lens + 1e-8)
        norm_feat_lens = torch.clamp(norm_feat_lens, min=0.0, max=1.0)
        
        return (norm_feat_lens,)