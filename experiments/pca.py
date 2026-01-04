from experiments.base_experiment import BaseExperiment
from typing import Sequence, Tuple
import torch
from torch import Tensor
from torch_pca import PCA

class PCAExperiment(BaseExperiment):
    @property
    def input_names(self) -> Sequence[str]:
        return ("preprocessed_inputs",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("pca",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs: Tensor) -> Tuple[Tensor]:
        outputs = self.dinov3_model(**preprocessed_inputs, output_hidden_states=True)
        
        x = outputs.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        patch_tokens = x[:, 1 + R:, :]
        B, _, H, W = preprocessed_inputs.pixel_values.shape
        ps = self.dinov3_model.config.patch_size
        Hp, Wp = H // ps, W // ps
        
        standardized_patch_tokens = (patch_tokens - patch_tokens.mean(dim=1)) / patch_tokens.std(dim=1)
        
        pca_model = PCA(n_components=3, svd_solver='full')
        standardized_transformed = pca_model.fit_transform(standardized_patch_tokens[0])
        
        x_min = standardized_transformed.min(dim=0, keepdim=True).values
        x_max = standardized_transformed.max(dim=0, keepdim=True).values
        standardized_transformed = (standardized_transformed - x_min) / (x_max - x_min + 1e-8)
        
        patch_grid = standardized_transformed.T.reshape(3, Hp, Wp)
        
        return (patch_grid,)