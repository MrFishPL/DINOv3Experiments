from experiments.base_experiment import BaseExperiment
from typing import Sequence, Tuple
import torch
from torch import Tensor

class LastAttentionExperiment(BaseExperiment):
    @property
    def input_names(self) -> Sequence[str]:
        return ("preprocessed_inputs", "head_idx", "token_idx")

    @property
    def output_names(self) -> Sequence[str]:
        return ("last_attention",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs: Tensor, head_idx: int = 0, token_idx: int = 0) -> Tuple[Tensor]:
        """
        Return an attention map for the given token_idx from the last layer, only for image patch tokens (excluding CLS and registers).
        If head_idx == -1: average heads. If head_idx == -2: max over heads. Otherwise, use the specific head.
        Output shape: (B, num_patches) -- attention from token_idx to each patch token.
        """
        outputs = self.dinov3_model(**preprocessed_inputs, output_attentions=True)
        last_attention = outputs.attentions[-1]

        B, num_heads, seq_len, _ = last_attention.shape
        
        R = self.dinov3_model.config.num_register_tokens
        B_img, _, H, W = preprocessed_inputs.pixel_values.shape
        patch_size = self.dinov3_model.config.patch_size
        Hp, Wp = H // patch_size, W // patch_size
        num_patches = Hp * Wp

        patch_token_indices = torch.arange(1 + R, 1 + R + num_patches, device=last_attention.device)

        attn_from_token = last_attention[:, :, token_idx, :]
        attn_map = attn_from_token[:, :, patch_token_indices]

        if head_idx == -1:
            result = attn_map.mean(dim=1)
        elif head_idx == -2:
            result, _ = attn_map.max(dim=1)
        else:
            result = attn_map[:, head_idx, :]
        
        return (result,)