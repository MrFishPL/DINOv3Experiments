from experiments.base_experiment import BaseExperiment
from typing import Sequence, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

from experiments.base_experiment import BaseExperiment
from typing import Sequence, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

class MatchingExperiment(BaseExperiment):
    @property
    def input_names(self) -> Sequence[str]:
        return ("preprocessed_inputs1", "preprocessed_inputs2")

    @property
    def output_names(self) -> Sequence[str]:
        return ("matching",)

    @torch.no_grad()
    def forward(self, preprocessed_inputs1, preprocessed_inputs2) -> Tuple[Tensor]:
        outputs1 = self.dinov3_model(**preprocessed_inputs1, output_hidden_states=True)
        outputs2 = self.dinov3_model(**preprocessed_inputs2, output_hidden_states=True)

        x1 = outputs1.last_hidden_state
        x2 = outputs2.last_hidden_state
        R = self.dinov3_model.config.num_register_tokens

        patch_tokens1 = x1[:, 1 + R:, :]  # [B, N1, D]
        patch_tokens2 = x2[:, 1 + R:, :]  # [B, N2, D]

        if patch_tokens1.shape[1] > patch_tokens2.shape[1]:
            raise ValueError(
                f"Expected image1 to have <= patches than image2, got N1={patch_tokens1.shape[1]} > N2={patch_tokens2.shape[1]}."
            )

        p1 = F.normalize(patch_tokens1, p=2, dim=-1)
        p2 = F.normalize(patch_tokens2, p=2, dim=-1)

        cos_sim = torch.matmul(p1, p2.transpose(-1, -2))
        cost = 1.0 - cos_sim

        dev_str = str(self.device)
        if dev_str.startswith("cuda"):
            assignment = batch_linear_assignment(cost)
        else:
            assignment = batch_linear_assignment(cost.cpu())

        row_ind, col_ind = assignment_to_indices(assignment)  # oba: [B, N1]

        B, N1 = patch_tokens1.shape[:2]
        mapping = torch.empty((B, N1), dtype=torch.long, device=row_ind.device)
        for b in range(B):
            mapping[b, row_ind[b]] = col_ind[b]

        return (mapping,)
