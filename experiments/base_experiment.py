from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from transformers import AutoModel
Outputs = Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]


class BaseExperiment(nn.Module, ABC):
    """
    Base class for DinoV3 experiments.

    Design goals:
      - Subclasses can define a *named* forward signature, e.g. forward(img, mask).
      - A uniform entrypoint (`run`) for pipelines/sweeps.
      - Variable #inputs and #outputs.
      - Optional normalization to a dict for logging/saving.

    Notes:
      - In PyTorch, `nn.Module` expects subclasses to implement `forward` and calls it via `__call__`.
      - We provide `run` as a stable API for orchestration code.
    """

    def __init__(self, *, dinov3_model: AutoModel, device: str = "cuda", name: Optional[str] = None) -> None:
        super().__init__()
        self._name = name or self.__class__.__name__
        self._dinov3_model = dinov3_model
        self._dinov3_model.eval()
        self._dinov3_model.to(device=device)
        self._device = device

    @property
    def name(self) -> str:
        return self._name
      
    @property
    def device(self) -> str:
        return self._device
      
    @property
    def dinov3_model(self) -> AutoModel:
        return self._dinov3_model

    @property
    def input_names(self) -> Optional[Sequence[str]]:
        """
        If provided, positional args passed to `run(*args)` will be mapped to these names
        and forwarded as kwargs.

        Example: ("image", "mask") lets you do exp.run(img, mask) even if forward expects
        forward(image=..., mask=...).
        """
        return None

    @property
    def output_names(self) -> Optional[Sequence[str]]:
        """
        If provided and the experiment returns a tuple, it will be named with these keys
        in `run_dict`.
        """
        return None

    def run(self, *args: Tensor, **kwargs: Tensor) -> Outputs:
        """
        Uniform entrypoint for all experiments.

        - Optionally maps positional args -> kwargs via `input_names`.
        - Calls `forward`.
        """
        f_args, f_kwargs = self._prepare_forward_call(args, kwargs)
        out = self.forward(*f_args, **f_kwargs)
        return out

    def run_dict(self, *args: Tensor, **kwargs: Tensor) -> Dict[str, Tensor]:
        """
        Like `run`, but normalizes outputs to a dict[str, Tensor] for easy logging/saving.
        """
        out = self.run(*args, **kwargs)
        return self._normalize_outputs(out)

    @abstractmethod
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Outputs:
        """
        Subclasses implement the experiment computation here.

        Subclasses are free to use a specific named signature, e.g.:
            def forward(self, image: Tensor, mask: Tensor) -> Tensor: ...
        """
        raise NotImplementedError

    def _prepare_forward_call(
        self, args: Tuple[Tensor, ...], kwargs: Mapping[str, Tensor]
    ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        if not args:
            return (), dict(kwargs)

        names = self.input_names
        if not names:
            return args, dict(kwargs)

        if len(args) != len(names):
            raise ValueError(
                f"{self.name}: got {len(args)} positional inputs, but input_names has {len(names)}"
            )

        mapped = {n: t for n, t in zip(names, args)}
        overlap = set(mapped).intersection(kwargs)
        if overlap:
            raise ValueError(f"{self.name}: inputs provided twice for keys: {sorted(overlap)}")

        merged = dict(kwargs)
        merged.update(mapped)
        return (), merged

    def _normalize_outputs(self, out: Outputs) -> Dict[str, Tensor]:
        if isinstance(out, torch.Tensor):
            return {"out": out}

        if isinstance(out, dict):
            return out

        names = self.output_names
        if names is not None:
            if len(names) != len(out):
                raise ValueError(
                    f"{self.name}: output_names has {len(names)} entries but forward returned {len(out)} tensors"
                )
            return {n: t for n, t in zip(names, out)}

        return {f"out{i}": t for i, t in enumerate(out)}
