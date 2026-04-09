import torch
import torch.nn as nn


class CenterAwareWeighting(nn.Module):
    def __init__(
        self,
        seq_len=41,
        center_index=20,
        center_window_size=5,
        center_weight=1.0,
        outer_weight=0.5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.center_index = center_index
        self.center_window_size = center_window_size
        self.center_weight = center_weight
        self.outer_weight = outer_weight
        self.register_buffer("position_weights", self._build_weights(seq_len), persistent=False)

    def _build_weights(self, seq_len):
        weights = torch.full((seq_len,), float(self.outer_weight), dtype=torch.float32)
        half_window = self.center_window_size // 2
        left = max(0, self.center_index - half_window)
        right = min(seq_len, self.center_index + half_window + 1)
        weights[left:right] = float(self.center_weight)
        return weights

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("CenterAwareWeighting expects input shape (B, L, D).")

        seq_len = x.size(1)
        if seq_len == self.position_weights.numel():
            weights = self.position_weights.to(device=x.device, dtype=x.dtype)
        else:
            weights = self._build_weights(seq_len).to(device=x.device, dtype=x.dtype)

        return x * weights.view(1, seq_len, 1)
