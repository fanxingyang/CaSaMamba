import os

import torch
import torch.nn as nn

from net.backbones.mamba2_backbone import Mamba2Block1, Mamba2Block2, Mamba2Block3
from net.fusion import MultiRepresentationFusion
from net.model_args import ModelArgs
from net.modules.center_pooling import CenterAwareWeighting
from net.rms_norm import RMSNorm


class ResidualBlockBase(nn.Module):
    def __init__(self, args, mixer_class):
        super().__init__()
        self.args = args
        self.mixer = mixer_class(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output


class ResidualBlock1(ResidualBlockBase):
    def __init__(self, args):
        super().__init__(args, Mamba2Block1)


class ResidualBlock2(ResidualBlockBase):
    def __init__(self, args):
        super().__init__(args, Mamba2Block2)


class ResidualBlock3(ResidualBlockBase):
    def __init__(self, args):
        super().__init__(args, Mamba2Block3)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers1 = nn.ModuleList([ResidualBlock1(args) for _ in range(args.n_layer)])
        self.layers2 = nn.ModuleList([ResidualBlock2(args) for _ in range(args.n_layer)])
        self.layers3 = nn.ModuleList([ResidualBlock3(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.fusion_module = MultiRepresentationFusion()

        center_index = int(os.getenv("CASAMAMBA_CENTER_INDEX", "20"))
        center_window_size = int(os.getenv("CASAMAMBA_CENTER_WINDOW_SIZE", "5"))
        center_weight = float(os.getenv("CASAMAMBA_CENTER_WEIGHT", "1.0"))
        outer_weight = float(os.getenv("CASAMAMBA_OUTER_WEIGHT", "0.5"))
        self.center_weighting = CenterAwareWeighting(
            seq_len=41,
            center_index=center_index,
            center_window_size=center_window_size,
            center_weight=center_weight,
            outer_weight=outer_weight,
        )

        self.linear_layer = nn.Linear(41, 1)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.LastDense = nn.Sequential(
            nn.Linear(64 * 41, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.nor = nn.Sigmoid()
        self.mapping = {0: 3, 1: 2, 2: 1, 3: 0}

    def forward(self, input_ids):
        input_ids = input_ids.long()
        embedded = self.embedding(input_ids)
        x_1 = embedded.clone()
        x_2 = embedded.clone()
        x_3 = embedded.clone()
        for layer in self.layers1:
            x_1 = layer(x_1)
        x_1 = self.norm_f(x_1)
        for layer in self.layers2:
            x_2 = layer(x_2)
        x_2 = self.norm_f(x_2)
        for layer in self.layers3:
            x_3 = layer(x_3)
        x_3 = self.norm_f(x_3)
        x = self.fusion_module(x_1, x_2, x_3)
        x = self.center_weighting(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.LastDense(x)
        logits = logits.squeeze(dim=1)
        logits = self.nor(logits)
        return logits
