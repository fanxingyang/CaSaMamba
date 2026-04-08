import inspect

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from net.ps_cnn import PSConvNet1, PSConvNet2, PSConvNet3

_Mamba2 = None
_MAMBA2_IMPORT_ERROR = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
except Exception:
    try:
        from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2
    except Exception as exc:
        _MAMBA2_IMPORT_ERROR = exc


def mamba2_available():
    return _Mamba2 is not None


class Mamba2Backbone(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        if _Mamba2 is None:
            self.mixer = None
            return

        sig = inspect.signature(_Mamba2.__init__)
        kwargs = {"d_model": d_model}
        optional_kwargs = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }
        for key, value in optional_kwargs.items():
            if key in sig.parameters:
                kwargs[key] = value

        if "headdim" in sig.parameters and d_model % 64 == 0:
            kwargs["headdim"] = 64

        self.mixer = _Mamba2(**kwargs)

    def forward(self, x):
        if self.mixer is None:
            message = (
                "Mamba-2 backend is not available. Install mamba-ssm in the runtime environment "
                "for B1, e.g. `pip install mamba-ssm` with a compatible PyTorch/CUDA stack."
            )
            if _MAMBA2_IMPORT_ERROR is not None:
                message += f" Original import error: {_MAMBA2_IMPORT_ERROR}"
            raise ImportError(message)
        return self.mixer(x)


class Mamba2BlockBase(nn.Module):
    def __init__(self, args, conv_class, conv_attr_name):
        super().__init__()
        self.args = args
        self.conv_attr_name = conv_attr_name
        setattr(self, conv_attr_name, conv_class())
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.backbone = Mamba2Backbone(
            d_model=args.d_inner,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, "b l d_in -> b d_in l")
        conv_layer = getattr(self, self.conv_attr_name)
        x = conv_layer(x)
        x = F.silu(x)
        y = self.backbone(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output


class Mamba2Block1(Mamba2BlockBase):
    def __init__(self, args):
        super().__init__(args, PSConvNet1, "conv1d1")


class Mamba2Block2(Mamba2BlockBase):
    def __init__(self, args):
        super().__init__(args, PSConvNet2, "conv1d2")


class Mamba2Block3(Mamba2BlockBase):
    def __init__(self, args):
        super().__init__(args, PSConvNet3, "conv1d3")
