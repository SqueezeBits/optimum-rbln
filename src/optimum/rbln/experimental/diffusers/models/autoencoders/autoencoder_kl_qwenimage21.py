# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Qwen-Image-2.1 VAE on RBLN NPUs, written against the diffusers implementation.

The original is a video VAE (5D tensors, a feature cache over frames); an image is one frame, for which the time axis
is a no-op, so each layer is re-expressed on 4D tensors with the same weights.

Original (`diffusers.models.autoencoders.autoencoder_kl_qwenimage21`) -> here:
- `QwenImage21CausalConv3d`, `QwenImage21RMS_norm` -> `_conv`, `_rms`
- `QwenImage21ResidualBlock` -> `_resblock` (`_resblock_branch`)
- `QwenImage21AttentionBlock` -> `_attention`
- `QwenImage21Resample` -> `_upsample` and the downsampler's `resample`
- `QwenImage21DupUp3D`, `QwenImage21AvgDown3D` -> `_DupUp2D`, `_AvgDown2D`
- `QwenImage21MidBlock`, `QwenImage21UpBlock`, `QwenImage21ResidualDownBlock` -> `_VAE2D.mid`, `.up`, `.down`
- `QwenImage21Decoder3d.forward`, `AutoencoderKLQwenImage21._decode` -> `DECODER_PROGRAM`, `TILED_DECODER_PROGRAM`
- `QwenImage21Encoder3d.forward`, `AutoencoderKLQwenImage21._encode` -> `ENCODER_PROGRAM`
- `tiled_decode` (blended tiles) -> `run_program` over exact row tiles (`tile_plan`, `_stitch`)
"""

import copy
import gc
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import rebel
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage21
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from rebel import RBLNCompileError
from torch import nn

from .....configuration_utils import RBLNCompileConfig
from .....modeling import RBLNModel
from .....modeling_base import Preprocessor
from .....utils.logging import get_logger
from ....utils import LazyRuntime, experimental, graphs_by_name, runtimes_by_name
from ...configurations import RBLNAutoencoderKLQwenImage21Config


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from .....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)

# The original VAE inherits the Wan video VAE: 5D tensors and a feature cache over frames. An image is a single frame,
# for which the time axis is a no-op, so the same computation runs on 4D tensors here.


# On RBLN-CR13 a 3x3 conv with many output channels at a high resolution fails to compile in the encoder (e.g. 96 → 192
# channels at 512x512, DEVICE_GRAPH_CONVERSION); the same conv split into groups of output channels compiles. The
# encoder's convs carry `split_out_channels` (see `_split_encoder_convs`); the decoder graphs compile without it (and
# with it, one of them does not).
_SPLIT_MIN_PIXELS, _SPLIT_OUT_CHANNELS = 512 * 512, 96


def _split_encoder_convs(vae: AutoencoderKLQwenImage21) -> AutoencoderKLQwenImage21:
    for module in vae.encoder.modules():
        if isinstance(module, nn.Conv2d):
            module.split_out_channels = _SPLIT_OUT_CHANNELS
    return vae


def _conv(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A `QwenImage21CausalConv3d` (a Conv2d with its own padding) on a 4D input."""
    x = F.pad(x, conv._padding)
    weight, bias, out_channels = conv.weight, conv.bias, conv.weight.shape[0]
    step = getattr(conv, "split_out_channels", None)
    if not (step and x.shape[-2] * x.shape[-1] >= _SPLIT_MIN_PIXELS and out_channels > step and weight.shape[-1] > 1):
        return F.conv2d(x, weight, bias, conv.stride, 0, conv.dilation, conv.groups)
    return torch.cat(
        [
            F.conv2d(
                x,
                weight[i : i + step],
                None if bias is None else bias[i : i + step],
                conv.stride,
                0,
                conv.dilation,
                conv.groups,
            )
            for i in range(0, out_channels, step)
        ],
        dim=1,
    )


def _rms(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # F.normalize over the channels, computed in float32 like the original.
    """`QwenImage21RMS_norm.forward` on one frame: `F.normalize` over the channels in float32, then `scale * gamma +
    bias`."""
    xf = x.float()
    normalized = (xf * torch.rsqrt((xf * xf).sum(dim=1, keepdim=True).clamp_min(1e-24))).to(x.dtype)
    bias = norm.bias.reshape(1, -1, 1, 1) if isinstance(norm.bias, torch.Tensor) else norm.bias
    return normalized * norm.scale * norm.gamma.reshape(1, -1, 1, 1) + bias


def _resblock_branch(block: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The two halves of `QwenImage21ResidualBlock.forward` on one frame, the conv branch and the shortcut, added by
    the caller."""
    shortcut = x if isinstance(block.conv_shortcut, nn.Identity) else _conv(block.conv_shortcut, x)
    x = _conv(block.conv1, F.silu(_rms(block.norm1, x)))
    return _conv(block.conv2, F.silu(_rms(block.norm2, x))), shortcut


def _resblock(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`QwenImage21ResidualBlock.forward` on one frame."""
    branch, shortcut = _resblock_branch(block, x)
    return branch + shortcut


def _attention(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`QwenImage21AttentionBlock.forward` on one frame: RMS norm, fused qkv, single-head scaled dot-product attention
    over the pixels, `proj`, residual."""
    batch, channels, height, width = x.shape
    qkv = block.to_qkv(_rms(block.norm, x)).reshape(batch, 1, channels * 3, height * width).permute(0, 1, 3, 2)
    q, k, v = qkv.chunk(3, dim=-1)
    out = F.scaled_dot_product_attention(q, k, v).squeeze(1).permute(0, 2, 1).reshape(batch, channels, height, width)
    return block.proj(out) + x


def _upsample(resample: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # nearest-exact at scale 2 is nearest
    """The upsampling of `QwenImage21Resample.forward` on one frame: `QwenImage21Upsample` (nearest x2), then its
    conv."""
    return resample.resample[1](x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))


class _DupUp2D(nn.Module):
    """`QwenImage21DupUp3D` on the first frame: output channel `c` at `(2h+i, 2w+j)` reads a constant input channel.
    A channel gather fails to compile at large sizes, so the layouts of the checkpoint (`a*c + b + s*i`) are computed
    with strided slices, a repeat and a pixel shuffle."""

    def __init__(self, dup: nn.Module):
        super().__init__()
        fs, ft, oc, r = dup.factor_s, dup.factor_t, dup.out_channels, dup.repeats
        index = torch.tensor(
            [
                [[(((c * ft + (ft - 1)) * fs + i) * fs + j) // r for j in range(fs)] for i in range(fs)]
                for c in range(oc)
            ]
        )
        if fs != 2 or not torch.equal(index[:, :, 0], index[:, :, 1]):
            raise NotImplementedError(f"Unsupported DupUp layout (factor_s={fs}).")
        a = int(index[1, 0, 0] - index[0, 0, 0]) if oc > 1 else 1
        b, s = int(index[0, 0, 0]), int(index[0, 1, 0] - index[0, 0, 0])
        if not torch.equal(index[:, :, 0], a * torch.arange(oc)[:, None] + b + s * torch.arange(fs)[None, :]) or not (
            s == 0 or (s == 1 and a == 2)
        ):
            raise NotImplementedError("Unsupported DupUp channel layout.")
        self.a, self.b, self.s, self.oc = a, b, s, oc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.s == 0:
            y = x[:, self.b : self.b + self.a * (self.oc - 1) + 1 : self.a]
            return y.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        # a == 2, s == 1: a 2x channel repeat and a pixel shuffle
        return F.pixel_shuffle(x[:, self.b : self.b + self.a * self.oc].repeat_interleave(2, dim=1), 2)


class _AvgDown2D(nn.Module):
    """`QwenImage21AvgDown3D` on the first frame (a zero frame is prepended, so it is a constant linear map after a
    pixel unshuffle), as 1x1 convs over the stride-`factor_s` slices (pixel_unshuffle does not compile)."""

    def __init__(self, avg: nn.Module):
        super().__init__()
        ft, fs, channels, oc, group = avg.factor_t, avg.factor_s, avg.in_channels, avg.out_channels, avg.group_size
        k = fs * fs
        weight = torch.zeros(oc, channels * k)
        for c in range(channels):
            for q in range(k):
                weight[(c * ft * k + (ft - 1) * k + q) // group, c * k + q] += 1.0 / group
        for q in range(k):
            self.register_buffer(f"w{q}", weight[:, q::k][:, :, None, None].contiguous(), persistent=False)
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fs = self.fs
        return sum(
            F.conv2d(x[:, :, i::fs, j::fs], getattr(self, f"w{i * fs + j}").to(x.dtype))
            for i in range(fs)
            for j in range(fs)
        )


class _VAE2D(nn.Module):
    """The VAE modules and the 2D replacements of its first-frame shortcut layers, shared by the chain steps."""

    def __init__(self, vae: AutoencoderKLQwenImage21):
        super().__init__()
        self.vae = vae
        self.dup = nn.ModuleList(
            [
                _DupUp2D(ub.avg_shortcut) if ub.avg_shortcut is not None else nn.Identity()
                for ub in vae.decoder.up_blocks
            ]
        )
        self.avg = nn.ModuleList([_AvgDown2D(db.avg_shortcut) for db in vae.encoder.down_blocks])

    def mid(self, mid_block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """`QwenImage21MidBlock.forward`: a resnet, then attention and resnet pairs."""
        x = _resblock(mid_block.resnets[0], x)
        for attention, block in zip(mid_block.attentions, mid_block.resnets[1:], strict=False):
            x = _resblock(block, _attention(attention, x))
        return x

    def up(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """`QwenImage21UpBlock.forward`: the resnets, the upsampler and the `avg_shortcut` (`QwenImage21DupUp3D`)."""
        up_block = self.vae.decoder.up_blocks[k]
        y = x
        for block in up_block.resnets:
            y = _resblock(block, y)
        if up_block.upsampler is not None:
            y = _upsample(up_block.upsampler, y)
        return y + self.dup[k](x) if up_block.avg_shortcut is not None else y

    def down(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """`QwenImage21ResidualDownBlock.forward`: the resnets, the `QwenImage21Resample` downsampling and the
        `avg_shortcut` (`QwenImage21AvgDown3D`)."""
        down_block, x_copy = self.vae.encoder.down_blocks[k], x
        for block in down_block.resnets:
            x = _resblock(block, x)
        if down_block.downsampler is not None:
            x = down_block.downsampler.resample(x)  # ZeroPad2d((0, 1, 0, 1)) + stride-2 conv
        return x + self.avg[k](x_copy)

    def decoder_out(self, x: torch.Tensor) -> torch.Tensor:
        """The tail of `QwenImage21Decoder3d.forward`: `norm_out`, SiLU, `conv_out`, clamped to [-1, 1] as
        `AutoencoderKLQwenImage21.decode` does."""
        decoder = self.vae.decoder
        return _conv(decoder.conv_out, F.silu(_rms(decoder.norm_out, x))).clamp(-1.0, 1.0)


class _Step(nn.Module):
    def __init__(self, vae2d: _VAE2D, fn: Callable):
        super().__init__()
        self.vae2d, self.fn = vae2d, fn

    def forward(self, *args):
        return self.fn(self.vae2d, *args)


def _decoder_head(f: _VAE2D, z: torch.Tensor) -> torch.Tensor:
    """The head of `AutoencoderKLQwenImage21._decode` on one frame: `post_quant_conv`, `conv_in` and the mid block of
    `QwenImage21Decoder3d`."""
    decoder = f.vae.decoder
    return f.mid(decoder.mid_block, _conv(decoder.conv_in, _conv(f.vae.post_quant_conv, z)))


def _decoder_up012(f: _VAE2D, x: torch.Tensor) -> torch.Tensor:
    """`up_blocks[0:3]` of `QwenImage21Decoder3d.forward`."""
    for k in range(3):
        x = f.up(k, x)
    return x


def _decoder_up3(f: _VAE2D, x: torch.Tensor) -> torch.Tensor:
    """`up_blocks[3]` of `QwenImage21Decoder3d.forward`."""
    up_block = f.vae.decoder.up_blocks[3]
    y = x
    for block in up_block.resnets:
        y = _resblock(block, y)
    return _upsample(up_block.upsampler, y) + f.dup[3](x)


def _decoder_up4(f: _VAE2D, x: torch.Tensor) -> torch.Tensor:
    """`up_blocks[4]` of `QwenImage21Decoder3d.forward` and the decoder's output layers."""
    up_block = f.vae.decoder.up_blocks[4]
    y = x
    for block in up_block.resnets:
        y = _resblock(block, y)
    return f.decoder_out(y)


def _decoder_up_all(f: _VAE2D, x: torch.Tensor) -> torch.Tensor:
    """`up_blocks` and the output layers of `QwenImage21Decoder3d.forward`: everything after the mid block."""
    return _decoder_up4(f, _decoder_up3(f, _decoder_up012(f, x)))


# A chain program: (step name, function, input names, output names). Up to the 1024 class the decoder is one graph;
# above it the mid block (`head`, which attends over the whole image) is one graph and everything after it another,
# run in row tiles with a halo of TILE_HALO latent rows, which gives exactly the untiled result (those steps have only
# local ops: convs, per-pixel norms, nearest upsampling, pixel shuffles). The residual blocks are kept as the
# original's conv branch + shortcut add: with the add folded into `conv2` (one conv over `[branch ; x]`) the whole
# decoder does not compile on RBLN-CR13 (rebel-compiler 0.12.1.dev163, "sharding directives of the operands
# disagree"), which is what used to split it into up to 7 graphs.
DECODER_PROGRAM = [
    ("whole", lambda f, z: _decoder_up_all(f, _decoder_head(f, z)), ["z"], ["out"]),
]
TILED_DECODER_PROGRAM = [
    ("head", _decoder_head, ["z"], ["x0"]),
    ("up_all", _decoder_up_all, ["x0"], ["out"]),
]
DECODER_TILED_STEPS = {"up_all"}
TILE_HALO = 8  # latent rows; the receptive radius of the tiled steps is about 6.7


def _encoder_e3(f: _VAE2D, y: torch.Tensor, h: torch.Tensor, x_copy: torch.Tensor) -> torch.Tensor:
    """The rest of `down_blocks[1]` of `QwenImage21Encoder3d.forward` after its first resblock's two halves."""
    down_block, x = f.vae.encoder.down_blocks[1], y + h
    for block in down_block.resnets[1:]:
        x = _resblock(block, x)
    return down_block.downsampler.resample(x) + f.avg[1](x_copy)


def _encoder_e4(f: _VAE2D, x: torch.Tensor) -> torch.Tensor:
    """`down_blocks[2:]`, the mid block and the output layers of `QwenImage21Encoder3d.forward`, then `quant_conv` and
    the mean half of the posterior (`AutoencoderKLQwenImage21._encode`)."""
    encoder = f.vae.encoder
    for k in range(2, len(encoder.down_blocks)):
        x = f.down(k, x)
    x = f.mid(encoder.mid_block, x)
    x = _conv(f.vae.quant_conv, _conv(encoder.conv_out, F.silu(_rms(encoder.norm_out, x))))
    return x[:, : f.vae.config.z_dim]


def _encoder_whole(f: _VAE2D, x: torch.Tensor) -> torch.Tensor:
    """`QwenImage21Encoder3d.forward` on one frame and the posterior mean (`AutoencoderKLQwenImage21._encode`)."""
    x1 = f.down(0, _conv(f.vae.encoder.conv_in, x))
    y, h = _resblock_branch(f.vae.encoder.down_blocks[1].resnets[0], x1)
    return _encoder_e4(f, _encoder_e3(f, y, h, x1))


# The encoder takes the image as RGBA (4 channels) and returns the posterior mean, as one graph (its mid block attends
# over the whole image, so it is not tiled).
ENCODER_PROGRAM = [
    ("whole", _encoder_whole, ["x"], ["z"]),
]
ENCODER_TILED_STEPS: set[str] = set()
ENCODER_TILE_UNIT, ENCODER_TILE_HALO = 16, 2


def tile_plan(rows: int, n: int, halo: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Row windows of n tiles over `rows` rows: the common window height and `(start, keep_from, keep_to)` per tile.

    Replaces the original's `tiled_decode`, which blends overlapping tiles: with a halo the tiles stitch exactly.
    """
    if n <= 1:
        return rows, [(0, 0, rows)]
    height = min(rows, -(-rows // n) + 2 * halo)
    plan = []
    for i in range(n):
        begin, end = rows * i // n, rows * (i + 1) // n
        start = min(max(begin - halo, 0), rows - height)
        plan.append((start, begin - start, end - start))
    return height, plan


def run_program(
    program: list[tuple],
    graphs: dict[str, Callable],
    env: dict[str, torch.Tensor],
    tiled: Collection[str] = (),
    n_tiles: int = 1,
    halo: int = 0,
    unit: int = 1,
    base: str | None = None,
) -> dict[str, torch.Tensor]:
    """Runs a chain program over `env` (name → tensor). Steps in `tiled` run in `n_tiles` row tiles planned in units of
    `unit` rows of `env[base]`; every tensor is sliced at its own scale and the outputs of the last tiled step are
    stitched back (each tile keeps its own rows)."""
    rows = env[base].shape[2] // unit if n_tiles > 1 else 0
    height, plan = tile_plan(rows, n_tiles, halo) if n_tiles > 1 else (0, [])
    tiles = last_outputs = None
    for name, _, inputs, outputs in program:
        if tiles is not None and name not in tiled:
            _stitch(env, tiles, last_outputs, height)
            tiles = None
        if name in tiled and n_tiles > 1:
            if tiles is None:
                tiles = []
                for start, keep_from, keep_to in plan:
                    window = {}
                    for key in inputs:
                        scale = env[key].shape[2] // rows
                        window[key] = env[key][:, :, start * scale : (start + height) * scale].contiguous()
                    tiles.append((window, keep_from, keep_to))
            for window, _, _ in tiles:
                result = graphs[name](*[window[key] for key in inputs])
                window.update(zip(outputs, result if isinstance(result, list | tuple) else (result,), strict=True))
            last_outputs = outputs
        else:
            result = graphs[name](*[env[key] for key in inputs])
            env.update(zip(outputs, result if isinstance(result, list | tuple) else (result,), strict=True))
    if tiles is not None:
        _stitch(env, tiles, last_outputs, height)
    return env


def _stitch(env: dict[str, torch.Tensor], tiles: list[tuple], keys: list[str], height: int) -> None:
    for key in keys:
        scale = tiles[0][0][key].shape[2] // height
        env[key] = torch.cat([window[key][:, :, scale * a : scale * b] for window, a, b in tiles], dim=2)


class _LatentDistribution:
    """The `DiagonalGaussianDistribution` the original returns, reduced to the mean the pipeline reads.

    The posterior as the pipeline reads it (`mode()`; the mean is all the graphs compute).
    """

    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError("Only the posterior mean is computed; use `mode()` (`sample_mode='argmax'`).")


class _Chain(NamedTuple):
    """The graphs of one VAE kind at one size: a program run in `tiles` row tiles (the `tiled` steps, with `halo`
    rows of `unit` rows of the tensor `base`)."""

    kind: str
    size: tuple[int, int]
    program: list[tuple]
    tiled: Collection[str]
    tiles: int
    halo: int
    unit: int
    base: str


def _graph_name(kind: str, size: tuple[int, int], step: str) -> str:
    return f"{kind}_{size[0]}x{size[1]}_{step}"


def _chiplet0_gib(compiled_models: Iterable[rebel.RBLNCompiledModel]) -> float:
    """GiB the graphs take on the first chiplet of the device together (their allocation tables)."""
    return sum(sum(v[0][0] for v in m.get_alloc_per_chiplet_by_key().values()) for m in compiled_models) / 2**30


@experimental
class RBLNAutoencoderKLQwenImage21(RBLNModel):
    """
    RBLN implementation of AutoencoderKLQwenImage21, the VAE of Qwen-Image-2.1 pipelines.

    The original is a video VAE; an image is one frame, so the decoder and encoder run here as 2D graphs compiled for
    each exact size of `sample_sizes` (decoder) and `encoder_sample_sizes` (encoder): the mid-block attention of both
    needs the whole image. Up to the 1024 class the decoder and the encoder are one graph each; at the 2048 class the
    decoder is two, its mid block and everything after it in row tiles (exact, see `decoder_row_tiles`).

    On RBLN-CR13 a VAE graph keeps most of its scratch on the first chiplet of the device, so a decoder takes 6 to
    8 GiB of it and the graphs of all sizes do not fit next to the transformer. The graphs of a size are therefore
    loaded on first use (`LazyRuntime`) and those of the least recently used sizes are released when the loaded ones
    would exceed `resident_gib` of the first chiplet together. When all of them fit the budget (the pipeline's default:
    one image size and four condition-image sizes), they are all loaded with the model.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = AutoencoderKLQwenImage21
    _rbln_config_class = RBLNAutoencoderKLQwenImage21Config

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.graphs = graphs_by_name(self.model, self.rbln_config)
        self._chain_gib: dict[tuple[str, tuple[int, int]], float] = {}
        if self.model:  # (no runtimes when loaded with `create_runtimes=False`)
            expected = {
                _graph_name(chain.kind, chain.size, step)
                for chain in self._programs(self.rbln_config)
                for step, *_ in chain.program
            }
            if missing := sorted(expected - set(self.graphs)):
                raise ValueError(
                    f"The compiled VAE has no graphs {missing}: it was compiled with another layout of the graph "
                    "chains; compile it again with this version."
                )
            for name, graph in self.graphs.items():
                path = Path(self.model_save_dir) / self.subfolder / f"{name}.rbln"
                graph.path = path if path.exists() else None
            for chain in self._programs(self.rbln_config):
                self._chain_gib[(chain.kind, chain.size)] = _chiplet0_gib(
                    self.graphs[_graph_name(chain.kind, chain.size, step)].compiled_model for step, *_ in chain.program
                )
        self._loaded: list[tuple[str, tuple[int, int]]] = []  # chains with open runtimes, least recently used first
        if self._chain_gib and sum(self._chain_gib.values()) <= self.rbln_config.resident_gib:
            # Everything fits the budget: load it all now, so that no request waits for a graph.
            for kind, size in self._chain_gib:
                for name, *_ in self._programs_by_size(kind, size):
                    self.graphs[name].open()
                self._loaded.append((kind, size))

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The pipeline config sets the image sizes.
        return rbln_config

    @staticmethod
    def _chain_for(kind: str, size: tuple[int, int], tiles: int) -> _Chain:
        if kind == "encoder":
            return _Chain(
                kind, size, ENCODER_PROGRAM, ENCODER_TILED_STEPS, tiles, ENCODER_TILE_HALO, ENCODER_TILE_UNIT, "x"
            )
        if tiles == 1:
            return _Chain(kind, size, DECODER_PROGRAM, set(), 1, 0, 1, "z")
        return _Chain(kind, size, TILED_DECODER_PROGRAM, DECODER_TILED_STEPS, tiles, TILE_HALO, 1, "z")

    @classmethod
    def _programs(cls, rbln_config: RBLNAutoencoderKLQwenImage21Config) -> list[_Chain]:
        """The chains of every size (see `_chain_for`)."""
        return [
            cls._chain_for("decoder", size, rbln_config.tiles("decoder", *size)) for size in rbln_config.sample_sizes
        ] + [
            cls._chain_for("encoder", size, rbln_config.tiles("encoder", *size))
            for size in rbln_config.encoder_sample_sizes
        ]

    @staticmethod
    def _modules(model: AutoencoderKLQwenImage21) -> dict[str, _VAE2D]:
        # The decoder runs in bfloat16 (PSNR 50.9 dB against the float32 original); the encoder in float32, which the
        # compiler lowers to dlfloat16 (in bfloat16 the condition latents were 1.5 % off, against 0.5-0.9 % for the
        # bfloat16 original on CPU, which shows in edits).
        return {
            "decoder": _VAE2D(copy.deepcopy(model).to(torch.bfloat16)).eval(),
            "encoder": _VAE2D(_split_encoder_convs(copy.deepcopy(model).to(torch.float32))).eval(),
        }

    @staticmethod
    def _dtype(kind: str) -> torch.dtype:
        return torch.bfloat16 if kind == "decoder" else torch.float32

    @classmethod
    def _chain_input_info(cls, modules: dict[str, _VAE2D], chain: _Chain) -> dict[str, list]:
        """Input shapes of every step at a size: the program runs on a small probe on CPU to get each tensor's channels
        and scale, and the shapes are scaled up (tiled steps take a window of the common tile height)."""
        kind, size, program, tiled, tiles, halo, unit, _ = chain
        dtype = cls._dtype(kind)
        probe = 2 * unit
        start = {"z": torch.zeros(1, 64, 2, 2, dtype=dtype)} if kind == "decoder" else {}
        if kind == "encoder":
            start = {"x": torch.zeros(1, 4, probe, probe, dtype=dtype)}
        steps = {name: _Step(modules[kind], fn) for name, fn, _, _ in program}
        with torch.no_grad():
            env = run_program(program, steps, dict(start))
        height, width = (size[0] // 16, size[1] // 16) if kind == "decoder" else size
        rows = height // unit
        tile_rows = tile_plan(rows, tiles, halo)[0]
        info = {}
        for name, _, inputs, _ in program:
            step_rows = tile_rows if name in tiled and tiles > 1 else rows
            info[name] = [
                (
                    key,
                    [
                        1,
                        env[key].shape[1],
                        step_rows * env[key].shape[2] * unit // probe,
                        width * env[key].shape[3] // probe,
                    ],
                    str(dtype).split(".")[-1],
                )
                for key in inputs
            ]
        return info

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLQwenImage21Config,
    ) -> RBLNAutoencoderKLQwenImage21Config:
        modules = cls._modules(model)
        rbln_config.set_compile_cfgs(
            [cfg for chain in cls._programs(rbln_config) for cfg in cls._chain_compile_cfgs(modules, chain)]
        )
        return rbln_config

    @classmethod
    def _chain_compile_cfgs(cls, modules: dict[str, _VAE2D], chain: _Chain) -> list[RBLNCompileConfig]:
        info = cls._chain_input_info(modules, chain)
        return [
            RBLNCompileConfig(compiled_model_name=_graph_name(chain.kind, chain.size, name), input_info=info[name])
            for name, *_ in chain.program
        ]

    @classmethod
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNAutoencoderKLQwenImage21Config
    ) -> dict[str, rebel.RBLNCompiledModel]:
        modules = cls._modules(model)
        functions = {
            (kind, name): fn
            for kind, program in (("decoder", DECODER_PROGRAM + TILED_DECODER_PROGRAM), ("encoder", ENCODER_PROGRAM))
            for name, fn, _, _ in program
        }
        compiled_models, compile_cfgs = {}, []
        for requested in cls._programs(rbln_config):
            kind, size, tiles = requested.kind, requested.size, requested.tiles
            # A chain whose graphs do not compile at a size (shape-specific compiler errors, e.g. the encoder at
            # 1248x832) or exceed `max_chain_gib` of chiplet 0 runs in more row tiles (the result is the same).
            candidates = [tiles] + [n for n in (2, 3, 4, 6, 8, 12, 16) if n > tiles]
            for more in candidates:
                chain = cls._chain_for(kind, size, more)
                cfgs = cls._chain_compile_cfgs(modules, chain)
                try:
                    chain_models = {
                        cfg.compiled_model_name: cls.compile(
                            _Step(modules[kind], functions[(kind, cfg.compiled_model_name.split("_", 2)[2])]).eval(),
                            rbln_compile_config=cfg,
                            create_runtimes=False,  # (runtimes are created on first use)
                            device=rbln_config.device,
                        )
                        for cfg in cfgs
                    }
                except RBLNCompileError:
                    if more == candidates[-1]:
                        raise
                    logger.warning(f"The VAE {kind} at {size[0]}x{size[1]} does not compile in {more} row tile(s).")
                    continue
                gib = _chiplet0_gib(chain_models.values())
                if gib > rbln_config.max_chain_gib and more != candidates[-1] and len(chain.tiled) > 0:
                    logger.warning(
                        f"The VAE {kind} at {size[0]}x{size[1]} in {more} row tile(s) takes {gib:.1f} GiB of chiplet 0 "
                        f"(`max_chain_gib={rbln_config.max_chain_gib}`); trying more tiles."
                    )
                    continue
                if more != tiles:
                    rbln_config.row_tiles[f"{kind}_{size[0]}x{size[1]}"] = more
                compiled_models.update(chain_models)
                compile_cfgs += cfgs
                break
        rbln_config.set_compile_cfgs(compile_cfgs)
        return compiled_models

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNAutoencoderKLQwenImage21Config
    ) -> list[LazyRuntime]:
        return runtimes_by_name(cls, compiled_models, rbln_config, runtime_cls=LazyRuntime)

    def _chain(self, kind: str, size: tuple[int, int]) -> tuple[list[tuple], dict[str, Callable], dict]:
        chain = next((c for c in self._programs(self.rbln_config) if c.kind == kind and c.size == size), None)
        if chain is None:
            option = "sample_sizes" if kind == "decoder" else "encoder_sample_sizes"
            sizes = getattr(self.rbln_config, option)
            raise ValueError(f"The VAE {kind} was not compiled for {size[0]}x{size[1]} (`{option}={sizes}`).")
        key = (kind, size)
        if key in self._loaded:
            self._loaded.remove(key)
        # Release the least recently used chains until this one fits the resident budget of the first chiplet.
        while self._loaded and (
            sum(self._chain_gib[k] for k in self._loaded) + self._chain_gib[key] > self.rbln_config.resident_gib
        ):
            for name, *_ in self._programs_by_size(*self._loaded.pop(0)):
                self.graphs[name].close()
            gc.collect()  # (the device memory is given back when the runtimes are destroyed)
        self._loaded.append(key)
        graphs = {name: self.graphs[_graph_name(kind, size, name)] for name, *_ in chain.program}
        tiling = {
            "tiled": chain.tiled,
            "n_tiles": chain.tiles,
            "halo": chain.halo,
            "unit": chain.unit,
            "base": chain.base,
        }
        return chain.program, graphs, tiling

    def _programs_by_size(self, kind: str, size: tuple[int, int]) -> list[tuple]:
        program = next(c.program for c in self._programs(self.rbln_config) if c.kind == kind and c.size == size)
        return [(_graph_name(kind, size, name), *rest) for name, *rest in program]

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        """
        Decodes latents into an image.

        Args:
            z (torch.Tensor): Denormalized latents of shape `(batch_size, 64, 1, height / 16, width / 16)`.
            return_dict (bool): Whether to return a `DecoderOutput` instead of a plain tuple.

        Returns:
            The image of shape `(batch_size, 4, 1, height, width)` in `[-1, 1]`.
        """
        images = []
        for sample in z:  # the graphs take one image
            latent = sample[:, 0][None].to(torch.bfloat16).contiguous()
            program, graphs, tiling = self._chain("decoder", (latent.shape[2] * 16, latent.shape[3] * 16))
            images.append(run_program(program, graphs, {"z": latent}, **tiling)["out"].float())
        image = torch.cat(images).unsqueeze(2)
        return DecoderOutput(sample=image) if return_dict else (image,)

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        """
        Encodes condition images into the posterior mean.

        Args:
            x (torch.Tensor): Images of shape `(batch_size, 4, 1, height, width)` in `[-1, 1]` (RGBA).
            return_dict (bool): Whether to return an `AutoencoderKLOutput` instead of a plain tuple.

        Returns:
            The posterior, of which only `mode()` (the mean) is computed.
        """
        means = []
        for sample in x:
            image = sample[:, 0][None].float().contiguous()
            program, graphs, tiling = self._chain("encoder", (image.shape[2], image.shape[3]))
            means.append(run_program(program, graphs, {"x": image}, **tiling)["z"].float())
        posterior = _LatentDistribution(torch.cat(means).unsqueeze(2))
        return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)


__all__ = [
    "RBLNAutoencoderKLQwenImage21",
]
