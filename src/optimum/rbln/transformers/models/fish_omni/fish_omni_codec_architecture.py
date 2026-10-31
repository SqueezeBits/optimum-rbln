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

"""FishOmni codec decoder (modded DAC vocoder) as one RBLN-traceable graph per frame window.

Covers fish-speech `DAC.from_indices` after the codebook lookups: RVQ `out_proj` per codebook, `post_module`
(WindowLimitedTransformer, causal, window 128), `upsample` (CausalTransConvNet k=s=2 + ConvNeXtBlock, twice) and the
causal `Decoder` (conv, 4 x [Snake, ConvTranspose k=2s, 3 ResidualUnits], Snake, conv, tanh).

Deltas against fish-speech `models/dac/{modded_dac,rvq}.py` (numbers unchanged, structure changed):
- the 10 codebook lookups (`F.embedding`, dim 8) run on the host; the graph takes the looked-up vectors;
- weight_norm is folded into plain conv weights at load (`fold_weight_norm`);
- interleaved RoPE of the post_module becomes rotate-half by permuting Q/K rows of `wqkv` at load; cos/sin are host
  inputs taken from upstream's bf16 `freqs_cis` table at the window's absolute positions;
- `scaled_dot_product_attention` with a bool mask becomes explicit attention with an additive float mask (same
  window-limited causal pattern), and the in-place LayerScale is an out-of-place multiply;
- causal padding uses the fixed amounts upstream computes for these kernel/stride pairs (`extra_padding` is 0 for
  every stride-1 conv and no ConvTranspose output is unpadded on the left).
- long code streams are decoded window by window (`FishOmniCodecWindow` for frames [0, W), then
  `FishOmniCodecNextWindow` for each following W frames). Exactness against one upstream pass over the whole stream:
  the post_module carries every layer's rotated K and V of the previous `post_window_size` frames as a state (each
  query only sees keys inside its window, so this is the whole attention context), and the conv stack (upsample +
  Decoder, receptive field below 10 frames) is recomputed over `context_frames` frames of the previous window's
  post_module output, whose samples the host drops. Upstream itself runs the whole stream in one pass.
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FishOmniCodecArgs:
    """Decoder half of fish_speech/configs/modded_dac_vq.yaml."""

    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [8, 8, 4, 2])
    codebook_dim: int = 8
    n_codebooks: int = 9
    semantic_codebook_size: int = 4096
    codebook_size: int = 1024
    downsample_factor: list[int] = field(default_factory=lambda: [2, 2])
    post_n_layer: int = 8
    post_n_head: int = 16
    post_head_dim: int = 64
    post_intermediate_size: int = 3072
    post_window_size: int = 128
    post_rope_base: float = 10000.0
    post_norm_eps: float = 1e-5

    @property
    def num_codebooks(self) -> int:
        return 1 + self.n_codebooks

    @property
    def samples_per_frame(self) -> int:
        return math.prod(self.downsample_factor) * math.prod(self.decoder_rates)


def post_module_freqs_table(seq_len: int, head_dim: int, base: float) -> Tensor:
    """Upstream `precompute_freqs_cis` (bf16 [seq_len, head_dim // 2, 2]) as fp32 cos / sin in rotate-half layout."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1).to(torch.bfloat16).float()
    cos, sin = cache[..., 0], cache[..., 1]
    return torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)


def window_causal_additive_mask(length: int, window: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Additive form of upstream `make_window_limited_mask`: key j visible to query i iff i - window < j <= i."""
    rows = torch.arange(length).view(-1, 1)
    cols = torch.arange(length).view(1, -1)
    visible = (cols <= rows) & (cols >= (rows - window + 1).clamp(min=0))
    return torch.zeros(length, length, dtype=dtype).masked_fill(~visible, -1e9)


def rope_row_permutation(head_dim: int) -> Tensor:
    return torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        out = (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight


class PostAttention(nn.Module):
    def __init__(self, args: FishOmniCodecArgs) -> None:
        super().__init__()
        self.n_head, self.head_dim = args.post_n_head, args.post_head_dim
        width = self.n_head * self.head_dim
        self.wqkv = nn.Linear(args.latent_dim, 3 * width, bias=False)
        self.wo = nn.Linear(width, args.latent_dim, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, kv_prev: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (output, k, v) with this window's rotated keys and values [B, H, T, D]; `kv_prev` holds the previous
        window's, prepended along the key axis (the mask then has prev + current key columns)."""
        bsz, seqlen, _ = x.shape
        width = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([width, width, width], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)
        cos, sin = cos[None, :, None], sin[None, :, None]
        half = self.head_dim // 2

        def rotate(t: Tensor) -> Tensor:
            tf = t.float()
            first, second = tf[..., :half], tf[..., half:]
            c, s = cos[..., :half], sin[..., :half]
            return torch.cat((first * c - second * s, second * c + first * s), -1).type_as(t)

        q, k, v = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2), v.transpose(1, 2)
        keys, values = (k, v) if kv_prev is None else (torch.cat((kv_prev[0], k), 2), torch.cat((kv_prev[1], v), 2))
        scores = torch.matmul(q, keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        probs = torch.softmax(scores + mask.to(scores.dtype), dim=-1)
        y = torch.matmul(probs, values)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
        return self.wo(y), k, v


class PostFeedForward(nn.Module):
    def __init__(self, args: FishOmniCodecArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(args.latent_dim, args.post_intermediate_size, bias=False)
        self.w3 = nn.Linear(args.latent_dim, args.post_intermediate_size, bias=False)
        self.w2 = nn.Linear(args.post_intermediate_size, args.latent_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class PostBlock(nn.Module):
    def __init__(self, args: FishOmniCodecArgs) -> None:
        super().__init__()
        self.attention = PostAttention(args)
        self.feed_forward = PostFeedForward(args)
        self.ffn_norm = RMSNorm(args.latent_dim, args.post_norm_eps)
        self.attention_norm = RMSNorm(args.latent_dim, args.post_norm_eps)
        self.attention_layer_scale = LayerScale(args.latent_dim)
        self.ffn_layer_scale = LayerScale(args.latent_dim)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, kv_prev: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        attn, k, v = self.attention(self.attention_norm(x), cos, sin, mask, kv_prev)
        h = x + self.attention_layer_scale(attn)
        return h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h))), k, v


class WindowLimitedTransformer(nn.Module):
    """Upstream `WindowLimitedTransformer` (channels first, input_dim == dim, so input/output projections are
    identities)."""

    def __init__(self, args: FishOmniCodecArgs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(PostBlock(args) for _ in range(args.post_n_layer))
        self.norm = RMSNorm(args.latent_dim, args.post_norm_eps)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, kv_prev: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """x [B, C, T] -> (z [B, T, C] time-major, kv [L, 2, B, H, T, D]); `kv_prev` is the previous window's `kv`.

        The output stays time-major (the caller transposes to channels-first) so that a window's context frames are
        concatenated on the time axis before the transpose: a last-axis concat of channels-first [B, C, 16] and
        [B, C, 128] is split to the host on RBLN-CA25 (WrongShape).
        """
        x = x.transpose(1, 2)
        states = []
        for index, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, mask, None if kv_prev is None else (kv_prev[index, 0], kv_prev[index, 1]))
            states.append(torch.stack((k, v)))
        return self.norm(x), torch.stack(states)


class CausalConvNet(nn.Module):
    """Upstream `CausalConvNet` with stride 1: left pad (kernel - 1) * dilation, no right pad."""

    def __init__(self, cin: int, cout: int, kernel_size: int, dilation: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size, dilation=dilation, groups=groups)
        self.padding = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class CausalTransConvNet(nn.Module):
    """Upstream `CausalTransConvNet`: ConvTranspose1d (output_padding 0), then drop the last kernel - stride
    samples."""

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel_size, stride=stride)
        self.trim = kernel_size - stride

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x if self.trim == 0 else x[..., : x.shape[-1] - self.trim]


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = CausalConvNet(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.dwconv(x).permute(0, 2, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        y = (self.gamma * y).permute(0, 2, 1)
        return x + y


class UpsampleStage(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, factor: int) -> None:
        super().__init__()
        self.conv = CausalTransConvNet(dim_in, dim_out, factor, factor)
        self.convnext = ConvNeXtBlock(dim_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.convnext(self.conv(x))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.snake1 = Snake1d(dim)
        self.conv1 = CausalConvNet(dim, dim, 7, dilation=dilation)
        self.snake2 = Snake1d(dim)
        self.conv2 = CausalConvNet(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, stride: int) -> None:
        super().__init__()
        self.snake = Snake1d(dim_in)
        self.conv = CausalTransConvNet(dim_in, dim_out, 2 * stride, stride)
        self.res = nn.ModuleList(ResidualUnit(dim_out, d) for d in (1, 3, 9))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.snake(x))
        for unit in self.res:
            x = unit(x)
        return x


class FishOmniCodecDecoder(nn.Module):
    """Device part of the codec decoder for one frame window."""

    def __init__(self, args: FishOmniCodecArgs, config=None) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.out_projs = nn.ModuleList(
            nn.Conv1d(args.codebook_dim, args.latent_dim, 1) for _ in range(args.num_codebooks)
        )
        self.post_module = WindowLimitedTransformer(args)
        dims = [args.latent_dim] * (len(args.downsample_factor) + 1)
        self.upsample = nn.ModuleList(
            UpsampleStage(dims[i + 1], dims[i], f) for i, f in reversed(list(enumerate(args.downsample_factor)))
        )
        self.conv_in = CausalConvNet(args.latent_dim, args.decoder_dim, 7)
        blocks, dim = [], args.decoder_dim
        for stride in args.decoder_rates:
            blocks.append(DecoderBlock(dim, dim // 2, stride))
            dim //= 2
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = Snake1d(dim)
        self.conv_out = CausalConvNet(dim, 1, 7)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv_out.conv.weight.dtype

    def latent(
        self, code_vectors: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, kv_prev: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """code_vectors [B, num_codebooks, codebook_dim, T] (host lookups) -> post_module (z [B, T, C], kv state)."""
        semantic = self.out_projs[0](code_vectors[:, 0])
        residual = 0.0
        for i in range(1, self.args.num_codebooks):
            residual = residual + self.out_projs[i](code_vectors[:, i])
        return self.post_module(semantic + residual, cos, sin, mask, kv_prev)

    def vocode(self, z: Tensor) -> Tensor:
        """post_module output z [B, C, T] -> audio [B, 1, T * samples_per_frame] (upsample + causal Decoder)."""
        for stage in self.upsample:
            z = stage(z)
        x = self.conv_in(z)
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.conv_out(self.snake_out(x)))

    def forward(self, code_vectors: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        """One pass from frame 0: cos / sin [T, head_dim] fp32, mask [T, T] additive. Returns audio."""
        return self.vocode(self.latent(code_vectors, cos, sin, mask)[0].transpose(1, 2))


class FishOmniCodecWindow(nn.Module):
    """Compiled graph `decoder`: the first window (frames 0 .. W - 1) with the window mask baked in as a constant.

    Returns (audio [B, 1, W * samples_per_frame], z [B, W, C] (time-major), kv [L, 2, B, H, W, D]); z and kv seed the
    next window.
    """

    def __init__(self, model: FishOmniCodecDecoder, window_frames: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mask", window_causal_additive_mask(window_frames, model.args.post_window_size), persistent=False
        )

    def forward(self, code_vectors: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z, kv = self.model.latent(code_vectors, cos, sin, self.mask)
        return self.model.vocode(z.transpose(1, 2)), z, kv


def next_window_additive_mask(window_frames: int, state_frames: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """[W, S + W] additive mask of `make_window_limited_mask` for queries t0 .. t0 + W - 1 over keys t0 - S .. t0 + W - 1.

    Key column j < S is absolute frame t0 - S + j, column S + j is t0 + j; query i sees key a iff t0 + i - S < a <= t0 + i
    (S = post_window_size).
    """
    query = torch.arange(window_frames).view(-1, 1) + state_frames
    key = torch.arange(state_frames + window_frames).view(1, -1)
    visible = (key <= query) & (key > query - state_frames)
    return torch.zeros(window_frames, state_frames + window_frames, dtype=dtype).masked_fill(~visible, -1e9)


class FishOmniCodecNextWindow(nn.Module):
    """Compiled graph `decoder_next`: W frames after a previous window, exact against one pass over the whole stream.

    Inputs: code_vectors [B, num_codebooks, codebook_dim, W], cos / sin [W, head_dim] at the window's absolute frames,
    z_context [B, context_frames, C] (the previous window's last post_module frames, time-major), kv_prev
    [L, 2, B, H, S, D] (the previous window's `kv`, S = post_window_size = W). Returns (audio
    [B, 1, (context_frames + W) * samples_per_frame], z [B, W, C], kv); the host drops the first
    context_frames * samples_per_frame samples.
    """

    def __init__(self, model: FishOmniCodecDecoder, window_frames: int) -> None:
        super().__init__()
        if window_frames != model.args.post_window_size:
            raise ValueError("decoder_next keeps one window of K/V state: window_frames must equal post_window_size")
        self.model = model
        self.register_buffer(
            "mask", next_window_additive_mask(window_frames, model.args.post_window_size), persistent=False
        )

    def forward(
        self, code_vectors: Tensor, cos: Tensor, sin: Tensor, z_context: Tensor, kv_prev: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        z, kv = self.model.latent(code_vectors, cos, sin, self.mask, kv_prev)
        return self.model.vocode(torch.cat((z_context, z), dim=1).transpose(1, 2)), z, kv


@torch.no_grad()
def fold_weight_norm(g: Tensor, v: Tensor, dim: int = 0) -> Tensor:
    """weight = g * v / ||v|| (norm over every dim but `dim`), as torch weight_norm computes it."""
    return torch._weight_norm(v, g, dim)


@torch.no_grad()
def permute_codec_rope_rows(state: dict[str, Tensor], args: FishOmniCodecArgs) -> dict[str, Tensor]:
    """Permute Q and K rows of every post_module `wqkv` into rotate-half order (in place)."""
    perm = rope_row_permutation(args.post_head_dim)
    width = args.post_n_head * args.post_head_dim
    for name in list(state):
        if name.startswith("post_module.") and name.endswith("attention.wqkv.weight"):
            q, k, v = state[name].split((width, width, width), 0)
            q = q.reshape(args.post_n_head, args.post_head_dim, -1)[:, perm].reshape(width, -1)
            k = k.reshape(args.post_n_head, args.post_head_dim, -1)[:, perm].reshape(width, -1)
            state[name] = torch.cat((q, k, v), 0).contiguous()
    return state


def codec_code_vectors(
    codes: torch.Tensor, codebooks: Sequence[torch.Tensor], args: FishOmniCodecArgs
) -> torch.Tensor:
    """Host codebook lookups of upstream `DownsampleResidualVectorQuantize.decode` (codes clamped to each codebook,
    as upstream clamps them): codes [B, num_codebooks, T] -> [B, num_codebooks, codebook_dim, T] fp32."""
    sizes = [args.semantic_codebook_size] + [args.codebook_size] * args.n_codebooks
    vectors = [
        F.embedding(codes[:, i].clamp(max=size - 1), codebooks[i].float()).transpose(1, 2)
        for i, size in enumerate(sizes)
    ]
    return torch.stack(vectors, dim=1)


def decode_codec_windows(
    run_first: Callable[..., Any],
    run_next: Callable[..., Any],
    vectors: torch.Tensor,
    args: FishOmniCodecArgs,
    window: int,
    context: int,
) -> torch.Tensor:
    """Drive the `decoder` / `decoder_next` graphs over vectors [B, num_codebooks, codebook_dim, T].

    The stream is zero-padded at the end to whole windows (every op is causal, so padding never reaches earlier
    samples) and the output is cut back to T * samples_per_frame samples.
    """
    frames = vectors.shape[-1]
    windows = math.ceil(frames / window)
    vectors = F.pad(vectors, (0, windows * window - frames))
    cos, sin = post_module_freqs_table(windows * window, args.post_head_dim, args.post_rope_base)
    spf = args.samples_per_frame
    pieces, z, kv = [], None, None
    for w in range(windows):
        span = slice(w * window, (w + 1) * window)
        if w == 0:
            audio, z, kv = run_first(vectors[..., span].contiguous(), cos[span], sin[span])
        else:
            audio, z, kv = run_next(
                vectors[..., span].contiguous(), cos[span], sin[span], z[:, -context:].contiguous(), kv
            )
            audio = audio[..., context * spf :]
        pieces.append(audio)
    return torch.cat(pieces, dim=-1)[..., : frames * spf]


_DECODER_BLOCK = re.compile(r"^decoder\.model\.([1-4])\.block\.(\d)\.(.*)$")
_RESIDUAL_UNIT = {"0": "snake1.", "1": "conv1.", "2": "snake2.", "3": "conv2."}


def _codec_key(key: str, args: FishOmniCodecArgs) -> str | None:
    """Device module name of a codec.pth tensor; None for tensors the decoder does not use (encoder, pre_module, ...)."""
    if key.startswith("quantizer.semantic_quantizer.quantizers.0.out_proj."):
        return "out_projs.0." + key.rsplit("out_proj.", 1)[1]
    if m := re.match(r"^quantizer\.quantizer\.quantizers\.(\d+)\.out_proj\.(.*)$", key):
        return f"out_projs.{int(m[1]) + 1}.{m[2]}"
    if key.startswith(("quantizer.post_module.layers.", "quantizer.post_module.norm.")):
        return key[len("quantizer.") :]
    if m := re.match(r"^quantizer\.upsample\.(\d)\.0\.conv\.(.*)$", key):
        return f"upsample.{m[1]}.conv.conv.{m[2]}"
    if m := re.match(r"^quantizer\.upsample\.(\d)\.1\.(.*)$", key):
        return f"upsample.{m[1]}.convnext.{m[2]}"
    if key.startswith("decoder.model.0.conv."):
        return "conv_in.conv." + key[len("decoder.model.0.conv.") :]
    if m := _DECODER_BLOCK.match(key):
        block, part, rest = int(m[1]) - 1, m[2], m[3]
        if part == "0":
            return f"blocks.{block}.snake.{rest}"
        if part == "1":
            return f"blocks.{block}.conv.{rest}"
        unit = re.match(r"^block\.(\d)\.(.*)$", rest)
        return f"blocks.{block}.res.{int(part) - 2}.{_RESIDUAL_UNIT[unit[1]]}{unit[2]}"
    if key.startswith("decoder.model.5."):
        return "snake_out." + key[len("decoder.model.5.") :]
    if key.startswith("decoder.model.6.conv."):
        return "conv_out.conv." + key[len("decoder.model.6.conv.") :]
    return None


def load_fish_omni_codec_state_dict(
    path: Path, args: FishOmniCodecArgs
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Decoder state dict (weight_norm folded, post_module RoPE rows permuted, fp32) and the 10 host codebooks from
    codec.pth, read with `weights_only=True`."""
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator" in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    codebooks = [state["quantizer.semantic_quantizer.quantizers.0.codebook.weight"].float().clone()]
    codebooks += [
        state[f"quantizer.quantizer.quantizers.{i}.codebook.weight"].float().clone() for i in range(args.n_codebooks)
    ]
    raw = {}
    for key, value in state.items():
        name = _codec_key(key, args)
        if name is not None:
            raw[name] = value.float()
    out = {}
    for name, value in raw.items():
        if name.endswith((".weight_v", ".parametrizations.weight.original1")):
            continue
        if name.endswith(".weight_g"):
            prefix = name[: -len("weight_g")]
            out[prefix + "weight"] = fold_weight_norm(value, raw[prefix + "weight_v"], 0)
        elif name.endswith(".parametrizations.weight.original0"):
            prefix = name[: -len("parametrizations.weight.original0")]
            out[prefix + "weight"] = fold_weight_norm(value, raw[prefix + "parametrizations.weight.original1"], 0)
        else:
            out[name] = value.clone()
    return permute_codec_rope_rows(out, args), codebooks
