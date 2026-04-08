import argparse
import json
from pathlib import Path
import os

import rebel
import torch
from diffusers.models.transformers.transformer_flux2 import dispatch_attention_fn


def expand_cos_sin(cos: torch.Tensor, sin: torch.Tensor, tensor_layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor_layout == "bshd":
        return cos[None, :, None, :], sin[None, :, None, :]
    if tensor_layout == "bhsd":
        return cos[None, None, :, :], sin[None, None, :, :]
    raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")


def apply_flux2_rotary_interleaved_slice(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tensor_layout: str
) -> torch.Tensor:
    cos, sin = expand_cos_sin(cos, sin, tensor_layout)
    cos_half = cos[..., ::2]
    sin_half = sin[..., ::2]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos_half - x_odd * sin_half
    out_odd = x_odd * cos_half + x_even * sin_half
    return torch.stack([out_even, out_odd], dim=-1).flatten(3)


def apply_flux2_rotary_pairwise_reshape(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tensor_layout: str
) -> torch.Tensor:
    cos, sin = expand_cos_sin(cos, sin, tensor_layout)

    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    cos_pairs = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_pairs = sin.reshape(*sin.shape[:-1], -1, 2)

    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    cos_pair = cos_pairs[..., 0]
    sin_pair = sin_pairs[..., 0]

    out_real = x_real * cos_pair - x_imag * sin_pair
    out_imag = x_imag * cos_pair + x_real * sin_pair
    return torch.stack([out_real, out_imag], dim=-1).reshape_as(x)


def apply_flux2_rotary_upstream_style(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tensor_layout: str
) -> torch.Tensor:
    cos, sin = expand_cos_sin(cos, sin, tensor_layout)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def apply_contiguous_half_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tensor_layout: str
) -> torch.Tensor:
    def rotate_half_contiguous(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos, sin = expand_cos_sin(cos, sin, tensor_layout)
    return (x * cos) + (rotate_half_contiguous(x) * sin)


class RotaryAttentionToy(torch.nn.Module):
    def __init__(
        self,
        num_blocks: int = 2,
        disable_second_rotary: bool = False,
        skip_second_attention: bool = False,
        rotary_impl: str = "interleaved_slice",
        tensor_layout: str = "bshd",
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.disable_second_rotary = disable_second_rotary
        self.skip_second_attention = skip_second_attention
        self.rotary_impl = rotary_impl
        self.tensor_layout = tensor_layout

    def _block(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        apply_rotary: bool,
        skip_attention: bool = False,
    ) -> torch.Tensor:
        if apply_rotary:
            if self.rotary_impl == "interleaved_slice":
                query = apply_flux2_rotary_interleaved_slice(query, cos, sin, self.tensor_layout)
                key = apply_flux2_rotary_interleaved_slice(key, cos, sin, self.tensor_layout)
            elif self.rotary_impl == "pairwise_reshape":
                query = apply_flux2_rotary_pairwise_reshape(query, cos, sin, self.tensor_layout)
                key = apply_flux2_rotary_pairwise_reshape(key, cos, sin, self.tensor_layout)
            elif self.rotary_impl == "upstream_style":
                query = apply_flux2_rotary_upstream_style(query, cos, sin, self.tensor_layout)
                key = apply_flux2_rotary_upstream_style(key, cos, sin, self.tensor_layout)
            elif self.rotary_impl == "contiguous_half":
                query = apply_contiguous_half_rotary(query, cos, sin, self.tensor_layout)
                key = apply_contiguous_half_rotary(key, cos, sin, self.tensor_layout)
            else:
                raise ValueError(f"Unsupported rotary_impl: {self.rotary_impl}")
        if skip_attention:
            return query + key

        if self.tensor_layout == "bshd":
            return dispatch_attention_fn(query, key, value)
        if self.tensor_layout == "bhsd":
            hidden_states = dispatch_attention_fn(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
            )
            return hidden_states.permute(0, 2, 1, 3)
        raise ValueError(f"Unsupported tensor_layout: {self.tensor_layout}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._block(query, key, value, cos, sin, apply_rotary=True)
        if self.num_blocks == 1:
            return hidden_states

        query = query + hidden_states
        key = key + hidden_states
        value = value + hidden_states
        return self._block(
            query,
            key,
            value,
            cos,
            sin,
            apply_rotary=not self.disable_second_rotary,
            skip_attention=self.skip_second_attention,
        )


def build_compiled_model(args: argparse.Namespace) -> Path:
    model = RotaryAttentionToy(
        num_blocks=args.num_blocks,
        disable_second_rotary=args.disable_second_rotary,
        skip_second_attention=args.skip_second_attention,
        rotary_impl=args.rotary_impl,
        tensor_layout=args.tensor_layout,
    ).eval()

    if args.tensor_layout == "bshd":
        qkv_shape = [1, args.sequence_length, args.num_heads, args.head_dim]
    elif args.tensor_layout == "bhsd":
        qkv_shape = [1, args.num_heads, args.sequence_length, args.head_dim]
    else:
        raise ValueError(f"Unsupported tensor_layout: {args.tensor_layout}")

    compiled_model = rebel.compile_from_torch(
        model,
        input_info=[
            ("query", qkv_shape, "float32"),
            ("key", qkv_shape, "float32"),
            ("value", qkv_shape, "float32"),
            ("cos", [args.sequence_length, args.head_dim], "float32"),
            ("sin", [args.sequence_length, args.head_dim], "float32"),
        ],
    )
    compiled_model.save(args.output)
    return args.output


def run_runtime(compiled_model_path: Path, args: argparse.Namespace) -> tuple[torch.Tensor, rebel.Runtime]:
    runtime = rebel.Runtime(compiled_model_path, tensor_type="pt", device=args.device)
    runtime.flush_reports()
    if args.tensor_layout == "bshd":
        qkv_shape = (1, args.sequence_length, args.num_heads, args.head_dim)
    elif args.tensor_layout == "bhsd":
        qkv_shape = (1, args.num_heads, args.sequence_length, args.head_dim)
    else:
        raise ValueError(f"Unsupported tensor_layout: {args.tensor_layout}")

    query = torch.randn(*qkv_shape)
    key = torch.randn(*qkv_shape)
    value = torch.randn(*qkv_shape)
    cos = torch.randn(args.sequence_length, args.head_dim)
    sin = torch.randn(args.sequence_length, args.head_dim)
    return runtime(query, key, value, cos, sin), runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["compile", "run", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--output", type=Path, default=Path("tmp_rotary_attn_toy.rbln"))
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--disable-second-rotary", action="store_true", default=False)
    parser.add_argument("--skip-second-attention", action="store_true", default=False)
    parser.add_argument(
        "--rotary-impl",
        choices=["interleaved_slice", "pairwise_reshape", "contiguous_half", "upstream_style"],
        default="interleaved_slice",
    )
    parser.add_argument("--tensor-layout", choices=["bshd", "bhsd"], default="bshd")
    parser.add_argument("--sequence-length", type=int, default=4608)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    compiled_model_path = args.output
    if args.mode in {"compile", "compile-and-run"}:
        compiled_model_path = build_compiled_model(args)
        print(f"Compiled model saved to {compiled_model_path}")

    if args.mode in {"run", "compile-and-run"}:
        # os.environ["RBLN_PROFILER"] = "1"
        os.environ["RBLN_RUNTIME_TIMER"] = "1"
        output, runtime = run_runtime(compiled_model_path, args)
        print(f"Runtime output shape: {tuple(output.shape)}")
        report = runtime.get_reports()
        json.dump(report, open("repro_report.json", "w"), indent=4)


if __name__ == "__main__":
    main()

# Results:
# Default settings are minimal reproducer of profiler failure in FLUX.2-Klein implementation. (python repro_flux2_profiler_error.py)
# - Upstream RoPE implementation(--rotary-impl upstream_style) fails to compile with "RBLNCompileError: Graph Generation: [DEVICE_GRAPH_CONVERSION]"
# - Interleaved RoPE workaround(--rotary-impl interleaved_slice) passes compile and runtime, but fails to profile with core dump.
# - With Interleaved RoPE, using only one block(--num-blocks 1) passes profiling, but it fails from 2 blocks.
# - Using both bhsd layout and contiguous_half rotary implementation(--tensor-layout bhsd --rotary-impl contiguous_half) passes profiling.
