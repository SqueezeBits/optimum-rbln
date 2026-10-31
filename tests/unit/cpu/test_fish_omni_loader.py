import inspect
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from optimum.rbln import (
    RBLNFishOmniCodecConfig,
    RBLNFishOmniForCausalLM,
    RBLNFishOmniForCausalLMConfig,
    RBLNFishOmniTower,
    RBLNFishOmniTowerConfig,
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5MoeForCausalLMConfig,
)
from optimum.rbln.transformers.models.fish_omni.fish_omni_architecture import (
    FishOmniFp8MoeMLP,
    FishOmniFp8SparseMoeBlock,
)
from optimum.rbln.transformers.models.fish_omni.fish_omni_tower_architecture import TOWER_DEVICE_PREFIXES
from optimum.rbln.transformers.models.fish_omni.modeling_fish_omni import (
    dequantize_block_fp8,
    fish_omni_text_config,
    load_fish_omni_text_state_dict,
)
from optimum.rbln.transformers.models.fish_omni.modeling_fish_omni_tower import build_fish_omni_tower
from optimum.rbln.transformers.models.qwen3_5_moe.qwen3_5_moe_architecture import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)

from ...test_base import FP8_BLOCK, TINY_TEXT, make_tiny_fish_omni_checkpoint, quantize_block_fp8


def bf16_experts() -> RBLNFishOmniForCausalLMConfig:
    return RBLNFishOmniForCausalLMConfig(expert_storage="bf16")


def test_get_pytorch_model_loads_text_backbone(tmp_path):
    source, expected = make_tiny_fish_omni_checkpoint(tmp_path)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32, rbln_config=bf16_experts())
    assert isinstance(model, Qwen3_5MoeForCausalLM)
    state = model.state_dict()
    for key, value in expected.items():
        torch.testing.assert_close(state[key], value, rtol=1e-6, atol=1e-6)
    for key, value in source.state_dict().items():
        if "experts.gate_up_proj" in key or "experts.down_proj" in key:
            continue
        torch.testing.assert_close(state[key], value)
    assert not any(t.device.type == "meta" for t in state.values())


def test_layer_truncation(tmp_path):
    make_tiny_fish_omni_checkpoint(tmp_path)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(
        tmp_path, dtype=torch.float32, num_hidden_layers=1, layer_types=["linear_attention"]
    )
    assert len(model.model.layers) == 1


def test_tiny_tokenizer_ids_fit_vocab(tmp_path):
    make_tiny_fish_omni_checkpoint(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, padding_side="left")
    text = "Who are you?\n" + "".join(chr(c) for c in range(256))
    ids = tokenizer(text)["input_ids"]
    assert min(ids) >= 0
    assert max(ids) < TINY_TEXT["vocab_size"]
    assert tokenizer.eos_token_id == 0
    assert tokenizer.decode(tokenizer("Who are you?")["input_ids"]) == "Who are you?"


def test_dequantize_block_fp8_matches_blockwise_product():
    q, scale = quantize_block_fp8(torch.randn(256, 384))
    out = dequantize_block_fp8(q, scale)
    for i in range(2):
        for j in range(3):
            rows, cols = slice(i * FP8_BLOCK, (i + 1) * FP8_BLOCK), slice(j * FP8_BLOCK, (j + 1) * FP8_BLOCK)
            torch.testing.assert_close(out[rows, cols], q[rows, cols].float() * scale[i, j])


def test_fp8_experts_stay_fp8_with_block_scales(tmp_path):
    make_tiny_fish_omni_checkpoint(tmp_path)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32)
    reference = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32, rbln_config=bf16_experts())
    for layer, ref_layer in zip(model.model.layers, reference.model.layers, strict=True):
        experts = layer.mlp.experts
        assert experts.gate_up_proj.dtype == experts.down_proj.dtype == torch.float8_e4m3fn
        gate_up = torch.stack(
            [dequantize_block_fp8(experts.gate_up_proj[e], experts.gate_up_proj_scale_inv[e]) for e in range(8)]
        )
        down = torch.stack(
            [dequantize_block_fp8(experts.down_proj[e], experts.down_proj_scale_inv[e]) for e in range(8)]
        )
        torch.testing.assert_close(gate_up, ref_layer.mlp.experts.gate_up_proj, rtol=0, atol=0)
        torch.testing.assert_close(down, ref_layer.mlp.experts.down_proj, rtol=0, atol=0)


def test_fp8_moe_block_matches_dequantized_reference(tmp_path):
    rebel = pytest.importorskip("rebel")  # noqa: F841  registers rbln_custom_ops
    make_tiny_fish_omni_checkpoint(tmp_path)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32)
    reference = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32, rbln_config=bf16_experts())
    x = torch.randn(1, 16, TINY_TEXT["hidden_size"])
    with torch.no_grad():
        for layer, ref_layer in zip(model.model.layers, reference.model.layers, strict=True):
            expected = ref_layer.mlp(x)
            torch.testing.assert_close(FishOmniFp8SparseMoeBlock(layer.mlp)(x), expected, rtol=1e-5, atol=1e-5)


def test_config_is_decoder_only_config():
    assert RBLNFishOmniForCausalLMConfig(batch_size=1, max_seq_len=8192).max_seq_len == 8192


@pytest.mark.parametrize(
    "config_cls", [RBLNFishOmniForCausalLMConfig, RBLNFishOmniTowerConfig, RBLNFishOmniCodecConfig]
)
def test_config_documents_its_arguments_in_init(config_cls):
    # As every other RBLN config: `Args:` in the `__init__` docstring, one entry per parameter.
    assert "Args:" not in config_cls.__doc__
    doc = config_cls.__init__.__doc__ or ""
    names = [n for n in inspect.signature(config_cls.__init__).parameters if n != "self"]
    assert [n for n in names if f"    {n} (" not in doc and f"    {n}:" not in doc] == []


def test_config_takes_several_decoder_batch_sizes():
    # Qwen3.5 refuses decoder bucketing; FishOmni serves rows [:b] of the linear state per decoder_batch_<b> graph.
    config = RBLNFishOmniForCausalLMConfig(batch_size=8, max_seq_len=8192, decoder_batch_sizes=[1, 2, 4, 8])
    assert config.decoder_batch_sizes == [8, 4, 2, 1]
    assert config.gdn_chunk_size == 128
    assert config.linear_attention_layers == []


def test_config_goes_through_the_qwen3_5_config_init(monkeypatch):
    # The one-decoder-batch guard is switched off by the class hook, not by skipping the parent __init__, so a field
    # the parent sets reaches FishOmni.
    calls = []
    parent_init = RBLNQwen3_5ForCausalLMConfig.__init__

    def spy(self, *args, **kwargs):
        calls.append(type(self).__name__)
        parent_init(self, *args, **kwargs)

    monkeypatch.setattr(RBLNQwen3_5ForCausalLMConfig, "__init__", spy)
    config = RBLNFishOmniForCausalLMConfig(
        batch_size=8, max_seq_len=8192, decoder_batch_sizes=[8, 4, 2, 1], gdn_chunk_size=64
    )
    assert calls == ["RBLNFishOmniForCausalLMConfig"]
    assert config.gdn_chunk_size == 64
    assert RBLNFishOmniForCausalLMConfig._allows_decoder_batch_bucketing is True


@pytest.mark.parametrize("config_cls", [RBLNQwen3_5ForCausalLMConfig, RBLNQwen3_5MoeForCausalLMConfig])
def test_qwen3_5_still_refuses_several_decoder_batch_sizes(config_cls):
    assert config_cls._allows_decoder_batch_bucketing is False
    with pytest.raises(ValueError, match=r"Qwen3\.5 takes one decoder batch size, got \[8, 4, 2, 1\]"):
        config_cls(batch_size=8, max_seq_len=8192, decoder_batch_sizes=[8, 4, 2, 1])


@pytest.mark.parametrize("norm_topk_prob", [None, True])
def test_text_config_takes_renormalized_top_k(norm_topk_prob):
    text = dict(TINY_TEXT) if norm_topk_prob is None else {**TINY_TEXT, "norm_topk_prob": norm_topk_prob}
    assert fish_omni_text_config({"text_config": text}).num_experts_per_tok == TINY_TEXT["num_experts_per_tok"]


def test_text_config_refuses_norm_topk_prob_false():
    # FishOmniFp8MoeMLP and the Qwen3.5-MoE block always renormalize the top-k routing weights.
    with pytest.raises(NotImplementedError, match="norm_topk_prob must be True"):
        fish_omni_text_config({"text_config": {**TINY_TEXT, "norm_topk_prob": False}})


def _checkpoint_spans(paths: list[Path]) -> list[tuple[int, int]]:
    names = {str(p.resolve()) for p in paths}
    spans = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5] in names:
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            spans.append((lo, hi))
    return spans


def _mmap_views(tensors: dict[str, torch.Tensor], checkpoint_dir: Path) -> list[str]:
    """Names of the tensors whose storage lies in a mapping of one of the checkpoint's safetensors files."""
    spans = _checkpoint_spans(sorted(checkpoint_dir.glob("*.safetensors")))
    return [name for name, t in tensors.items() if any(lo <= t.untyped_storage().data_ptr() < hi for lo, hi in spans)]


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="reads the process memory map from /proc (Linux)")
@pytest.mark.parametrize("expert_storage", ["fp8", "bf16"])
def test_loader_keeps_no_view_of_the_checkpoint_mmap(tmp_path, expert_storage):
    # RBLNMoeLoadMixin copies whatever transformers left as a view of the safetensors mmap; FishOmni's loader copies
    # every tensor itself. The tiny checkpoint is float32 and is loaded as float32, so no dtype cast copies for it.
    make_tiny_fish_omni_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config.update(
        semantic_start_token_id=100,
        semantic_end_token_id=115,
        audio_decoder_config={"vocab_size": 16, "vq_backbone_codebooks": 2},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(
        {"audio_decoder.codebook_embeddings.weight": torch.randn(64, TINY_TEXT["hidden_size"])},
        str(tmp_path / "model-audio.safetensors"),
    )
    rbln_config = RBLNFishOmniForCausalLMConfig(expert_storage=expert_storage)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32, rbln_config=rbln_config)
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    tensors["codebook_weight"] = model.fish_omni_backbone_embedding["codebook_weight"]
    views = _mmap_views(tensors, tmp_path)
    assert views == [], f"{len(views)} of {len(tensors)} tensors are views of the checkpoint mmap: {views[:5]}"


TINY_TOWER = {
    "dim": 32,
    "intermediate_size": 48,
    "n_head": 4,
    "n_local_heads": 2,
    "head_dim": 8,
    "norm_eps": 1e-6,
    "rope_base": 10000.0,
    "rotary_percent": 0.5,
    "vocab_size": 16,
    "num_codebooks": 3,
    "shared_enrichment_layers": 1,
    "vq_backbone_codebooks": 1,
}


def write_tiny_tower_checkpoint(path: Path, dtype: torch.dtype) -> None:
    """Write the audio decoder tensors the tower loader reads (device layers and heads, host embedding tables),
    stored in `dtype`."""
    torch.manual_seed(0)
    args = build_fish_omni_tower(TINY_TOWER, dtype=torch.float32).args
    checkpoint_prefix = {new: old for old, new in TOWER_DEVICE_PREFIXES.items()}
    tensors = {}
    for name, value in build_fish_omni_tower(TINY_TOWER, dtype=torch.float32).state_dict().items():
        prefix = next(p for p in checkpoint_prefix if name.startswith(p))
        tensors[checkpoint_prefix[prefix] + name[len(prefix) :]] = torch.randn_like(value)
    vocab, dim = args.vocab_size, args.dim
    for i in range(args.shared_enrichment_layers):
        tensors[f"audio_decoder_mtp_shared.layers.{i}.semantic_embedding.weight"] = torch.randn(vocab, dim)
    for i in range(args.num_residual_layers):
        tensors[f"audio_decoder_mtp.layers.{i}.codebook_embedding.weight"] = torch.randn(vocab, dim)
    tensors["audio_decoder.codebook_embeddings.weight"] = torch.randn(args.num_codebooks * vocab, dim)
    save_file({k: v.to(dtype).contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"audio_decoder_config": TINY_TOWER}))


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="reads the process memory map from /proc (Linux)")
@pytest.mark.parametrize("dtype", [torch.float32, None], ids=["float32", "default-bfloat16"])
def test_tower_loader_keeps_no_view_of_the_checkpoint_mmap(tmp_path, dtype):
    # The tower loader casts to the model dtype (bfloat16 unless given). A checkpoint stored in that dtype is not
    # cast, so the host embedding tables have to be copied out of the mapping explicitly.
    stored = dtype or torch.bfloat16
    write_tiny_tower_checkpoint(tmp_path, stored)
    model = RBLNFishOmniTower.get_pytorch_model(tmp_path, **({"dtype": dtype} if dtype else {}))
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    tensors.update({f"host_embeddings.{k}": v for k, v in model.host_embeddings.items()})
    assert len(model.host_embeddings) == 4
    assert {t.dtype for t in tensors.values()} == {stored}
    views = _mmap_views(tensors, tmp_path)
    assert views == [], f"{len(views)} of {len(tensors)} tensors are views of the checkpoint mmap: {views[:5]}"


def test_expert_storage_is_an_rbln_config_field():
    assert RBLNFishOmniForCausalLMConfig().expert_storage == "fp8"
    assert RBLNFishOmniForCausalLMConfig(expert_storage="bf16").expert_storage == "bf16"
    assert RBLNFishOmniForCausalLMConfig()._prepare_for_serialization()["expert_storage"] == "fp8"


@pytest.mark.parametrize("value", ["int8", "FP8", "bfloat16", ""])
def test_expert_storage_refuses_other_values(value):
    with pytest.raises(ValueError, match='expert_storage must be "fp8" or "bf16"'):
        RBLNFishOmniForCausalLMConfig(expert_storage=value)
    with pytest.raises(ValueError, match='expert_storage must be "fp8" or "bf16"'):
        load_fish_omni_text_state_dict(Path("/nonexistent"), None, torch.float32, value)


def test_rbln_config_saved_before_expert_storage_loads_as_fp8(tmp_path):
    # The rbln_config.json of an artifact compiled before the field has no expert_storage key; those artifacts were
    # compiled with the block-FP8 experts, the default.
    RBLNFishOmniForCausalLMConfig(batch_size=1, max_seq_len=8192).save(tmp_path)
    saved = json.loads((tmp_path / "rbln_config.json").read_text())
    saved.pop("expert_storage", None)
    (tmp_path / "rbln_config.json").write_text(json.dumps(saved))
    assert RBLNFishOmniForCausalLMConfig.from_pretrained(tmp_path).expert_storage == "fp8"


def test_get_pytorch_model_takes_expert_storage_from_rbln_config(tmp_path):
    make_tiny_fish_omni_checkpoint(tmp_path)
    default = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32)
    bf16 = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32, rbln_config=bf16_experts())
    assert default.model.layers[0].mlp.experts.gate_up_proj.dtype == torch.float8_e4m3fn
    assert bf16.model.layers[0].mlp.experts.gate_up_proj.dtype == torch.float32
    with pytest.raises(ValueError, match="expert_storage is an rbln_config field"):
        RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32, expert_storage="bf16")


def test_compile_refuses_experts_loaded_another_way(tmp_path):
    # rbln_config.json records expert_storage, so a model loaded with the other storage is refused before tracing.
    make_tiny_fish_omni_checkpoint(tmp_path)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32)
    message = r"loaded with expert_storage='fp8' but rbln_config\.expert_storage is 'bf16'"
    with pytest.raises(ValueError, match=message):
        RBLNFishOmniForCausalLM._wrap_model_if_needed(model, bf16_experts())


def test_fp8_sparse_moe_block_is_built_by_the_qwen3_5_moe_init(tmp_path, monkeypatch):
    # FishOmni changes only the routed-experts class, through the sibling's `_experts_cls` hook; the router, the
    # shared expert and the attributes come from Qwen3_5MoeSparseMoeBlock.__init__.
    make_tiny_fish_omni_checkpoint(tmp_path)
    model = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32)
    calls = []
    parent_init = Qwen3_5MoeSparseMoeBlock.__init__

    def spy(self, model):
        calls.append(type(self).__name__)
        parent_init(self, model)

    monkeypatch.setattr(Qwen3_5MoeSparseMoeBlock, "__init__", spy)
    block = FishOmniFp8SparseMoeBlock(model.model.layers[0].mlp)
    sibling = Qwen3_5MoeSparseMoeBlock(model.model.layers[1].mlp)  # only built, never run, so FP8 experts are fine
    assert calls == ["FishOmniFp8SparseMoeBlock", "Qwen3_5MoeSparseMoeBlock"]
    assert type(block.experts) is FishOmniFp8MoeMLP
    assert type(sibling.experts) is Qwen3_5MoeMLP
    assert [n for n, _ in block.named_children()] == [n for n, _ in sibling.named_children()]
    assert (block.num_experts, block.top_k) == (sibling.num_experts, sibling.top_k) == (8, 2)
