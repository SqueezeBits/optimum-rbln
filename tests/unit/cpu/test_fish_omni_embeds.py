"""CPU check of the FishOmni backbone input built on the host (`FishOmniBackboneEmbedding`).

The expected rows come from `reference_embed_backbone`, a row-by-row restatement of fish-speech @0f8a12e
`FishOmniAudioDecoder.embed_backbone` (fish_speech/models/text2semantic/fish_omni/tower.py) for
`vq_feature_placement: backbone_partial`: a row whose main token lies in the semantic band takes
`(embed_tokens(main) + embedding_bag_sum(codebook_embeddings, [k * V + code_k for k in 1..K])) / sqrt(1 + K)`, every
other row takes `embed_tokens(main)`. It reads the full upstream table at upstream row indices, so the offset
translation of `FishOmniBackboneEmbedding` (rows [V, (1 + K) * V) only) is checked as well.
"""

import json
import math
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from optimum.rbln.transformers.models.fish_omni.modeling_fish_omni import (
    CODEBOOK_EMBEDDINGS_KEY,
    FishOmniBackboneEmbedding,
    fish_omni_backbone_embedding_meta,
    load_fish_omni_backbone_codebooks,
)


VOCAB, CODEBOOKS, DIM, TEXT_VOCAB = 64, 10, 32, 512
SEM_BEGIN, SEM_END, EOS = 300, 300 + VOCAB - 1, 7


def reference_embed_backbone(
    frames: torch.Tensor, text_embeddings: torch.Tensor, table: torch.Tensor, count: int, sem_begin: int, sem_end: int
) -> torch.Tensor:
    """fish-speech @0f8a12e `embed_backbone` (backbone_partial), one row at a time; `table` is the whole upstream
    `codebook_embeddings.weight` [V * num_codebooks, D] and frame row r is `[main, code 0, ..., code N-1]`."""
    out = text_embeddings.clone()
    for r in range(frames.shape[0]):
        main = int(frames[r, 0])
        if not sem_begin <= main <= sem_end:
            continue
        if count:
            rows = torch.tensor([[k * VOCAB + int(frames[r, 1 + k]) for k in range(1, 1 + count)]])
            extra = F.embedding_bag(rows, table, mode="sum")[0]
        else:
            extra = torch.zeros_like(text_embeddings[r])
        out[r] = (text_embeddings[r] + extra) / math.sqrt(1 + count)
    return out


def frames_like_generate(rows: int, seed: int) -> torch.Tensor:
    """Prompt-like rows: text rows with zero codes, semantic rows [main, main - begin, c1..c9], one eos row."""
    g = torch.Generator().manual_seed(seed)
    frames = torch.zeros(rows, 1 + CODEBOOKS, dtype=torch.int64)
    kind = torch.randint(0, 3, (rows,), generator=g)
    for r in range(rows):
        if kind[r] == 0:
            frames[r, 0] = torch.randint(0, SEM_BEGIN, (1,), generator=g)
        else:
            c0 = int(torch.randint(0, VOCAB, (1,), generator=g))
            frames[r, 0] = SEM_BEGIN + c0
            frames[r, 1] = c0
            frames[r, 2:] = torch.randint(0, VOCAB, (CODEBOOKS - 1,), generator=g)
    frames[rows // 2] = 0
    frames[rows // 2, 0] = EOS
    return frames


@pytest.mark.parametrize("count", [1, 2, 0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backbone_embedding_matches_upstream_formula(count, dtype):
    torch.manual_seed(count)
    table = torch.randn(VOCAB * CODEBOOKS, DIM).to(dtype)
    embed_tokens = nn.Embedding.from_pretrained(torch.randn(TEXT_VOCAB, DIM).to(dtype))
    frames = frames_like_generate(48, seed=count)

    expected = reference_embed_backbone(frames, embed_tokens(frames[:, 0]), table, count, SEM_BEGIN, SEM_END)
    ours = FishOmniBackboneEmbedding(
        embed_tokens, table[VOCAB : (1 + count) * VOCAB].clone(), VOCAB, count, SEM_BEGIN, SEM_END
    )
    got = ours(frames)
    assert got.dtype == expected.dtype
    assert torch.equal(got, expected)
    # Batched [B, T, 11] input gives the same rows.
    assert torch.equal(ours(frames.reshape(4, 12, -1)).reshape(48, -1), expected)
    # Semantic rows really mix in the codebook rows (the test would pass trivially otherwise).
    semantic = (frames[:, 0] >= SEM_BEGIN) & (frames[:, 0] <= SEM_END)
    assert semantic.any()
    assert (~semantic).any()
    if count:
        assert not torch.equal(got[semantic], embed_tokens(frames[semantic, 0]))
    assert torch.equal(got[~semantic], embed_tokens(frames[~semantic, 0]))


def test_codebook_rows_loaded_from_checkpoint(tmp_path: Path):
    table = torch.randn(VOCAB * CODEBOOKS, DIM, dtype=torch.bfloat16)
    save_file({CODEBOOK_EMBEDDINGS_KEY: table, "other": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    config = {
        "semantic_start_token_id": SEM_BEGIN,
        "semantic_end_token_id": SEM_END,
        "audio_decoder_config": {
            "vocab_size": VOCAB,
            "vq_backbone_codebooks": 1,
            "vq_feature_placement": "backbone_partial",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    meta = fish_omni_backbone_embedding_meta(config)
    assert meta == {
        "audio_vocab_size": VOCAB,
        "vq_backbone_codebooks": 1,
        "semantic_start_token_id": SEM_BEGIN,
        "semantic_end_token_id": SEM_END,
    }
    rows = load_fish_omni_backbone_codebooks(tmp_path, meta, torch.float32)
    assert rows.dtype == torch.float32
    assert torch.equal(rows, table[VOCAB : 2 * VOCAB].float())


def test_artifact_round_trip_gives_upstream_embeddings(tmp_path: Path, monkeypatch):
    """get_pytorch_model -> save_torch_artifacts -> __post_init__ -> embed_frames, without compiling."""
    from optimum.rbln import RBLNFishOmniForCausalLM

    from ...test_base import TINY_TEXT, make_tiny_fish_omni_checkpoint

    make_tiny_fish_omni_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config.update(
        semantic_start_token_id=64,
        semantic_end_token_id=64 + VOCAB - 1,
        audio_decoder_config={"vocab_size": VOCAB, "num_codebooks": CODEBOOKS, "vq_backbone_codebooks": 1},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    hidden = TINY_TEXT["hidden_size"]
    table = torch.randn(VOCAB * CODEBOOKS, hidden)
    save_file({CODEBOOK_EMBEDDINGS_KEY: table}, str(tmp_path / "model-audio.safetensors"))

    hf = RBLNFishOmniForCausalLM.get_pytorch_model(tmp_path, dtype=torch.float32)
    save_dir = tmp_path / "artifact"
    save_dir.mkdir()
    rbln_config = types.SimpleNamespace(use_inputs_embeds=True, dtype=torch.float32)
    RBLNFishOmniForCausalLM.save_torch_artifacts(hf, save_dir, "", rbln_config)

    monkeypatch.setattr(RBLNFishOmniForCausalLM, "setup_runtime", lambda self: None)
    rbln = object.__new__(RBLNFishOmniForCausalLM)
    rbln.__dict__.update(rbln_config=rbln_config, model_save_dir=save_dir, subfolder="", config=hf.config)
    rbln.__post_init__()

    g = torch.Generator().manual_seed(3)
    frames = torch.zeros(20, 1 + CODEBOOKS, dtype=torch.int64)
    frames[:, 0] = torch.randint(0, TINY_TEXT["vocab_size"], (20,), generator=g)
    frames[:, 1:] = torch.randint(0, VOCAB, (20, CODEBOOKS), generator=g)
    expected = reference_embed_backbone(frames, hf.get_input_embeddings()(frames[:, 0]), table, 1, 64, 64 + VOCAB - 1)
    assert ((frames[:, 0] >= 64) & (frames[:, 0] < 64 + VOCAB)).any()
    assert torch.equal(rbln.embed_frames(frames), expected)
