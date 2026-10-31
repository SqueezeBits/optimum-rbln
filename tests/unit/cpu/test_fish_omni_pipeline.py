"""CPU checks of the host side of the FishOmni text-to-speech loop (`fish_omni_runtime_utils`).

The tower runs through its RBLN graph modules in eager fp32 (`TowerAttention.eager_attention`, the CPU stand-in of the
paged attention ops), on a tiny random tower; the backbone and the codec are deterministic stand-ins that record what
the loop feeds them. Checks: candidate ids, the tower's host embeddings against their formula, free running feeds the
loop's own frames, teacher forcing feeds the teacher's, teacher forcing on the loop's own frames reproduces the free
run, batching (rows, padding rows, different frame counts) does not change a request, eos stops before the frame is
kept, a teacher-forced row records its own eos and runs to max_frames, a logits-only backbone artifact is refused, a
batch whose rows would decode past the backbone cache is refused before the loop, and the loop is not exported from
`optimum.rbln`.
"""

from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.fish_omni.fish_omni_runtime_utils import (
    BackboneRunner,
    FishOmniTokens,
    FishOmniTTSPipeline,
    RBLNBackboneRunner,
    TowerHostEmbeddings,
    TowerRunner,
    TTSRequest,
    semantic_top_share,
)
from optimum.rbln.transformers.models.fish_omni.fish_omni_tower_architecture import (
    FishOmniTowerArgs,
    FishOmniTowerDecodeFirst,
    FishOmniTowerDecodeResidual,
    FishOmniTowerPrefill,
    TowerAttention,
)
from optimum.rbln.transformers.models.fish_omni.modeling_fish_omni_tower import build_fish_omni_tower


V, CODEBOOKS, DIM = 16, 4, 32
HIDDEN = DIM  # the tower reads the backbone hidden directly
EOS, SEM = 5, 20
TOKENS = FishOmniTokens(EOS, SEM, SEM + V - 1)
FULL_VOCAB = SEM + V + 4
MAX_SEQ = 64
TOWER_CFG = {
    "dim": DIM,
    "intermediate_size": 48,
    "n_head": 4,
    "n_local_heads": 2,
    "head_dim": 8,
    "norm_eps": 1e-6,
    "rope_base": 10000.0,
    "rotary_percent": 0.5,
    "vocab_size": V,
    "num_codebooks": CODEBOOKS,
    "shared_enrichment_layers": 1,
    "vq_backbone_codebooks": 1,
}


def tiny_tower():
    torch.manual_seed(0)
    model = build_fish_omni_tower(TOWER_CFG, dtype=torch.float32).eval()
    args = model.args
    g = torch.Generator().manual_seed(1)
    host = {}
    for i in range(args.shared_enrichment_layers):
        host[f"shared_layers.{i}.semantic_embedding.weight"] = torch.randn(V, DIM, generator=g)
    for i in range(args.num_residual_layers):
        host[f"residual_layers.{i}.codebook_embedding.weight"] = torch.randn(V, DIM, generator=g)
    host["codebook_embeddings.weight"] = torch.randn(CODEBOOKS * V, DIM, generator=g)
    return model, host


def cpu_tower_runner(model, host, batch: int, chunk: int = 8) -> TowerRunner:
    """The tower graphs in eager fp32 with explicit caches, driven like the device runtimes."""
    TowerAttention.eager_attention = True
    args = model.args
    prefill_graph = FishOmniTowerPrefill(model)
    first = FishOmniTowerDecodeFirst(model)
    rest = {k: FishOmniTowerDecodeResidual(model, k) for k in range(2, args.num_residual_layers + 1)}
    caches = [torch.zeros(batch, args.n_local_heads, MAX_SEQ, args.head_dim) for _ in range(2 * args.num_layers)]
    shared = args.shared_enrichment_layers

    def prefill(*inputs):
        return prefill_graph(*inputs, *caches)

    def decode(k, *inputs):
        if k == 1:
            return first(*inputs, *caches[: 2 * (shared + 1)])
        layer = shared + k - 1
        return rest[k](*inputs, caches[2 * layer], caches[2 * layer + 1])

    emb = TowerHostEmbeddings(host, args, TOKENS)
    return TowerRunner(prefill, decode, emb, batch, MAX_SEQ, chunk, torch.float32)


class FakeBackbone(BackboneRunner):
    """Row-independent causal stand-in: hidden = tanh(running mean of frame embeddings); logits = hidden @ W.T with eos
    pushed down, or up at the decode steps listed in `eos_at` {row: step}. Records every decode input per row."""

    def __init__(self, batch_sizes=(1, 2, 4), eos_at=None, seed=2, max_seq_len=MAX_SEQ):
        g = torch.Generator().manual_seed(seed)
        self.max_seq_len = max_seq_len
        self.tok = torch.randn(FULL_VOCAB, HIDDEN, generator=g)
        self.code = torch.randn(CODEBOOKS, V, HIDDEN, generator=g)
        self.w = torch.randn(FULL_VOCAB, HIDDEN, generator=g)
        self.batch_sizes, self.eos_at = batch_sizes, eos_at or {}
        self.state, self.fed, self.steps = {}, {}, {}

    def graph_batch(self, n):
        return min(b for b in self.batch_sizes if b >= n)

    def _embed(self, frames):
        codes = frames[:, 1:].clamp(0, V - 1)
        return self.tok[frames[:, 0]] + sum(self.code[k][codes[:, k]] for k in range(CODEBOOKS))

    def _logits(self, hidden, row):
        lg = hidden @ self.w.T
        lg[EOS] = 50.0 if self.eos_at.get(row) == self.steps.get(row, 0) else -50.0
        return lg

    def prefill(self, row, frames):
        x = self._embed(frames)
        cum = x.cumsum(0)
        hidden = torch.tanh(cum / torch.arange(1, len(x) + 1)[:, None])
        self.state[row], self.fed[row], self.steps[row] = (cum[-1], len(x)), [], 0
        return self._logits(hidden[-1], row), hidden

    def decode(self, frames, positions):
        logits, hidden = [], []
        x = self._embed(frames)
        for i in range(frames.shape[0]):
            total, count = self.state[i]
            assert int(positions[i]) == count, "decode position must be the next cache row"
            assert count < self.max_seq_len, f"row {i} decoded at cache row {count}, past the backbone cache"
            total, count = total + x[i], count + 1
            self.state[i] = (total, count)
            self.fed[i].append(frames[i].clone())
            self.steps[i] += 1
            h = torch.tanh(total / count)
            hidden.append(h)
            logits.append(self._logits(h, i))
        return torch.stack(logits), torch.stack(hidden)


def fake_codec(codes):
    return codes.float().sum(1, keepdim=True).repeat_interleave(4, -1)


def make_prompt(rows, seed):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.zeros(rows, 1 + CODEBOOKS, dtype=torch.long)
    prompt[:, 0] = torch.randint(0, SEM, (rows,), generator=g)  # text rows
    speech = torch.arange(rows) >= rows // 2  # the tail holds semantic rows with codes
    prompt[speech, 0] = torch.randint(SEM, SEM + V, (int(speech.sum()),), generator=g)
    prompt[speech, 1:] = torch.randint(0, V, (int(speech.sum()), CODEBOOKS), generator=g)
    prompt[speech, 1] = prompt[speech, 0] - SEM
    return prompt


def pipeline(batch, backbone=None):
    model, host = tiny_tower()
    return FishOmniTTSPipeline(backbone or FakeBackbone(), cpu_tower_runner(model, host, batch), fake_codec, TOKENS)


REQS = [("a", 5, 6), ("b", 11, 3), ("c", 4, 6)]  # name, prompt rows (b spans two prefill chunks), frames


def requests(teacher=None):
    return [
        TTSRequest(name, make_prompt(rows, i), frames, None if teacher is None else teacher[name])
        for i, (name, rows, frames) in enumerate(REQS)
    ]


def test_candidates_eos_first():
    cand = TOKENS.candidates()
    assert cand.numel() == 1 + V
    assert int(cand[0]) == EOS
    assert torch.equal(cand[1:], torch.arange(SEM, SEM + V))


def test_tower_host_embeddings_follow_their_formula():
    model, host = tiny_tower()
    emb = TowerHostEmbeddings(host, model.args, TOKENS)
    frames = make_prompt(6, 3)
    main = frames[:, 0]
    codes = (main - SEM).clamp(0, V - 1)
    semantic = TOKENS.is_semantic(main)
    table = host["codebook_embeddings.weight"]
    unseen = sum(table[k * V + frames[:, 1 + k]] for k in range(2, CODEBOOKS))
    want = torch.where(semantic[:, None], host["shared_layers.0.semantic_embedding.weight"][codes] + unseen, 0.0)
    torch.testing.assert_close(emb.shared(codes, semantic, frames)[0], want)
    for k in range(1, CODEBOOKS):
        want = torch.where(semantic[:, None], host[f"residual_layers.{k - 1}.codebook_embedding.weight"][codes], 0.0)
        torch.testing.assert_close(emb.residual(k, codes, semantic), want)


def test_free_running_feeds_its_own_frames():
    backbone = FakeBackbone()
    pipe = pipeline(4, backbone)
    out = pipe.generate(requests())
    for row, (res, (_, _rows, frames)) in enumerate(zip(out, REQS, strict=True)):
        assert res.frames.shape == (frames, 1 + CODEBOOKS)
        assert res.codes.shape == (CODEBOOKS, frames)
        assert res.step_main.shape == (frames,)
        assert res.step_cand_logits.shape == (frames, 1 + V)
        assert res.step_head_logits.shape == (frames, CODEBOOKS - 1, V)
        assert res.finish_reason == "length"
        assert res.waveform.shape == (4 * frames,)
        assert torch.equal(res.frames[:, 0], res.step_main)
        assert torch.equal(res.frames[:, 1], res.step_main - SEM)
        assert torch.equal(res.frames[:, 2:], res.step_head_logits.argmax(-1))
        fed = torch.stack(backbone.fed[row][: frames - 1])
        assert torch.equal(fed, res.frames[: frames - 1]), "the backbone must be fed the loop's own frames"
        assert 0.0 < semantic_top_share(res.step_main, frames) <= 1.0


def test_teacher_forcing_feeds_the_teacher_and_records_its_own_argmax():
    g = torch.Generator().manual_seed(7)
    teacher = {}
    for name, _, frames in REQS:
        t = torch.randint(0, V, (frames, 1 + CODEBOOKS), generator=g)
        t[:, 0] = t[:, 1] + SEM
        teacher[name] = t
    backbone = FakeBackbone()
    out = pipeline(4, backbone).generate(requests(teacher))
    for row, (res, (name, _, frames)) in enumerate(zip(out, REQS, strict=True)):
        assert torch.equal(torch.stack(backbone.fed[row][: frames - 1]), teacher[name][: frames - 1])
        assert torch.equal(res.frames[:, 0], res.step_main)  # own choice recorded
    free = pipeline(4).generate(requests())
    assert any(not torch.equal(f.frames, t.frames) for f, t in zip(free, out, strict=True)), (
        "teacher frames had no effect"
    )


def test_teacher_forcing_on_own_frames_reproduces_the_free_run():
    free = pipeline(4).generate(requests())
    forced = pipeline(4).generate(requests({r.name: r.frames for r in free}))
    for f, t in zip(free, forced, strict=True):
        assert torch.equal(f.frames, t.frames)
        torch.testing.assert_close(f.step_cand_logits, t.step_cand_logits, rtol=0, atol=0)
        torch.testing.assert_close(f.step_head_logits, t.step_head_logits, rtol=0, atol=0)


def test_batch_rows_and_padding_do_not_change_a_request():
    batched = pipeline(4).generate(requests())  # 3 requests in a batch-4 graph: one padding row
    for i, res in enumerate(batched):
        alone = pipeline(1).generate([requests()[i]])[0]
        assert torch.equal(res.frames, alone.frames)
        torch.testing.assert_close(res.step_head_logits, alone.step_head_logits, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(res.step_cand_logits, alone.step_cand_logits, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(res.waveform, alone.waveform)


def test_eos_stops_before_the_frame_is_kept():
    out = pipeline(4, FakeBackbone(eos_at={0: 2})).generate(requests())
    res = out[0]
    assert res.finish_reason == "eos"
    assert res.eos_position == 2
    assert res.step_main.shape == (3,)
    assert int(res.step_main[-1]) == EOS
    assert res.frames.shape[0] == 2
    assert res.step_head_logits.shape[0] == 2
    assert [r.frames.shape[0] for r in out[1:]] == [3, 6]


def test_teacher_forcing_records_its_own_eos_and_runs_to_max_frames():
    # Row 0's own argmax is eos at step 2. Teacher forced, it records that eos and still runs every step to
    # max_frames on the teacher's frames; outside the eos column the run equals the one without the eos.
    teacher = {r.name: r.frames for r in pipeline(4).generate(requests())}
    plain = pipeline(4).generate(requests(teacher))
    backbone = FakeBackbone(eos_at={0: 2})
    out = pipeline(4, backbone).generate(requests(teacher))
    res, frames = out[0], REQS[0][2]
    assert res.finish_reason == "eos"
    assert res.eos_position == 2
    assert res.step_main.shape == (frames,), "a teacher-forced row runs every step"
    assert (res.step_main == EOS).nonzero().flatten().tolist() == [2]
    assert res.step_cand_logits.shape == (frames, 1 + V)
    assert res.step_hidden.shape == (frames, HIDDEN)
    assert res.step_head_logits.shape == (frames, CODEBOOKS - 1, V)
    assert res.frames.shape == (frames, 1 + CODEBOOKS)
    assert torch.equal(res.frames[:, 0], res.step_main)  # own choice recorded, the eos step too
    assert torch.equal(torch.stack(backbone.fed[0]), teacher["a"][: frames - 1]), "fed the teacher after its eos"
    assert torch.equal(res.step_cand_logits[:, 1:], plain[0].step_cand_logits[:, 1:])
    assert torch.equal(res.step_hidden, plain[0].step_hidden)
    assert torch.equal(res.step_head_logits, plain[0].step_head_logits)
    for got, want in zip(out[1:], plain[1:], strict=True):
        assert (got.finish_reason, got.eos_position) == ("length", None)
        assert torch.equal(got.frames, want.frames)
        assert torch.equal(got.step_cand_logits, want.step_cand_logits)


def test_logits_only_backbone_artifact_is_refused():
    cfg = SimpleNamespace(use_inputs_embeds=True, output_hidden_states=False, max_seq_len=8192)
    fake = SimpleNamespace(rbln_config=cfg)
    with pytest.raises(ValueError, match="output_last_hidden_state"):
        RBLNBackboneRunner(fake)
    runner = RBLNBackboneRunner(fake, logits_only=True)  # backbone-only checks
    model, host = tiny_tower()
    with pytest.raises(ValueError, match="logits-only"):
        FishOmniTTSPipeline(runner, cpu_tower_runner(model, host, 1), fake_codec, TOKENS)


def test_tower_args_of_the_tiny_config():
    args = FishOmniTowerArgs.from_config(TOWER_CFG)
    assert args.num_residual_layers == CODEBOOKS - 1
    assert args.num_layers == 1 + CODEBOOKS - 1


def test_backbone_cache_bound_counts_every_row():
    # Each request fits alone, but in one batch the short-prompt request's 6 frames make the long-prompt row (finished
    # after 2 frames) decode on to step 5: 12 rows + 6 steps - 1 = 17 >= 16.
    long_prompt = TTSRequest("long", make_prompt(12, 0), 2)
    many_frames = TTSRequest("many", make_prompt(4, 1), 6)
    for req in (long_prompt, many_frames):
        assert pipeline(1, FakeBackbone(max_seq_len=16)).generate([req])[0].frames.shape[0] == req.max_frames
    backbone = FakeBackbone(max_seq_len=16)
    with pytest.raises(ValueError, match=r"long: 12 rows \+ 6 steps .* exceed the backbone cache"):
        pipeline(2, backbone).generate([long_prompt, many_frames])
    assert backbone.state == {}, "the batch must be refused before any prefill"
    # The bound is prompt rows + steps - 1 < max_seq_len.
    with pytest.raises(ValueError, match="exceed the backbone cache"):
        pipeline(2, FakeBackbone(max_seq_len=17)).generate([long_prompt, many_frames])
    assert len(pipeline(2, FakeBackbone(max_seq_len=18)).generate([long_prompt, many_frames])) == 2


def test_loop_is_not_exported_from_optimum_rbln():
    import optimum.rbln
    from optimum.rbln.transformers.models import fish_omni

    for name in ("FishOmniTTSPipeline", "FishOmniTokens", "TTSRequest", "TTSResult", "fish_omni_tokens"):
        assert not hasattr(optimum.rbln, name), f"optimum.rbln exports {name}"
        assert not hasattr(fish_omni, name), f"the fish_omni package exports {name}"
    assert hasattr(optimum.rbln, "RBLNFishOmniForCausalLM")
