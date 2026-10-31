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

"""Greedy FishOmni (drama-3) text-to-speech loop over the three RBLN models: backbone, codebook tower, codec.

Port of fish-speech @0f8a12e `generate_drama3`'s eager path, with argmax in place of the upstream sampler:

1. backbone prefill of the prompt rows (host input `RBLNFishOmniForCausalLM.embed_frames`); the logits of the last row
   are step 0's logits and every row's final-norm hidden feeds the tower;
2. tower prefill of the prompt rows: row p with the backbone hidden of row p, labelled with the codes of row p + 1;
3. per frame: main token = argmax over eos + the semantic band; stop at eos before the frame is kept; code 0 = main
   minus the first semantic id; `decode_1` (shared layers + residual layer 1 on code 0) then `decode_k` (residual layer
   k on code k - 1), argmax of every codebook head; the frame `[main, code 0 .. code 9]` is the backbone's next input;
4. the codec decodes the codes of the kept frames of every request.

Teacher forcing (a request with `teacher_frames`) feeds the given frames wherever the free-running loop feeds its own
choice (the backbone input of the next step, the tower's main token, previous frame and residual input codes) and still
records its own argmax at every step, so per-step error separates from drift. A teacher-forced request runs every step
up to `max_frames` and keeps its own frame at every step: its first own eos is recorded (`finish_reason` "eos",
`eos_position`) but does not stop it.

Requests share one batch: request i runs in batch row i; rows past the last request (up to the decoder graph's batch)
are prefilled with copies of the first requests and then stay idle, and a finished request keeps feeding its last
input until every request is done; outputs of idle rows are dropped.

Host-side runtime logic of the FishOmni models (like `qwen3_5_runtime_utils`): nothing here is exported from
`optimum.rbln`; import it from `optimum.rbln.transformers.models.fish_omni.fish_omni_runtime_utils`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .fish_omni_tower_architecture import FishOmniTowerArgs, tower_rope_cos_sin


@dataclass(frozen=True)
class FishOmniTokens:
    """Token ids the loop needs: eos and the semantic band [semantic_begin, semantic_end]."""

    eos_token_id: int
    semantic_begin: int
    semantic_end: int

    def candidates(self) -> torch.Tensor:
        """Ids the main token is chosen from, ascending (eos first): eos + the semantic band."""
        band = torch.arange(self.semantic_begin, self.semantic_end + 1)
        return torch.cat((torch.tensor([self.eos_token_id]), band)).sort().values

    def is_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)


@dataclass
class TTSRequest:
    """One request: prompt rows [R, 1 + num_codebooks] int64 (main token, codes), frames to generate, and for teacher
    forcing the frames [F, 1 + num_codebooks] to feed instead of the loop's own choice."""

    name: str
    prompt_rows: torch.Tensor
    max_frames: int
    teacher_frames: torch.Tensor | None = None


@dataclass
class TTSResult:
    """Per-request outputs in the reference layout (S = steps run, F = frames kept). A free-running request stops at
    its eos (S = F + 1); a teacher-forced one runs to `max_frames` (S = F), with `finish_reason` "eos" and
    `eos_position` at its first own eos if it chose one."""

    name: str
    prompt_rows: torch.Tensor
    step_main: torch.Tensor  # [S] int64
    step_cand_logits: torch.Tensor  # [S, 1 + vocab] fp32, column 0 = eos
    step_hidden: torch.Tensor  # [S, hidden] fp32
    step_head_logits: torch.Tensor  # [F, num_codebooks - 1, vocab] fp32
    frames: torch.Tensor  # [F, 1 + num_codebooks] int64 (own argmax)
    finish_reason: str
    eos_position: int | None
    waveform: torch.Tensor | None = None
    timing: dict[str, Any] = field(default_factory=dict)

    @property
    def codes(self) -> torch.Tensor:
        """[num_codebooks, F] int64"""
        return self.frames[:, 1:].T.contiguous()


class TowerHostEmbeddings:
    """Upstream `TowerLayer` code embeddings and `run_shared`'s unseen-code sum, on the host in fp32.

    Delta: the unseen-code sum and its add to the first shared layer's embed stay fp32 and are cast once to the graph
    dtype by the caller; upstream, run in the model dtype, rounds the sum before the add. The two agree exactly for a
    float32 tower and differ by about one ulp of a 16-bit graph dtype.

    `host` holds `shared_layers.{i}.semantic_embedding.weight`, `residual_layers.{i}.codebook_embedding.weight` and
    `codebook_embeddings.weight` (the tower artifact's `torch_tower_host_embeddings.pth`).
    """

    def __init__(self, host: dict[str, torch.Tensor], args: FishOmniTowerArgs, tokens: FishOmniTokens) -> None:
        self.args, self.tokens = args, tokens
        self.semantic = [
            host[f"shared_layers.{i}.semantic_embedding.weight"].float() for i in range(args.shared_enrichment_layers)
        ]
        self.codebook = [
            host[f"residual_layers.{i}.codebook_embedding.weight"].float() for i in range(args.num_residual_layers)
        ]
        self.table = host["codebook_embeddings.weight"].float()

    def unseen(self, frames: torch.Tensor) -> torch.Tensor:
        """Sum of the embeddings of the codes the backbone input did not mix in, for frames [B, 1 + num_codebooks]."""
        first, end = 1 + self.args.vq_backbone_codebooks, self.args.num_codebooks
        offsets = torch.arange(first, end) * self.args.vocab_size
        rows = frames[:, 1 + first : 1 + end] + offsets
        summed = F.embedding_bag(rows, self.table, mode="sum")
        return torch.where(self.tokens.is_semantic(frames[:, 0])[:, None], summed, 0.0)

    @staticmethod
    def _lookup(table: torch.Tensor, codes: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        codes = codes.clamp(0, table.shape[0] - 1)
        return torch.where(semantic[:, None], F.embedding(codes, table), 0.0)

    def shared(self, codes: torch.Tensor, semantic: torch.Tensor, frames: torch.Tensor) -> list[torch.Tensor]:
        """Embeds of every shared layer for code 0 (the first also adds the unseen codes of `frames`)."""
        unseen = self.unseen(frames)
        return [
            self._lookup(table, codes, semantic) + unseen if i == 0 else self._lookup(table, codes, semantic)
            for i, table in enumerate(self.semantic)
        ]

    def residual(self, k: int, codes: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        """Embed of residual layer k (1-based) for code k - 1."""
        return self._lookup(self.codebook[k - 1], codes, semantic)


class TowerRunner:
    """Drives the tower graphs: `prefill(*inputs)` (batch 1, writes one row's caches) and `decode(k, *inputs)` ->
    (logits [B, 1, vocab], hidden [B, 1, dim]) for k = 1 .. num_codebooks - 1 (graph batch B)."""

    def __init__(
        self,
        prefill: Callable[..., Any],
        decode: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        emb: TowerHostEmbeddings,
        batch: int,
        max_seq_len: int,
        chunk: int,
        dtype: torch.dtype,
    ) -> None:
        self._prefill, self._decode, self.emb = prefill, decode, emb
        self.args = emb.args
        self.batch, self.max_seq_len, self.chunk, self.dtype = batch, max_seq_len, chunk, dtype

    def prefill(self, row: int, prompt: torch.Tensor, hidden: torch.Tensor) -> None:
        """Teacher-forced tower caches of one request in batch row `row`: prompt rows 0 .. R-2 with the backbone hidden
        of those rows, each labelled with the codes of the next row (upstream `tower.prefill`)."""
        args, emb, chunk = self.args, self.emb, self.chunk
        paired = len(prompt) - 1
        for start in range(0, paired, chunk):
            end = min(start + chunk, paired)
            n = end - start
            frames, nxt = prompt[start:end], prompt[start + 1 : end + 1]
            semantic = emb.tokens.is_semantic(nxt[:, 0])
            embeds = emb.shared(nxt[:, 1], semantic, frames)
            embeds += [emb.residual(k, nxt[:, k], semantic) for k in range(1, args.num_residual_layers + 1)]
            pad = chunk - n
            h = F.pad(hidden[start:end].float(), (0, 0, 0, pad))[None]
            layer_embeds = torch.stack([F.pad(e, (0, 0, 0, pad)) for e in embeds])[:, None]
            cos, sin = tower_rope_cos_sin(torch.arange(start, start + chunk)[None], args)
            self._prefill(
                h.to(self.dtype).contiguous(),
                layer_embeds.to(self.dtype).contiguous(),
                cos.contiguous(),
                sin.contiguous(),
                torch.tensor([[start]], dtype=torch.int32),
                torch.tensor([row], dtype=torch.int16),
            )

    def decode_frame(
        self,
        hidden: torch.Tensor,
        main: torch.Tensor,
        prev: torch.Tensor,
        positions: torch.Tensor,
        active: torch.Tensor,
        teacher_codes: torch.Tensor | None = None,
        teacher_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One frame for every batch row.

        hidden [B, dim] backbone hidden, main [B] main token fed to the tower (code 0 = main - first semantic id),
        prev [B, 1 + num_codebooks] previous frame, positions [B] (frame position R - 1 + step), active [B] bool (idle
        rows write a cache slot no request reads). Rows with `teacher_rows` set feed `teacher_codes[k - 1]`
        [num_codebooks, B] to residual layer k >= 2 instead of their own argmax of head k - 1.
        Returns (head logits [B, num_codebooks - 1, vocab] fp32, own argmax codes 1 .. num_codebooks - 1 [K, B]).
        """
        args, emb = self.args, self.emb
        b = self.batch
        positions = torch.where(active, positions, torch.full_like(positions, self.max_seq_len - 1))
        c0 = (main - emb.tokens.semantic_begin).clamp(0, args.vocab_size - 1)
        semantic = emb.tokens.is_semantic(main) & active
        cos, sin = tower_rope_cos_sin(positions[:, None], args)
        common = (
            cos.contiguous(),
            sin.contiguous(),
            positions[:, None].to(torch.int32),
            torch.arange(b, dtype=torch.int16)[:, None],
        )
        embeds = [*emb.shared(c0, semantic, prev), emb.residual(1, c0, semantic)]
        layer_embeds = torch.stack(embeds)[:, :, None].to(self.dtype).contiguous()
        out, h = self._decode(1, hidden[:, None].to(self.dtype).contiguous(), layer_embeds, *common)
        logits = [out[:, 0].float().clone()]
        own = [logits[0].argmax(-1)]
        for k in range(2, args.num_residual_layers + 1):
            feed = own[-1]
            if teacher_rows is not None:
                feed = torch.where(teacher_rows, teacher_codes[k - 1], feed)
            e = emb.residual(k, feed, semantic)
            out, h = self._decode(k, h.contiguous(), e[:, None].to(self.dtype).contiguous(), *common)
            logits.append(out[:, 0].float().clone())
            own.append(logits[-1].argmax(-1))
        return torch.stack(logits, dim=1), torch.stack(own)


class BackboneRunner:
    """Backbone interface of the loop: `prefill(row, frames [R, W]) -> (logits [vocab], hidden [R, H])` writes one
    request's caches in batch row `row`; `decode(frames [B, W], positions [B]) -> (logits [B, vocab], hidden [B, H])`
    runs one step for every row of a decoder graph of batch B. Logits cover the full vocabulary; hidden is the
    backbone's final-norm output (what fish-speech feeds to lm_head and the tower). `max_seq_len` is the cache length
    of one batch row: every row of a batch is decoded at every step, so the loop checks it before it starts."""

    max_seq_len: int

    def graph_batch(self, n: int) -> int:
        raise NotImplementedError

    def prefill(self, row: int, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def decode(self, frames: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class RBLNBackboneRunner(BackboneRunner):
    """`RBLNFishOmniForCausalLM` compiled with `use_inputs_embeds` and `output_last_hidden_state` (or
    `output_hidden_states`, whose last entry is the same final-norm hidden): the tower reads the final-norm hidden of
    every prompt row and every step, which a logits-only artifact does not return.

    `logits_only=True` drives a logits-only artifact for backbone-only checks (hidden comes back as None); the loop
    itself never takes such a runner."""

    def __init__(self, model: Any, logits_only: bool = False) -> None:
        cfg = model.rbln_config
        if not cfg.use_inputs_embeds:
            raise ValueError(
                "the FishOmni loop builds the backbone input on the host: compile with use_inputs_embeds=True"
            )
        has_hidden = bool(getattr(cfg, "output_last_hidden_state", False) or cfg.output_hidden_states)
        if not has_hidden and not logits_only:
            raise ValueError(
                "the codebook tower reads the backbone's final-norm hidden of every prompt row and every decode step; "
                "this artifact returns logits only: compile the backbone with rbln_output_last_hidden_state=True"
            )
        self.model = model
        self.logits_only = not has_hidden
        self.max_seq_len = cfg.max_seq_len
        self.positions_offset: dict[int, int] = {}

    def graph_batch(self, n: int) -> int:
        fits = [b for b in sorted(self.model.decoders) if b >= n]
        if not fits:
            raise ValueError(f"{n} requests exceed the largest decoder batch {max(self.model.decoders)}")
        return fits[0]

    def prefill(self, row: int, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = len(frames)
        embeds = self.model.embed_frames(frames)[None]
        out = self.model.prefill_decoder(
            inputs_embeds=embeds,
            cache_position=torch.arange(rows, dtype=torch.int32)[None],
            batch_idx=row,
        )
        self.positions_offset[row] = int(out.padded_cache_lengths)
        logits = out.logits.reshape(-1, out.logits.shape[-1])[-1]
        hidden = None if self.logits_only else out.hidden_states[-1][0].float().clone()
        return logits.float().clone(), hidden

    def decode(self, frames: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = frames.shape[0]
        offsets = torch.tensor([self.positions_offset.get(i, 0) for i in range(b)])
        embeds = self.model.embed_frames(frames)[:, None]
        cache_position = (positions + offsets)[:, None].to(torch.int32)
        out = self.model.decoders[b](inputs_embeds=embeds, cache_position=cache_position)
        # decoder runtimes write into reused output buffers: copy before the next call
        hidden = None if self.logits_only else out.hidden_states[-1][:, -1].float().clone()
        return out.logits[:, -1].float().clone(), hidden


class FishOmniTTSPipeline:
    """Greedy end-to-end loop. `codec(codes [1, num_codebooks, F]) -> audio [1, 1, samples]` is called per request."""

    def __init__(
        self,
        backbone: BackboneRunner,
        tower: TowerRunner,
        codec: Callable[[torch.Tensor], torch.Tensor] | None,
        tokens: FishOmniTokens,
    ) -> None:
        if getattr(backbone, "logits_only", False):
            raise ValueError("the codebook tower needs the backbone hidden; a logits-only backbone runner cannot")
        self.backbone, self.tower, self.codec, self.tokens = backbone, tower, codec, tokens
        self.candidates = tokens.candidates()

    @classmethod
    def from_rbln(
        cls, backbone: Any, tower: Any, codec: Any, host_embeddings: dict[str, torch.Tensor]
    ) -> "FishOmniTTSPipeline":
        """From loaded `RBLNFishOmniForCausalLM`, `RBLNFishOmniTower`, `RBLNFishOmniCodec` and the tower artifact's
        host embedding tables (`torch_tower_host_embeddings.pth`)."""
        tokens = fish_omni_tokens(backbone)
        runner = rbln_tower_runner(tower, host_embeddings, tokens)
        return cls(RBLNBackboneRunner(backbone), runner, codec.decode if codec is not None else None, tokens)

    def _check(self, requests: Sequence[TTSRequest]) -> int:
        width = 1 + self.tower.args.num_codebooks
        for r in requests:
            if r.prompt_rows.ndim != 2 or r.prompt_rows.shape[1] != width:
                raise ValueError(f"{r.name}: prompt rows {tuple(r.prompt_rows.shape)}, expected [R, {width}]")
            if len(r.prompt_rows) + r.max_frames > self.tower.max_seq_len:
                raise ValueError(f"{r.name}: {len(r.prompt_rows)} rows + {r.max_frames} frames exceed the tower cache")
            if r.teacher_frames is not None and len(r.teacher_frames) < r.max_frames:
                raise ValueError(f"{r.name}: {len(r.teacher_frames)} teacher frames for {r.max_frames} frames")
        b = self.backbone.graph_batch(len(requests))
        if b > self.tower.batch:
            raise ValueError(f"backbone graph batch {b} exceeds the tower graph batch {self.tower.batch}")
        # The backbone decodes every batch row (finished and padding rows too) at prompt rows + s for every step s of
        # the longest request, and the decoder runtimes are called directly, so the cache bound is checked here.
        steps = max(r.max_frames for r in requests)
        for r in (requests[i % len(requests)] for i in range(b)):  # rows: the requests, then the padding copies
            if len(r.prompt_rows) + steps - 1 >= self.backbone.max_seq_len:
                raise ValueError(
                    f"{r.name}: {len(r.prompt_rows)} rows + {steps} steps (the longest request in the batch) exceed "
                    f"the backbone cache ({self.backbone.max_seq_len}); every batch row is decoded at every step"
                )
        return b

    @torch.inference_mode()
    def generate(
        self, requests: Sequence[TTSRequest], clock: Callable[[], float] = time.perf_counter
    ) -> list[TTSResult]:
        n = len(requests)
        b = self._check(requests)
        rows = list(requests) + [requests[i % n] for i in range(b - n)]  # padding rows: prefilled copies, then idle
        tb = self.tower.batch
        args = self.tower.args
        width = 1 + args.num_codebooks
        t_start = clock()

        # prefill: backbone then tower, one row at a time
        firsts, lasts = [], []
        prompt_len = torch.tensor([len(r.prompt_rows) for r in rows])
        for row, req in enumerate(rows):
            lg, hid = self.backbone.prefill(row, req.prompt_rows)
            firsts.append(lg)
            lasts.append(hid[-1])
            self.tower.prefill(row, req.prompt_rows, hid)
        logits, hidden = torch.stack(firsts), torch.stack(lasts)
        t_prefill = clock() - t_start

        steps = max(r.max_frames for r in requests)
        prev = torch.stack([r.prompt_rows[-1] for r in rows])  # [b, W] frame at position R - 1
        done = torch.zeros(b, dtype=torch.bool)
        done[n:] = True
        keys = ("main", "cand", "hidden", "heads", "frames")
        rec = [{**{k: [] for k in keys}, "finish": "length", "eos": None} for _ in range(n)]
        per_frame = []
        teacher_rows = _pad_rows(torch.tensor([r.teacher_frames is not None for r in rows]), tb)
        for s in range(steps):
            cand = logits[:, self.candidates]
            main = self.candidates[cand.argmax(-1)]
            for i in range(n):
                if not done[i] and s < requests[i].max_frames:
                    rec[i]["main"].append(int(main[i]))
                    rec[i]["cand"].append(cand[i].clone())
                    rec[i]["hidden"].append(hidden[i].clone())
                    if int(main[i]) == self.tokens.eos_token_id and rec[i]["eos"] is None:
                        rec[i]["finish"], rec[i]["eos"] = "eos", s
                        # upstream stops before keeping the frame; a teacher-forced row runs on to max_frames
                        done[i] = requests[i].teacher_frames is None
            active = torch.tensor([bool(not done[i]) and s < rows[i].max_frames for i in range(b)])
            if not active[:n].any():
                break

            # tower inputs: own main token and previous frame, or the teacher's
            tower_main = main.clone()
            teacher_codes = torch.zeros(args.num_codebooks, tb, dtype=torch.long)
            for i, r in enumerate(rows):
                if r.teacher_frames is not None and active[i]:
                    tower_main[i] = r.teacher_frames[s, 0]
                    teacher_codes[:, i] = r.teacher_frames[s, 1:]
            t1 = clock()
            head_logits, heads = self.tower.decode_frame(
                _pad_rows(hidden, tb),
                _pad_rows(tower_main, tb),
                _pad_rows(prev, tb),
                _pad_rows(prompt_len - 1 + s, tb),
                _pad_rows(active, tb),
                teacher_codes,
                teacher_rows,
            )
            t_tower = clock() - t1
            c0 = (main - self.tokens.semantic_begin).clamp(0, args.vocab_size - 1)
            frame = torch.cat((main[:, None], c0[:, None], heads[:, :b].T), dim=1)  # [b, W] own frame
            for i in range(n):
                if active[i]:
                    rec[i]["heads"].append(head_logits[i].clone())
                    rec[i]["frames"].append(frame[i].clone())

            # next backbone input: own frame or the teacher's; finished rows repeat their last input
            feed = prev.clone()
            for i, r in enumerate(rows):
                if active[i]:
                    feed[i] = r.teacher_frames[s] if r.teacher_frames is not None else frame[i]
            prev = feed
            t_backbone = None
            if s + 1 < steps:
                t2 = clock()
                logits, hidden = self.backbone.decode(feed, prompt_len + s)
                t_backbone = clock() - t2
            active_rows = int(active[:n].sum())
            per_frame.append({"step": s, "tower_s": t_tower, "backbone_s": t_backbone, "active": active_rows})
        t_loop = clock() - t_start

        results = []
        for i, req in enumerate(requests):
            r = rec[i]
            heads_dim = (args.num_residual_layers, args.vocab_size)
            res = TTSResult(
                name=req.name,
                prompt_rows=req.prompt_rows,
                step_main=torch.tensor(r["main"], dtype=torch.long),
                step_cand_logits=torch.stack(r["cand"]),
                step_hidden=torch.stack(r["hidden"]),
                step_head_logits=torch.stack(r["heads"]) if r["heads"] else torch.zeros(0, *heads_dim),
                frames=torch.stack(r["frames"]) if r["frames"] else torch.zeros(0, width, dtype=torch.long),
                finish_reason=r["finish"],
                eos_position=r["eos"],
            )
            t_codec = None
            if self.codec is not None and len(res.frames):
                t3 = clock()
                res.waveform = self.codec(res.codes[None])[0, 0].float()
                t_codec = clock() - t3
            res.timing = {
                "batch_rows": b,
                "prefill_all_rows_s": t_prefill,
                "loop_s": t_loop,
                "codec_s": t_codec,
                "first_audio_s": clock() - t_start if t_codec is not None else None,
                "per_frame": per_frame,
            }
            results.append(res)
        return results


def fish_omni_tokens(backbone: Any) -> FishOmniTokens:
    """eos and the semantic band of a loaded `RBLNFishOmniForCausalLM` (config and host backbone embedding)."""
    emb = backbone.backbone_embedding
    return FishOmniTokens(
        int(backbone.config.eos_token_id), int(emb.semantic_start_token_id), int(emb.semantic_end_token_id)
    )


def rbln_tower_runner(tower: Any, host_embeddings: dict[str, torch.Tensor], tokens: FishOmniTokens) -> TowerRunner:
    """A `TowerRunner` over the runtimes of a loaded `RBLNFishOmniTower`."""
    args = FishOmniTowerArgs.from_config(tower.config)
    cfg = tower.rbln_config
    dtype = cfg.dtype if isinstance(cfg.dtype, torch.dtype) else getattr(torch, cfg.dtype)

    def tower_decode(k, *inputs):
        out = tower.runtime(f"decode_{k}")(*inputs)
        return out[0], out[1]

    return TowerRunner(
        tower.runtime("prefill"),
        tower_decode,
        TowerHostEmbeddings(host_embeddings, args, tokens),
        cfg.batch_size,
        cfg.max_seq_len,
        cfg.prefill_chunk_size,
        dtype,
    )


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Pad dim 0 with zeros (False for bool) up to `rows` (the tower graph batch)."""
    if x.shape[0] == rows:
        return x
    pad = torch.zeros((rows - x.shape[0], *x.shape[1:]), dtype=x.dtype)
    return torch.cat((x, pad))


def semantic_top_share(step_main: torch.Tensor, frames: int) -> float:
    """Share of the most frequent main token among the kept frames (the reference's degeneracy statistic)."""
    if frames == 0:
        return 0.0
    _, counts = torch.unique(step_main[:frames], return_counts=True)
    return float(counts.max()) / frames
