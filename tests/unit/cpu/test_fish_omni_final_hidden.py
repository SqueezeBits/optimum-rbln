"""CPU checks of the FishOmni backbone's final-hidden output (`output_last_hidden_state`).

- `FishOmniForCausalLM` computes the logits exactly as `Qwen3_5ForCausalLM` (bitwise, prefill and decode) and
  hands back the final-norm hidden of every row (prefill: the whole chunk, before the query_position slice).
- The backbone keeps the Qwen3.5 forwards (causal LM and wrapper) and overrides only the causal LM's output hook
  `_build_outputs`; the graph outputs keep `(logits, *states)` in front and put the final hidden last. The Qwen3.5
  default outputs are unchanged.
- `assemble_prefill_hidden` joins the per-window outputs and places them at the attention-mask slots.
- The config flag round-trips and refuses to be combined with `output_hidden_states`; the loop's backbone runner
  accepts an artifact that has it.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from optimum.rbln.transformers.models.fish_omni.configuration_fish_omni import RBLNFishOmniForCausalLMConfig
from optimum.rbln.transformers.models.fish_omni.fish_omni_architecture import (
    FishOmni_CausalLMWrapper,
    FishOmniForCausalLM,
)
from optimum.rbln.transformers.models.fish_omni.fish_omni_runtime_utils import RBLNBackboneRunner
from optimum.rbln.transformers.models.fish_omni.modeling_fish_omni import (
    FishOmniRuntimeModel,
    assemble_prefill_hidden,
)
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_architecture import (
    Qwen3_5_CausalLMWrapper,
    Qwen3_5ForCausalLM,
)
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_runtime_utils import RBLNQwen3_5RuntimeModel


HIDDEN, VOCAB, CHUNK = 16, 40, 8


class StandInModel(nn.Module):
    """Returns a fixed final-norm hidden for the given input shape plus two state tensors, like `Qwen3_5Model`."""

    def __init__(self, output_hidden_states=False, output_last_hidden_state=True):
        super().__init__()
        self.rbln_config = SimpleNamespace(
            lora_config=None,
            logits_to_keep=1,
            output_hidden_states=output_hidden_states,
            output_last_hidden_state=output_last_hidden_state,
        )
        self.phase = "prefill"

    def forward(self, inputs_embeds=None, output_hidden_states=None, **kwargs):
        g = torch.Generator().manual_seed(inputs_embeds.shape[0] * 100 + inputs_embeds.shape[1])
        hidden = torch.randn(*inputs_embeds.shape[:2], HIDDEN, generator=g)
        states = [torch.full((1, 3), 1.0), torch.full((1, 3), 2.0)]
        all_hidden = (hidden * 0.5, hidden) if output_hidden_states else None
        return hidden, states, all_hidden


def heads(**flags):
    torch.manual_seed(0)
    causal_lm = nn.Module()
    causal_lm.config = SimpleNamespace(get_text_config=SimpleNamespace)
    causal_lm.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    inner = StandInModel(**flags)
    return FishOmniForCausalLM(causal_lm, inner), Qwen3_5ForCausalLM(causal_lm, inner), causal_lm.lm_head


def test_graph_head_logits_bitwise_and_final_hidden_of_every_row():
    graph, parent, lm_head = heads()
    for phase, shape, query in (("prefill", (1, CHUNK), torch.tensor(5, dtype=torch.int16)), ("decode", (4, 1), None)):
        graph.phase = parent.phase = phase
        x = torch.zeros(*shape, HIDDEN)
        logits, states, all_hidden, final = graph(inputs_embeds=x, query_position=query)
        ref_logits, ref_states, _ = parent(inputs_embeds=x, query_position=query)
        assert torch.equal(logits, ref_logits), phase
        assert all(torch.equal(a, b) for a, b in zip(states, ref_states, strict=True))
        assert all_hidden is None
        assert final.shape == (*shape, HIDDEN), "every row of the chunk / every batch row"
        if phase == "prefill":
            assert logits.shape == (1, 1, VOCAB)
            assert torch.equal(lm_head(final[:, 5:6]), logits), "final hidden is the lm_head input"
        else:
            assert torch.equal(lm_head(final), logits)


def test_backbone_keeps_the_qwen3_5_forwards_and_overrides_only_the_output_hook():
    assert "forward" not in FishOmniForCausalLM.__dict__
    assert "forward" not in FishOmni_CausalLMWrapper.__dict__
    assert FishOmniForCausalLM.forward is Qwen3_5ForCausalLM.forward
    assert FishOmni_CausalLMWrapper.forward is Qwen3_5_CausalLMWrapper.forward
    assert "_build_outputs" in FishOmniForCausalLM.__dict__


def bare_wrapper(cls, causal_lm, inputs_embeds):
    """A causal LM wrapper without its conversion: `forward` with `prepare_forward_args` stubbed to these inputs."""
    wrapper = cls.__new__(cls)
    nn.Module.__init__(wrapper)
    wrapper.model = causal_lm
    wrapper.rbln_config = causal_lm.model.rbln_config
    args = [None] * 16  # prepare_forward_args order: input_ids, inputs_embeds, ..., batch_idx
    args[1] = inputs_embeds
    wrapper.prepare_forward_args = lambda *_: tuple(args)
    return wrapper


@pytest.mark.parametrize(
    ("output_hidden_states", "output_last_hidden_state"), [(False, False), (False, True), (True, False)]
)
def test_graph_outputs_put_the_final_hidden_after_the_states(output_hidden_states, output_last_hidden_state):
    # FishOmni: (logits, *states, *per-layer hiddens if output_hidden_states, final hidden if
    # output_last_hidden_state). Qwen3.5 (the hook's default): the same without the final hidden.
    graph, parent, lm_head = heads(
        output_hidden_states=output_hidden_states, output_last_hidden_state=output_last_hidden_state
    )
    x = torch.randn(4, 1, HIDDEN)
    for causal_lm, cls, final_last in (
        (graph, FishOmni_CausalLMWrapper, output_last_hidden_state),
        (parent, Qwen3_5_CausalLMWrapper, False),
    ):
        causal_lm.phase = "decode"
        out = bare_wrapper(cls, causal_lm, x)()
        hidden, states, all_hidden = causal_lm.model(inputs_embeds=x, output_hidden_states=output_hidden_states)
        want = (lm_head(hidden), *states)
        if output_hidden_states:
            want += tuple(all_hidden)
        if final_last:
            want += (hidden,)
        assert len(out) == len(want), cls.__name__
        assert all(torch.equal(a, b) for a, b in zip(out, want, strict=True)), cls.__name__
    assert len(parent(inputs_embeds=x, output_hidden_states=output_hidden_states)) == 3, "Qwen3.5 default"


def test_prefill_hidden_joins_windows_and_follows_the_mask():
    windows = [torch.arange(4 * 2.0).reshape(1, 4, 2), 100 + torch.arange(4 * 2.0).reshape(1, 4, 2)]
    hidden = assemble_prefill_hidden(windows, 6)
    assert hidden.shape == (1, 6, 2)
    assert torch.equal(hidden[0, 4], windows[1][0, 0])
    mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    placed = assemble_prefill_hidden(windows, 6, mask)
    assert placed.shape == (1, 8, 2)
    assert torch.equal(placed[:, 2:], hidden)
    assert not placed[:, :2].any()


def test_config_flag_round_trip_and_exclusive_with_all_hidden_states():
    cfg = RBLNFishOmniForCausalLMConfig(output_last_hidden_state=True)
    assert cfg.output_last_hidden_state is True
    assert cfg.output_hidden_states is False
    assert RBLNFishOmniForCausalLMConfig().output_last_hidden_state is False
    with pytest.raises(ValueError, match="not both"):
        RBLNFishOmniForCausalLMConfig(output_last_hidden_state=True, output_hidden_states=True)


def test_backbone_runner_takes_an_artifact_with_the_final_hidden():
    cfg = SimpleNamespace(
        use_inputs_embeds=True, output_hidden_states=False, output_last_hidden_state=True, max_seq_len=8192
    )
    runner = RBLNBackboneRunner(SimpleNamespace(rbln_config=cfg))
    assert runner.logits_only is False
    assert runner.max_seq_len == 8192  # the loop's backbone cache bound


class StandInRuntime:
    """Called like a `rebel.Runtime`: records the positional inputs and returns fixed outputs."""

    _index_to_input_name = {0: "inputs_embeds", 1: "cache_position"}

    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.outputs


def runtime_model(cls, outputs, **attrs):
    runtime = object.__new__(cls)  # no rebel.Runtime: only `_run` and its output hook are exercised
    runtime.runtime = StandInRuntime(outputs)
    for key, value in attrs.items():
        setattr(runtime, key, value)
    return runtime


def test_runtime_reads_the_final_hidden_through_the_qwen3_5_output_hook():
    # FishOmniRuntimeModel keeps the parent's `_run` (input order, call) and overrides only `_split_outputs`.
    assert "_run" not in FishOmniRuntimeModel.__dict__
    logits, hidden = torch.randn(1, 1, VOCAB), torch.randn(1, CHUNK, HIDDEN)
    runtime = runtime_model(
        FishOmniRuntimeModel,
        (logits, torch.zeros(2), torch.ones(2), hidden),
        rbln_config=SimpleNamespace(output_hidden_states=False),
        _final_hidden=[],
    )
    embeds, position = torch.randn(1, CHUNK, HIDDEN), torch.arange(CHUNK)[None]
    got_logits, got_hidden = runtime._run({"cache_position": position, "inputs_embeds": embeds})
    assert got_logits is logits
    assert got_hidden is None
    assert len(runtime._final_hidden) == 1
    assert runtime._final_hidden[0] is hidden
    assert runtime.runtime.calls[0][0] is embeds
    assert runtime.runtime.calls[0][1] is position


@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_qwen3_5_runtime_output_hook_default(output_hidden_states):
    # The parent's default hook: logits are output 0, the per-layer hiddens the last num_hidden_layers + 1 outputs.
    logits = torch.randn(1, 1, VOCAB)
    layers = tuple(torch.full((1, 1, HIDDEN), float(i)) for i in range(3))
    runtime = runtime_model(
        RBLNQwen3_5RuntimeModel,
        (logits, torch.zeros(2), torch.ones(2), *layers),
        rbln_config=SimpleNamespace(output_hidden_states=output_hidden_states),
        config=SimpleNamespace(num_hidden_layers=2),
    )
    got_logits, got_hidden = runtime._run({"inputs_embeds": torch.zeros(1), "cache_position": torch.zeros(1)})
    assert got_logits is logits
    if output_hidden_states:
        assert len(got_hidden) == 3
        assert all(a is b for a, b in zip(got_hidden, layers, strict=True))
    else:
        assert got_hidden is None
