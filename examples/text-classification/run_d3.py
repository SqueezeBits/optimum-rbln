"""vllm-sr/d3 typed decisions on RBLN NPUs.

d3 reads a Qwen3.5-27B backbone as an encoder: its full-attention layers see the whole prompt in both directions, its
GatedDeltaNet layers keep their causal scan, and a 255-way readout scores the answer codes at the last prompt position.
The backbone runs on the NPU through `RBLNQwen3_5TextModel` compiled with bidirectional prefill (`use_attention_mask`
and `use_position_ids`); the readout runs on the host. A bidirectional prompt cannot be split into windows, so the
prefill window, which defaults to `max_seq_len`, bounds the prompt length.

The checkpoint ships its own prompt format and runtime (`d3_format.py`, `d3_runtime.py`). This example imports them
from the downloaded snapshot unchanged, so read that code before running it, as you would for `trust_remote_code`.

    python run_d3.py                                    # compile once into ./d3-rbln, then answer
    python run_d3.py --compiled-dir d3-rbln             # reuse the compiled backbone
"""

import argparse
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from optimum.rbln import RBLNQwen3_5TextModel


STATE = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}


class D3Decider:
    """The release's decision rule around the compiled backbone: render the question with answer codes, read the
    final hidden state at the last prompt token, score the codes with the readout, and for a choice question average
    the original and reversed option orders."""

    def __init__(self, model_dir: str, rbln_model: RBLNQwen3_5TextModel):
        sys.path.insert(0, model_dir)
        import d3_format  # shipped with the checkpoint
        import d3_runtime

        self.format, self.runtime = d3_format, d3_runtime
        self.model = rbln_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.codes, _ = d3_format.answer_codes(self.tokenizer)
        with open(os.path.join(model_dir, "decision_config.json")) as f:
            config = json.load(f)
        self.temperature = float(config.get("temperature", 1.0))
        self.readout = load_file(os.path.join(model_dir, "readout.safetensors"))["weight"].float()

    def distribution(self, state, rendered: dict, num_options: int) -> list[float]:
        text = self.format.render(self.tokenizer, state, rendered, self.codes)
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        hidden = self.model(input_ids=input_ids).last_hidden_state[0, -1].float()
        logits = hidden @ self.readout.T
        return torch.softmax(logits[:num_options] / self.temperature, dim=-1).tolist()

    def probabilities(self, state, question: dict) -> list[float]:
        """Option probabilities in prompt order."""
        normalized = self.runtime.normalize_question(question)
        probs = self.distribution(state, normalized.rendered, len(normalized.keys))
        reversed_question = self.runtime.reversed_question(normalized)
        if reversed_question is not None:
            backward = self.distribution(state, reversed_question, len(normalized.keys))
            probs = self.runtime.average_orders(probs, backward)
        return probs

    def decide(self, state, questions: dict) -> dict:
        return {
            key: self.runtime.product_answer(
                self.runtime.normalize_question(question), self.probabilities(state, question)
            )
            for key, question in questions.items()
        }


def load_model(model_dir: str, compiled_dir: str, max_seq_len: int) -> RBLNQwen3_5TextModel:
    if os.path.isdir(compiled_dir):
        return RBLNQwen3_5TextModel.from_pretrained(compiled_dir, export=False)
    model = RBLNQwen3_5TextModel.from_pretrained(
        model_dir,
        export=True,
        rbln_config={
            "max_seq_len": max_seq_len,
            # Bidirectional prefill over a single window of max_seq_len tokens.
            "use_attention_mask": True,
            "use_position_ids": True,
            "dtype": "bfloat16",
            # A 27B backbone fits one RBLN-CR13 device; RBLN-CA25 needs 8.
            "num_devices": 1,
        },
    )
    model.save_pretrained(compiled_dir)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="vllm-sr/d3")
    parser.add_argument("--compiled-dir", default="d3-rbln")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_id if os.path.isdir(args.model_id) else snapshot_download(args.model_id)
    decider = D3Decider(model_dir, load_model(model_dir, args.compiled_dir, args.max_seq_len))
    print(json.dumps(decider.decide(STATE, QUESTIONS), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
