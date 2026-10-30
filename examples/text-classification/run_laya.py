"""Laya (convaiinnovations/laya) typed decisions on RBLN NPUs.

Laya is a ModernBERT-large encoder plus a small decision head that reads one [MASK] marker per option and returns
calibrated probabilities in a single forward pass. The encoder runs on the NPU through `RBLNModernBertModel`; the head
(two transformer layers and a scorer) runs on the host on the encoder's last hidden state.

The checkpoint ships its own inference code (`rl_agent_api.py`, `rl_common.py`). This example imports it from the
downloaded snapshot unchanged and only swaps the encoder, so read that code before running it, as you would for
`trust_remote_code`.

    python run_laya.py                                   # compile once into ./laya-rbln, then answer
    python run_laya.py --compiled-dir laya-rbln          # reuse the compiled encoder
"""

import argparse
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from transformers.modeling_outputs import BaseModelOutput

from optimum.rbln import RBLNModernBertModel


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


class RBLNEncoder(torch.nn.Module):
    """Stands in for Laya's encoder. The compiled graph has batch 1 and a fixed length, so each row is padded to
    `max_seq_len` (masked out) and the hidden states are cut back to the batch length."""

    def __init__(self, rbln_model: RBLNModernBertModel, config, pad_token_id: int):
        super().__init__()
        self.rbln_model = rbln_model
        self.config = config
        self.pad_token_id = pad_token_id
        self.max_seq_len = rbln_model.rbln_config.max_seq_len

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BaseModelOutput:
        length = input_ids.shape[1]
        if length > self.max_seq_len:
            raise ValueError(f"Input length {length} exceeds the compiled max_seq_len {self.max_seq_len}.")
        pad = self.max_seq_len - length
        rows = []
        for i in range(input_ids.shape[0]):
            ids = torch.nn.functional.pad(input_ids[i : i + 1], (0, pad), value=self.pad_token_id)
            mask = torch.nn.functional.pad(attention_mask[i : i + 1], (0, pad), value=0)
            out = self.rbln_model(input_ids=ids, attention_mask=mask)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            rows.append(hidden[:, :length].float())
        return BaseModelOutput(last_hidden_state=torch.cat(rows))


def load_agent(model_dir: str, compiled_dir: str):
    sys.path.insert(0, model_dir)
    from rl_agent_api import RLAgent  # shipped with the checkpoint

    agent = RLAgent(model_dir, device="cpu")
    encoder = agent.model.encoder
    if os.path.isdir(compiled_dir):
        rbln_encoder = RBLNModernBertModel.from_pretrained(compiled_dir, export=False)
    else:
        rbln_encoder = RBLNModernBertModel.from_model(
            encoder,
            rbln_max_seq_len=agent.cfg["max_len"],
            rbln_batch_size=1,
        )
        rbln_encoder.save_pretrained(compiled_dir)
    agent.model.encoder = RBLNEncoder(rbln_encoder, encoder.config, agent.tok.pad_token_id)
    return agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--compiled-dir", default="laya-rbln")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = (
        args.model_id
        if os.path.isdir(args.model_id)
        else snapshot_download(
            args.model_id,
            allow_patterns=[
                "*.py",
                "*.json",
                "model.safetensors",
                "encoder/*",
                "tokenizer/*",
            ],
        )
    )
    agent = load_agent(model_dir, args.compiled_dir)
    result = agent.system_one(STATE, QUESTIONS)
    print(json.dumps(result["answers"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
