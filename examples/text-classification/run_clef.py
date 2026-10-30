"""Cloudflare/clef typed decisions on RBLN NPUs.

Clef is a Qwen3.5-27B backbone plus a joint schema head. The head reads the backbone's final hidden state at every
position and scores each allowed option of each question in a single forward pass. For text-only requests the release
runs only the language model of the backbone, so the language model runs on the NPU through `RBLNQwen3_5TextModel`
and the head runs on the host.

The checkpoint ships its own inference code (`joint_schema_model.py`). This example imports it from the downloaded
snapshot unchanged and only swaps the language model, so read that code before running it, as you would for
`trust_remote_code`.

    python run_clef.py                                  # compile once into ./clef-rbln, then answer
    python run_clef.py --compiled-dir clef-rbln         # reuse the compiled backbone
"""

import argparse
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

from optimum.rbln import RBLNQwen3_5TextModel


REQUEST = {
    "model": "Cloudflare/clef",
    "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "questions": {
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
    },
}


class RBLNLanguageModel(torch.nn.Module):
    """Stands in for the language model of the backbone. Returns float32 hidden states for the float32 head."""

    def __init__(self, rbln_model: RBLNQwen3_5TextModel):
        super().__init__()
        self.rbln_model = rbln_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> BaseModelOutput:
        output = self.rbln_model(input_ids=input_ids, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=output.last_hidden_state.float())


class RBLNClefBackbone(torch.nn.Module):
    """Stands in for the `Qwen3_5ForConditionalGeneration` that the release's `ClefModel` wraps. For a text-only request
    `ClefModel` calls `backbone.model.language_model` and reads `backbone.get_output_embeddings().weight`, which the
    head uses as option vectors."""

    def __init__(self, rbln_model: RBLNQwen3_5TextModel, output_embeddings: torch.Tensor):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = RBLNLanguageModel(rbln_model)
        self.lm_head = torch.nn.Module()
        self.lm_head.weight = torch.nn.Parameter(output_embeddings, requires_grad=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


def load_output_embeddings(model_dir: str) -> torch.Tensor:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    with safe_open(os.path.join(model_dir, index["weight_map"]["lm_head.weight"]), "pt") as f:
        return f.get_tensor("lm_head.weight").float()


def load_clef(model_dir: str, compiled_dir: str, max_seq_len: int):
    sys.path.insert(0, model_dir)
    import joint_schema_model as clef  # shipped with the checkpoint

    if os.path.isdir(compiled_dir):
        rbln_model = RBLNQwen3_5TextModel.from_pretrained(compiled_dir, export=False)
    else:
        rbln_model = RBLNQwen3_5TextModel.from_pretrained(
            model_dir,
            export=True,
            rbln_config={
                "max_seq_len": max_seq_len,
                "kvcache_partition_len": 4096,
                "dtype": "bfloat16",
                # A 27B backbone fits one RBLN-CR13 device; RBLN-CA25 needs 8.
                "num_devices": 1,
            },
        )
        rbln_model.save_pretrained(compiled_dir)

    with open(os.path.join(model_dir, "joint_head_config.json")) as f:
        head = clef.JointSchemaHead(**json.load(f))
    head.load_state_dict(load_file(os.path.join(model_dir, "joint_head.safetensors")), strict=True)
    backbone = RBLNClefBackbone(rbln_model, load_output_embeddings(model_dir))
    model = clef.ClefModel(backbone, head.float()).eval()
    return clef, model, AutoProcessor.from_pretrained(model_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Cloudflare/clef")
    parser.add_argument("--compiled-dir", default="clef-rbln")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_id if os.path.isdir(args.model_id) else snapshot_download(args.model_id)
    clef, model, processor = load_clef(model_dir, args.compiled_dir, args.max_seq_len)
    response = clef.systemone(model, processor, REQUEST, max_length=args.max_seq_len)
    print(json.dumps(response["answers"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
