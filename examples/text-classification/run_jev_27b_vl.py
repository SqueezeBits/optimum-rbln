"""autotrust/JEV-27B-VL System 1 decisions on RBLN NPUs, with the decision adapter as a compiled LoRA.

JEV-27B-VL is a Qwen3.5-27B model plus a LoRA adapter (`adapter_vllm/`) that turns it into a one-pass decision model
("System 1"): the adapter trains every projection, including the LM head, and the decision is read from the
next-token logits at a few verbalizer tokens (`decision_head.json`), plus a bias, divided by a per-kind temperature
(`calibration.json`). Without the adapter (`lora_int_id` 0) the same compiled model is the base model ("System 2").

The prompt and the readout follow the checkpoint's `serve_decide.py` for text requests with up to 16 options.

    python run_jev_27b_vl.py                            # compile once into ./jev-27b-vl-rbln, then answer
    python run_jev_27b_vl.py --compiled-dir jev-27b-vl-rbln
"""

import argparse
import json
import os
import string

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from optimum.rbln import RBLNLoRAAdapterConfig, RBLNQwen3_5ForCausalLM


STATE = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
QUESTIONS = [
    ("choice", "Which department should handle this?", ["billing", "technical support", "other"]),
    ("score", "How urgent is this? 0 = not urgent, 5 = drop everything.", None),
    ("noul", "Does the user threaten to cancel or leave?", None),
]


class JevSystem1:
    def __init__(self, model_dir: str, model: RBLNQwen3_5ForCausalLM):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        with open(os.path.join(model_dir, "adapter_vllm", "decision_head.json")) as f:
            self.head = json.load(f)
        self.temperatures = {"noul": 1.0, "score": 1.0, "choice": 1.0}
        calibration = os.path.join(model_dir, "calibration.json")
        if os.path.exists(calibration):
            with open(calibration) as f:
                self.temperatures.update(json.load(f)["per_kind"])

    def decide(self, kind: str, state, question: str, options: list[str] | None = None) -> dict:
        if kind == "choice":
            low, high = self.head["slots"]["ranges"]["choice"]
            if not 2 <= len(options) <= high - low:
                raise ValueError(f"this example handles 2 to {high - low} options")
            names = options
            lines = [f"{label}) {option}" for label, option in zip(string.ascii_uppercase, options, strict=False)]
        else:
            names = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
            lines = names
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        prompt = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines)
        prompt += "\n[decision]:"

        low, _ = self.head["slots"]["ranges"][kind]
        slot_ids = self.head["verbalizer_ids"][low : low + len(names)]
        bias = torch.tensor(self.head["bias"][low : low + len(names)])
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        logits = self.model(input_ids=input_ids).logits[0, -1].float()
        probs = torch.softmax((logits[slot_ids] + bias) / self.temperatures[kind], dim=-1).tolist()
        best = max(range(len(names)), key=probs.__getitem__)
        return {"kind": kind, "choice": names[best], "probabilities": dict(zip(names, probs, strict=True))}


def load_model(model_dir: str, compiled_dir: str, max_seq_len: int) -> RBLNQwen3_5ForCausalLM:
    if os.path.isdir(compiled_dir):
        model = RBLNQwen3_5ForCausalLM.from_pretrained(compiled_dir, export=False)
    else:
        model = RBLNQwen3_5ForCausalLM.from_pretrained(
            model_dir,
            export=True,
            rbln_config={
                "max_seq_len": max_seq_len,
                "kvcache_partition_len": 4096,
                "dtype": "bfloat16",
                # System 1 reads one forward pass. Add "decode" to also generate with the base model.
                "phases": ["prefill"],
                "lora_config": {
                    "adapters": [RBLNLoRAAdapterConfig(1, "system1", os.path.join(model_dir, "adapter_vllm"))]
                },
                # A 27B model fits one RBLN-CR13 device; RBLN-CA25 needs 8.
                "num_devices": 1,
            },
        )
        model.save_pretrained(compiled_dir)
    model.set_adapter("system1")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="autotrust/JEV-27B-VL")
    parser.add_argument("--compiled-dir", default="jev-27b-vl-rbln")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = (
        args.model_id
        if os.path.isdir(args.model_id)
        else snapshot_download(args.model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    )
    system1 = JevSystem1(model_dir, load_model(model_dir, args.compiled_dir, args.max_seq_len))
    for kind, question, options in QUESTIONS:
        print(json.dumps(system1.decide(kind, STATE, question, options), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
