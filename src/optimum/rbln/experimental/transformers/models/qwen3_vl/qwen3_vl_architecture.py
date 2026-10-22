# Copyright 2026 Rebellions Inc. All rights reserved.

import torch
from torch import nn
from transformers import PreTrainedModel


class Qwen3VLTextEncoderWrapper(nn.Module):
    """The Qwen3-VL text tower as a single-pass prompt encoder graph.

    It runs the decoder layers with a causal mask restricted to the valid (attention mask 1) keys, like the Hugging
    Face model does with an attention mask, so the padding positions are computed too. The rotary embeddings come in
    as inputs, computed on the host from the position ids. The output is the requested `hidden_states` entries
    concatenated along the hidden dimension; layers past the last requested one are not part of the graph.
    """

    def __init__(self, model: PreTrainedModel, hidden_states_layers: list[int]) -> None:
        super().__init__()
        language_model = model.model.language_model
        self.embed_tokens = language_model.embed_tokens
        self.layers = language_model.layers[: max(hidden_states_layers)]
        self.norm = language_model.norm
        self.num_hidden_layers = len(language_model.layers)
        self.hidden_states_layers = hidden_states_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Mirrors `Qwen3VLTextModel.forward` with `output_hidden_states=True`: embeddings, the decoder layers under
        the causal mask restricted to the attention-mask keys, the final norm on the last entry only."""
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=attention_mask.dtype))
        allowed = causal[None, None, :, :] * attention_mask[:, None, None, :]
        mask = (allowed - 1) * torch.finfo(allowed.dtype).max
        position_embeddings = (position_cos, position_sin)

        outputs = [hidden_states] if 0 in self.hidden_states_layers else []
        for layer_idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=mask)
            if layer_idx in self.hidden_states_layers:
                # The last entry of the Hugging Face `hidden_states` is taken after the final norm.
                outputs.append(self.norm(hidden_states) if layer_idx == self.num_hidden_layers else hidden_states)
        return torch.cat(outputs, dim=-1)
