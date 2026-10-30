import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model

from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import RBLNQwen3_5TextModel


def _tiny_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "vocab_size": 128,
        },
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64},
    )


@pytest.mark.parametrize("vl_class", [Qwen3_5ForConditionalGeneration, Qwen3_5Model])
def test_text_model_loads_the_decoder_of_vision_language_checkpoints(tmp_path, vl_class):
    # Qwen3_5ForConditionalGeneration saves the decoder as `model.language_model.*`, a bare Qwen3_5Model (e.g.
    # vllm-sr/d3) as `language_model.*`. Both must load, not leave the decoder randomly initialized.
    torch.manual_seed(0)
    source = vl_class(_tiny_config()).eval()
    source.save_pretrained(tmp_path)
    decoder = source.model.language_model if vl_class is Qwen3_5ForConditionalGeneration else source.language_model

    model = RBLNQwen3_5TextModel.get_pytorch_model(str(tmp_path))

    loaded = model.state_dict()
    for name, tensor in decoder.state_dict().items():
        torch.testing.assert_close(loaded[name].float(), tensor.float(), msg=name)
