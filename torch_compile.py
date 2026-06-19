import inspect
import time

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from optimum.rbln.diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModelWrapper,
    RBLNFlux2Transformer2DModel,
)


TRANSFORMER_TENSOR_PARALLEL_SIZE = 4
TEXT_ENCODER_DEVICE_ID = 2
VAE_DEVICE_ID = 3
RBLN_COMPILE_OPTIONS = {"mode": "strict"}


def make_rbln_compile_options(*, device_id=None, tensor_parallel_size=1):
    options = {
        **RBLN_COMPILE_OPTIONS,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if device_id is not None:
        options["device"] = device_id
        try:
            from rebel.compile_context import CompileContext
        except ImportError:
            pass
        else:
            compile_context_kwargs = {}
            if "device_id" in inspect.signature(CompileContext).parameters:
                compile_context_kwargs["device_id"] = device_id
            options["compile_context"] = CompileContext(**compile_context_kwargs)
    return options


def torch_compile_rbln(fn, *, name, device_id=None, tensor_parallel_size=1):
    device_log = "" if device_id is None else f" on device {device_id}"
    print(f"Compiling {name} with torch.compile backend='rbln'{device_log}")
    return torch.compile(
        fn,
        backend="rbln",
        dynamic=False,
        options=make_rbln_compile_options(
            device_id=device_id,
            tensor_parallel_size=tensor_parallel_size,
        ),
    )


def install_text_encoder_layer_limit(pipe):
    original_get_qwen3_prompt_embeds = pipe._get_qwen3_prompt_embeds

    def _get_qwen3_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt,
        dtype=None,
        device=None,
        max_sequence_length: int = 512,
        hidden_states_layers: list[int] = (9, 18, 27),
    ):
        max_hidden_state_layer = max(hidden_states_layers or (0,))
        model = getattr(text_encoder, "model", None)
        config = getattr(model, "config", None)
        text_encoder_config = getattr(text_encoder, "config", None)
        original_model_layers = getattr(config, "num_hidden_layers", None)
        original_text_encoder_layers = getattr(text_encoder_config, "num_hidden_layers", None)

        if config is not None and original_model_layers is not None:
            config.num_hidden_layers = min(original_model_layers, max_hidden_state_layer)
        if text_encoder_config is not None and original_text_encoder_layers is not None:
            text_encoder_config.num_hidden_layers = min(original_text_encoder_layers, max_hidden_state_layer)

        try:
            return original_get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
        finally:
            if config is not None and original_model_layers is not None:
                config.num_hidden_layers = original_model_layers
            if text_encoder_config is not None and original_text_encoder_layers is not None:
                text_encoder_config.num_hidden_layers = original_text_encoder_layers

    pipe._get_qwen3_prompt_embeds = _get_qwen3_prompt_embeds


def compile_required_components(pipe):
    install_text_encoder_layer_limit(pipe)
    pipe.text_encoder.forward = torch_compile_rbln(
        pipe.text_encoder.forward,
        name="FLUX.2 Qwen3 text_encoder",
        device_id=TEXT_ENCODER_DEVICE_ID,
    )
    pipe.vae.decode = torch_compile_rbln(
        pipe.vae.decode,
        name="FLUX.2 vae.decode",
        device_id=VAE_DEVICE_ID,
    )
    pipe.vae.encode = torch_compile_rbln(
        pipe.vae.encode,
        name="FLUX.2 vae.encode",
        device_id=VAE_DEVICE_ID,
    )


model_id = "black-forest-labs/FLUX.2-klein-4B"
pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32)
compile_required_components(pipe)


def transformer_forward(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    img_ids=None,
    txt_ids=None,
    guidance=None,
    joint_attention_kwargs=None,
    return_dict=True,
    **kwargs,
):
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = transformer.pos_embed(img_ids)
    text_rotary_emb = transformer.pos_embed(txt_ids)
    image_rotary_emb_0 = torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0).contiguous()
    image_rotary_emb_1 = torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0).contiguous()

    sample = compiled_transformer(
        hidden_states.contiguous(),
        encoder_hidden_states.contiguous(),
        timestep.contiguous(),
        image_rotary_emb_0,
        image_rotary_emb_1,
        guidance=None if guidance is None else guidance.contiguous(),
        joint_attention_kwargs=joint_attention_kwargs,
        return_dict=False,
    )[0]

    if return_dict:
        return Transformer2DModelOutput(sample=sample)
    return (sample,)

pipe.transformer.forward = transformer_forward
pipe.transformer = RBLNFlux2Transformer2DModel._reconstruct_model_if_needed(pipe.transformer)
transformer = pipe.transformer
wrapper = Flux2Transformer2DModelWrapper(transformer).eval()
compiled_transformer = torch_compile_rbln(
    wrapper.forward,
    name="FLUX.2 transformer",
    tensor_parallel_size=TRANSFORMER_TENSOR_PARALLEL_SIZE,
)

generator = torch.Generator().manual_seed(42)
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
_ = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=generator,
)

time_list = []
for i in range(10):
    start = time.perf_counter()
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")
    time_list.append(end - start)
print(f"Average Time taken: {sum(time_list) / len(time_list)} seconds")
image.save("image.png")
