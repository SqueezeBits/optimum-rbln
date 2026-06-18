import time

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from optimum.rbln.diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModelWrapper, RBLNFlux2Transformer2DModel



model_id = "black-forest-labs/FLUX.2-klein-4B"
pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32)

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
compiled_transformer = torch.compile(
    wrapper.forward,
    backend="rbln",
    dynamic=False,
    options={
        "tensor_parallel_size": 4,
        "mode": "strict",
    },
)

generator = torch.Generator().manual_seed(42)
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
_ = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=generator,
)

start = time.perf_counter()
image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=generator,
).images[0]
end = time.perf_counter()

print(f"Time taken: {end - start} seconds")
image.save("image.png")