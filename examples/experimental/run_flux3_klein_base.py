import argparse
import os

import torch
from diffusers import ClassifierFreeGuidance

from optimum.rbln.experimental import RBLNFlux3ModularPipeline


# FLUX 3 Image Klein Base (50 steps, classifier-free guidance). The submodules are compiled exactly as for the
# step-distilled model; the guider runs the transformer once for the prompt and once for the negative prompt. The
# reference code of FLUX 3 samples the base model with CFG 4.0 (3.5 in the first release).


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="black-forest-labs/flux-3-image-klein-base")
    parser.add_argument("--save-dir", default="flux-3-image-klein-base-rbln")
    parser.add_argument("--prompt", default="A man walking with his dog, afternoon sun")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.isdir(args.model_id):
        pipe = RBLNFlux3ModularPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNFlux3ModularPipeline.from_pretrained(
            args.model_id,
            export=True,
            model_save_dir=args.save_dir,
            rbln_config={"height": args.height, "width": args.width},
        )
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=args.guidance_scale))
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        generator=torch.manual_seed(42),
        output="images",
    )[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
