import argparse
import os

import torch

from optimum.rbln.experimental import RBLNFlux3DistilledModularPipeline


# FLUX 3 Image Klein (step-distilled, 16 steps). The first run compiles the prompt upsampler, the text encoder, the
# transformer and the VAE for one image size and saves them to --save-dir; pass that directory as --model-id to load
# the compiled pipeline. The prompt is a short request: the prompt upsampler expands it into the full caption.
#
# On RBLN-CR13 the whole pipeline fits one device (about 72 GiB at 1024x1024).


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="black-forest-labs/flux-3-image-klein")
    parser.add_argument("--save-dir", default="flux-3-image-klein-rbln")
    parser.add_argument("--prompt", default="A man walking with his dog, afternoon sun")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.isdir(args.model_id):
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(
            args.model_id,
            export=True,
            model_save_dir=args.save_dir,
            rbln_config={"height": args.height, "width": args.width},
        )
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=torch.manual_seed(42),
        output="images",
    )[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
