import argparse
import os

import torch

from optimum.rbln.experimental import RBLNFlux3DistilledModularPipeline


# FLUX 3 Image Klein (step-distilled, 16 steps). The first run compiles the prompt upsampler, the text encoder, the
# transformer and the VAE for one image size and saves them to --save-dir; pass that directory as --model-id to load
# the compiled pipeline. The prompt is a short request: the prompt upsampler expands it into the full caption.
#
# On RBLN-CR13 the whole pipeline fits one device: about 68 GiB at 1024x1024, and 62 GiB at 2048x2048 with the
# transformer compiled for up to 4 references and 640-token prompts (set below for that size; with the defaults of 10
# references and 1040 tokens the 2048 pipeline takes 134 GiB and two devices).


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="black-forest-labs/flux-3-image-klein")
    parser.add_argument("--save-dir", default="flux-3-image-klein-rbln")
    parser.add_argument("--prompt", default="A man walking with his dog, afternoon sun")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=16)
    return parser.parse_args()


def compile_config(height, width):
    config = {"height": height, "width": width}
    if height * width >= 2048 * 2048:  # one device at the 2048 class
        config.update(max_references=4, max_seq_len=640)
    return config


def main():
    args = parse_args()
    if os.path.isdir(args.model_id):
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(
            args.model_id,
            export=True,
            model_save_dir=args.save_dir,
            rbln_config=compile_config(args.height, args.width),
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
