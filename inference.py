import argparse
import os

import torch

from optimum.rbln import RBLNAutoPipelineForText2Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="(int) random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate image
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.0, generator=generator).images[0]

    # Save image result
    image.save(f"output.png")


if __name__ == "__main__":
    main()
