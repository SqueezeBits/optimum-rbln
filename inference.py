import argparse
import os
import time

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
    rbln_config = {
        "text_encoder": {"device": 3},
        "vae": {"device": 3},
        "transformer": (
            {"device": [0, 1, 2, 3], "tensor_parallel_size": 4}
            if os.environ.get("RSD", None) else 
            {"device": 1, "tensor_parallel_size": 1} 
        ),
    }

    # Load compiled model
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config=rbln_config,
    )

    # Generate image
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    start_time = time.perf_counter()
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.0, generator=generator, height=1024, width=1024).images[0]
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")

    # Save image result
    image.save(f"output.png")


if __name__ == "__main__":
    main()
