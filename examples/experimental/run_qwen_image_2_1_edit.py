import argparse
import os
from io import BytesIO

import requests
import torch
from PIL import Image

from optimum.rbln.experimental import RBLNQwenImage21TurboModularPipeline


# Image editing with Qwen-Image 2.1 Turbo: one or more condition images (--image, repeatable) and an edit prompt. A
# condition image is resized to the 1024-class size of its aspect ratio (--output-resolution), as the original
# pipeline does; a reference whose aspect ratio is not among the compiled ones is center-cropped to the closest.
#
# The default input is the sketch of the model card's editing example (3200x1120). Its result is a 2048x2048 image in
# the model card; profile "1024" generates it at 1024x1024 on fewer resources.

EXAMPLE_IMAGE = "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/main/assets/turbo-boat-input.png"

# The devices are a runtime setting, so they are given when loading a compiled pipeline too: the transformer's layer
# groups are spread over both, the prompt encoder runs on the first and the VAE decoder on the second (with its encoder
# on the first).
DEVICES = {"transformer": {"device": [0, 1]}, "text_encoder": {"device": 0}, "vae": {"device": [1, 0]}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1-Turbo")
    parser.add_argument("--save-dir", default="qwen-image-2.1-turbo-rbln")
    parser.add_argument("--profile", choices=["1024", "2048"], default="1024")
    parser.add_argument(
        "--prompt",
        default="Create a photorealistic photograph of the motor yacht from the input sketch, sailing on a calm "
        "deep-blue sea in full side profile, hull black, superstructures and mast white, funnel yellow",
    )
    parser.add_argument("--image", action="append", help="condition image URL or path; repeat for several")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output-resolution", type=int, default=1024)
    return parser.parse_args()


def load_image(source):
    if os.path.exists(source):
        return Image.open(source).convert("RGB")
    return Image.open(BytesIO(requests.get(source, timeout=60).content)).convert("RGB")


def main():
    args = parse_args()
    images = [load_image(source) for source in args.image or [EXAMPLE_IMAGE]]
    if os.path.isdir(args.model_id):
        pipe = RBLNQwenImage21TurboModularPipeline.from_pretrained(args.model_id, rbln_config=DEVICES)
    else:
        pipe = RBLNQwenImage21TurboModularPipeline.from_pretrained(
            args.model_id, export=True, model_save_dir=args.save_dir, rbln_config={"profile": args.profile, **DEVICES}
        )
    image = pipe(
        prompt=args.prompt,
        image=images if len(images) > 1 else images[0],
        height=args.height,
        width=args.width,
        output_resolution=args.output_resolution,
        generator=torch.manual_seed(42),
        output="images",
    )[0]
    image.save(f"{args.prompt[:60]}.png")


if __name__ == "__main__":
    main()
