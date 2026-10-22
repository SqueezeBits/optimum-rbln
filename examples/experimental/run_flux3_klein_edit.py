import argparse
import os
from io import BytesIO

import requests
import torch
from PIL import Image

from optimum.rbln.experimental import RBLNFlux3DistilledModularPipeline


# Image editing with FLUX 3 Image Klein: one to ten reference images (--image, repeatable) and a short instruction.
# The prompt upsampler sees the references and writes the caption of the edited image. The compiled pipeline takes
# up to --max-references images (10 by default); fewer compile less. At 2048x2048 the example compiles for up to 4
# references and 640-token prompts, which fits one RBLN-CR13 device (62 GiB; 134 GiB and two devices otherwise).
#
# The default references are the examples of https://github.com/black-forest-labs/flux-3-image-klein.

EXAMPLES = "https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/examples/inputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="black-forest-labs/flux-3-image-klein")
    parser.add_argument("--save-dir", default="flux-3-image-klein-rbln")
    parser.add_argument("--prompt", default="put a red bandana around the dog's neck")
    parser.add_argument("--image", action="append", help="reference image URL or path; repeat for several")
    parser.add_argument("--max-references", type=int, default=10)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()


def load_image(source):
    if os.path.exists(source):
        return Image.open(source).convert("RGB")
    return Image.open(BytesIO(requests.get(source, timeout=60).content)).convert("RGB")


def compile_config(height, width, max_references):
    config = {"height": height, "width": width, "max_references": max_references}
    if height * width >= 2048 * 2048:  # one device at the 2048 class
        config.update(max_references=min(max_references, 4), max_seq_len=640)
    return config


def main():
    args = parse_args()
    references = [load_image(source) for source in args.image or [f"{EXAMPLES}/3.png"]]
    if os.path.isdir(args.model_id):
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(
            args.model_id,
            export=True,
            model_save_dir=args.save_dir,
            rbln_config=compile_config(args.height, args.width, args.max_references),
        )
    image = pipe(
        prompt=args.prompt,
        image=references if len(references) > 1 else references[0],
        height=args.height,
        width=args.width,
        generator=torch.manual_seed(42),
        output="images",
    )[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
