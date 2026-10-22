import argparse
import os

import torch

from optimum.rbln.experimental import RBLNQwenImage21TurboModularPipeline


# Qwen-Image 2.1 Turbo (8 steps, no guidance). The first run compiles the Qwen3-VL prompt encoder, the transformer
# and the VAE for one profile of image sizes and saves them to --save-dir; pass that directory as --model-id to load
# the compiled pipeline. Profile "1024" serves the 1024-class aspect ratios, "2048" the 2048-class presets of the
# model card (2048x2048, 2528x1696, ...).
#
# On RBLN-CR13 the pipeline takes two devices (see DEVICES).

# The devices are a runtime setting, so they are given when loading a compiled pipeline too: the transformer's layer
# groups are spread over both, the prompt encoder runs on the first and the VAE decoder on the second (with its encoder
# on the first).
DEVICES = {"transformer": {"device": [0, 1]}, "text_encoder": {"device": 0}, "vae": {"device": [1, 0]}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1-Turbo")
    parser.add_argument("--save-dir", default="qwen-image-2.1-turbo-rbln")
    parser.add_argument("--profile", choices=["1024", "2048"], default="1024")
    parser.add_argument("--prompt", default="A red fox in a snow-covered forest clearing at golden hour")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.isdir(args.model_id):
        pipe = RBLNQwenImage21TurboModularPipeline.from_pretrained(args.model_id, rbln_config=DEVICES)
    else:
        pipe = RBLNQwenImage21TurboModularPipeline.from_pretrained(
            args.model_id, export=True, model_save_dir=args.save_dir, rbln_config={"profile": args.profile, **DEVICES}
        )
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        generator=torch.manual_seed(42),
        output="images",
    )[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
