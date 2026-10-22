import argparse
import json
import os

import torch

from optimum.rbln.experimental import RBLNFlux3DistilledModularPipeline


# Layout control with FLUX 3 Image Klein: each box places one element, as {"id", "desc", "bbox"} with the box as
# [top, left, bottom, right] in 0-1 image coordinates. The prompt upsampler writes the caption around the boxes;
# the boxes guide placement and scale rather than masking pixels. Every box adds about 80 tokens to the caption, so
# compile with a larger `max_seq_len` (e.g. 2080) for layouts of more than about eight boxes.

BOXES = [
    {"id": "obj_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]},
    {"id": "obj_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="black-forest-labs/flux-3-image-klein")
    parser.add_argument("--save-dir", default="flux-3-image-klein-rbln")
    parser.add_argument("--prompt", default="hot-air balloons over a green valley at sunrise")
    parser.add_argument("--boxes", help="JSON file with a list of boxes; the two balloons above by default")
    parser.add_argument("--max-seq-len", type=int, default=1040)
    return parser.parse_args()


def main():
    args = parse_args()
    boxes = BOXES
    if args.boxes:
        with open(args.boxes) as f:
            boxes = json.load(f)
    if os.path.isdir(args.model_id):
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNFlux3DistilledModularPipeline.from_pretrained(
            args.model_id,
            export=True,
            model_save_dir=args.save_dir,
            rbln_config={"max_seq_len": args.max_seq_len},
        )
    image = pipe(
        prompt=args.prompt,
        boxes=boxes,
        generator=torch.manual_seed(42),
        output="images",
    )[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
