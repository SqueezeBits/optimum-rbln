import argparse
import os

from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-diffusers-transformer",
        action="store_true",
        default=False,
        help="flag to use diffusers transformer",
    )
    args = parser.parse_args()

    if args.use_diffusers_transformer:
        os.environ["USE_DIFFUSERS_TRANSFORMER"] = "1"

    model_id = "black-forest-labs/FLUX.2-klein-4B"

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=1.0,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
