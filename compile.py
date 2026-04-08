import os

from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
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
