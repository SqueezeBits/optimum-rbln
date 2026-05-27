import os

from optimum.rbln import RBLNAutoPipelineForText2Image


def main():
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    rbln_config = {
        "text_encoder": {"device": 0},
        "vae": {"device": 0},
        "transformer": (
            {"device": [0, 1, 2, 3], "tensor_parallel_size": 4}
            if os.environ.get("RSD", None) else 
            {"device": 1, "tensor_parallel_size": 1} 
        ),
    }

    # Compile and export
    pipe = RBLNAutoPipelineForText2Image.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=1.0,
        rbln_config=rbln_config,
        rbln_height=1024,
        rbln_width=1024,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
