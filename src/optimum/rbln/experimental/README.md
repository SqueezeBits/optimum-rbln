# optimum.rbln.experimental

Models that are being ported to RBLN NPUs and are not supported yet.

They are written like the rest of the library (`RBLNModel`, `RBLN*Config`, `RBLNDiffusionMixin`, lazy package
exports) and are exported from `optimum.rbln` like every other model, so their compiled artifacts load the usual way.
Their interfaces, options and artifacts may still change without notice; the first `from_pretrained` of a process
warns about it (`@experimental`).

Rules of this tree:

- Nothing here changes a file outside it, apart from the `TYPE_CHECKING` import in `optimum/rbln/__init__.py` that
  lists the package. What a model needs from the library it gets by subclassing or overriding, so rebasing on
  upstream never conflicts.
- A model graduates by moving its files to the regular packages and dropping the `@experimental` marker. Its class
  names stay, so the artifacts compiled while it was here keep loading.

## Models

| Model | Pipeline classes | Components | Verified on | Open points |
| --- | --- | --- | --- | --- |
| FLUX 3 Image Klein (`black-forest-labs/flux-3-image-klein`, `-base`) | `RBLNFlux3DistilledModularPipeline`, `RBLNFlux3ModularPipeline` | `RBLNFlux3Transformer2DModel`, `RBLNAutoencoderKLFlux2`, `RBLNQwen3VLTextEncoder`, `RBLNFlux3PromptUpsampler` | RBLN-CR13, one device, 1024-class sizes: text-to-image, editing with up to 10 references, layout control, distilled (16 steps) and base (CFG); 2048x2048 on two devices (text-to-image and editing) | FLUX.3 is only in diffusers `main`; the image-conditioned transformer graphs now and then give a wrong output on RBLN-CR13, so at the 2048 class they are re-run until two runs agree (`verify_outputs`); the 2048 output has not been measured against a reference |
