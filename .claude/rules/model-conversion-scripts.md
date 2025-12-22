# Model Conversion Scripts

Some models require custom conversion scripts. Consider whether the main conversion script at `mlx_audio/convert.py` can be used or whether a custom conversion script is needed.

## Guidelines for Custom Conversion Scripts

Look at the other custom conversion scripts in this repository and follow their conventions regarding command line arguments, readme generation, etc. Keep the generated readme concise.

The default Hugging Face cache exists at `~/.cache/huggingface/hub`. This should be used for downloading the original model files from Hugging Face.

Carefully consider which layers should be quantized when quantization is enabled. Consider optimal solutions for layers that may be more sensitive to quantization.

If the model requires tokenizer files, these should use the modern format with `tokenizer.json` and `tokenizer_config.json`.