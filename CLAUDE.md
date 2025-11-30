**Current task**

We have ported the Chatterbox model from PyTorch to MLX in Python. You can find more information on the porting process and its current status at `docs/chatterbox-mlx-port.md`.

You can find the relevant original repos at `./reference/`. If they're not there, you can clone them there for reference.

Now we are working on porting Chatterbox from Python MLX to Swift MLX.

**Porting from PyTorch to MLX in Python**

Use MLX built-ins whenever possible instead of importing extra dependencies or recreating functionality that already exists in MLX. In general, using MLX built-ins will be more efficient. Also in general, try to use more efficient methods rather than naive methods like loops.

Many TTS models have already been ported to MLX in this repo. You can find them at `mlx_audio/tts`. Refer to them as a guide on how to port new models to MLX.

You can also find language models that have been ported to MLX at https://github.com/ml-explore/mlx-lm. These may also be useful references for porting.

**Porting from Python MLX to Swift MLX**

Refer to `../mlx-lm` and `../mlx-swift-lm` for examples of LLMs that have been ported from Python MLX to Swift MLX.

There is a comprehensive porting guide at `../mlx-swift-lm/Libraries/MLXLMCommon/Documentation.docc/porting.md`.

Also see the Python-to-Swift API reference at https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/converting-python for method/function name mapping.
