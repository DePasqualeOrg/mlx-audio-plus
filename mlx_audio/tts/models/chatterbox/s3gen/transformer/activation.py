# Copyright © Anthony DePasquale
# Ported to MLX from https://github.com/resemble-ai/chatterbox

import mlx.nn as nn

# Swish is equivalent to SiLU, which is built into MLX
Swish = nn.SiLU
