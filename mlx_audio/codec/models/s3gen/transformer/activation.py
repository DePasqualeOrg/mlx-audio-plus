# Copyright © 2025 Resemble AI (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/resemble-ai/chatterbox
# License: licenses/chatterbox.txt

import mlx.nn as nn

# Swish is equivalent to SiLU, which is built into MLX
Swish = nn.SiLU
