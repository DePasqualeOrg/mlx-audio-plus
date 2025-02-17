import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .istftnet import AdainResBlk1d
from transformers import AlbertModel

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x):
        return self.linear_layer(x)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=None):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = []
        for _ in range(depth):
            self.cnn.extend([
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                nn.LayerNorm(channels),
                nn.ReLU() if actv is None else actv,
                nn.Dropout(0.2),
            ])
        # MLX doesn't have built-in LSTM, so we'll implement a simplified version
        self.lstm = SimplifiedBiLSTM(channels, channels//2)

    def __call__(self, x, input_lengths, m):
        x = self.embedding(x)
        x = mx.transpose(x, (0, 2, 1))
        m = mx.expand_dims(m, axis=1)
        x = mx.where(m, 0.0, x)

        for conv in self.cnn:
            x = mx.transpose(x, (0, 2, 1))
            x = conv(x)
            x = mx.transpose(x, (0, 2, 1))
            x = mx.where(m, 0.0, x)

        x = mx.transpose(x, (0, 2, 1))
        x = self.lstm(x)
        x = mx.transpose(x, (0, 2, 1))
        x = mx.where(m, 0.0, x)
        return x

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def __call__(self, x, s):
        h = self.fc(s)
        h = mx.reshape(h, (h.shape[0], h.shape[1], 1))
        gamma, beta = mx.split(h, 2, axis=1)

        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return (1 + gamma) * x + beta

class SimplifiedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = nn.LSTM(input_size, hidden_size)
        self.backward_lstm = nn.LSTM(input_size, hidden_size)

    def __call__(self, x):
        # Forward pass
        forward_out = self.forward_lstm(x)[0]

        # Backward pass - reverse the sequence manually
        x_reversed = x[:, ::-1]  # Reverse along sequence dimension
        backward_out = self.backward_lstm(x_reversed)[0]
        backward_out = backward_out[:, ::-1]  # Reverse back

        # Concatenate forward and backward outputs
        return mx.concatenate([forward_out, backward_out], axis=-1)

class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        self.lstm = SimplifiedBiLSTM(d_hid + style_dim, d_hid // 2)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = SimplifiedBiLSTM(d_hid + style_dim, d_hid // 2)

        # F0 and N blocks
        self.F0 = [
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        ]

        self.N = [
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout, bias=True),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        ]

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, padding=0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, padding=0)

    def __call__(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        x = self.lstm(d, text_lengths)

        # Apply dropout during inference
        x = mx.random.bernoulli(0.5, x.shape) * x

        duration = self.duration_proj(x)
        en = mx.matmul(mx.transpose(d), alignment)
        return mx.squeeze(duration, axis=-1), en

    def F0Ntrain(self, x, s):
        x = mx.array(x)
        s = mx.array(s)
        x = self.shared(mx.transpose(x, (0, 2, 1)))

        # F0 prediction
        F0 = mx.transpose(x, (0, 2, 1))
        for block in self.F0:
            F0 = block(F0, s)

        F0 = mx.transpose(F0, (0, 2, 1))
        F0 = self.F0_proj(F0 )
        F0 = mx.transpose(F0, (0, 2, 1))

        # N prediction
        N = mx.transpose(x, (0, 2, 1))
        for block in self.N:
            N = block(N, s)
        N = mx.transpose(N, (0, 2, 1))
        N = self.N_proj(N)
        N = mx.transpose(N, (0, 2, 1))

        return mx.squeeze(F0, axis=1), mx.squeeze(N, axis=1)




class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = []
        for _ in range(nlayers):
            self.lstms.extend([
                SimplifiedBiLSTM(d_model + sty_dim, d_model // 2),
                AdaLayerNorm(sty_dim, d_model)
            ])
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def __call__(self, x, style, text_lengths, m):
        x = mx.array(x)
        style = mx.array(style)
        text_lengths = mx.array(text_lengths)
        m = mx.array(m)
        x = x.transpose(2, 0, 1)
        s = mx.broadcast_to(style, (x.shape[0], x.shape[1], style.shape[-1]))

        x = mx.concatenate([x, s], axis=-1)
        x = mx.where(m[..., None].transpose(1, 0, 2), 0.0, x)
        x = x.transpose(1, 0, 2)
        input_lengths = text_lengths
        x = x.transpose(0, 2, 1)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x, style).transpose(0, 1, -1)
                x = mx.concatenate([x, s.transpose(1, -1, 0)], axis=1)
                x = mx.where(m[..., None].transpose(0, 2, 1), 0.0, x)
            else:
                x = x.transpose(0, 2, 1)[0]

                x = block(x)
                x = mx.sort(x, axis=1)
                x = x[None, :].transpose(0, 2, 1)
                x_pad = mx.zeros([x.shape[0], x.shape[1], m.shape[-1]])
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
        return x.transpose(0, 2, 1)



# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
# TODO: Implement this in MLX
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state