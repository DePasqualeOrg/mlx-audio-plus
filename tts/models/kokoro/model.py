from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder, AdaLayerNorm, CustomAlbert, AlbertModelArgs
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from numbers import Number
from typing import Dict, Optional, Union
import json
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from ..interpolate import interpolate
from ..base import check_array_shape
from ...utils import get_class_predicate
import sys

# Force reset logger configuration at the top of your file
logger.remove()  # Remove all handlers
logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])  # Add back with explicit level


class KokoroModel(nn.Module):
    '''
    KokoroModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KokoroModel instance, and it can be reused across
    multiple KokoroPipelines to avoid redundant memory allocation.

    Unlike KokoroPipeline, KokoroModel is language-blind.

    KokoroModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KokoroModel.
    '''

    REPO_ID = 'prince-canuma/Kokoro-82M'

    def __init__(self, repo_id: Optional[str] = None):
        super().__init__()
        if not repo_id:
            logger.debug("No config provided, downloading from HF")
            config = hf_hub_download(repo_id=KokoroModel.REPO_ID, filename='config.json')
        else:
            config = hf_hub_download(repo_id=repo_id, filename='config.json')

        with open(config, 'r', encoding='utf-8') as r:
            config = json.load(r)

        self.config = config
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertModelArgs(vocab_size=config['n_token'], **config['plbert']))

        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], **config['istftnet']
        )

        weight_files = hf_hub_download(repo_id=KokoroModel.REPO_ID if not repo_id else repo_id, filename='kokoro-v1_0.safetensors')

        logger.debug(f"Loading model from {weight_files}")
        weights = mx.load(weight_files)
        sanitized_weights = {}

        # TODO: Create separate sanitize functions for each layer
        quantization = config.get("quantization", None)
        if quantization is None:
            for key, state_dict in weights.items():

                if key.startswith("bert"):
                    if "position_ids" in key:
                        # Remove unused position_ids
                        continue
                    else:
                        # print(k, v.shape)
                        sanitized_weights[key] = state_dict


                if key.startswith("bert_encoder"):
                    sanitized_weights[key] = state_dict

                if key.startswith("text_encoder"):

                    if key.endswith(('.gamma', '.beta')):
                        base_key = key.rsplit('.', 1)[0]
                        if key.endswith(".gamma"):
                            new_key = f"{base_key}.weight"
                        else:
                            new_key = f"{base_key}.bias"

                        sanitized_weights[new_key] = state_dict
                    elif "weight_v" in key:
                        if check_array_shape(state_dict):
                            sanitized_weights[key] = state_dict
                        else:
                            sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                    # Replace weight_ih_l0_reverse and weight_hh_l0_reverse with Wx and Wh
                    elif key.endswith('.weight_ih_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wx_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.weight_hh_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wh_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_ih_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_ih_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_hh_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_hh_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.weight_ih_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wx_forward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.weight_hh_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wh_forward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_ih_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_ih_forward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_hh_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_hh_forward"
                        sanitized_weights[new_key] = state_dict
                    else:
                        sanitized_weights[key] = state_dict


                if key.startswith("predictor"):
                    if "F0_proj.weight" in key:
                        sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                    elif "N_proj.weight" in key:
                        sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                    elif "weight_v" in key:
                        if check_array_shape(state_dict):
                            sanitized_weights[key] = state_dict
                        else:
                            sanitized_weights[key] = state_dict.transpose(0, 2, 1)

                        # Replace weight_ih_l0_reverse and weight_hh_l0_reverse with Wx and Wh
                    elif key.endswith('.weight_ih_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wx_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.weight_hh_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wh_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_ih_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_ih_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_hh_l0_reverse'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_hh_backward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.weight_ih_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wx_forward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.weight_hh_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.Wh_forward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_ih_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_ih_forward"
                        sanitized_weights[new_key] = state_dict
                    elif key.endswith('.bias_hh_l0'):
                        base_key = key.rsplit('.', 1)[0]
                        new_key = f"{base_key}.bias_hh_forward"
                        sanitized_weights[new_key] = state_dict
                    else:
                        sanitized_weights[key] = state_dict

                if key.startswith("decoder"):
                    sanitized_weights[key] = self.decoder.sanitize(key, state_dict)

        else:
            sanitized_weights = weights

        if (quantization := config.get("quantization", None)) is not None:
            # Handle legacy models which may not have everything quantized`
            class_predicate = get_class_predicate()

            nn.quantize(
                self,
                **quantization,
                class_predicate=class_predicate,
            )

        # Load weights
        self.load_weights(list(sanitized_weights.items()))
        mx.eval(self.parameters())
        self.eval()


    @dataclass
    class Output:
        audio: mx.array
        pred_dur: Optional[mx.array] = None

    def __call__(
        self,
        phonemes: str,
        ref_s: mx.array,
        speed: Number = 1,
        return_output: bool = False, # MARK: BACKWARD COMPAT
    ) -> Union['KokoroModel.Output', mx.array]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = mx.array([[0, *input_ids, 0]])
        input_lengths = mx.array([input_ids.shape[-1]])
        text_mask = mx.arange(int(input_lengths.max()))[None, ...]
        text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(input_lengths.dtype)
        text_mask = text_mask + 1 > input_lengths[:, None]
        bert_dur, _ = self.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
        d_en = self.bert_encoder(bert_dur).transpose(0, 2, 1)
        ref_s = ref_s
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=None).astype(mx.int32)[0]
        indices = mx.concatenate([mx.repeat(mx.array(i), int(n)) for i, n in enumerate(pred_dur)])
        pred_aln_trg = mx.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, mx.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg[None, :]
        en = d.transpose(0, 2, 1) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask) # Working fine in MLX
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])[0] # Working fine in MLX
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio