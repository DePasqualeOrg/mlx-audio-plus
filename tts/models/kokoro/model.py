from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from numbers import Number
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from ..interpolate import interpolate

def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False

def sanitize_state_dict(state_dict: dict) -> dict:
    # Process LSTM weight/bias keys
    processed_dict = {}
    for k, v in state_dict.items():
        # Replace weight_v and weight_g with weight
        if k.endswith(('.weight_v', '.weight_g')):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.weight"
            # Handle conv weights
            if "pool" in new_key or "conv" in new_key or "cnn" in new_key :
                if check_array_shape(v):
                    processed_dict[new_key] = v
                else:
                    logger.debug(f"Transposing {k} with shape {v.shape}")
                    processed_dict[new_key] = v.transpose(0, 2, 1)
            else:
                processed_dict[new_key] = v

        elif k.endswith(('.gamma', '.beta')):
            base_key = k.rsplit('.', 1)[0]
            if k.endswith(".gamma"):
                new_key = f"{base_key}.weight"
            else:
                new_key = f"{base_key}.bias"
            processed_dict[new_key] = v

        elif "F0_proj.weight" in k or "N_proj.weight" in k:
            processed_dict[k] = v.transpose(0, 2, 1)
            print(f"Transposing {k} with shape {v.shape} transposed shape {processed_dict[k].shape}")

        # Replace weight_ih_l0_reverse and weight_hh_l0_reverse with Wx and Wh
        elif k.endswith('.weight_ih_l0_reverse'):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.backward_lstm.Wx"
            processed_dict[new_key] = v
        elif k.endswith('.weight_hh_l0_reverse'):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.backward_lstm.Wh"
            processed_dict[new_key] = v
        elif k.endswith(('.bias_ih_l0_reverse', '.bias_hh_l0_reverse')):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.backward_lstm.bias"
            processed_dict[new_key] = v
        elif k.endswith('.weight_ih_l0'):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.forward_lstm.Wx"
            processed_dict[new_key] = v
        elif k.endswith('.weight_hh_l0'):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.forward_lstm.Wh"
            processed_dict[new_key] = v
        elif k.endswith(('.bias_ih_l0', '.bias_hh_l0')):
            base_key = k.rsplit('.', 1)[0]
            new_key = f"{base_key}.forward_lstm.bias"
            processed_dict[new_key] = v
        else:
            processed_dict[k] = v

    return processed_dict

class KModel(nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    REPO_ID = 'hexgrad/Kokoro-82M'

    def __init__(self, config: Union[Dict, str, None] = None, model: Optional[str] = None):
        super().__init__()
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=KModel.REPO_ID, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")

        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
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
        if not model:
            model = hf_hub_download(repo_id=KModel.REPO_ID, filename='kokoro-v1_0.pth')

        logger.debug(f"Loading model from {model}")
        state_dict = torch.load(model, map_location='cpu', weights_only=True)
        logger.debug(f"State dict: {state_dict.keys()}")
        logger.debug(model)
        for key, state_dict in state_dict.items():
            assert hasattr(self, key), key
            if key == 'bert':
                try:
                    getattr(self, key).load_state_dict(state_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                    getattr(self, key).load_state_dict(state_dict, strict=False)
            if key == "bert_encoder":
                logger.debug(f"Loading {key} from state_dict")
                state_dict = {
                    'weight': mx.array(state_dict["module.weight"]),
                    'bias': mx.array(state_dict["module.bias"])
                }
                getattr(self, key).load_weights(list(state_dict.items()))

            if key == "text_encoder":
                logger.debug(f"Loading {key} from state_dict")
                logger.debug(getattr(self, key).parameters().keys())
                logger.debug(state_dict.keys())
                mlx_state_dict = {
                    k.replace('module.', ''): mx.array(v)
                    for k, v in state_dict.items()
                }
                mlx_state_dict = sanitize_state_dict(mlx_state_dict)
                logger.debug(f"MLX state dict: {mlx_state_dict.keys()}")
                getattr(self, key).load_weights(list(mlx_state_dict.items()))

            if key == "predictor":
                logger.debug(f"Loading {key} from state_dict")
                logger.debug(getattr(self, key).parameters().keys())
                # logger.debug(getattr(self, "predictor").F0[0].parameters())


                mlx_state_dict = {
                    k.replace('module.', ''): mx.array(v)
                    for k, v in state_dict.items()
                }


                mlx_state_dict = sanitize_state_dict(mlx_state_dict)

                logger.debug(f"MLX state dict: {mlx_state_dict.keys()}")
                getattr(self, key).load_weights(list(mlx_state_dict.items()))

    @dataclass
    class Output:
        audio: mx.array
        pred_dur: Optional[mx.array] = None

    def __call__(
        self,
        phonemes: str,
        ref_s: mx.array,
        speed: Number = 1,
        return_output: bool = False # MARK: BACKWARD COMPAT
    ) -> Union['KModel.Output', mx.array]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = mx.array([[0, *input_ids, 0]])
        input_lengths = mx.array([input_ids.shape[-1]])
        text_mask = mx.arange(int(input_lengths.max()))[None, ...]
        text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(input_lengths.dtype)
        text_mask = text_mask + 1 > input_lengths[:, None]
        # TODO: Convert input_ids to MLX array
        bert_dur = self.bert(torch.from_numpy(np.array(input_ids)), attention_mask=torch.from_numpy(np.array(~text_mask)).int())
        bert_dur = mx.array(bert_dur)
        d_en = self.bert_encoder(bert_dur).transpose(0, 2, 1)
        ref_s = ref_s
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        print("d.shape", d.shape)
        x = self.predictor.lstm(d)
        print("x.shape", x.shape)
        duration = self.predictor.duration_proj(x)
        print("duration.shape", duration.shape)
        duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=None).astype(mx.int32)[0]
        print("pred_dur.shape", pred_dur.shape)
        logger.debug(f"pred_dur: {pred_dur}")
        print("pred_dur.shape", pred_dur.shape)
        indices = mx.concatenate([mx.repeat(mx.array(i), int(n)) for i, n in enumerate(pred_dur)])
        pred_aln_trg = mx.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, mx.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg[None, :]
        en = d.transpose(0, 2, 1) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])[0]
        logger.info(f"audio.shape: {audio.shape}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio