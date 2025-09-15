import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence

from fireredtts2.codec.rvq import ResidualVQ
from fireredtts2.codec.decoder import AcousticDecoder
from fireredtts2.codec.utils import make_nonpad_mask
from fireredtts2.codec.whisper import (
    WhisperEncoderLayer,
    PretrainedWhisperEncoder,
    WhisperAcousticEncoder,
)


class SslAdaptor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int = None,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        # Input Projection
        self.in_proj = nn.Linear(in_dim, embed_dim)
        # Transformer
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    embed_dim, num_heads, ffn_dim, attn_dropout, dropout
                )
                for _ in range(num_layers)
            ]
        )
        # Output norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, out_dim)
        # Init weight
        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_length: torch.Tensor,
    ):
        # Downsampling
        hidden_states = self.in_proj(hidden_states)
        # Transformer
        attention_mask = make_nonpad_mask(hidden_length).unsqueeze(1)  # (b, 1, t)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states, hidden_length

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ResidualDownConv(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        avg_pooler=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.avg_pooler = avg_pooler
        self.intermediate_dim = embed_dim * avg_pooler
        # Convolution layer for downsampling
        self.gate_proj = nn.Conv1d(
            embed_dim, self.intermediate_dim, avg_pooler, avg_pooler, bias=False
        )
        self.up_proj = nn.Conv1d(
            embed_dim, self.intermediate_dim, avg_pooler, avg_pooler, bias=False
        )
        # Downsampled linear projection
        self.down_proj = nn.Linear(
            self.intermediate_dim, self.intermediate_dim, bias=False
        )
        # Activation function and layer normalization
        self.act_fn = nn.SiLU()
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        # Final output projection
        self.out_proj = nn.Linear(self.intermediate_dim, embed_dim)

    def forward(self, x: torch.Tensor, input_length: torch.Tensor):
        output_length = input_length // self.avg_pooler
        batch_size, seq_len, _ = x.shape  # (B, T, D)

        xt = x.permute(0, 2, 1)  # (B, D, T)
        g = self.gate_proj(xt).permute(0, 2, 1)  # (B, T//4, D*4)
        u = self.up_proj(xt).permute(0, 2, 1)  # (B, T//4, D*4)
        x = x.reshape(batch_size, -1, self.intermediate_dim)  # (B, T//4, D*4)

        c = self.down_proj(self.act_fn(g) * u)  # (B, T//4, D*4)
        res = self.layer_norm(c + x)  # (B, T//4, D*4)

        res = self.out_proj(res)
        return res, output_length


class UpConv(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        stride: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.stride = stride
        self.in_proj = nn.Linear(embed_dim, self.stride * embed_dim)
        # Simple transpose convolution layer to keep channel number consistent
        self.up_conv = nn.ConvTranspose1d(
            self.stride * embed_dim,
            embed_dim,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor, input_length: torch.Tensor):
        x = self.in_proj(x)
        x = x.transpose(1, 2)
        res = self.up_conv(x)
        res = res.transpose(1, 2)
        output_length = input_length * self.stride
        return res, output_length


class RedCodec(nn.Module):
    def __init__(
        self,
        ssl: PretrainedWhisperEncoder,
        ssl_adaptor: SslAdaptor,
        acoustic_encoder: WhisperAcousticEncoder,
        downsample: ResidualDownConv,
        rvq: ResidualVQ,
        upsample: UpConv,
        semantic_decoder: SslAdaptor,
        acoustic_decoder: AcousticDecoder,
    ):
        super().__init__()
        self.ssl = ssl
        self.ssl_adaptor = ssl_adaptor
        self.acoustic_encoder = acoustic_encoder
        self.downsample = downsample
        self.rvq = rvq
        self.upsample = upsample
        self.semantic_decoder = semantic_decoder
        self.acoustic_decoder = acoustic_decoder

    @classmethod
    def from_config(cls, config_json: str) -> "RedCodec":
        with open(config_json, "rb") as f:
            config = json.load(f)["codec"]
        ssl = PretrainedWhisperEncoder.from_pretrained()
        ssl_adaptor = SslAdaptor(**config["ssl_adaptor"])
        acoustic_encoder = WhisperAcousticEncoder(**config["acoustic_encoder"])
        downsample = ResidualDownConv(**config["downsample"])
        rvq = ResidualVQ(**config["rvq"])
        upsample = UpConv(**config["upsample"])
        semantic_decoder = SslAdaptor(**config["semantic_decoder"])
        acoustic_decoder = AcousticDecoder(**config["acoustic_decoder"])
        return cls(
            ssl,
            ssl_adaptor,
            acoustic_encoder,
            downsample,
            rvq,
            upsample,
            semantic_decoder,
            acoustic_decoder,
        )


class RedCodecInfer(RedCodec):
    def __init__(self, codec: RedCodec):
        super().__init__(
            codec.ssl,
            codec.ssl_adaptor,
            codec.acoustic_encoder,
            codec.downsample,
            codec.rvq,
            codec.upsample,
            codec.semantic_decoder,
            codec.acoustic_decoder,
        )

    @classmethod
    def from_pretrained(cls, conf_path: str, ckpt_path: str) -> "RedCodecInfer":
        with open(conf_path, "r") as f:
            codec = RedCodec.from_config(conf_path)
        # support cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device)["generator"]
        ckpt = torch.load(ckpt_path)["generator"]
        codec.load_state_dict(ckpt)
        return cls(codec)

    def _encode_one_batch(self, audio16k: torch.Tensor):
        B, T = audio16k.shape
        audio16k_length = torch.tensor(
            [T] * B, dtype=torch.long, device=audio16k.device
        )
        # Semantic
        ssl, ssl_length = self.ssl.forward(audio16k, audio16k_length)
        ssl = ssl.clone()  # For onnx export
        sem_feats, sem_length = self.ssl_adaptor(ssl, ssl_length)
        # Acoustic
        aco_feats, aco_length = self.acoustic_encoder(audio16k, audio16k_length)
        # VQ
        vq_in_feats = torch.cat([sem_feats, aco_feats], dim=2)
        vq_in_feats, vq_in_length = self.downsample(vq_in_feats, aco_length)
        # RVQ,
        indices = self.rvq.encode_codes(vq_in_feats.transpose(1, 2))  # (nq, B, L)
        indices = indices.permute(1, 0, 2)
        return indices  # (B, nq, L)

    @staticmethod
    def _pad_and_chunk(audio: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        pad_len = math.ceil(audio.shape[1] / chunk_size) * chunk_size - audio.shape[1]
        audio = F.pad(audio, (0, pad_len), mode="constant", value=0)
        audio_chunks = audio.split(chunk_size, dim=1)
        return audio_chunks

    @torch.inference_mode()
    def encode(
        self,
        audio16k: torch.Tensor,
        audio16k_length: torch.Tensor = None,
        batch_size: int = 96,
    ):
        """
        Args:
            audio16k: shape (b, t)
            audio16k_length: (b,)
        Returns:
            token: shape (b, nq, l)
            token_length: (b,)
        """
        if audio16k_length is None:
            assert audio16k.shape[0] == 1
            audio16k_length = torch.tensor(
                [audio16k.shape[1]], dtype=torch.long, device=audio16k.device
            )

        CHUNK_SIZE = 6 * 16000
        B, T = audio16k.shape
        # Pad, chunk, and batch
        audio16k_batch = []
        batch_size_list = []
        for i in range(B):
            # Remove extra paddings
            one_audio_chunks = self._pad_and_chunk(
                audio16k[i : (i + 1), : audio16k_length[i]], CHUNK_SIZE
            )
            audio16k_batch += one_audio_chunks
            batch_size_list.append(len(one_audio_chunks))
        audio16k_batch = torch.cat(audio16k_batch, dim=0)
        # Batch encode
        token_batch = []
        for i in range(0, audio16k_batch.shape[0], batch_size):
            one_audio_batch = audio16k_batch[i : (i + batch_size)]
            one_token_batch = self._encode_one_batch(one_audio_batch)
            token_batch.append(one_token_batch)
        token_batch = torch.cat(token_batch, dim=0)
        # Recover & concat
        token_list = torch.split(
            token_batch, batch_size_list, dim=0
        )  # [(B=1, nq, l), (B=3, nq, l), ...]
        token_list = [
            torch.cat(token_ts.split(1, dim=0), dim=-1)  # (B=1, nq, l)
            for token_ts in token_list
        ]
        # Pad tokens
        token = pad_sequence(
            [ts.squeeze(0).transpose(1, 0) for ts in token_list],
            batch_first=True,
            padding_value=0,
        ).transpose(
            1, 2
        )  # (B, nq, L)
        token_length = (audio16k_length / 1280).ceil().long()
        token = token[
            ..., : token_length.max()
        ]  # Remove extra paddings (we pad to multiples of 6s)
        return token, token_length

    @torch.inference_mode()
    def decode(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B=1, nq, L)
        Returns:
            audio: (B=1, t)
        """
        tokens = tokens.permute(1, 0, 2)  # (B, nq, L) -> (nq, B, L)
        vq_out_feats = self.rvq.decode_codes(tokens)
        vq_out_feats = vq_out_feats.transpose(1, 2)
        vq_out_length = torch.tensor(
            [vq_out_feats.shape[1]], dtype=torch.long, device=vq_out_feats.device
        )
        vq_out_feats, vq_out_length = self.upsample(vq_out_feats, vq_out_length)
        # audio: (b, t)
        audio, audio_length = self.acoustic_decoder(vq_out_feats, vq_out_length)
        return audio

    @torch.inference_mode()
    def decode_one_token(
        self, token: torch.Tensor, cache_dict: Dict[str, torch.Tensor], last_token: bool
    ):
        """Decode one single token to audio.

        Args:
            token: (B=1, nq, L=1)
        Returns:
            audio:  (B=1, t)
        """
        # token->latent->upsample, (naturally causal)
        token = token.permute(1, 0, 2)  # (B, nq, L) -> (nq, B, L)
        vq_out_feats = self.rvq.decode_codes(token)
        vq_out_feats = vq_out_feats.transpose(1, 2)
        vq_out_length = torch.tensor(
            [vq_out_feats.shape[1]], dtype=torch.long, device=vq_out_feats.device
        )
        vq_out_feats, vq_out_length = self.upsample(vq_out_feats, vq_out_length)
        # acoustic decoder
        up_conv_cache = cache_dict.get("up_conv_cache", None)
        bb_conv_cache1 = cache_dict.get("bb_conv_cache1", None)
        bb_conv_cache2 = cache_dict.get("bb_conv_cache2", None)
        bb_kv_cache = cache_dict.get("bb_kv_cache", None)
        is_cache = cache_dict.get("is_cache", None)

        (
            audio,
            new_up_conv_cache,
            new_bb_conv_cache1,
            new_bb_conv_cache2,
            new_bb_kv_cache,
            new_is_cache,
        ) = self.acoustic_decoder.forward_chunk(
            vq_out_feats,
            up_conv_cache,
            bb_conv_cache1,
            bb_conv_cache2,
            bb_kv_cache,
            is_cache,
            last_token,
        )

        new_cache_dict = {
            "up_conv_cache": new_up_conv_cache,
            "bb_conv_cache1": new_bb_conv_cache1,
            "bb_conv_cache2": new_bb_conv_cache2,
            "bb_kv_cache": new_bb_kv_cache,
            "is_cache": new_is_cache,
        }
        return audio, new_cache_dict
