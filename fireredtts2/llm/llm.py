import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from huggingface_hub import PyTorchModelHubMixin
from fireredtts2.llm.modules import FLAVORS


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


# Does multinomial sampling without a cuda synchronization
def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


def sample_top_nsigma(logits: torch.Tensor, n: float, temperature: float):
    """_summary_

    Args:
        logits (torch.Tensor): _description_
        n (float): _description_
        temperature (float): _description_

    Returns:
        _type_: _description_
    """
    logits = logits / temperature
    threshold = logits.max(dim=-1, keepdim=True).values - n * logits.std(
        dim=-1, keepdim=True
    )
    logits[logits < threshold] = float("-inf")
    # scores_processed = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    decoder_loss_weight: float
    use_text_loss: bool


class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.text_head = nn.Linear(backbone_dim, config.text_vocab_size, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )

        self.decoder_loss_weight = config.decoder_loss_weight
        self.use_text_loss = config.use_text_loss

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(
                max_batch_size,
                dtype,
                decoder_max_seq_len=self.config.audio_num_codebooks,
            )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
        )

    def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
        """
        Forward pass for Sesame's CSM model.
        This will be added to the model with `model.forward = types.MethodType(forward, model)`

        Args:
            tokens: (batch_size, seq_len, n_codebooks+1)
            tokens_mask: (batch_size, seq_len, n_codebooks+1)
        """

        dtype = next(self.parameters()).dtype
        bsz, seq_len, _ = tokens.size()
        device = tokens.device

        # embed tokens
        embeds = self._embed_tokens(tokens)  # (bsz,seq_len,17,2048)

        # get targets and codebook embeddings corresponding to audio tokens
        audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len]
        target_tokens = tokens[audio_mask][:, :-1]  # [audio_len, n_codebooks]
        # [audio_len, n_codebooks, embed_dim]
        c_embeds = embeds[:, :, :-1, :][audio_mask]

        # get targets corresponding to text tokens
        text_mask = tokens_mask[:, :, -1]
        text_target_mask = torch.roll(input=text_mask, shifts=1, dims=1)
        text_target_tokens = tokens[text_target_mask][:, -1]

        # retain just non-padding embeddings
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        # backbone forward pass
        # [bsz, seq_len]
        padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]
        # [seq_len, seq_len]
        backbone_attn_mask = _create_causal_mask(seq_len, device)
        # [bsz, seq_len, seq_len]
        padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)
        backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
        backbone_attn_mask = backbone_attn_mask | torch.eye(
            seq_len, device=device
        ).bool().unsqueeze(0).expand(bsz, -1, -1)
        input_pos = (
            torch.arange(0, seq_len).unsqueeze(0).expand(bsz, seq_len).long().to(device)
        )
        h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(
            dtype=dtype
        )

        # get backbone embeddings used for audio codebook prediction predict first codebook and compute loss
        audio_mask = torch.roll(audio_mask, -1, 1)  # shift audio mask to the right by 1
        audio_h = h[audio_mask]  # [audio_len, embed_dim]
        c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
        c0_target = target_tokens[:, 0]  # [audio_len]
        c0_loss = F.cross_entropy(c0_logits, c0_target)

        # predict text loss
        text_h = h[text_mask]
        text_logits = self.text_head(text_h)
        text_loss = F.cross_entropy(text_logits, text_target_tokens, ignore_index=0)

        # "compute amortization" (train decoder on random 1/8 subset of audio tokens)
        # important change to 1/8
        indices = torch.randperm(c_embeds.size(0))[: c_embeds.size(0) // 8]
        # [audio_len//16, n_codebooks-1, embed_dim]
        c_embeds = c_embeds[indices][:, :-1, :]
        audio_h = audio_h[indices]  # [audio_len//16, embed_dim]
        target_tokens = target_tokens[indices][:, 1:]  # [audio_len//16, n_codebooks-1]

        # concatenate backbone embeddings and codebook embeddings for decoder input
        # [audio_len//16, n_codebooks, embed_dim]
        decoder_embeds = torch.cat([audio_h.unsqueeze(1), c_embeds], dim=1)
        N, n_codebooks, _ = decoder_embeds.size()
        c_pos = (
            torch.arange(0, n_codebooks)
            .unsqueeze(0)
            .expand(N, n_codebooks)
            .long()
            .to(device)
        )

        decoder_causal_mask = _create_causal_mask(
            decoder_embeds.size(1), device
        ).expand(N, -1, -1)
        decoder_h = self.decoder(
            self.projection(decoder_embeds), input_pos=c_pos, mask=decoder_causal_mask
        ).to(dtype=dtype)
        c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)

        c_loss = F.cross_entropy(
            c_logits.reshape(-1, c_logits.size(-1)), target_tokens.reshape(-1)
        )

        if self.use_text_loss:
            loss = (
                2
                * (
                    (1 - self.decoder_loss_weight) * c0_loss
                    + self.decoder_loss_weight * c_loss
                )
                + 0.01 * text_loss
            )
        else:
            loss = 2 * (
                (1 - self.decoder_loss_weight) * c0_loss
                + self.decoder_loss_weight * c_loss
            )
        return loss, text_loss, c0_loss, c_loss

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(
            dtype=dtype
        )

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            ).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, 10, 0.75)  # fix to 10 and 0.75
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
