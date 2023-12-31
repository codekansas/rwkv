# mypy: disable-error-code="import, override"
r"""Defines a simple API for using the RWKV model.

This code is adapted from the minimimal implementation
`here <https://johanwind.github.io/2023/03/23/rwkv_details.html>`_, adapted
to be fine-tunable.

.. highlight:: python
.. code-block:: python

    from rwkv.model import pretrained_rwkv

    model = pretrained_rwkv("7B")
    predictor = model.predictor()

    for token in predictor.generate("The quick brown fox jumped over the"):
        print(token)

Using the tokenizer requires installing the ``tokenizers`` library:

.. code-block:: bash

    pip install tokenizers

Additionally, using the training mode CUDA kernel requires installing ``triton``:

.. code-block:: bash

    pip install triton

The choices for the model key are:

- ``"169m"``
- ``"430m"``
- ``"1.5b"``
- ``"3b"``
- ``"7b"``
- ``"14b"``
"""

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence

import torch
import torch.nn.functional as F
from ml.models.lora import maybe_lora
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.timer import Timer
from torch import Tensor, nn

from rwkv.wkv import WkvImpl, get_wkv_fn

logger = logging.getLogger(__name__)

PretrainedRwkvKey = Literal["169m", "430m", "1.5b", "3b", "7b", "14b"]

AttentionState = tuple[Tensor, Tensor]
FeedForwardState = Tensor
State = tuple[AttentionState, FeedForwardState]


@dataclass
class ModelArgs:
    url: str
    sha256: str
    emb_dim: int
    num_layers: int


PRETRAINED_MODEL_SIZES: dict[PretrainedRwkvKey, ModelArgs] = {
    "169m": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth",
        sha256="713c6f6137a08d3a86ab57df4f09ea03563329beb3bbabc23509d6c57aa0f9e2",
        emb_dim=768,
        num_layers=12,
    ),
    "430m": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth",
        sha256="261e6b8fef1c7c9e08a4dde31bf5caf8e79c4da38126d77977a4707de82a7f64",
        emb_dim=1024,
        num_layers=24,
    ),
    "1.5b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-20220929-ctx4096.pth",
        sha256="6c97043e1bb0867368249290c97a2fe8ffc5ec12ceb1b5251f4ee911f9982c23",
        emb_dim=2048,
        num_layers=24,
    ),
    "3b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221110-ctx4096.pth",
        sha256="9500633f23d86fbae3cb3cbe7908b97b971e9561edf583c2c5c60b10b02bcc27",
        emb_dim=2560,
        num_layers=32,
    ),
    "7b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-7b/resolve/main/RWKV-4-Pile-7B-20230109-ctx4096.pth",
        sha256="9ea1271b25deb6c72bd29f629147d5013cc7d7c69f9715192f6b6b92fca08f64",
        emb_dim=4096,
        num_layers=32,
    ),
    "14b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-14b/resolve/main/RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth",
        sha256="9e1b9b44f2a98124d86fe35e298f230e3a4fa7b60431962da282817ae1b0bf32",
        emb_dim=5120,
        num_layers=40,
    ),
}

TOKENIZER_URL = "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json"


class Attention(nn.Module):
    init_x: Tensor
    init_state: Tensor

    def __init__(
        self,
        dim: int,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze: bool = False,
        wkv_impl: WkvImpl | None = None,
    ) -> None:
        super().__init__()

        self.time_decay = nn.Parameter(torch.empty(dim))
        self.time_first = nn.Parameter(torch.empty(dim))

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, dim))

        if freeze:
            self.time_decay.requires_grad_(False)
            self.time_first.requires_grad_(False)
            self.time_mix_k.requires_grad_(False)
            self.time_mix_v.requires_grad_(False)
            self.time_mix_r.requires_grad_(False)

        self.key = maybe_lora(nn.Linear(dim, dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)
        self.value = maybe_lora(nn.Linear(dim, dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)
        self.receptance = maybe_lora(nn.Linear(dim, dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)
        self.output = maybe_lora(nn.Linear(dim, dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)

        wkv_fn, init_state = get_wkv_fn(dim, wkv_impl)

        self.register_buffer("init_x", torch.zeros(1, 1, dim), persistent=False)
        self.register_buffer("init_state", init_state, persistent=False)

        self.wkv_fn = wkv_fn

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: AttentionState | None) -> tuple[Tensor, AttentionState]:
        bsz, _, _ = x.shape

        if state is None:
            last_x = self.init_x.repeat_interleave(bsz, dim=0)
            last_state = self.init_state.repeat_interleave(bsz, dim=0)
        else:
            last_x, last_state = state
        last_x = self.time_shift(last_x, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        w = torch.exp(w)
        wkv, next_state = self.wkv_fn(w, u, k, v, last_state)
        rwkv = wkv * sr

        return self.output(rwkv), (x[..., -1:, :], next_state)


class FeedForward(nn.Module):
    init_state: Tensor

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, dim))

        if freeze:
            self.time_mix_k.requires_grad_(False)
            self.time_mix_r.requires_grad_(False)

        self.key = maybe_lora(nn.Linear(dim, ffn_dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)
        self.receptance = maybe_lora(nn.Linear(dim, dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)
        self.value = maybe_lora(nn.Linear(ffn_dim, dim, False), lora_rank, lora_alpha, lora_dropout, freeze=freeze)

        self.register_buffer("init_state", torch.zeros(1, 1, dim), persistent=False)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: FeedForwardState | None = None) -> tuple[Tensor, FeedForwardState]:
        bsz = x.shape[0]

        last_x = self.time_shift(self.init_state.repeat(bsz, 1, 1) if state is None else state, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        vk = self.value(F.relu(k) ** 2)

        return torch.sigmoid(r) * vk, x[..., -1:, :]


class Block(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        pre_norm: bool,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_attn: bool = True,
        lora_ffn: bool = True,
        freeze_layer_norm: bool = False,
        freeze_attn: bool = False,
        freeze_ffn: bool = False,
        wkv_impl: WkvImpl | None = None,
    ) -> None:
        super().__init__()

        self.ln0 = nn.LayerNorm(emb_dim) if pre_norm else None
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

        if freeze_layer_norm:
            if self.ln0 is not None:
                self.ln0.requires_grad_(False)
            self.ln1.requires_grad_(False)
            self.ln2.requires_grad_(False)

        self.att = Attention(
            emb_dim,
            lora_rank=lora_rank if lora_attn else None,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze=freeze_attn,
            wkv_impl=wkv_impl,
        )

        self.ffn = FeedForward(
            emb_dim,
            emb_dim * 4,
            lora_rank=lora_rank if lora_ffn else None,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze=freeze_ffn,
        )

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        if self.ln0 is not None:
            x = self.ln0(x)
        dx, att_state_out = self.att.forward(self.ln1(x), None if state is None else state[0])
        x = x + dx
        dx, ffn_state_out = self.ffn.forward(self.ln2(x), None if state is None else state[1])
        x = x + dx
        return x, (att_state_out, ffn_state_out)


class Rwkv(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_tokens: int,
        num_layers: int,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_embeddings: bool = True,
        lora_linear: bool = True,
        lora_top_k_blocks: int | None = None,
        lora_attn: bool = True,
        lora_ffn: bool = True,
        freeze_non_lora: bool = False,
        freeze_layer_norm: bool | None = None,
        freeze_attn: bool | None = None,
        freeze_ffn: bool | None = None,
        wkv_impl: WkvImpl | None = None,
    ) -> None:
        super().__init__()

        if lora_rank is None:
            freeze_non_lora = False
        if freeze_layer_norm is None:
            freeze_layer_norm = freeze_non_lora
        if freeze_attn is None:
            freeze_attn = freeze_non_lora
        if freeze_ffn is None:
            freeze_ffn = freeze_non_lora

        if lora_top_k_blocks is None:
            min_block = 0
        else:
            min_block = num_layers - lora_top_k_blocks

        self.emb = maybe_lora(
            nn.Embedding(num_tokens, emb_dim),
            lora_rank if lora_embeddings else None,
            lora_alpha,
            lora_dropout,
            freeze=freeze_non_lora,
        )
        blocks = [
            Block(
                emb_dim,
                i == 0,
                lora_rank=lora_rank if i >= min_block else None,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_attn=lora_attn,
                lora_ffn=lora_ffn,
                freeze_layer_norm=freeze_layer_norm,
                freeze_attn=freeze_attn,
                freeze_ffn=freeze_ffn,
                wkv_impl=wkv_impl,
            )
            for i in range(num_layers)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.ln_out = nn.LayerNorm(emb_dim)
        if freeze_layer_norm:
            self.ln_out.requires_grad_(False)
        self.head = maybe_lora(
            nn.Linear(emb_dim, num_tokens, bias=False),
            lora_rank if lora_linear else None,
            lora_alpha,
            lora_dropout,
            freeze=freeze_non_lora,
        )

    def tensor_to(self, x: Tensor) -> Tensor:
        ref_tensor = self.head.weight
        if x.is_floating_point():
            return x.to(ref_tensor)
        return x.to(ref_tensor.device)

    def forward(
        self,
        tokens: Tensor,
        states_in: list[State] | None = None,
        return_logits: bool = False,
    ) -> tuple[Tensor, list[State]]:
        x = self.emb(tokens)
        states_out: list[State] = []
        for i, block in enumerate(self.blocks):
            x, state_out = block(x, None if states_in is None else states_in[i])
            states_out.append(state_out)
        x = self.head(self.ln_out(x))
        if return_logits:
            return x, states_out
        e_x = torch.exp(x - torch.max(x))
        probs = e_x / e_x.sum()
        return probs, states_out

    def predictor(self) -> "RwkvPredictor":
        return RwkvPredictor(self)


def get_tokenizer() -> Any:
    from tokenizers import Tokenizer

    with Timer("downloading tokenizer"):
        tokenizer_path = ensure_downloaded(TOKENIZER_URL, "rwkv", "tokenizer.json")
    return Tokenizer.from_file(str(tokenizer_path))


class RwkvPredictor:
    def __init__(self, rwkv_model: Rwkv) -> None:
        """Provides an API for sampling from the RWKV model.

        Args:
            rwkv_model: The RWKV model to use for sampling.
        """
        super().__init__()

        self.tokenizer = get_tokenizer()
        self.model = rwkv_model

    def sample_probs(self, probs: Tensor, temperature: float = 1.0, top_p: float = 0.85) -> Tensor:
        try:
            probs = probs ** (1 / temperature)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True) + 1e-6)
            next_token = torch.multinomial(probs_sort.squeeze(-3), num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token[..., None, :, :]).squeeze(-1)
            return next_token

        except Exception:
            logger.exception("Error sampling from probabilities.")
            return probs.new_zeros(probs.shape[:-1], dtype=torch.long)

    @torch.no_grad()
    def generate(
        self,
        prompt: str | Tensor,
        max_len: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.85,
        end_toks: Sequence[int] | None = None,
        end_strs: Sequence[str] | None = None,
    ) -> Iterator[str]:
        if isinstance(prompt, str):
            prompt = torch.tensor([self.tokenizer.encode(prompt).ids])
        assert prompt.dim() == 2 and prompt.shape[0] == 1

        probs, state = self.model.forward(self.model.tensor_to(prompt))
        probs = probs[:, -1:]

        end_toks_set = set() if end_toks is None else set(end_toks)
        end_strs_set = [] if end_strs is None else list(end_strs)

        for i in range(max_len):
            token = self.sample_probs(probs, temperature=temperature, top_p=top_p)
            if token in end_toks_set:
                break
            token_str = self.tokenizer.decode([token.item()])
            yield token_str
            if any(e in token_str for e in end_strs_set):
                break
            if i < max_len - 1:
                probs, state = self.model(self.model.tensor_to(torch.tensor([[token]])), state)


def pretrained_rwkv(
    key: PretrainedRwkvKey,
    *,
    device: BaseDevice | None = None,
    lora_rank: int | None = None,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    lora_embeddings: bool = True,
    lora_linear: bool = True,
    lora_top_k_blocks: int | None = None,
    lora_attn: bool = True,
    lora_ffn: bool = True,
    freeze_non_lora: bool = False,
    freeze_layer_norm: bool | None = None,
    freeze_attn: bool | None = None,
    freeze_ffn: bool | None = None,
    empty: bool = False,
    wkv_impl: WkvImpl | None = None,
) -> Rwkv:
    """Returns a pretrained RWKV model.

    Args:
        key: The key of the pretrained model to load.
        device: The device to load the model onto. If None, the model will be
            loaded onto the device returned by ``AutoDevice.detect_device()``.
        lora_rank: The rank of the LoRA decomposition to use.
        lora_alpha: The alpha parameter of the LoRA decomposition.
        lora_dropout: The dropout rate to use in the LoRA decomposition.
        lora_embeddings: Whether to use LoRA for the embedding matrices.
        lora_linear: Whether to use LoRA for the linear layers.
        lora_top_k_blocks: The number of top-k blocks to use in the LoRA
            decomposition.
        lora_attn: Whether to use LoRA for the attention layers.
        lora_ffn: Whether to use LoRA for the feed-forward layers.
        freeze_non_lora: Whether to freeze the non-LoRA parameters. This value
            will override the other freeze parameters if they are None.
        freeze_layer_norm: Whether to freeze the layer normalization parameters.
        freeze_attn: Whether to freeze the attention parameters.
        freeze_ffn: Whether to freeze the feed-forward parameters.
        empty: Whether to return an empty model with the same structure as the
            pretrained model.
        wkv_impl: The implementation of the WKV to use.

    Returns:
        The pretrained RWKV model.
    """
    device = AutoDevice.detect_device() if device is None else device
    model_args = PRETRAINED_MODEL_SIZES[key]

    with Timer("building model skeleton", spinner=True), init_empty_weights():
        model = Rwkv(
            emb_dim=model_args.emb_dim,
            num_tokens=50277,
            num_layers=model_args.num_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_embeddings=lora_embeddings,
            lora_linear=lora_linear,
            lora_top_k_blocks=lora_top_k_blocks,
            lora_attn=lora_attn,
            lora_ffn=lora_ffn,
            freeze_non_lora=freeze_non_lora,
            freeze_layer_norm=freeze_layer_norm,
            freeze_attn=freeze_attn,
            freeze_ffn=freeze_ffn,
            wkv_impl=wkv_impl,
        )

    if empty:
        model._apply(meta_to_empty_func(torch.device("cpu"), torch.bfloat16))
        device.module_to(model)
        return model

    with Timer("downloading checkpoint"):
        ckpt_path = ensure_downloaded(model_args.url, "rwkv", f"{key}.pth", sha256=model_args.sha256)

    with Timer("loading model checkpoint", spinner=True):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    # Build the transformer and loads the checkpoint.
    with Timer("loading state dict", spinner=True):
        model._apply(meta_to_empty_func(torch.device("cpu"), torch.bfloat16))
        model.load_state_dict(ckpt)
        device.module_to(model)

    return model
