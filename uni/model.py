"""The pinned model, generating greedily at batch size one, with a seam on the residual stream."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uni.pinned import Device, Pinned

AVAILABLE = {
    "mps": torch.backends.mps.is_available,
    "cpu": lambda: True,
    "cuda": torch.cuda.is_available,
}


def stop_ids(eos_token_id: int | list[int] | None) -> frozenset[int]:
    """The checkpoint's stop tokens, which its generation_config may give as one id or several."""
    if eos_token_id is None:
        raise ValueError("the checkpoint's generation_config names no eos_token_id, so generation could never stop")
    return frozenset([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)


def require_device(device: Device) -> torch.device:
    if not AVAILABLE[device]():
        raise RuntimeError(f"pinned device {device!r} is not available on this machine")
    return torch.device(device)


@dataclass(frozen=True)
class ResidualAdd:
    """Add `vector` to the residual stream leaving decoder layer `layer`, at every position of every step."""

    layer: int
    vector: torch.Tensor  # shape (hidden_size,)


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]  # each id decoded alone, for display
    logprobs: tuple[float, ...]  # log-probability of each generated token when it was chosen

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class Model:
    def __init__(self, pinned: Pinned) -> None:
        self.pinned = pinned
        self.device = require_device(pinned.device)  # before the checkpoint download, not after
        self.dtype = getattr(torch, pinned.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(pinned.model_id, revision=pinned.revision)
        self.model = (
            AutoModelForCausalLM.from_pretrained(pinned.model_id, revision=pinned.revision, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        # Only the stop tokens are taken from the checkpoint's generation_config. Its sampling
        # settings are ignored: generate() below is the whole decoding policy. [LAW:one-source-of-truth]
        self.stop_ids = stop_ids(self.model.generation_config.eos_token_id)
        self.layers = self.model.model.layers
        self.hidden_size = self.model.config.hidden_size

    @torch.inference_mode()
    def generate(self, prompt: str, additions: Sequence[ResidualAdd] = ()) -> Generation:
        chat = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        step_ids = encoded["input_ids"].to(self.device)
        past = None
        token_ids: list[int] = []
        logprobs: list[float] = []
        with self._residual(additions):
            for _ in range(self.pinned.max_new_tokens):
                out = self.model(input_ids=step_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                token = int(torch.argmax(logp))  # the first index among ties, so ties cannot flicker
                token_ids.append(token)
                logprobs.append(float(logp[token]))
                if token in self.stop_ids:
                    break
                step_ids = torch.tensor([[token]], device=self.device)
        return Generation(
            text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=tuple(token_ids),
            tokens=tuple(self.tokenizer.decode([token]) for token in token_ids),
            logprobs=tuple(logprobs),
        )

    @contextmanager
    def _residual(self, additions: Sequence[ResidualAdd]) -> Iterator[None]:
        # [LAW:dataflow-not-control-flow] no additions is no hooks; the decoding loop never asks.
        with ExitStack() as hooks:
            for addition in additions:
                vector = self._residual_vector(addition)
                handle = self.layers[addition.layer].register_forward_hook(lambda _m, _i, hidden, v=vector: hidden + v)
                hooks.callback(handle.remove)
            yield

    def _residual_vector(self, addition: ResidualAdd) -> torch.Tensor:
        if addition.layer not in range(len(self.layers)):
            raise ValueError(f"layer must be in 0..{len(self.layers) - 1}, got {addition.layer}")
        if addition.vector.shape != (self.hidden_size,):
            raise ValueError(f"vector must have shape ({self.hidden_size},), got {tuple(addition.vector.shape)}")
        return addition.vector.to(self.device, self.dtype)
