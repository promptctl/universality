"""The pinned model, generating greedily at batch size one, with a seam on the residual stream."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uni.parse import ConfigError
from uni.pinned import Pinned


class ModelError(ConfigError):
    """The pinned model cannot run what the run asks of it. The message says what it could not do."""


def stop_ids(eos_token_id: int | list[int] | None) -> frozenset[int]:
    """The checkpoint's stop tokens, which its generation_config may give as one id or several."""
    if eos_token_id is None:
        raise ModelError("the checkpoint's generation_config names no eos_token_id, so generation could never stop")
    return frozenset([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)


def require_metal() -> torch.device:
    # Metal is the only device: CPU is too slow for this work, and the run host has no CUDA.
    # Reported rather than raised, because "this machine has no Metal" is a run that cannot be run
    # as described - the CLI's own definition of what it prints as `uni: ...` rather than as a
    # traceback. Saying so is not a fallback: there is still no device to choose. [LAW:no-silent-failure]
    if not torch.backends.mps.is_available():
        raise ModelError("Metal (mps) is not available on this machine, and it is the only device this runs on")
    return torch.device("mps")


@dataclass(frozen=True)
class ResidualAdd:
    """Add `vector` to the residual stream leaving decoder layer `layer`, at every position of every step."""

    layer: int
    vector: torch.Tensor  # shape (hidden_size,)


# What ended a reply: the model, with a stop token, or one of the two limits on how long it may run -
# the pinned budget, or the room the context has left after the prompt. The two limits are told
# apart because they are fixed in different places: one by the pin, the other by how long the state
# has grown. [LAW:types-are-the-program]
Ending = Literal["stop", "budget", "context"]


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]  # each id decoded alone, for display
    logprobs: tuple[float, ...]  # log-probability of each generated token when it was chosen
    ended: Ending

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class Model:
    def __init__(self, pinned: Pinned) -> None:
        self.pinned = pinned
        self.device = require_metal()  # before the checkpoint download, not after
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
        self.context_limit = self.model.config.max_position_embeddings  # prompt and generation together

    def encode(self, prompt: str) -> torch.Tensor:
        """The prompt as one user turn in the chat template, shape (1, length), on the device."""
        chat = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def encode_reply(self, prompt: str, reply: str) -> tuple[torch.Tensor, slice]:
        """The prompt as a user turn and `reply` as the assistant's answer, and the span of the reply's own tokens."""

        def chat(content: str) -> torch.Tensor:
            turns = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
            return self.tokenizer.apply_chat_template(turns, return_dict=True, return_tensors="pt")["input_ids"].to(self.device)

        ids, empty, prefix = chat(reply), chat(""), self.encode(prompt)
        # A reply generated against a prompt that nearly fills the context is a reply the model
        # can write and cannot then be shown its own transcript of: the closing tokens this adds
        # push the two past the limit together. [LAW:single-enforcer] the one place that knows.
        if ids.shape[1] > self.context_limit:
            raise ModelError(f"prompt and reply are {ids.shape[1]} tokens together; the context limit of {self.context_limit} cannot hold them")
        # What the template puts after an empty reply is what closes every reply, such as <|im_end|>.
        start = prefix.shape[1]
        closing = empty[:, start:]
        end = ids.shape[1] - closing.shape[1]
        if end <= start:
            raise ModelError("the reply encodes to no tokens, so it has no residual stream to read")
        if not (torch.equal(ids[:, :start], prefix) and torch.equal(ids[:, end:], closing)):
            raise RuntimeError("the chat template encodes a prompt or its closing differently around this reply")
        return ids, slice(start, end)

    @torch.inference_mode()
    def reply_residual(self, prompt: str, reply: str, layer: int) -> torch.Tensor:
        """The residual stream leaving decoder `layer`, averaged over the reply's tokens, shape (hidden_size,).

        It is read where ResidualAdd writes, so a direction built from it steers the same stream.
        """
        ids, span = self.encode_reply(prompt, reply)
        captured = []
        handle = self.layers[self._layer(layer)].register_forward_hook(lambda _m, _i, hidden: captured.append(hidden))
        try:
            self.model(input_ids=ids, logits_to_keep=1)
        finally:
            handle.remove()
        return captured[0][0, span].mean(dim=0)

    @torch.inference_mode()
    def reply_logprob(self, prompt: str, reply: str, additions: Sequence[ResidualAdd]) -> float:
        """The mean log-probability the model gives each token of `reply`, written after `prompt`.

        Mean and not total, so the number is not length wearing a second name; length is its
        own observable. `additions` are the ones the reply was written under: the same model
        turned to a different setting is a different model, and scores the reply differently.
        """
        ids, span = self.encode_reply(prompt, reply)
        # Position i carries the logits that choose token i + 1, so scoring the span needs the
        # positions from one before it. Only those are kept: all of them is more than one Metal
        # kernel can encode for a prompt near the context limit.
        with self._residual(additions):
            out = self.model(input_ids=ids, logits_to_keep=ids.shape[1] - span.start + 1)
        logp = torch.log_softmax(out.logits[0, : span.stop - span.start].float(), dim=-1)
        mean = float(logp.gather(1, ids[0, span].unsqueeze(1)).mean())
        # [LAW:no-silent-failure] as in generate, and for the same reason: printed in a column of
        # six-decimal numbers, a nan out of overflowed additions reads as one more reading.
        if not math.isfinite(mean):
            raise ModelError("the reply has no finite log-probability; the residual additions overflow this model's arithmetic")
        return mean

    def room(self, ids: torch.Tensor) -> int:
        """How many tokens the context leaves to generate after this prompt, refused if none.

        [LAW:single-enforcer] the one place that decides whether a prompt fits, so a sweep asking
        before it starts is told exactly what the run would have told it, a cell in.
        """
        room = self.context_limit - ids.shape[1]
        if room <= 0:
            raise ModelError(f"prompt is {ids.shape[1]} tokens; the context limit of {self.context_limit} leaves no room to generate")
        return room

    @torch.inference_mode()
    def generate(self, prompt: str, additions: Sequence[ResidualAdd] = ()) -> Generation:
        step_ids = self.encode(prompt)
        room = self.room(step_ids)
        past = None
        token_ids: list[int] = []
        logprobs: list[float] = []
        limit = min(self.pinned.max_new_tokens, room)
        with self._residual(additions):
            for _ in range(limit):
                # Logits for the last position only: all positions of a full-context prompt is
                # more than one Metal kernel can encode, and only the last one picks a token.
                out = self.model(input_ids=step_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                token = int(torch.argmax(logp))  # the first index among ties, so ties cannot flicker
                logprob = float(logp[token])
                # [LAW:no-silent-failure] argmax returns an index out of NaN logits just as readily as
                # out of real ones, so without this a collapsed forward pass is recorded as an orbit.
                if not math.isfinite(logprob):
                    raise ModelError(f"step {len(token_ids)} has no finite logits, so no token can be chosen; the residual additions overflow this model's arithmetic")
                token_ids.append(token)
                logprobs.append(logprob)
                if token in self.stop_ids:
                    break
                step_ids = torch.tensor([[token]], device=self.device)
        return Generation(
            text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=tuple(token_ids),
            tokens=tuple(self.tokenizer.decode([token]) for token in token_ids),
            logprobs=tuple(logprobs),
            # Decided by the token that ended the loop, which is never absent: the limit is at least
            # one token, because the pin is positive and `room` refuses none. When the context
            # leaves no more room than the pin allows, it is the context that ended the reply, since
            # a larger budget would have ended it just the same.
            ended="stop" if token_ids[-1] in self.stop_ids else "budget" if limit < room else "context",
        )

    @contextmanager
    def _residual(self, additions: Sequence[ResidualAdd]) -> Iterator[None]:
        # [LAW:dataflow-not-control-flow] no additions is no hooks; the decoding loop never asks.
        with ExitStack() as hooks:
            for addition in additions:
                vector = self.residual_vector(addition)
                handle = self.layers[addition.layer].register_forward_hook(lambda _m, _i, hidden, v=vector: hidden + v)
                hooks.callback(handle.remove)
            yield

    # Both are reachable from a hand-written contrast or a hand-edited direction file, so they are
    # reported rather than raised: this is the earliest place that knows the model's own shape.
    def _layer(self, layer: int) -> int:
        if layer not in range(len(self.layers)):
            raise ModelError(f"layer must be in 0..{len(self.layers) - 1}, got {layer}")
        return layer

    def residual_vector(self, addition: ResidualAdd) -> torch.Tensor:
        """What this addition adds, on this model's device in its dtype, refused unless it fits.

        Public because a map family asks it before it makes a map, not only because the decoding
        loop needs the tensor: what an addition adds to is fixed by the direction rather than by
        the knob's setting, so a layer this checkpoint does not have is wrong in every cell of a
        sweep, and that is worth knowing at the first one. [LAW:parse-dont-validate]
        """
        self._layer(addition.layer)
        if addition.vector.shape != (self.hidden_size,):
            raise ModelError(f"vector must have shape ({self.hidden_size},), got {tuple(addition.vector.shape)}")
        return addition.vector.to(self.device, self.dtype)
