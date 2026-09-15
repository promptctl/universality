# universality

Does Feigenbaum universality apply to LLM feedback loops? The founding document is
[PROJECT.md](PROJECT.md); this file is how to run the code.

## Setup

Everything runs with [uv](https://docs.astral.sh/uv/).

    uv run uni gen "hello"

That one command installs everything and generates from the pinned model. The first run
downloads the checkpoint, about a gigabyte, into the Hugging Face cache. It prints the
text, its sha256, and each generated token with its log-probability.

The model, its revision, dtype, device, and generation limit are pinned in
[uni/pinned.toml](uni/pinned.toml), and nothing else in the code names them. Decoding is
greedy at batch size one; the checkpoint's own sampling settings are ignored.

## Determinism

Every measurement in this project compares hashes of generated text, so the same input
must produce the same output every time. The gate checks that:

    uv run uni determinism --remote

It generates each case 100 times on one device and prints every run's sha256, then a
verdict per case. It exits 1 if any case produced more than one hash. The cases are an
ordinary prompt, the empty prompt, and a prompt that fills the model's context limit
with 16 tokens left over. `--device` points it at another device and `--runs` changes
the count. `uv run pytest` asserts the same thing on the device given by
`pytest --device`, which defaults to the pinned one.

Hashes are only comparable within one machine. The run host and a dev Mac, both on
Metal, agree on tokens and hashes but differ in the sixth decimal of the
log-probabilities.

The gate passes on the run host on both Metal and CPU. Metal is pinned: it is the
only device that can evaluate large models, and it runs the full-context case about
four times faster than CPU.

`tests/test_determinism.py` also holds a negative test, which shows the failure the
gate protects against. The same prompt, batched with a longer request, gives
log-probabilities that differ from batch size one around the fifth decimal on both
devices. The greedy text survives that on a short prompt, but in a long feedback loop
a near-tie flips a token, and one flipped token turns an orbit into noise. That is
why the wrapper generates at batch size one and has no batch setting.

## Running on the experiment host

Every `uni` command accepts `--remote`. With it, this working tree, uncommitted edits
included, is mirrored to a host with rsync and the same command runs there over ssh,
streaming its output back. Without it, the command runs here.

The host is described only by three variables, read from a gitignored `.env` at the
repo root. Copy the example and fill it in:

    cp .env.example .env

| Variable          | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `UNI_REMOTE_HOST` | Hostname or ssh config alias of the host             |
| `UNI_REMOTE_USER` | The user to ssh as                                   |
| `UNI_REMOTE_DIR`  | Absolute path on the host to mirror the tree into, plain characters only |

A real environment variable overrides the file. Run `--remote` from inside the checkout;
the tree that is mirrored is the one git sees from where you stand. The host needs key-based ssh access,
`rsync`, and `uv` on the PATH that a non-interactive ssh command sees.

    uv run uni host --remote

The repo is public. No hostname, user, or ssh target belongs in the tracked tree.
`tests/test_no_identity.py` fails the suite if a tracked file contains an ssh target, a
`.local` name, a private IPv4 address, or any value from your own `.env`. That last
check is the one that knows your host; run the suite with your `.env` in place.

## Tests

    uv run pytest
