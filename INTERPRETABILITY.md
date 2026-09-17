# Interpretability as the practical skill

Written 2026-09-17, a companion to DIRECTION.md. The owner's stance: interpretability is not
separate from using LLMs well; at a certain level, interpreting the model is the skill. This
file says what in this repo already serves that, and what to build next if that stance leads.

## What this repo already is, read that way

`uni response` is a dose-response instrument: push one feature into the residual stream at one
layer, read what the later layers write back along it. Everything below reuses it. The cascade
work was a very precise certificate that the instrument measures a smooth function; that part
is done.

Facts already in hand (`[measured]`, EXPERIMENTS.md):
- Along formality, layers 13 to 23 undo about 14 percent of a small push, up to about 21 percent
  at the top (x = -8.5), and less past it. Along the formal side there is no top out to +40.
- Along `past`, nothing pushes back: the later layers follow the push.
- Along positivity and certainty the answer turns and then flattens; along brevity it turns twice.
- A second model (SmolLM2-360M) pushes back along formality too, from a different layer.

Those are statements about what the model does with a feature, and each has a usage reading:
a feature the model resists needs a larger push to move and returns toward baseline when the
push stops; a feature it passes through moves at full strength and does not self-correct.

## The one idea to carry: measure the model's response, then use the model within it

A practical rule derived from an internal measurement beats one guessed from outputs, because it
generalizes across prompts the way the mechanism does. Three such rules this repo can produce:

1. **Per-feature steering ceiling** (DIRECTION.md A1): the push at which a one-pass reply stops
   being a rewrite, in contrast units, per direction. Usage: a bound on steering strength that is
   measured, not tuned by eye.
2. **Damping map** (DIRECTION.md C1): slope at zero per direction and layer. Usage: which
   qualities of an output the model will drift back from over a multi-turn loop, and which it
   will hold; predicts self-refinement drift before running the loop.
3. **Survival from token probabilities** (DIRECTION.md B1): a settled answer's stability under
   temperature from one greedy pass. Usage: a reproducibility number per prompt, with no loop run.

## Where this goes if it works

- Run C1's table on each new model as part of pinning it; the table is the model's profile.
- Turn 1 to 3 into `uni` subcommands that print a number for a given prompt and direction, so
  "interpreting the model" is a command a user runs before a deployment, not a research project.
- The open scientific question is whether these curves are new relative to the self-repair
  literature (DIRECTION.md C0). Check before claiming; build regardless, since the usage value
  does not depend on novelty.

## What not to conflate

The Feigenbaum constants are not interpretability results: they are the same for every smooth
hump. The hump, its top, its slope and its asymmetry are the interpretability results. Keep the
distinction in every writeup.
