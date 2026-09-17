# Where to take this project next

Written 2026-09-17 at the end of a review session, for the agent that turns it into the
lit backlog. It records a change of direction: the cascade result is finished, and the
next work is the two questions it left open, not more confirmations of it.

## 0. How to use this document

- You are writing and re-ranking lit tickets. Run `lit quickstart new` and
  `lit quickstart update` before you start, and use the flags they print. Do not guess flags.
- Everything you need is in this file. Do not re-read EXPERIMENTS.md end to end (it is 1,900
  lines). Each fact below carries the section heading to spot-check it under if you doubt it.
- Every claim is labeled. `[measured]` is in EXPERIMENTS.md under the named heading.
  `[derived]` is arithmetic on measured numbers. `[theory]` is a standard mathematical result.
  `[opinion]` and `[prediction]` are the author's judgment. `[unknown]` means nobody knows yet,
  and a ticket exists to find out. Copy the label into the ticket when you copy the claim.
- Section 1 holds the rules for every ticket. Section 10 is the checklist you run on your own
  output before you finish. They say the same things on purpose: read section 1 now and section
  10 last.

## 1. Rules for the backlog

1. No ticket whose purpose is to measure delta, alpha or kappa on another hump. Section 2 says why.
   If an existing ticket does this, amend it or close it (section 8); do not leave it.
2. Every experiment ticket carries all of: the command shape (an existing `uni` subcommand, or the
   new map or subcommand it needs); the grid; the starts; the steps or draws; the observables it
   records; the number it reads out; a pre-registered reading, written as "if the reading is X it
   means Y; if it is Z it means W"; and a done-when that names the EXPERIMENTS.md section it adds
   and the README sentence it changes.
3. Rank order among experiments: the harness fix `universality-sweep-81g` first, then A1, A2-greedy,
   B1, D1, then the rest in the order of section 9.
4. Disposition every existing ticket as section 8 says: amend, re-rank, re-parent, or close with the
   reason written in the closing comment. Do not create a new ticket that duplicates an existing one.
5. Titles are declarative sentences in the repo's style. Section 7 has examples of good and bad ones.
6. Compute-heavy children of one epic share one lane, because they run on the same two machines.
   Keep that convention when you create epics A, B and C.
7. Raw outputs (curves, sweeps, trajectories) are content-addressed and committed, as `curves/` and
   `sweeps/` are now. Every experiment ticket keeps that.

## 2. Why the direction changes

### 2.1 What the cascade is

`[measured]` ("The loop closed: a gain, a fixed point, and the first flip"; `uni/response.py`)
The loop that cascades is `x_next = gain * r(x) / |v|^2`. `r(x)` is read with no text generated:
push the residual stream leaving layer 12 by `x` times the formality direction `v`, and read how far
the stream sits along `v` at layer 23, averaged over the prompt's tokens, with the push itself
subtracted. The project's code closes the loop, not the model.

`[theory]` Feigenbaum's three constants are set by the shape class of the hump alone. Lanford (1982)
proved it for smooth one-dimensional maps with a single quadratic maximum. Any smooth single-peaked
`r` with a rounded top, scaled by a gain, gives 4.6692, 2.5029 and 6.619. A flat (quartic) top gives
about 7.28 instead.

`[derived]` So Rung 5's result, that the three constants came back after swapping the knob, the loop
and the model, was guaranteed the moment each swap's curve was a single rounded hump. The
information in each swap was the hump. The constants added nothing model-specific, and measuring
them on more humps adds nothing either. That is rule 1.

### 2.2 The cascade lives where the model no longer writes

`[measured]` ("A push and the model's answer to it: the hump"; "The loop closed") The hump's top is
at x = -8.5. The period-2 attractor that starts the cascade appears at gain 2.875 and already spans
+4.8 to -7.0. The cascade's orbits pass through the top and beyond it.

`[measured]` ("The rewrite loop over the steering coefficient") The text rewrite loop, pushed along
the same direction, writes replies that end only for x from -2.5 to 2.0. At -3.0 the reply is
"Cause the room was booked!" repeated until the 256-token budget stops it. At +2.5 every reply is
cut off, and by +6.0 it slides into repeated boilerplate in another language.

`[caveat]` The two pushes are not the same size. The text loop adds the push at every generated
token, so it compounds over the reply; the response curve is one pass over the prompt. Track A1
measures the single-pass edge so the comparison can be made properly.

`[opinion]` This is why practical insight has not come out of the cascade: it is a fact about the
network's arithmetic in a region of perturbation where the network is not behaving as a language
model. Reading it harder will not change that.

### 2.3 What the text loops showed

`[measured]` ("The loops from many starts: is there a hump?") Rewrite loop, 24 starts: 18 settle on
a fixed text within four rounds, no two on the same one, 4 alternate between two texts, 2 still
changing after six. Summarize loop: all 23 runnable starts fixed within three rounds. No shared
hump, no cascade. Text state space is discrete, so any continuous knob gives a staircase map, and
Feigenbaum's theory does not apply to it as written.

### 2.4 What is real about the model in the existing results

`[measured]` The model's float32 arithmetic supports five doublings read directly, delta_n = 4.625,
and its scatter at periods 4 to 64 is 0.66 to 1.27 times the curve's jitter carried by the gain
("The cascade: delta from six superstable gains"; "Kappa"). The temperature check, one drawn token
per step, matched the noise prediction through period 8 within about 10 percent, and period 16
survived at T = 0.5 where the prediction said 0.3 ("Temperature: how many doublings a drawn token
lets through"). These are results about the model, and they stand.

`[measured]` ("Rung 5, a second knob: brevity", table) Along `past` the curve never turns.
Along `positivity` and `certainty` it turns and then flattens at layer 23. Along `brevity` it turns
twice with steep sides at layers 18 and 20. Along `formality` it is a single hump at layer 23.

`[measured]` The fixed point's slope at gain 1 is -0.14: layers 13 to 23 undo about 14 percent of a
small formality push. `[derived]` At the top, r = 16.01 against a push that reads back as
8.5 x 8.96 = 76, about 21 percent.

### 2.5 Which of Goal 2's tools need universality

`[opinion]` Going down PROJECT.md's Goal 2 list: stability margin, damping, Lyapunov exponent and
the model-collapse framing are general dynamical-systems ideas and need no cascade. Early warning
("oscillation precedes chaos") is contradicted by the project's own loop, where the period-2
attractor coexisted with a stable fixed point from gain 2.875 while the fixed point held until
13.59 (the README already says this). Temperature as a noise budget is the one tool that used the
cascade's theory, and it worked, in the continuous loop. None of them transfers to loops that write
text until someone shows the continuous map predicts the written one. That is Track A. If it does
not, Goal 2 lives in Track B.

## 3. Vocabulary and numbers to carry into tickets

Vocabulary (PROJECT.md "Vocabulary" and EXPERIMENTS.md use these; match them):
- **map**: one step of a loop. Existing maps in `uni/maps.py`: `logistic`, `model` (text in, text
  out), `response` (push in, push-back out, no text).
- **template**: the prompt wrapper. `rewrite`, `summarize`, in `uni/templates.toml`.
- **knob / direction**: `--knob formality` adds `value * v` to the residual stream leaving one layer.
  Contrasts in `uni/directions/*.toml`; derived vectors per model in `uni/directions/<model-id>/`.
- **push**: the value `x`. **push-back / answer**: `r(x)`, read with the push subtracted.
- **gain**: `--value` of the `response` map; the next push is `gain * r(x) / |v|^2`.
- **start / starts**: `--start "text"` or `--starts harbor` / `--starts memo` (passages cut at 1, 2,
  3, 5, 8, 12, 18, 27, 40, 60, 90, 130 words; 24 starts, 3 to 830 characters; `uni/starts.toml`).
- **sweep**: a grid of knob values, each cell a trajectory; `uni sweep`, `uni plot`, `uni observe`.
- **curve**: readings of `r(x)` on a grid, in `curves/`; **fit / smooth**: `uni smooth --degree N`.
- **refused cell**: a reply the 256-token budget cut off is not a state; the cell is refused, not
  recorded. Exit code 79 marks a sweep with refused cells.

Numbers:
- Qwen2.5-0.5B-Instruct: 24 layers, residual width 896; push at layer 12, read at 23.
- SmolLM2-360M-Instruct: 32 layers, width 960; push at 16 (`formality-16`), read at 24, 28, 31.
- Direction lengths: formality |v|^2 = 8.96 (|v| = 2.99); brevity |v| = 5.02; SmolLM2 formality
  |v| = 33.99; SmolLM2 formality-16 |v| = 40.28. A push of x = 1 adds exactly one mean
  (toward minus away) difference, since v is that mean; call x "contrast units".
- Formality response, layer 23: single maximum 16.01 at x = -8.5; quadratic top; falls on both
  sides to -40 and +40. Curve jitter 1.35e-5. Fixed point at gain 1: x = -0.18, slope -0.14.
- mu_1 = 13.59 +- 0.02 (fixed point's slope crosses -1). Period-2 attractor through the top from
  gain 2.875; period 4 between 3.85 and 3.9; period 8 between 4.375 and 4.425; no repeat within
  200 steps by about 4.55; attractor gone between 5.05 and 5.075.
- Text rewrite loop along formality: replies end for x in [-2.5, 2.0]; cut off at -3.0 and below,
  +2.5 and above; longest reply at 2.0 is 24 tokens; settled projection rises from about 1.0 at
  -2.5 to 4.6 at 1.5.
- Temperature check: T in {1.0, 0.7, 0.5, 0.3, 0.2}, hundreds of runs each; period 8 within 10
  percent (model low); period 16 present at 0.5, lost at 0.7.
- Pinned generation budget: 256 new tokens.

## 4. Track A: the bridge. Does the continuous map predict the loop that writes?

This is the existing ticket `universality-rung6-0cd.2d4`, moved to the front, freed from its
dependency on batching for the first pass, and split in three. It answers PROJECT.md's first
"Held open" question: is the state the text or the hidden activations, and does one predict the
other. `[unknown]` either way; that is the point.

Create epic A. Suggested title: "The bridge: whether the model's answer to a push survives a
written reply, and the push at which writing breaks". Topic: `bridge`.

### A1. The push at which a one-pass reply stops being a rewrite, per direction and sentence

- Why: section 2.2. The only breakdown edge measured so far is the compounding text loop's, on one
  direction and one sentence. The single-pass edge is what the response curve can be compared with,
  and it is the "how hard can a direction be pushed" number in the place it means something.
- Run: for each direction in {formality, brevity, positivity, certainty, past} and each of three
  sentences (the meeting sentence plus two from the `harbor` and `memo` sets at 12 to 27 words),
  render the `rewrite` prompt once, push at layer 12 by x on a grid of about 100 values spanning
  at least -20 to +20 (widen per direction until both edges are found), generate a greedy reply
  capped at 32 new tokens. This needs a small new subcommand or a flag on `uni response`
  (suggest `uni reply`), reusing the steered generation the `model` map already does.
- Record per cell: the reply text; whether it ended within the cap; the number of tokens; a
  repetition flag (any 4-gram occurring three or more times); content retention (fraction of the
  original sentence's non-stopword tokens present in the reply); the reply's projection along the
  direction at layer 23, push subtracted token by token, averaged over the reply's tokens only.
- Read out: on each side, the largest |x| at which the reply ends, is not repetitive, and retains
  at least half the content words. Report the three conditions' own edges separately as well, since
  they may fail at different pushes. Report each edge in raw x and in x times |v|.
- Pre-registered: if the edges sit at similar raw x across directions, contrast units are the
  natural steering scale and a practical rule of the form "stay within k contrast units" exists;
  if they sit at similar x times |v|, the absolute norm is the scale; if neither, the limit is per
  direction and no simple rule exists. Separately: formality's known data already has the
  informal-side edge (about -3 compounding) at a third of the hump's top (-8.5) and no top at all
  on the formal side, so "the top marks the edge" is expected to fail; record whether any direction
  has its edge and its top together.
- Done when: EXPERIMENTS.md has a section with the command, a table of edges (direction x sentence
  x side, in x and x|v|), and the figure of one direction's reply projection and content retention
  against x; README's "What isn't shown yet" gains a sentence stating the measured edge and how it
  compares with the top.
- Cost `[rough]`: 5 directions x 3 sentences x 100 pushes x 32 tokens, about 1,500 short
  generations. Hours on the Metal machine. No batching needed.
- Depends on: `universality-sweep-81g` (soft; this run deliberately walks into refused cells).

### A2-greedy. The written-reply map: a staircase, its steps, and whether a hump survives

- Why: the `response` map reads the prompt. This map reads a reply the model writes. If it keeps a
  single hump inside the coherent region from A1, the continuous relaxation predicts the written
  loop and Goal 2's continuous tools transfer. If not, the two are different systems.
- Run: new map `reply` in `uni/maps.py`. State: the push x, four decimals as `response` uses.
  Step: push at layer 12 during a k-token greedy generation from the rendered `rewrite` prompt for
  the fixed sentence; read the residual at layer 23 over the generated tokens only, push subtracted
  as `uni/response.py` does (apply the `admit` rounding bound to generated tokens too); average;
  next push is `gain * reading / |v|^2`. Read the map's curve on a grid of 200 pushes across the
  coherent region from A1, for k in {1, 4, 16, 32}. Then `uni fixed` and `uni sweep --map reply`
  over gain from 1 to 20 from two starts (beside the fixed point and at the curve's top), 400 steps.
- Record: the curve per k (a staircase: each token flips at a threshold of x); step count and the
  largest step height relative to the curve's range; fixed point and slope per gain; periods.
- Read out: whether the curve has one turn inside the coherent region; whether the loop shows
  period 2 and period 4 at gains that the smooth `response` curve would predict.
- Pre-registered: if a hump survives and the loop doubles at least to period 4 at gains within 10
  percent of the `response` map's, the continuous map predicts the written loop (PROJECT.md's held
  question closes "yes"); if the staircase's steps are larger than about a tenth of the range or no
  hump appears inside the coherent region, it does not (closes "no") and Track B carries Goal 2.
  `[prediction]` the author expects "no", with low confidence.
- Done when: EXPERIMENTS.md has the curves at four reply lengths, the fixed-point slopes and the
  sweep pictures; README's "Not the loop people run" bullet says what was learned.
- Cost `[rough]`: 200 pushes x 4 lengths for the curves, then sweeps of 141 gains x 2 starts x 400
  steps x k tokens. Greedy only. A night for k = 32; hours for the rest.
- Depends on: A1.

### A2-sampled. The same map averaged over drawn replies

- Why: sampling at temperature T and averaging N replies per push smooths the staircase; at k = 1
  this is exactly the existing temperature check, which anchors the method.
- Run: as A2-greedy with `--temperature T` for T in {0.3, 0.7} and N in {16, 64} draws per push,
  the reading being the mean over draws; k in {1, 4, 16}.
- Read out: whether the averaged curve has a hump and where; whether its loop doubles.
- Pre-registered: same as A2-greedy. Also, a hump that appears only after averaging is a fact about
  the ensemble, not about any single run, and the writeup says so.
- Cost `[rough]`: N times A2-greedy. This is where `universality-speed-rzl` (batching) pays for
  itself; make it a hard dependency of this ticket and of no other in Track A.

## 5. Track B: the loop people run, under temperature

`[opinion]` This is where Goal 2's practical numbers are. The text loop at T = 0 is a
deterministic map on a finite set, so every orbit is eventually periodic. At T > 0 it is a Markov
chain on texts. Do not write tickets here that expect delta, alpha or kappa. Write them expecting
survival curves, exit probabilities and a zero-parameter prediction.

Create epic B. Suggested title: "The written loop under temperature: how long a settled answer
survives, where it goes when it leaves, and whether the loop's own token probabilities predict
it". Topic: `textloop`.

### B1. Survival of a text fixed point against temperature, predicted from its own token probabilities

- Why: "the run that is unreproducible" (PROJECT.md Goal 2) is a fixed point that sampling knocks
  loose. The per-step probability of staying at a fixed text is exactly the probability of
  sampling that text verbatim: `p_stay(T) = product over reply tokens i of softmax(logits_i / T)[t_i]`,
  with the logits from one greedy pass, since at a fixed point the input and the output are the
  same text. `p_exit(T) = 1 - p_stay(T)`, and predicted survival after k rounds is
  `(1 - p_exit)^k`. No free parameters. `[derived]` from the definition of a fixed point; the
  only approximation is ignoring a second tokenization of the same string, which is negligible.
- Run: for each of the 18 rewrite fixed points and 23 summarize fixed points (from "The loops from
  many starts"): one greedy pass to get the prediction at every T; then `uni loop --map model` with
  `--temperature T` for T in {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, starting at the fixed text, 50 steps,
  64 runs each (seeds recorded). Check `uni/temperature.py` first: it already measures how much
  randomness each T adds, and may hold the per-token spreads this prediction needs.
- Record per step: exact-hash equality with the fixed text; along:<direction> for formality;
  length; and the two distances from B2 if B2 lands first.
- Read out: measured first-exit step distribution and survival curve per T per start, against the
  prediction; the T at which median survival drops below 10 rounds, per template; whether the
  collapse across T is sharp (a band narrower than 0.2 in T) or gradual.
- Pre-registered: if measured survival matches `(1 - p_exit)^k` within run-to-run scatter, the
  loop's stability under noise is readable from one greedy pass, with no loop run at all, and that
  is a shippable diagnostic; if measured survival is longer than predicted, sampled replies return
  to the fixed text (a basin), which B2 measures; if shorter, something other than the reply's
  own tokens moves the chain, and the writeup says what was looked at.
- Done when: EXPERIMENTS.md has the prediction formula, the command, a table of predicted versus
  measured survival per T for both templates, and one figure; README's "Why it would matter" says
  which of Goal 2's tools this makes real, in one sentence.
- Cost `[rough]`: 41 starts x 6 temperatures x 64 runs x 50 steps of full replies. Nights on both
  machines at the current speed. Start with 6 starts per template and 16 runs to see the shape,
  then widen. `universality-speed-rzl` reduces this; soft dependency.

### B1b. Top-p and repetition penalty as the noise dial

- Why: the existing ticket `universality-rung6-0cd.8b2` asks whether top-p and repetition penalty
  act as noise. In Track B that question is: do they move survival the way temperature does, and
  does the prediction from B1 extend to them (top-p truncates the softmax; repetition penalty
  reshapes the logits; both change `p_stay` in a computable way).
- Run: B1 on six starts per template, holding T = 0.7 and sweeping top-p in {0.5, 0.8, 0.95, 1.0},
  then repetition penalty in {1.0, 1.1, 1.3}, prediction recomputed for each.
- Pre-registered: as B1.
- Depends on: B1.

### B2. Where the loop goes after it leaves: paraphrase cloud or content loss

- Why: "at T = 0.7 the loop loses half the original's content within N rounds" is a practical
  number about the loop people run.
- Run: add two observables to `uni/observe.py`: content retention (fraction of the start's
  non-stopword tokens present in the state; the same measure A1 uses) and normalized character
  edit distance to the start. Neither needs an embedding model. Read them on B1's trajectories, and
  on fresh runs started from the original passages (not the fixed points) at the same T grid.
- Read out: per T, the round at which median content retention falls below one half; whether
  distance to start grows without bound or plateaus (a paraphrase cloud); whether orbits that left
  a fixed text return to within edit distance 0.1 of it, and how often.
- Pre-registered: if retention plateaus above one half at T <= 0.5 and falls without bound at
  T >= 0.7, there is a usable temperature ceiling for self-refinement on this model; if it falls
  without bound at every T > 0, there is no safe temperature and the writeup says so.
- Done when: EXPERIMENTS.md has the retention and distance curves against round per T for both
  templates; README's "Why it would matter" quotes the ceiling if one exists.
- Depends on: B1 (shares its trajectories).

### B3. Flip-flops under noise (lower priority)

- Why: the compounding text loop showed period 2 at knob values -0.5, 1.0 and 1.5 and period 3 at
  0.5 ("The rewrite loop over the steering coefficient"). Goal 2's "early warning" asks whether a
  loop that alternates is closer to breaking. Under noise, the question becomes whether a
  period-2 text orbit survives more or fewer rounds than a fixed text does at the same T. The B1
  prediction extends: `p_stay` for a 2-cycle is the product over both replies.
- Run: B1's protocol on those four cells, T in {0.2, 0.5, 0.7}.
- Depends on: B1.

### B4. Prompt sensitivity as divergence of nearby starts (lower priority)

- Why: PROJECT.md Goal 2, "prompt sensitivity as a Lyapunov exponent".
- Run: for six passages, make three one-word variants each (swap one content word for a synonym);
  run rewrite and summarize at T = 0 for 20 steps; read B2's edit distance between the variant's
  orbit and the original's at each step.
- Read out: whether nearby starts converge to the same fixed text, to different fixed texts at a
  fixed distance, or diverge over steps; report the fraction in each class per template.
- Depends on: B2 (the observable).

## 6. Track C: the response curve as a probe (optional epic, off the critical path)

`[opinion]` The instrument `uni response` already produces a dose-response curve: push a feature at
one layer, read what later layers write back. Read across directions and layers it is an
interpretability result about which features these models resist and where, not a chaos result.
Keep it if that interests the owner. It does not need the loop.

`[unknown]` Novelty. Self-repair after ablation is published (McGrath et al. 2023, "the Hydra
effect"; Rushing and Nanda 2024, "Explorations of Self-Repair in Language Models"). Whether
dose-response curves along steering directions, with a measured top and asymmetry, have been
published, the author does not know. The first ticket in this epic is a literature check, and the
epic's README claims are written after it.

This is the existing `universality-rung6-0cd.zna` census, stripped of its constants pipeline.

### C0. Literature check
- Read out: whether the per-direction dose-response curve, its top and its slope at zero appear in
  published self-repair or steering work; list what does and does not. Done when the epic's
  description states what would be new.

### C1. The census as a table of push-back, not of constants
- Run: `uni response` for every committed direction, read at every layer from the push layer to
  the last, on six sentences, for both pinned models (and Qwen2.5-1.5B if it fits; skip the larger
  sizes until C1 shows something worth scaling).
- Record per (direction, layer, sentence): class (monotone / one turn / two or more turns / turn
  then flat); slope at x = 0 (the damping fraction; -0.14 for formality at layer 23); position and
  height of the largest turn; the asymmetry between the two sides at |x| = 3 contrast units.
- Read out: which directions the later layers push back on, how strongly, from which layer on, and
  whether that ordering holds across sentences and across the two models.
- Pre-registered: `past` is expected monotone everywhere (measured once so far); if some direction
  is damped in one model and passed through in the other, that is the first model-specific
  finding of the project.
- Done when: EXPERIMENTS.md has the table and the commands; README has a paragraph in "Why it
  would matter" only if C0 found the result new.
- Cost `[rough]`: no generation; one forward pass per cell; cheap. Banks on idle nights.

### C2. Layer-resolved: where the push-back arises (lower priority)
- Run: per-layer contribution `r_L - r_(L-1)` from C1's readings; for formality, the dip at layers
  18 to 20 that layer 23 smooths into a shoulder.
- Read out: which layers add push-back and which subtract it. Head or MLP ablation is out of scope
  for this ticket; note it as a follow-up if C2 shows one or two layers dominate.

## 7. Track D: the writeup, alongside the rest

### D1. README says what the cascade certifies
- Change: the closing sentence of "Why it would matter" currently reads: "What exists now is a
  first piece of evidence that the math Gleick wrote about reaches inside a neural network."
  Proposed replacement: "What exists now is a precise certificate of one thing: along a direction
  like formality, the model's later layers answer a push with a single smooth rounded hump, and a
  loop closed on that hump does what every such hump does. The orbits of that loop pass through
  pushes larger than any at which the model still writes a rewrite, so it says nothing yet about a
  loop that writes." Keep the compounding caveat (section 2.2) in EXPERIMENTS.md, not the README,
  until A1 measures the single-pass edge.
- Also: in "What isn't shown yet", the first bullet's closing line "the rung's next steps are more
  knobs, loops and models, and the combinations of them" changes to say the next step is whether
  the map survives a written reply (Track A), since more swaps confirm a theorem.
- Done when: both sentences are changed and nothing else in the README is touched.
- Rank: after A1 is filed, before it runs; it is a ten-minute ticket.

### D2. PROJECT.md records the outcome and the reordering
- Change: add a short "Rung 5 outcome" note under "The ladder": the constants held under every
  swap, as the theory requires once each curve is a single rounded hump; the informative content
  of a swap is the hump. Replace Rung 6's text with Tracks A and B (the bridge, and the written loop
  under noise), and move the census to an optional probe. Under "Held open", note that Track A
  decides the first question.
- Done when: PROJECT.md's ladder matches the backlog's epics.

### D3. Figures (the existing epic `universality-figures-qvj`)
- Keep all five. Re-rank: `2r9` (why text loops do not cascade: the diagonal return map beside the
  hump) and `85y` (different humps, same numbers) first, since both explain the reframe; `5dl`,
  `ta4`, `8fc` after. All below A1, A2-greedy, B1 and D1.

## 8. Disposition of every existing ticket

| id | now | do |
| :-- | :-- | :-- |
| `universality-speed-ndl` | in_progress, claimed by this checkout | Leave to its holder. If `lit` reports it stale, follow `lit quickstart work`. Small; finish it. |
| `universality-speed-rzl` | open; blocks zna, 8b2, 2d4 | Keep. Remove it as a dependency of A1 and A2-greedy. Make it a hard dependency of A2-sampled only, and a soft one of B1. Rank after B1. |
| `universality-speed-1hm` (MLX) | open | Rank bottom. |
| `universality-precision-4bp` (float64 on CPU) | open; unblocks a6s | Rank bottom; it only serves a6s. |
| `universality-rung4-a6s` (the d_64 gap) | open | Close. Closing comment: the kappa section shows the model's direct scatter at periods 4 to 64 is 0.66 to 1.27 times the curve's jitter carried by the gain, and at period 64 that moves the return by about 1e-3, the size of the gap; the gap is the model's float32 roughness, and resolving it further changes no conclusion. If the owner would rather keep it, rank it bottom. |
| `universality-sweep-81g` (refused cells forgotten) | open | Rank first among all open work. A1 deliberately walks into refused cells, and reruns must not pay for them again. |
| `universality-docs-003` (period 16 wording) | open | Keep. Rank after D1. |
| `universality-figures-qvj` and children | open | Keep; re-rank as D3. |
| `universality-rung6-0cd` (epic) | open | Retitle to the bridge framing, or close it and create epics A, B, C. Either way its children move as below. |
| `universality-rung6-0cd.zna` (census) | open; blocked by a6s and rzl | Move under epic C as C1. Amend: delete "every single rounded hump carried to delta, alpha and kappa" and the constants pipeline paragraph; add the slope-at-zero, top and asymmetry columns from C1; remove the dependency on a6s; keep rzl soft. Add C0 ahead of it. |
| `universality-rung6-0cd.8b2` (noise past 16; top-p; repetition penalty) | open | Split. "Noise past period 16 with thousands of draws down to T = 0.05" is more kappa confirmation in the continuous loop: close with that reason, or rank bottom. "Top-p and repetition penalty as noise" becomes B1b under epic B. |
| `universality-rung6-0cd.2d4` (writing back into the loop) | open; blocked by rzl | Becomes Track A: split into A1, A2-greedy, A2-sampled under epic A. Remove the dependency on rzl from A1 and A2-greedy. Rank A1 second overall (after sweep-81g). Carry its description's questions into A2. |
| `universality-rung6-0cd.y81` (MSS sequence) | open | Close: it confirms a theorem (the periodic-window order is universal for the same reason the constants are). Or rank bottom. |

## 9. Rank order to end with

1. `universality-sweep-81g` (harness; A1 needs it)
2. A1
3. D1
4. `universality-speed-ndl` (finishing)
5. A2-greedy
6. B1
7. D2
8. `universality-speed-rzl`
9. A2-sampled
10. B2
11. `universality-docs-003`
12. figures `2r9`, `85y`
13. B1b, B3, B4
14. C0, C1, C2 (optional epic; rank as a block here or below the figures, owner's call)
15. figures `5dl`, `ta4`, `8fc`
16. `universality-precision-4bp`, `universality-speed-1hm`, and any of a6s / 8b2-first-half / y81 the
    owner chose to keep rather than close

## 10. Title style, with examples

The repo's titles are declarative sentences naming the result or the question, with a prefix for the
rung or track. Read `git log --oneline | head -30` for thirty examples.

Bad titles (do not write these):
- "Investigate steering vector limits" (no question, no number, no command)
- "Carry the brevity hump at layer 18 to delta, alpha and kappa" (rule 1: confirms a theorem)
- "Add an edit-distance observable" as a standalone ticket (infrastructure with no experiment that
  reads it; fold into B2)
- "Explore temperature effects on the rewrite loop" (no prediction, no grid, no done-when)

Good titles (write these or their equals):
- "Track A1: the push at which a one-pass 32-token reply stops being a rewrite, for five directions
  and three sentences, against the hump's top and the push in contrast units"
- "Track B1: a text fixed point's survival against temperature, predicted with no free parameters
  from the product of its own token probabilities and checked on 64 runs per T"
- "The README says the cascade certifies a shape, and that its orbits pass through pushes at which
  the model no longer writes"

A ticket body has, in this order: Why (one paragraph, copied or condensed from this file with its
labels); Run (command shape, grid, starts, steps or draws); Record; Read out; Pre-registered
reading; Done when; Cost; Depends on. Section 4's A1 is the model.

## 11. Code touchpoints, from the module list (not read this session; verify before citing)

- `uni/maps.py`: maps `logistic`, `model`, `response`. A2 adds `reply`.
- `uni/response.py`: `admit` (rounding bound, refused past 1e-2), `response` (push subtracted
  token by token, then averaged), `turns` (where the slope changes sign). A1 and A2 reuse all three
  over generated tokens.
- `uni/model.py`: `prompt_residual`, `residual_vector`; steered generation used by the `model` map.
- `uni/observe.py`: observables (`length`, `along:<direction>`, exact hash). B2 adds content
  retention and normalized edit distance.
- `uni/temperature.py`: the per-temperature randomness measurement; B1 may reuse it.
- `uni/period.py`, `uni/fixed.py`, `uni/sweep.py`, `uni/loop.py`, `uni/draw.py`: runners and
  detectors; unchanged.
- `uni/starts.toml` (harbor, memo), `uni/templates.toml` (rewrite, summarize),
  `uni/directions/*.toml` (five contrasts plus formality-16), `uni/pinned.toml` (models).
- `tests/`: every new map and observable gets a test on the logistic map or on a fixed string,
  as the existing ones do (`tests/test_cascade.py` checks against mpmath to 40 digits).

## 12. What is not known, stated so that no ticket asserts it

- Whether a hump survives a written reply (A2). Author's guess: no. Low confidence.
- Whether the breakdown edge scales with contrast units or with absolute norm (A1).
- Whether text fixed points' survival matches the zero-parameter prediction (B1). Plausible.
- Whether the single-pass edge differs much from the compounding text loop's -3 / +2.5 (A1).
- Whether Track C's curves are new relative to the self-repair literature (C0).
- Whether anything here holds beyond 0.5B and 360M models. Nothing in this file claims it does.

## 13. Checklist before you finish

- [ ] No open ticket's purpose is to measure delta, alpha or kappa on another hump.
- [ ] Every experiment ticket has: command shape, grid, starts, steps or draws, observables, the
      number read out, a pre-registered reading, a done-when naming the EXPERIMENTS.md section and
      the README sentence.
- [ ] Every claim copied from this file kept its label.
- [ ] Every row of section 8's table was acted on, and each closed ticket's closing comment gives
      the reason from the table.
- [ ] The rank order matches section 9, with `universality-sweep-81g` first and A1 second.
- [ ] Epics A and B exist with the one-lane convention; C exists only if the owner wants it, and
      C0 precedes C1.
- [ ] No ticket duplicates an existing one; 2d4, zna and 8b2 were split or amended, not copied.
- [ ] D1's two README sentences are quoted in its ticket exactly as section 7 gives them.
- [ ] This file: commit it as `DIRECTION.md` next to PROJECT.md once the backlog matches it, since
      it is the rationale for the ranking; or, if the owner prefers PROJECT.md to carry that,
      fold section 2 and section 9 into D2 and delete this file in the same commit.
