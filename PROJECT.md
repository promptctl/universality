# Universality

An LLM in a feedback loop is an iterated map. Turn one knob on that loop smoothly and
watch what the loop does. If it goes from settling down, to alternating between two
answers, to alternating among four, then eight, then sixteen, and if the knob distances
between those transitions shrink by a factor of 4.669 each time, then the loop belongs
to the same universality class as the dripping faucet, the logistic map, and convecting
mercury. That number is Feigenbaum's constant, δ. It does not depend on the knob, the
model, or the prompt. It depends only on the shape of the instability.

This project measures it.

## The two goals

**Goal 1: prove it.** Devise and carry out experiments that establish, conclusively and
then comprehensively, that Feigenbaum universality applies to LLM feedback loops. Start
with one loop, one knob, one model, and the cleanest possible measurement of δ. Then
widen: every knob we can turn, every kind of loop, several models, until the result is
undeniable. All of it in Python that anyone can run and get the same numbers.

**Goal 2: use it.** Extract what the dynamical picture says about real LLM usage. Agent
loops, self-refinement, multi-agent debate, and training on synthetic data are all the
same mathematical object as our experiment. The failure modes people describe
anecdotally, the agent that flip-flops between two fixes, the run that is unreproducible,
have names and measurable warning signs in this framework. Goal 2 turns those into
diagnostics.

Goal 1 stands on its own. It is worth doing even if Goal 2 yields nothing.

## What "prove" means here

Feigenbaum's theorem is proven, by Lanford in 1982, for one-parameter families of smooth
one-dimensional maps with a single quadratic hump. Nobody proves universality for a
physical system. They measure it. Libchaber confirmed it in mercury in 1982 with four
period doublings and δ ≈ 4.4. That is the standard we hold ourselves to, and we intend
to beat it.

The claim is falsifiable and we state it that way. Either the ratios of successive
bifurcation intervals converge toward 4.669 across unrelated models, loops, and knobs,
or they do not. A result of "sort of" is a failure. A result of "no, and here is the
number we got instead" is a success of a different kind, and we document it just as
carefully.

Two independent universal numbers make the case far stronger than one. Besides δ, the
theory predicts α ≈ 2.503 for the geometric scaling of the orbit itself, and it predicts
how noise truncates the cascade, with its own exponent near 6.62. Temperature is noise.
If three predicted numbers all land, there is no vibe left to argue about.

## Vocabulary

These are the only terms a reader needs, and every experiment is described in them.

- **Loop.** A fixed template that takes the model's output and feeds it back as the next
  input. "Rewrite this." "Summarize this." "Critique this and revise it." One pass
  through the loop is one step of the map.
- **State.** What the loop carries from step to step. In the simplest version it is the
  text. In the continuous version it is an embedding.
- **Observable.** A single number read off the state at each step, so we can plot
  step n against step n+1. Response length, a projection of the embedding, a score.
- **Knob.** A parameter of the loop we can vary smoothly. The knob plays the role of r
  in the logistic map.
- **Orbit.** The sequence of states the loop visits from a starting point.
- **Fixed point.** The loop stops changing. Period 1.
- **Period doubling.** The knob crosses a value where the fixed point becomes unstable
  and the orbit alternates between two states, then four, then eight. Each crossing is
  a bifurcation at knob value r₁, r₂, r₃, and so on.
- **δ (delta).** The limit of (rₙ − rₙ₋₁) / (rₙ₊₁ − rₙ). Predicted 4.6692.
- **α (alpha).** The limit of the ratio of orbit widths at successive doublings.
  Predicted 2.5029.
- **Return map.** The plot of observable at step n+1 against observable at step n. If
  it has one smooth hump, Feigenbaum applies. If it does not, the theory says nothing.

## The ladder: conclusive first

Each rung is a separate experiment with its own reproducible script and figure. Nothing
on a higher rung is attempted before the rung below it holds.

**Rung 0. Determinism.** Same input, same output, every time. Greedy decoding, batch
size one, deterministic kernels, pinned seeds, a local model. GPU inference is
nondeterministic across batch sizes by default and that alone would make cycle detection
impossible. This rung is a test that runs the same generation a hundred times and
asserts the hashes match. Nothing else is measurable until it passes.

**Rung 1. The hump.** Pick a loop, an observable, and a knob. Run the loop from many
starting points and plot the return map. We are looking for a single smooth maximum.
This is the go/no-go for everything above, and the first rung where a lay reader can
look at a picture and see the claim.

**Rung 2. The first flip.** Find the fixed point. Sweep the knob and find the value
where it gives way to a period-2 orbit. In the continuous version of the loop, where
the state is an embedding and the map is differentiable, compute the map's Jacobian at
the fixed point with autodiff and watch its leading eigenvalue move toward −1 as the
knob turns. An eigenvalue crossing −1 is period doubling by definition. A complex pair
crossing the unit circle is something else, and we would need to know that.

**Rung 3. The cascade.** Locate r₁ through r₄ or r₅ by bisection on the knob. Each
bisection step runs the loop past a burn-in and reads the period by hashing states.
Compute the ratios. With four doublings the logistic map's ratios sit within a percent
of δ, so four is the target and five is decisive.

**Rung 4. α and noise.** Measure α from the geometry of the cycles in the observable.
Then add temperature as noise and measure how the cascade truncates. Three numbers.

**Rung 5. Universality.** Repeat rungs 1 through 4 with a different model, a different
loop, and a different knob. The prediction is that δ does not move. This rung is the
proof, in the only sense the word has for an empirical system.

**Rung 6. Comprehensive.** Turn every knob in the catalogue below. Document which ones
show a cascade, which show a different route to chaos, and which show nothing. The
result is a map of the territory, not just one path through it.

## The knob catalogue

Knobs are classified by two properties. Is it continuous, so that bifurcation points can
be located to arbitrary precision? Is it deterministic, so that the loop is a map and
not a stochastic process? The cascade needs both. Knobs that lack one are still worth
turning, because how they fail is informative.

**Below the prompt.** Continuous and deterministic. These are where δ gets measured.

- Steering-vector coefficient. Add λ times a fixed direction to the residual stream.
  The cleanest knob we have.
- Logit bias on a token or set of tokens.
- Interpolation between two prompt embeddings. The prompt becomes continuous.
- Feedback gain. Feed back a mix of the previous state and the new output, weighted by
  a scalar. This is the knob most directly analogous to the logistic map's r, since it
  sets how hard the loop pushes on itself.
- Layer or attention scaling factors.

**In the prompt.** Discrete, so the map is piecewise constant in the knob and
bifurcation points blur to token resolution. Deterministic at temperature zero.

- A number written into the template. "Make it 37% more formal."
- Number of few-shot examples.
- Prompt length, or the number of prior turns carried in context.
- Wording strength along a hand-built ladder of synonyms.

**Around the loop.** The structure of the loop itself.

- Number of models in a debate.
- Number of critique-revise rounds per step.
- Fraction of the output fed back versus discarded.

**Stochastic.** Continuous but noisy. These test the noise-scaling prediction, not δ.

- Temperature.
- Top-p and top-k.
- Repetition penalty.

## Loops and observables

Loops to try, roughly in order of how likely they are to show a clean hump:

- Rewrite in place. "Rewrite the following text."
- Summarize. Length is a natural observable and contraction is nearly guaranteed.
- Translation round trip through a second language.
- Critique and revise, which is the skeleton of every self-refinement agent.
- Edit, run tests, edit again. The coding-agent loop, with tests as the environment.
- Two models alternating. The debate loop.

Observables: text length, exact-text hash for period detection, log-probability of the
output under the model, projection of the embedding onto a fixed direction, and any
scalar a judge model assigns. The hash is the ground truth for period. The others are
for plotting return maps and measuring α.

## Where it will probably break

We write these down now so that finding one is a result, not a surprise.

- **No hump.** The return map may be monotone or many-humped. Then the loop is not in
  the class and we say so.
- **A different bifurcation first.** In high-dimensional dissipative systems a complex
  eigenvalue pair crossing the unit circle is at least as generic as a real one crossing
  −1. That gives quasi-periodic motion, not period doubling. Rung 2 exists to catch this.
- **Finite state.** Text space is finite, so a true infinite cascade cannot exist. We
  will see a handful of doublings at best. That is enough for δ, as it was for Libchaber.
- **Float nondeterminism.** Rung 0 exists for this.
- **Tokenization steps.** In-prompt knobs are staircases. We do not measure δ with them.
- **A different constant.** δ ≈ 7.28 means the hump is quartic, a different universality
  class. That would be a more interesting finding than 4.669.

## Goal 2: what becomes useful

The number δ has no engineering use. The theory it confirms does. If Goal 1 holds, the
following become honest, measurable quantities for any LLM loop, including production
agents:

- **Stability margin.** The leading eigenvalue of the map at the loop's fixed point. How
  close is this agent configuration to flip-flopping? This is intrinsic to the fixed
  point and does not care which knob moved you there, so it is honest for prompts too.
- **Early warning.** If the route to chaos is period doubling, oscillation precedes
  chaos, and it is detectable. A loop that has started alternating is a loop that is
  about to become unreproducible.
- **Damping.** The cure for an oscillating loop is to reduce feedback gain. In agent
  terms: carry more prior context forward, anchor harder to the original instruction,
  feed back less of the last output. The theory says which direction to turn and by how
  much.
- **Prompt sensitivity as a Lyapunov exponent.** How fast do two nearby starting prompts
  diverge under the loop? One number, comparable across models and templates, replacing
  "this model is finicky."
- **Temperature as a noise budget.** The noise-scaling result says how much temperature
  a loop can tolerate before its structure is washed out.
- **Model collapse.** Training on synthetic data is the loop with the slowest clock.
  Fixed points and their stability are the same objects.

Each of these is a separate small deliverable once the harness exists. None is promised
until Goal 1's rungs hold.

## Reproducibility

Everything runs locally with `uv`. One command reproduces each figure from scratch. The
model, seeds, and kernel settings are pinned in the repo. A stranger with a GPU should be
able to clone, run, and see 4.669 or see why not, without talking to us.

## Held open

Two framings we are not collapsing yet, because each may turn out to be the right one:

- Is the state the text, or the hidden activations? The token loop is what people
  actually run. The continuous relaxation is what we can differentiate. They may not be
  the same dynamical system, and part of the work is finding out whether the one
  predicts the other.
- Is "prompting" a knob at all, or only a preset? A prompt cannot be turned smoothly.
  The honest reading is that below-the-prompt knobs map the landscape and prompts drop
  you at a point on it. Whether that transfer is as clean as the theory suggests is
  itself an experiment.

## Not yet decided

The first model, the first loop, and the hardware. These are decided at Rung 0, not
here.
