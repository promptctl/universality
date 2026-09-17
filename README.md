# universality

Inside one small AI language model, this project found a feedback loop that goes through the same
period-doubling route to chaos as the logistic map. It also measured Feigenbaum's constant there.
The gaps between doublings shrink by 4.66920 each time, which matches Feigenbaum's number to five
decimal places. Two other numbers the theory predicts showed up as well. What hasn't been shown yet
is whether this holds for other models, or for the everyday loop where a chatbot keeps reworking its
own answer.

This page explains that for someone who knows chaos theory from James Gleick's *Chaos* and nothing
about AI. The plan behind the work is in [PROJECT.md](PROJECT.md), and every command and measurement
is in [EXPERIMENTS.md](EXPERIMENTS.md).

## The setup, in Gleick's terms

Recall the logistic map: take a number x, compute r·x·(1−x), feed the answer back in, and repeat.
With a small r it settles on one value. Turn r up and it alternates between two values, then four,
then eight, and then goes chaotic. Feigenbaum noticed that each gap between doublings is 4.669 times
shorter than the one before. The same 4.669 turns up in any system whose feedback curve has a single
smooth hump, like dripping faucets or Libchaber's convecting fluid. The exact shape of the hump
doesn't matter, only that there is one.

The question is whether a language model fed its own output belongs to that family. The model is
Qwen2.5-0.5B, a small open model running on a Mac. The first step was making it fully repeatable:
the same input has to give byte-identical output every time, or nothing else can be measured.

## The obvious loop didn't work

The first attempt was the natural one. Ask the model to rewrite a sentence, then rewrite the
rewrite, and so on, with a "formality" dial to push it around. That loop has nothing to double.
Almost every starting text settles on a fixed rewrite within two or three rounds, and 24 starting
texts end up at 24 different resting places. To this model, a rewrite of its own rewrite comes out
the same. There's no hump and no cascade. That's a real negative result, and the project records it
as one.

## A loop with one number in it

So the loop was cut down to a single number, the way the logistic map carries a single x. A language
model passes text up through a stack of layers, and partway up it holds an internal representation
of that text. Researchers have found directions inside that representation that match qualities like
formality. The experiment pushes the model's representation of a fixed sentence some amount x along
the formality direction, then reads how hard the later layers push back.

Plotted against x, that push-back has a single smooth hump. Its top is rounded like a parabola,
which is exactly the shape Feigenbaum's 4.669 requires. A flatter top would give a different
constant, about 7.28. The loop is then closed: the push-back, multiplied by a gain, becomes the next
push. The gain plays the role of r.

As the gain goes up, the loop starts alternating between two values, then splits into four, eight
and sixteen, with the splits crowding closer each time. Measured directly on the model, the
splitting goes up to period 64, which is six doublings (Libchaber got four). The ratio between gaps
came out at 4.625, within 1% of 4.669.

Past period 64, the model's own arithmetic gets in the way. It computes with limited precision, and
that tiny roughness drowns out cycles that small. So the project measured the push-back curve at
2,901 points and fitted a smooth mathematical curve through them. The fit matches the model to
within the model's own rounding noise, and it reproduces every doubling the model could be measured
on directly. Following the cascade on that curve out to period 8,192, the ratio reads 4.66920.
Feigenbaum's constant is 4.6692016.

Four fits of different looseness were tried, and this is what makes it universality rather than
coincidence. Their early ratios disagree by 2%, yet all four arrive at the same 4.66920. The early
doublings reflect the particular shape of this model's hump, and the later ones lose track of it.
That is exactly what the theory says should happen.

## Two more of Feigenbaum's numbers

The theory predicts more than 4.669, and the project checked two more of its numbers.

The first, about 2.503, describes the shape of the fork diagram rather than its timing. At each
doubling, the new cycle's points sit 2.503 times closer to the top of the hump. The fitted curves
give that number to six decimal places.

The second is about noise. Real systems are never perfectly clean, and noise blurs the smallest
cycles, so you only see so many doublings before the diagram turns to fuzz. The theory says that to
see one more doubling, you need noise 6.619 times smaller. The fitted curves give 6.619 to within
about a millionth.

Language models come with a built-in noise dial: temperature, which sets how randomly the model
picks its next word. The project first measured exactly how much randomness each temperature adds.
It then used the 6.619 rule to predict the last doubling you could still see at each setting.
Finally, it ran the real model hundreds of times with random word choices to check. Through period
8, the measured blurring matched the prediction within a few percent. The cutoff also landed where it
was predicted. At high temperature the cascade blurs out after period 8. Cooling the model lets
period 16 through, and that change happens between temperatures 0.5 and 0.3, as predicted.

## What isn't shown yet

- **Only one case so far.** It's one model, one sentence, one direction and one loop. Universality
  claims none of those matter, but testing that with other models, loops and knobs is the project's
  next stage ([Rung 5](PROJECT.md#the-ladder-conclusive-first)), and it hasn't been done.
- **Not the loop people run.** The loop that cascades is a one-number loop built inside the model.
  The loop people actually use, text in and text out, showed no cascade at all.
- **The deepest numbers come from a fit.** Everything past period 64 is measured on a curve fitted
  to the model, not on the model itself. That curve reproduces every doubling that could be measured
  directly, which is strong reason to trust it, but it is one step removed.

## Why it would matter

If this holds up more broadly, it gives practical tools for AI agents that loop on their own output.
An agent that starts flip-flopping between two answers would be a measurable early warning that its
loop is near instability. Temperature would also come with a calculable limit on how much randomness
a loop can take before its structure washes out. None of that is built yet. What exists now is a
first piece of evidence that the math Gleick wrote about reaches inside a neural network.

## Going further

[PROJECT.md](PROJECT.md) is the plan: the claim, the ladder of experiments that tests it, and where
it is expected to break. [EXPERIMENTS.md](EXPERIMENTS.md) holds the command that reproduces each
result above and the full measurements behind it:

- [Making the model repeatable](EXPERIMENTS.md#determinism)
- [The rewrite loop settling on fixed points](EXPERIMENTS.md#the-loops-from-many-starts-is-there-a-hump)
- [The hump in the push-back](EXPERIMENTS.md#a-push-and-the-models-answer-to-it-the-hump)
- [The first split as the gain turns](EXPERIMENTS.md#the-loop-closed-a-gain-a-fixed-point-and-the-first-flip)
- [The ratios measured on the model to period 64](EXPERIMENTS.md#the-cascade-delta-from-six-superstable-gains)
- [The smooth fits to period 8,192](EXPERIMENTS.md#past-period-64-the-cascade-on-a-smooth-fit-of-the-models-answer)
- [The shrinking cycles, 2.503](EXPERIMENTS.md#alpha-the-cycles-shrink-by-the-same-factor)
- [The noise rule, 6.619](EXPERIMENTS.md#kappa-how-much-noise-each-doubling-needs-removed)
- [Temperature and the last doubling](EXPERIMENTS.md#temperature-how-many-doublings-a-drawn-token-lets-through)

To run any of it yourself, start with [Setup](EXPERIMENTS.md#setup).
