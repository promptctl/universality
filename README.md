# universality

Inside one small AI language model, this project found a feedback loop whose cycles double the way
the logistic map's do. Measured on the model directly, the ratio between successive gaps is 4.625,
within 1% of Feigenbaum's 4.669. Followed deeper, on a smooth curve fitted to the model's behavior,
it converges to 4.66920, matching his constant to five decimal places. Two other numbers the theory
predicts showed up on that curve as well. What hasn't been shown yet is whether any of this holds
for other models, or for the everyday loop where a chatbot keeps reworking its own answer.

This page explains that for someone who knows chaos theory from James Gleick's *Chaos* and nothing
about AI. The plan behind the work is in [PROJECT.md](PROJECT.md), and every command and measurement
is in [EXPERIMENTS.md](EXPERIMENTS.md).

## The setup, in Gleick's terms

Recall the logistic map: take a number x, compute r·x·(1−x), feed the answer back in, and repeat.
With a small r it settles on one value. Turn r up and it alternates between two values, then four,
then eight, and then goes chaotic. Feigenbaum noticed that the gaps between doublings shrink by a
ratio that closes in on 4.669. The same 4.669 turns up in any system whose feedback curve has a
single smooth hump with a rounded top, like dripping faucets or Libchaber's convecting fluid. Beyond
that, the exact shape of the hump doesn't matter.

The question is whether a language model fed its own output belongs to that family. The model is
Qwen2.5-0.5B, a small open model running on a Mac. The first step was making it fully repeatable:
the same input has to give byte-identical output every time, or nothing else can be measured.

## The obvious loop didn't work

The first attempt was the natural one. Ask the model to rewrite a sentence, then rewrite the
rewrite, and so on. Of 24 starting texts, 18 settle within four rounds on a rewrite the model then
repeats word for word, and no two settle on the same one. Four more alternate between two rewrites,
and two are still changing after six rounds. A summarize loop is stiller: all 23 texts it could run
settle within three rounds. Adding a "formality" dial that pushes the rewrites toward or away from
formal makes some texts restless, but each wanders on its own rather than toward a shared pattern.
Period doubling needs one resting point that a dial slowly destabilizes, and these loops have a
separate resting point for nearly every starting text. There's no hump and no cascade. That's a real
negative result, and the project records it as one.

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

As the gain goes up, the loop's one resting point stays stable for a long time; it gives way only at
gain 13.59. Well before that, from gain 2.875, a loop started at the top of the hump falls into a
second pattern instead, alternating between two values. That alternation then splits the way the
logistic map's does, into four, eight and sixteen, with the splits crowding closer each time.
Measured directly on the model, the splitting goes up to period 64, five doublings past period 2,
where Libchaber saw four. The ratio between gaps came out at 4.625, within 1% of 4.669.

Past period 64, the model's own arithmetic gets in the way. It computes with limited precision, and
that tiny roughness drowns out cycles that small. So the project measured the push-back curve at
2,901 points and fitted a smooth mathematical curve through them. The best fit matches the model to
within the model's own rounding noise, and it places every doubling the model could be measured on
where the model puts it. Following the cascade on that curve out to period 8,192, the ratio closes
in on 4.66920. Feigenbaum's constant is 4.6692016.

Four fits of different looseness were tried, and between them they show the pattern universality
predicts. Their early ratios disagree by 2%, and over the same early doublings the two closest fits
sit low at 4.625 just as the model did, so that shortfall belongs to this hump's particular shape.
By period 4,096 all four read 4.66920. The early doublings reflect the shape, and the deep ones lose
track of it.

## Two more of Feigenbaum's numbers

The theory predicts more than 4.669, and the project checked two more of its numbers. Both were read
to their limits on the fitted curves, where the doublings go deep enough to settle.

The first, about 2.503, describes the shape of the fork diagram rather than its timing. Each
doubling brings the cycle's nearest point closer to the top of the hump, and the ratio of successive
distances settles toward 2.503 as the doublings go deeper. The fitted curves give that number to six
decimal places.

The second is about noise. Real systems are never perfectly clean, and noise blurs the smallest
cycles, so you only see so many doublings before the diagram turns to fuzz. The theory says that
each extra doubling you want to see needs the noise cut by a factor that closes in on 6.619. The
fitted curves give 6.619 to within about a millionth.

Language models come with a built-in noise dial: temperature, which sets how randomly the model
picks its next word. The project first measured exactly how much randomness each temperature adds.
It then carried that randomness along the cycles, the way the 6.619 rule describes, to predict the
last doubling a single run could still show at each setting. Finally, it checked on the real model.
In each round of the loop the model picks one next word at random at that temperature, and the
push-back is read with that word included. The loop was run hundreds of times at each of five
temperatures, from 1 down to 0.2. Through period 8 the measured blurring agreed with the prediction
to within the runs' statistical scatter, though the real model tended to come in low, by up to 10%.
Period 16 survived at a warmer temperature than predicted. The prediction had it showing only once
the model cooled to about 0.3; the real model already showed it at 0.5, just under the edge, and
lost it at 0.7 and above. Near that edge the prediction overstates the blurring by up to a quarter,
which accounts for the difference.

## What isn't shown yet

- **Only one case so far.** It's one model, one sentence, one direction and one loop. Universality
  claims none of those matter, but testing that with other models, loops and knobs is the project's
  next stage ([Rung 5](PROJECT.md#the-ladder-conclusive-first)), and it hasn't been done.
- **Not the loop people run.** The loop that cascades is a one-number loop built inside the model.
  The loop people actually use, text in and text out, showed no cascade at all.
- **The deepest numbers come from a fit.** Everything past period 64, and the limits 2.503 and
  6.619, are measured on curves fitted to the model, not on the model itself. The best fit places
  every doubling the model shows where the model puts it, which is strong reason to trust it. But by
  period 64 the model's measurements sit a little off the fits', by two to three and a half times
  the measurement error, and whether that is the model's rounding noise or shape the fits miss is
  not yet settled.

## Why it would matter

If this holds up more broadly, it gives practical tools for AI agents that loop on their own output.
How close a loop is to flip-flopping would be a number you can measure: the steepness of its
feedback at its resting point, which this project watched tip past the critical value of −1 just as
the resting point gave way. Temperature would come with a calculable limit on how much randomness a
loop can take before its structure washes out. This loop also shows that the warning signs aren't
simple. Its two-value pattern appeared while its resting point was still stable, so an agent seen
flip-flopping isn't necessarily one whose loop is about to break. None of these tools is built yet.
What exists now is a first piece of evidence that the math Gleick wrote about reaches inside a
neural network.

## Going further

[PROJECT.md](PROJECT.md) is the plan: the claim, the ladder of experiments that tests it, and where
it is expected to break. [EXPERIMENTS.md](EXPERIMENTS.md) holds the command that reproduces each
result above and the full measurements behind it:

- [Making the model repeatable](EXPERIMENTS.md#determinism)
- [The rewrite loop settling on fixed points](EXPERIMENTS.md#the-loops-from-many-starts-is-there-a-hump)
- [The hump in the push-back](EXPERIMENTS.md#a-push-and-the-models-answer-to-it-the-hump)
- [The resting point, the second pattern, and the first splits](EXPERIMENTS.md#the-loop-closed-a-gain-a-fixed-point-and-the-first-flip)
- [The ratios measured on the model to period 64](EXPERIMENTS.md#the-cascade-delta-from-six-superstable-gains)
- [The smooth fits to period 8,192](EXPERIMENTS.md#past-period-64-the-cascade-on-a-smooth-fit-of-the-models-answer)
- [The shrinking cycles, 2.503](EXPERIMENTS.md#alpha-the-cycles-shrink-by-the-same-factor)
- [The noise rule, 6.619](EXPERIMENTS.md#kappa-how-much-noise-each-doubling-needs-removed)
- [Temperature and the last doubling](EXPERIMENTS.md#temperature-how-many-doublings-a-drawn-token-lets-through)

To run any of it yourself, start with [Setup](EXPERIMENTS.md#setup).