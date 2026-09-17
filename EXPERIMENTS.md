# Experiments

Does Feigenbaum universality apply to LLM feedback loops? The founding document is
[PROJECT.md](PROJECT.md), and [README.md](README.md) says in plain language what has been found.
This file is how to run the code, and every measurement it has made.

## Setup

Everything runs with [uv](https://docs.astral.sh/uv/).

    uv run uni gen "hello"

That one command installs everything and generates from the pinned model. The first run
downloads the checkpoint, about a gigabyte, into the Hugging Face cache. It prints the
text, its sha256, and each generated token with its log-probability.

The models, their revisions and dtype, and the generation limit are pinned in
[uni/pinned.toml](uni/pinned.toml), and nothing else in the code names them. Each model is a
table under a name, and one of them is the default. Every command that runs a model takes
`--model NAME` and runs the default without it; `uni observe` and `uni plot` read an orbit with the
pinned model whose checkpoint it recorded, and refuse one no pinned model is. Everything in this file
before Rung 5's second model runs on the default, Qwen2.5-0.5B-Instruct. Generation
runs on Metal, greedy at batch size one; the checkpoint's own sampling settings are
ignored. CPU is too slow for this work, so it is not an option.

## What a run's exit code means

Each code is its own answer rather than a generic failure, because what a reader does
about them differs.

| code | what happened |
|---|---|
| `0` | it did what was asked |
| `1` | the determinism gate ran and some case produced more than one hash |
| `2` | the command line did not parse: a flag this program does not have, or one it needs left out or given no value |
| `70` | a bug in this program: the traceback printed with it says where, and is worth reporting |
| `74` | the machine would not do the work: a full disk, a read-only volume, a path in the way, a checkpoint it cannot fetch |
| `78` | the command parsed, and the run it describes cannot be run: a file, a value, or a combination this program refuses |
| `79` | a sweep ran and some cell of it has no orbit, so it came up short of its grid |
| `141` | something downstream stopped reading, as `uni ... \| head` does — the run was told to stop, not refused |

`78` and `74` are the pair worth telling apart: the first is something you asked for, the
second is something about where you asked it. A loop running unattended can fix a `78` and
try again, while a `74` is not the command's to fix — an outage may clear, a disk will not — and
is where such a loop should stop.

With `--remote` the code is the one `uni` exited with on the host, so the table holds there
too — unless the trip itself fails, and then the code is ssh's or rsync's: `255` from ssh, and
rsync's own codes, one of which is `1`.

## Determinism

Every measurement in this project compares hashes of generated text, so the same input
must produce the same output every time. The gate checks that:

    uv run uni determinism --remote

It generates each case 100 times and prints every run's sha256, then a verdict per
case. It exits 1 if any case produced more than one hash. The cases are an ordinary
prompt, the empty prompt, and a prompt that fills the model's context limit with 16
tokens left over. `--runs` changes the count. A full run takes about 15 minutes, and
it has passed on the run host. Run it again when the model, torch, transformers, or
macOS on the run host changes; nothing else can change the result.

`uv run pytest` checks the same cases with 3 runs each, so the suite stays fast, and
also checks that a fresh process produces the same hash, since the command's runs all
share one process.

Hashes are only comparable within one machine. The run host and a dev Mac agree on
tokens and hashes but differ in the sixth decimal of the log-probabilities.

`tests/test_determinism.py` also holds a negative test, which shows the failure the
gate protects against. The same prompt, batched with a longer request and left-padded
with positions counted from the mask, gives log-probabilities that differ from batch
size one by up to about 1e-4. The greedy text survives that on a short prompt, but in
a long feedback loop a near-tie flips a token, and one flipped token turns an orbit
into noise. That is why the wrapper generates at batch size one and has no batch
setting.

The batch the expensive commands would use is a different one: one prompt repeated, every row
the same length with nothing padded, the rows differing only in the push added to the residual
stream. `uni batch` reads each push alone, as every command here reads it, then again at each
batch size. It prints the largest difference in the answer and in any token's log-probability,
from the reading alone and from the largest batch, and how many pushes each size reads a second.
On the run host:

    uv run uni --remote batch --template rewrite --knob formality \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-10:10:64 --layer 23

    64 pushes from -10 to 10 along formality, the answer read at layer 23, on Qwen/Qwen2.5-0.5B-Instruct at float32
     rows   answer from alone  logprob from alone    answer from 64   logprob from 64   pushes/s
        1           identical           identical          5.05e-05          8.93e-05       54.2
        2            5.05e-05             8.9e-05         identical          3.91e-05       81.7
        4            5.05e-05             8.9e-05         identical          3.91e-05      106.9
        8            5.05e-05             8.9e-05         identical          3.91e-05      122.5
       16            5.05e-05            8.93e-05         identical         identical      149.9
       32            5.05e-05            8.93e-05         identical         identical      161.8
       64            5.05e-05            8.93e-05         identical         identical      166.8
    read in reverse order, 64 rows to a pass: answer identical, logprob identical

    uv run uni --remote batch --model smollm2-360m --template rewrite --knob formality-16 \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-3:3:64 --layer 28

    64 pushes from -3 to 3 along formality-16, the answer read at layer 28, on HuggingFaceTB/SmolLM2-360M-Instruct at float32
     rows   answer from alone  logprob from alone    answer from 64   logprob from 64   pushes/s
        1           identical           identical         identical          4.37e-05       48.7
        2           identical           identical         identical          4.37e-05       81.7
        4           identical           identical         identical          4.37e-05      108.8
        8           identical           identical         identical          4.37e-05      129.4
       16           identical            4.37e-05         identical         identical      153.5
       32           identical            4.37e-05         identical         identical      167.2
       64           identical            4.37e-05         identical         identical      171.2
    read in reverse order, 64 rows to a pass: answer identical, logprob identical

One row through the batched pass is the reading alone to the bit, so what the larger batches
differ by is the pass itself, not the arithmetic around it. On both models no row depends on the
rows beside it: the largest batch read in reverse order reads the same bits.

The answer is where the two models part. SmolLM2-360M's is identical at every batch size up to 64.
Qwen2.5-0.5B's is up to 5.05e-05 away from the reading alone at every size from 2 to 64, and every
one of those sizes reads the same bits as 64 rows do. That is above the 1.35e-05 jitter of the
reference curve at layer 23 (see "Past period 64" below). So a Qwen curve read in batches is not
the committed curve with its last bits moved and still inside the noise its fits are measured
against. It is a curve of its own, the same at any batch size from 2, and anything read off the
committed curves would have to be read again from it.

The next token's log-probabilities move on both models: by up to 8.93e-05 on Qwen from 2 rows,
and by 4.37e-05 on SmolLM2 from 16. From 16 rows up they read the same bits as 64 rows. Below
that they read otherwise, so a drawn token's probabilities depend on the batch size they were
read at.

A dev Mac reads the same pattern: the answer 5.05e-05 away on Qwen and identical on SmolLM2, the
log-probabilities up to 9.02e-05 away on Qwen and 4.3e-05 on SmolLM2, at 116.7 and 137.8 pushes a
second at 64 rows. `tests/test_batch.py` checks both models at 16 pushes and batch sizes 1, 2, 3
and 16. One row reads the reading alone to the bit. The reversed batch reads the same bits. No
batch moves the answer by more than 1e-4 or a log-probability by more than 2e-4.

## The loop

A template turns a state into a prompt, and the model's reply is the next state. Feeding
the reply back in, step after step, makes the model an iterated map:

    uv run uni loop --template rewrite --steps 20 --start "The lighthouse keeper climbed the stairs."

It prints the start as step 0 and every state after it, one per line, then writes the
trajectory to `trajectories/<name>.json` and prints the path. The file holds the start,
every state, the knob value, the template's text, and the pinned config, so the run can
be reproduced from the file alone. The name is a hash of those inputs and the step
count. Rerunning the same command rewrites the same file with the same bytes. With
`--remote` the file is written on the run host. The sync neither sends nor deletes
`trajectories/`, so each machine keeps its own. The directory is also gitignored.

The templates live in [uni/templates.toml](uni/templates.toml), each holding `{state}`
exactly once. `identity` asks for the state back unchanged, `empty` sends the state as
the whole prompt, `rewrite` asks for a rewrite, and `summarize` for a summary. The start may be
empty.

`--map` chooses which map is iterated, and defaults to the model. Whichever it is, the runner
only ever calls `step(state)` on it, so nothing below the command line knows there is more than
one. `--value` is that map's parameter: the knob's setting for the model, r for the logistic map
below, and the gain for the response map further down. Each map reads the flags that describe it
and refuses the rest, so a `--knob` carried over from a model run does not ride along unapplied on
an orbit of numbers.

## The logistic map

    uv run uni loop --map logistic --start 0.5 --steps 400 --value 3.5

`x -> r x (1 - x)`, the textbook map, iterated by the same runner, written to the same kind of
trajectory file, and read by the same `uni observe`. It is here because it is the one map whose
answer is known before the code runs: Feigenbaum's period doubling was published for it in 1978,
so an orbit of it is the fixture where a wrong answer is visible as a wrong answer rather than as
a result. Every brick in this repo has it as a second consumer, which is what keeps the bricks
from being shaped around the model.

    r = 2.8   period 1     r = 3.5   period 4     r = 3.9   no period in 1000 steps
    r = 3.2   period 2     r = 3.55  period 8

Those are what `uni observe` reports, exactly, on orbits from `--start 0.5`, given enough steps
to pass the transient and then close the cycle once: 155 at r = 2.8, 35 at 3.2, 33 at 3.5, and
289 at 3.55. The transient lengthens as the cascade goes on, so 2.8 is the slow one only until
3.55 overtakes it, and `--steps 400` covers every row that has a period. The r = 3.9 row is a
claim about a thousand steps and takes a thousand. Too few steps is not a wrong
answer — `uni observe` says it found no period in the steps it was given, and means it.

The orbit is periodic in the strict sense and not merely close to it: float64 lands on the cycle
and stays there, so the detector's exactness reports it with nothing rounded to help.

The states are text, like every other map's. A logistic state is the shortest text that reads
back as exactly its float, and a second spelling of a number the map holds is refused rather than
read: `--start 0.50` is told to write `0.5`. The detector compares the states themselves while the
`x` column compares their numbers, so a spelling the map would never write is one state to the
plot and two to the verdict — a fixed point reported as a transient it never had.

r is refused outside 0..4 and a state outside 0..1, which is not fussiness either: outside them
the orbit runs to -inf and then to nan, and a run of nans is a state that repeats, which a
detector would report as period 1. r has no default for the same reason — 0 is a legal r whose
orbit collapses to zero, so a forgotten `--value` would be answered `period 1` rather than asked
about.

This map costs no checkpoint. `uni loop --map logistic` imports no torch at all, refusals
included, which a test holds to.

## The steering knob

A knob is a number turned on the loop. The first one steers the model: it adds the
value times a fixed direction to the residual stream leaving one decoder layer, at every
position of every step.

    uv run uni loop --template rewrite --steps 20 --start "..." --knob formality --value 2

`--knob` names a direction in [uni/directions](uni/directions), or `none`; left off, the model
map runs unturned. `--value` defaults to 0 for this map, and at 0 the orbit is exactly the
unsteered one. Positive values
push the rewrites toward the direction's quality and negative values push away from it.
At layer 12, `formality` makes the rewrites clearly more formal by 2 and casual by -2. By
4 the rewrites drift away from the text they started from, and they keep drifting: a value
in the millions still generates, it just generates nothing worth reading. Only a value
large enough to stop the logits being numbers at all is refused, at the step it happens,
rather than written down as an orbit of real text. A value with no knob to turn is refused
before the checkpoint is read, rather than recorded as if it had steered.

A direction comes from a contrast, `uni/directions/<name>.toml`. A contrast is a
template, a layer, and pairs of replies to the same text, one toward the quality and
one away from it. Each reply is read as the model's own answer to the template, and the
direction is the mean over pairs of the difference in the layer's residual stream,
averaged over the reply's tokens. Derive it once:

    uv run uni direction formality

This writes `uni/directions/Qwen--Qwen2.5-0.5B-Instruct/formality.json`, which holds the vector,
a copy of the contrast that produced it, and the checkpoint it was read from, and is committed. A
contrast is words and holds for every model; a direction is a vector in one model's residual
stream, so each model's directions are kept in a directory named by its model id, and `uni direction
formality --model NAME` derives that model's. Every
trajectory steered by it records the direction's name, layer, and sha256, which is the
hash of the file itself: a direction file that has been edited by hand, or derived on a
different checkpoint than the one pinned, is refused. Re-derive when the contrast or the
checkpoint changes.

## Sweeps

One map run at every value on a grid, from every start, kept together:

    uv run uni sweep --remote --map logistic --grid 2.8:4.0:200 --start 0.5 --steps 400

`--grid FROM:TO:COUNT` names the values, `TO` included, and the ends are exactly the numbers
asked for rather than what the arithmetic lands near — a sweep to r = 4 is a sweep to the edge of
the logistic map's range, and one ulp past it is a cell the map refuses. `--start` is repeated
once per start. Everything else means what it means for `uni loop`.

`--starts NAME` runs a whole set of starts from [uni/starts.toml](uni/starts.toml): a passage cut
after each of a list of word counts, so the starts spread over how long a state is rather than
sitting wherever one typed sentence happens to. It can be repeated, and mixed with `--start`; the
typed starts come first, then each set in the order named. The sweep is named for the starts
themselves, so a set and the same starts typed out are one sweep, and editing a set is a new one.

A sweep runs each cell exactly once, so a grid or a set of starts that names one twice is refused
rather than run twice into one file, and so is a grid the map cannot take — every value is offered
to the map and to the knob before the first cell runs, because a sweep that dies two hundred cells
in dies again on every resume.

Two failures those checks cannot cover, both because a model's states are its own replies. A
state can grow until the rendered state leaves no room to generate, and a reply can fail to end
within the pinned `generation.max_new_tokens`, or within what room the context has left — which a
map refuses, because a reply a limit cut off is not the model's reply, and names the limit. Whether step 300 does either is knowable only by running to step
300. So a cell the map refuses **partway through its orbit** does not stop the sweep. The run says
so on that cell's line, carries on to the cells after it — which are
separate runs of a separate map, with nothing wrong with them — names every refused cell again at
the end, and exits non-zero. Nothing is written for a cell with no orbit: a sweep directory holds
trajectories and nothing else, so the cell simply stays pending and the next run tries it again,
which is what you want the moment whatever refused it is fixed. Were the failure recorded instead,
the cell would be marked done by a run that did not do it, and nothing would ever go back for it.

What makes that a statement about the cell rather than about the sweep is *what* was refused, not
how far the orbit got. A steering direction's layer, and the length of its vector, are fixed by
the direction and not by the setting, so an addition the checkpoint cannot take is one no cell
could have taken: the family is asked for a map before the manifest is written, so that sweep is
refused whole, with no token generated and no directory left behind to fill. A map that comes back
set to a value other than the one it was asked for says the same kind of thing about every cell,
but is found only by running one, so it stops the run where it is found. How early the orbit
stopped would be the wrong test for the same thing, because residual additions large enough to
overflow the model's arithmetic also refuse the first token, and *that* is a property of the
value the cell runs at.

Two things a refused cell does not get, and both are deliberate. It is not remembered: a rerun
runs it again, pays for its orbit again, and is refused again, so an unattended resume loop does
not converge — read the exit code, which is `79` when a sweep ran and came up short, distinct from
the `78` that says the command itself cannot be run. The two do not add up: a run that refused
some cells and was then stopped outright exits with whatever stopped it — `78`, `74`, or `141` —
because what ended it outranks how far it got, and its refusals are on stderr rather than in the
code. And the states it did produce are thrown away rather than written under their own shorter
name, which would put a file in the directory this sweep never named and no rerun would ever look
for. Both are the price of a sweep directory that holds finished orbits and nothing else;
`universality-sweep-81g` carries the question of whether a sweep should be able to say "tried and
cannot" somewhere.

The sweep writes `sweeps/<name>/`: one trajectory per cell, in the same format `uni loop` writes
and `uni observe` reads, beside a `sweep.json` naming the map, the values, the starts, and the
step count. The directory is a hash of exactly those, so rerunning the same command resumes the
same sweep, and changing any of them starts a different one.

It is resumable, and by construction rather than by bookkeeping. A trajectory's file name is a
hash of what fixes the orbit, all of which is known before the cell is run, so a cell is done
when its file is on disk — and `uni loop` already writes each file under a temporary name and
renames it into place, so a run killed part way leaves no half-written cell to mistake for a
finished one. The manifest records what the sweep *is* and never what it has finished: a count of
progress kept beside the files would be a second thing to believe, free to disagree with them.

    uv run uni sweep --remote --map logistic --grid 2.8:4.0:200 --start 0.5 --steps 400 --status

The same command with `--status` says how many cells are done and runs none of them — and writes
nothing, not even the sweep's own directory: a question that left something behind would be an
answer that had changed what it just measured. It costs no checkpoint either, because a map family
holds the pinned configuration rather than a loaded model and reads the checkpoint only when a
cell is actually run. A sweep is identified by what it is, so there is no way to ask about one you
cannot describe.

Sweeps are kept where trajectories are: written on the machine that ran them, gitignored, and
excluded from the `--remote` sync — which runs one way, this tree to the host, with `--delete`. So
a local sweep is never pushed and the host's own is never deleted, and each machine keeps its own.

## Observables and the period

A trajectory is read back from its file rather than re-run, so an observable thought of today
can be asked of an orbit recorded months ago:

    uv run uni observe trajectories/<name>.json

It prints the period first, which is read off the states alone and needs no model, and then one
row per step: a number for the state, then each observable. The state numbers count distinct
states in the order they first appeared, so a period-2 orbit reads `1 2 1 2` straight down the
column. Step 0 is the start, which was given rather than stepped into, so its row carries a
number and no readings.

What an orbit can be read for is decided by what its states are made of, so the columns differ
by map and nothing printing them asks which map it was. The character length of the state is the
one every map answers. An orbit of numbers is read for the numbers, under the column `x` — there
is nothing to derive, and the return map of such a map is the map itself drawn. A model's orbit
is read for the mean log-probability the model gives
the state it wrote (mean, not total, so it is not length under another name), and, for a run that
was steered, how far the state sits along the direction that steered it. The log-probability is
read with the knob at the setting the run recorded: the same model turned elsewhere is another
model, and scores the state differently. The projection is read with the knob off, because it is
read at the layer the knob writes to, and reading through the knob would add the same vector at
every step — a constant that says nothing about the state. An orbit written on a checkpoint other
than the one pinned now is refused, as is a direction that has changed since it steered the run:
every reading is in the units of the weights it was taken under.

The period is exact rather than estimated. The orbit of a deterministic map is exact about this: if a state comes back,
the state after it is the same state as last time, and so is every state after that, forever.
So one repeat fixes both numbers at once, and no window width or count of confirming cycles
enters into it. That the model's map is deterministic is what `uni determinism` establishes.

There are three things the detector can say, and only one of them carries a number:

- `period P, entered at step N` — the orbit came back and kept coming back for the rest of
  the data. Step 0 is the start, as `uni loop` prints it, so `N` also says how long the
  transient was.
- `no period: N steps examined and no state repeated, so any period is at least N` — nothing
  came back. At least, not longer: showing a period of P takes P + 1 states, so a window of N
  that shows no repeat leaves a period of exactly N standing. The detector will not guess which.
- `the state at step N came back P steps later, but step M is not the state P steps before it`
  — the orbit came back and then left the cycle, naming the step that broke it. That cannot
  happen to a map that is a function of its state, so it is reported as its own answer rather
  than filed as "no period", and it means `uni determinism` should be run before anything is
  read into the orbit.

`--burn-in N` passes over the first N steps before looking, for when an early transient is
already known and not wanted. It is rarely needed: the detector reports where the cycle began,
which is the same fact measured rather than assumed.

## The two pictures

    uv run uni plot sweeps/<name> --observable x --burn-in 200

Reads a persisted sweep and writes two figures under `figures/`. Nothing about either one asks
which map ran. The **return map** is the observable's sequence plotted against itself one step
later, which for a map whose states are numbers is the map itself drawn. The **orbit diagram** is
the same readings against the value they were taken at, which is the picture PROJECT.md's Rung 1
is about: period doubling is branches splitting as the value grows.

Both are the same kind of value — points with an x, a y, and a number that colours them — so
there is one drawing function rather than two to keep in step. The return map is coloured by the
value each orbit ran at, so a sweep of one value is one colour and a grid is a fan of them. The
orbit diagram is one colour, because the value is already its x axis.

`--burn-in N` keeps each cell from step N onward, which is where a picture wants a settled orbit
and not the transient that got there. It means exactly what it means to `uni observe`, which
counts the start as step 0 - so a period read off one command and a picture drawn by the other are
about the same states. Nothing has a reading for the start, so `--burn-in 0` and `--burn-in 1`
draw the same picture. A sweep that is still running is drawn from the cells
that are on disk, which is the point of persisting them one at a time; a sweep with nothing on
disk, or a burn-in that ate every step, is refused rather than written as an empty figure that
looks like an answer.

The figure files are named for what fixes the picture - the sweep, the observable, the burn-in -
so a picture says what it is of, and drawing the same sweep another way puts a new file beside the
old one rather than over it.

### The logistic cascade

![orbit diagram of the logistic map](figures/e84d01205c88ec92-x-burn200-orbit.png)

    uv run uni sweep --map logistic --grid 2.5:4.0:600 --start 0.2 --start 0.7 --steps 400
    uv run uni plot sweeps/e84d01205c88ec92 --observable x --burn-in 200

This is the reference picture, and it is the textbook one. The single branch to r = 3; the first
split exactly at 3.0; the second at about 3.449; the third at about 3.544; the branches crowding
into the accumulation near 3.5699 and then the chaotic band; and inside the band the period-3
window at about 3.83, with its own miniature cascade. The start is 0.2 and 0.7 rather than 0.5,
because 0.5 is exactly the pre-image of the map's maximum: at r = 4 it lands on 1.0 and then on
0.0, and the whole right-hand edge of the picture would be a single dot at zero.

![return map of the logistic map](figures/e84d01205c88ec92-x-burn200-return.png)

The return map of the same sweep is the family of parabolas, one per r, fanning up to the r = 4
one that touches 1.0. PROJECT.md asks of Rung 1 whether the return map has one smooth hump,
because that is the property the whole theory rests on. For this map it does, by construction —
which is what makes the logistic map the fixture: a wrong answer here is visible as a wrong
answer rather than as a result.

### The rewrite loop over the steering coefficient

The same two pictures for the map this project is actually about: the model rewriting its own
output, with the formality direction added to the residual stream at layer 12, swept over the
coefficient.

    uv run uni sweep --remote --map model --template rewrite --knob formality \
        --grid=-6:6:25 --start "The meeting moved to Thursday because the room was booked." --steps 30
    uv run uni fetch 37e340c190aad178
    uv run uni plot sweeps/37e340c190aad178 --observable along:formality --burn-in 10

The middle line is not decoration. The sweep ran on the host and its cells stay there - `sweeps/`
is excluded from the sync, and the sync has no leg coming back - so the directory has to be
brought home before anything here can draw it. Plotting is a local command by design: a figure
is an output this repo commits, and drawing one on the host puts it where no commit can reach
it, which is also why `figures/` is excluded from the sync rather than deleted by it. `uni plot
--remote` is refused for the same reason rather than left to draw somewhere unreachable and exit
0 - it is the one command that says where its answer lands. The sweep's name is the same on both
machines, because it is the hash of what the sweep is.

`uni fetch NAME` runs here and reaches the host itself, so it takes no `--remote`. It copies the
sweep's directory from the host into `sweeps/NAME/`, adding the cells not here and leaving the
ones that are, and then says how much of the sweep is here, as `uni sweep --status` does. A cell is
a file written whole and named by what fixed it, so a fetch cut short is finished by running it
again, and one run while the sweep is still going on the host brings home what it has so far.

**Only the middle of that grid is a map.** The sweep exits `79`, with 10 of its 25 cells written:
every coefficient from -2.5 to 2.0. At -3.0 and below, and at 2.5 and above, the model's reply to
its own rewrite does not end within the pinned 256 new tokens, and a reply the budget cut off is
not a state of this map, so those cells are refused rather than recorded. What fills the budget
there is not a long answer a larger budget would finish: at -3.0 it is "Cause the room was
booked!" repeated until the budget stops it, and at +6.0 a slide into or-chains and then into
repeating boilerplate in another language. The edge is sharp, too. The longest reply at 2.0 is 24
tokens, and at 2.5 every one is cut.

An earlier run of this sweep kept those cuts as states, and 446 of its 750 states sat on the
budget; its pictures showed wings that measured the ceiling. A trajectory now records that cut
replies are refused, so that run's orbits are not taken for this map's. The 10 cells both runs
wrote are identical, state for state.

![orbit diagram of the rewrite loop along the formality direction](figures/37e340c190aad178-along-formality-burn10-orbit.png)

The knob works. Where the settled state sits along the formality direction rises from about 1.0
at a coefficient of -2.5 to about 4.6 at 1.5, which is the knob doing what a knob should. At 2.0,
the last coefficient before replies stop ending, it falls back to 3.9.

What the picture does not show is a cascade. The periods are measured rather than eyeballed —
`uni observe` reports each one — and across the 10 cells they are 6 fixed points, a period 2 at
-0.5, 1.0 and 1.5, and a period 3 at 0.5. The cycles longer than one sit around the unsteered
point, not in a doubling sequence, and 0.5 apart on the knob is far too coarse a grid to call any
of it a bifurcation.

![return map of the rewrite loop along the formality direction](figures/37e340c190aad178-along-formality-burn10-return.png)

The return map says the same thing: the fixed points lie on the diagonal, and each cycle is a
handful of points mirrored across it. Rung 1 of PROJECT.md asks whether this map has one smooth
hump, because that is the shape the whole theory rests on. This is not that shape. A hump needs
states that move.

![orbit diagram of the rewrite loop read for length](figures/37e340c190aad178-length-burn10-orbit.png)

Read for the character length of the state instead, the sweep shows what the knob does to this
loop on the way to the edge. The informal side settles on one short sentence of 58 characters,
and on the formal side the settled state grows to 143 characters at 2.0, just before replies
stop ending at all.

What this sweep settles is therefore narrow and worth stating plainly: at this template, this
direction, this start, this step count and this resolution, the rewrite loop does not period
double where it is defined, which is between -2.5 and 2.0. It converges or cycles near zero.
Whether a finer grid inside that range, a longer run, more starts, or another knob would show
anything else is the next question, and it is the question Rung 1 exists to ask.

### The loops from many starts: is there a hump?

A sweep from one start shows where that start's orbit settles and nothing about the map around
it. Rung 1 asks for the map's shape, and a map's shape is only drawn where there are states, so
these sweeps hold the knob still and spread the starts instead: the two start sets, each a passage
cut at 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90 and 130 words, 24 starts from 3 to 830 characters.

    uv run uni sweep --template summarize --knob none --grid 0:0:1 --starts harbor --starts memo --steps 6
    uv run uni plot sweeps/2ec2ab48ac42aa38 --observable length --burn-in 0
    uv run uni sweep --template rewrite --knob none --grid 0:0:1 --starts harbor --starts memo --steps 6
    uv run uni plot sweeps/d118c0eae9b434f2 --observable length --burn-in 0

![return map of the summarize loop from 24 starts](figures/2ec2ab48ac42aa38-length-burn0-return.png)

**Summarize has no hump, and no one fixed point either.** Every one of its 23 orbits is a fixed
point by step 3, exactly - the same text, byte for byte, from then on - and 20 of them by step 2.
They are 23 *different* fixed points: no two starts settle on the same text, and their lengths run
from 114 to 1151 characters. The one cell refused is the memo cut at 12 words, whose summary does
not end within the 256-token budget. So the return map is the diagonal with a few transient
points off it, which is what a map looks like when nearly every state it writes is one it will
write again: a summary of a summary, to this model under greedy decoding, is the same summary.

![return map of the rewrite loop from 24 starts](figures/d118c0eae9b434f2-length-burn0-return.png)

**Rewrite is the same with a little more motion.** Of its 24 orbits, 18 are fixed points within
four steps - again 18 different ones, none shared - four are period 2 by step 4, and two have not
repeated in six steps. Its return map stays near the diagonal. The map draws the model's own
states, from step 1 on, and among those no step moves one of 60 characters or more by more than
37% (94 to 129, on its way to a fixed point); the one point far from the diagonal, 44 to 170,
belongs to a 28-character start that has still not settled at step 6. The first rewrite of a
start is another matter - the 12-word harbor start goes from 64 characters to 615 - which is the
model deciding what the text is, once.

Turning the knob does not change that. The same 24 starts under the formality direction, at five
coefficients inside the range where replies end:

    uv run uni sweep --template rewrite --knob formality --grid=-2:2:5 --starts harbor --starts memo --steps 6
    uv run uni plot sweeps/5417892e26309123 --observable along:formality --burn-in 0

| coefficient | written | refused | fixed points (all distinct) | period 2 | no repeat in 6 steps |
|---:|---:|---:|---:|---:|---:|
| -2.0 | 21 | 3 | 21 | 0 | 0 |
| -1.0 | 22 | 2 | 22 | 0 | 0 |
| 0.0 | 24 | 0 | 18 | 4 | 2 |
| 1.0 | 23 | 1 | 15 | 1 | 7 |
| 2.0 | 10 | 14 | 9 | 0 | 1 |

The sweep exits `79` with 100 of 120 cells written; every refusal is a reply that did not end
within the budget. The row at 0.0 is the unsteered sweep above over again, all 24 orbits text for
text, which is what adding zero times a direction should be. At every coefficient each start
that settles settles on a fixed point of its own, and no two starts share one: 85 fixed points,
84 different texts, because one start - the memo's first three words, "Following the review" -
is left exactly as it is at both 0.0 and 1.0. Steering toward
formal makes the orbits restless, seven of 23 still moving at step 6 at 1.0, but restless along
the diagonal rather than onto a common attractor.

![return map of the steered rewrite loop from 24 starts](figures/5417892e26309123-along-formality-burn0-return.png)

Read along the direction that steers it, the return map is a band on the diagonal from -8 to 7,
tight where the coefficient is negative and loosening above 2 where the formal coefficients spread
it. There is no hump in it anywhere.

This is a result about the structure the theory needs, and it is negative. Period doubling is a
single attracting fixed point losing stability as a knob turns - its slope in the return map
passing through -1. These loops do not have a single attracting fixed point to lose. Each start
lands on a fixed point of its own within a step or two, so the set of fixed points is as large as
the set of starts, the slope along it is +1, and there is no hump for a slope to steepen on. A
content-preserving instruction under greedy decoding is close to idempotent: the model's first
answer is already the answer it would give to itself, steered or not.

What it points to is a loop that forgets where it started, which a loop carrying the whole text
forward cannot: a state small enough that the map is a smooth function of one number, as
PROJECT.md's continuous version of the loop has it. That is what the next section measures.

### A push and the model's answer to it: the hump

    uv run uni response --template rewrite --knob formality \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-40:40:321 --layer 12 --layer 14 --layer 16 --layer 18 --layer 20 --layer 23

Here the state is a number rather than a text. The prompt is rendered once; the residual stream
leaving layer 12 is pushed by that number times the formality direction; and the reading is how
far the stream sits along the same direction at a later layer, averaged over the prompt, with no
token generated. Nothing is sampled, so the reading is a smooth function of the push - which no
number read off greedy text is, since the text holds still until one token overtakes another.

The push itself is taken back out of the reading. The residual connections carry it to every
later layer, so the stream's own projection is the push read back, 8.96 times the push (the
direction's squared length), a straight line whatever the model does. What is left is what the
layers after the push wrote in answer to it, and it is what `uni response` prints and draws.

The push is taken out token by token, before the average, and a push too large to take out is
refused rather than read. Every addition the stream makes while it holds the push rounds at the
push's size, so a large enough push rounds the model's writes out of the stream, and what is left
reads as a model that answers nothing. The command bounds that rounding from the push alone
before it runs a forward pass, refuses any cell it could move by a hundredth or more, and prints
the worst bound over the grid: 0.00051 for the run above, where a push of 1e12 would be 1.3e7.
Like `uni plot`, it runs here and refuses `--remote`: the figure it draws, and the curves it writes
to `curves/` (see the smooth fit below), would stay on the host.

![the model's answer to a push along formality, at six layers](figures/response-6e6cae900235d2b4.png)

At layer 12 there is no answer - the curve is flat at -4.3035, float jitter aside, because
nothing after the push has run, which is also the check that the reading holds the push at all.
From layer 14 on the answer has a **hump**: pushed informal, the layers push back toward formal,
most strongly near -8.5; pushed further, less; pushed formal, they push back hard toward informal.
By layer 23, the last one, the curve changes direction exactly once on the 321-point grid, at a
maximum of 16.01 at -8.5, and falls on both sides all the way to -40 and 40. Layers 18 and 20 add
a small dip between -3 and 0, which layer 23 has smoothed into a shoulder.

Feigenbaum's 4.669 belongs to maps whose maximum is quadratic; a quartic top has its own constant,
about 7.28. On 97 points across -14.5 to -2.5, layer 23's top sits at -8.625, and a quartic fit
there gives a squared coefficient of -0.43, -0.40 and -0.40 on windows of 6, 4 and 2 either side,
against a fourth-power coefficient of 0.002 to 0.004. Within 2 of the top a parabola alone fits
to an rms of 0.05 on a range of 1.87, and how fast the curve falls away from the top goes as the
distance to the power 1.75 on the left and 2.09 on the right. The top is quadratic.

This is Rung 1 met, for a loop of this kind: one smooth maximum, of the order the theory needs. It
is a map in waiting rather than a loop yet - closing it means turning the answer back into the
next push, with a gain to turn, and finding the fixed point and its first flip, which is Rung 2.

### The loop closed: a gain, a fixed point, and the first flip

    uv run uni loop --map response --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --value 3 --start=-8.5000 --steps 40

`--map response` turns the answer back into the next push. The state is the push x. One step reads
the answer r(x) exactly as `uni response` does, and the next push is `gain * r(x) / |v|^2`, where
the gain is `--value`. Dividing by the direction's squared length, 8.96, puts the answer in the
units of the push, since a push of x reads back as x |v|^2 along the direction. So at gain 1 the
next push is exactly the push the answer amounts to, and the gain is PROJECT.md's feedback gain
made literal: how much of the answer is fed back. `uni sweep --map response` sweeps the gain.

The state is the push written to four decimals, `-8.5000` and not `-8.5`, and a start spelled any
other way is refused with the spelling to use. A push held to every bit of a float64 is more than
the model can tell apart: the answer is float32, and so is the push on its way into the stream.
Measured at gains 1 to 12, a settled orbit's float64 states spread over 2e-7 to 7e-6, and the
detector counted that spread as cycles of three and six and as real periods doubled. Four decimals
is fourteen times the widest spread. Even so, near an attractor an orbit can flicker between two
neighbouring spellings, a cycle whose points are 0.0001 to 0.0007 apart, which the exact detector
reports as a period. Such a period is not one this section counts; the orbit diagrams and the
slope below do not see it.

    uv run uni fixed --map response --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --grid 1:20:20 --bracket=-3:0 --step 0.05

`uni fixed` answers Rung 2 off the map itself rather than off an orbit. At each value it bisects
for the state the map carries least far, between two states it carries in opposite directions,
and reads the map's slope there across the states `--step` either side. A fixed point gives way
to a period-2 orbit exactly where that slope passes through -1. On the logistic map it returns
x* = 1 - 1/r and a slope of 2 - r to the last digit, and the crossing at r = 3.

The response map has one fixed point between -3 and 0 at every positive gain, and it moves from
-0.18 at gain 1 to -1.76 at gain 20. Its slope is -0.14 at gain 1, never steeper than -0.25 up to
gain 10, and then it falls: -0.61 at 12, -1.10 at 14, -2.72 at 20. It passes through -1 at gain
**13.59**. On a grid of 0.02 across 13.4 to 13.8 the crossing lands at 13.60, 13.60 and 13.58 for
steps of 0.02, 0.05 and 0.1. A four-decimal state makes the slope uncertain by 0.005, and the slope
changes by 0.25 per unit of gain there, so mu_1 = 13.59 +- 0.02. It never passes through +1: the
fixed point is not born or destroyed on this range, only destabilised.

The orbits split where that slope says they should. Swept across it from two starts, beside the
fixed point (-1.5000) and at the top of the hump (-8.5000):

    uv run uni sweep --map response --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --grid 10:17:141 --start=-1.5000 --start=-8.5000 --steps 400
    uv run uni plot sweeps/919bbadcb5408a3a --observable x --burn-in 200

![orbit diagram of the response map over gain 10 to 17](figures/919bbadcb5408a3a-x-burn200-orbit.png)

From beside the fixed point, the orbit settles on it up to gain 13.35. From 13.65 on it alternates
between two pushes either side of where the fixed point was: -1.63 and -1.41 at gain 13.7, -1.74
and -1.27 at 14, -1.93 and -0.91 at 15. The gap between them grows as the square root of the
distance past the flip, which is how a period-2 orbit born in a flip grows: the gap squared is a
straight line in the gain, 0.62 per unit with an rms of 0.006 from 13.65 to 14.5. Fitted from 13.65
to 13.8 or to 14.0 that line reaches zero at 13.61, which drifts up to 13.63 as the fit is taken
out to 14.5, where the next order starts to show. The slope put the flip at 13.59 +- 0.02. The
cells at 13.4 to 13.6 hold gaps of 0.001 to 0.03, orbits still creeping toward a fixed point
whose slope is nearly -1 after 200 steps, and gaps under 0.001 elsewhere are the flicker described
above.

Between 15.15 and 15.2 that small orbit is gone, and both starts land on a period-5 orbit 150 wide,
from -126 to 27, each of whose five points splits in two between 16.6 and 16.8. From the top of
the hump, below gain 11.45, the orbit mostly reaches a large orbit of its own rather than the fixed
point: period 3, near 12, -5 and -52, doubled to period 6 by 10.6 and repeating nothing within
200 steps by 11.1. At 13.15, 13.35 to 13.45 and 14.75 it reaches other large orbits too, which are
the stray columns in the picture.

That flip is not the first thing this loop does, though. Swept from 1.5 to 6 from two starts, the
top of the hump (-8.5000) and zero:

    uv run uni sweep --map response --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --grid 1.5:6:181 --start=-8.5000 --start=0.0000 --steps 400
    uv run uni plot sweeps/d503da82b73fe909 --observable x --burn-in 200

![orbit diagram of the response map over gain 1.5 to 6](figures/d503da82b73fe909-x-burn200-orbit.png)

The fixed point runs straight through the picture, stable throughout. Beside it, from gain 2.875,
there is a second attractor: a period-2 orbit through the top of the hump, which appears already
12 wide, between 4.8 and -7.0, rather than growing out of anything. At gain 2.85 the orbit from the
top still falls onto the fixed point, and at 2.875 it does not. Its birth is therefore not a flip of
the fixed point, whose slope at that gain is -0.24. The start decides which attractor an orbit
reaches: from zero, every orbit on this range ends at the fixed point.

That second attractor then does exactly what a unimodal map's cascade does. It doubles to period
4 between gains 3.85 and 3.9, where each point has split in two by 0.95, and to period 8 between
4.375 and 4.425, split by 0.36. Just below each doubling the orbit settles slowly, so 200 steps
leave splits of a few thousandths that are not yet the cycle's own. By about 4.55 it no longer
repeats within 200 steps, and between 5.05 and 5.075 it vanishes: from there on, the orbit from
the top falls onto the fixed point too. Measuring those doublings to more than a grid step, and
their ratios against 4.669, is Rung 3, below.

This is Rung 2 met: the fixed point is found, its slope is measured across the gain, and it gives
way at mu_1 = 13.59 +- 0.02, with the orbits splitting where the slope says they must. What this
loop adds to PROJECT.md is a coexisting attractor with its own cascade, reached from the top of the
hump long before the fixed point flips. In one dimension that is the only kind of "different
bifurcation first" there can be.

### The cascade: delta from six superstable gains

An orbit is a poor instrument for a doubling. Just past one it settles slowly, so the cells either
side of 3.875 above hold splits that are not yet the cycle's own, and at four decimals a settled
orbit can still flicker between neighbouring spellings, which an exact detector counts as a period.
So the cascade is read without waiting for anything to settle.

A cycle that passes through the top of the hump, the critical point x_c where the map's slope is
zero, has a multiplier of zero: it is superstable. Each period 2^n of a cascade has one gain where
its cycle is, between that cycle's birth and its own doubling, and the spacings of those gains
shrink by the same ratio as the doublings do. Feigenbaum measured delta that way. At a
superstable gain the orbit from x_c is back at x_c after 2^n steps, so F^p(x_c) - x_c is zero
there, and of opposite signs either side. It is read at a grid of gains around each one, with no
burn-in and no period detected, and a parabola fitted through those readings crosses zero at the
superstable gain. The fit's scatter gives the crossing its error. A parabola and not a line,
because a line cannot follow the curve and misplaces the logistic map's superstable values by
three times the error it reports, where a parabola misplaces them by one.

The starting point x_c is itself a measurement, and an error in it moves every superstable gain by
that error over how steeply F^p(x_c) - x_c crosses zero. The obvious way around that, starting a
step later from the top value F(x_c), does not work: the map is flat at its top, so F(x_c) barely
depends on x_c, but for the same reason F^p(F(x_c)) - F(x_c) only touches zero and never changes
sign. So the top is measured first:

    uv run uni critical --map response --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --decimals 6 --value 20 --grid=-9.1:-8.1:101

`uni critical` fits a cubic to the map over a grid of states, since the hump falls away faster on
one side than the other and would pull a parabola's top toward the gentler side, and finds where
its slope crosses zero. At gains 4.5 and 20, over 61 and 101 states, the top is at -8.61403 +-
0.00002. On the logistic map it finds 0.5.

`--decimals` sets how many decimals the response map writes a push to. The default is four, for the
flicker described above. A superstable gain detects no period, so it can use more: at six, the
readings below scatter three to thirty-five times less than at four, and what is left is the
model's own float32 roughness. The spelling is recorded in every trajectory the map writes.

    uv run uni cascade --map response --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --decimals 6 --critical=-8.614030 --period 2 \
        --grid 3.067:3.075:21 --grid 4.161:4.169:21 --grid 4.4653:4.4673:21 \
        --grid 4.5302:4.5324:21 --grid 4.5442:4.5466:21 --grid 4.5478:4.5491:21

| period | superstable gain | error |
| -----: | ---------------: | ----: |
| 2 | 3.0712744 | 0.0000001 |
| 4 | 4.1657353 | 0.0000021 |
| 8 | 4.4663795 | 0.0000025 |
| 16 | 4.5313866 | 0.0000042 |
| 32 | 4.5454385 | 0.0000050 |
| 64 | 4.5484729 | 0.0000049 |

| spacing ratio over periods | delta_n |
| :------------------------- | ------: |
| 2, 4, 8 | 3.6403855 +- 4.5e-05 |
| 4, 8, 16 | 4.6247912 +- 3.7e-04 |
| 8, 16, 32 | 4.6262033 +- 2.4e-03 |
| 16, 32, 64 | 4.6309232 +- 1.2e-02 |

Starting from -8.613930 instead, five times the top's error away, moves those three ratios by
-0.0019, +0.0056 and -0.0111, so the top's own error adds at most 0.0004, 0.0011 and 0.0022. Grids
twice and half as wide move the ratio over 4, 8, 16 by 0.0008 at most.

The ratios are 4.625, within one percent of 4.6692, which is the target PROJECT.md set for Rung 3,
and nowhere near the 7.28 a quartic top would give. On the logistic map the same command returns
the superstable values to 2e-9 and their ratios 4.6808, 4.6630 and 4.6684. But three ratios in a row
at 4.625, with these errors, are not yet converging on 4.669: they are 0.9% below it and flat. For
the logistic map the ratios at these periods are already within 0.25% of it. This map is not the
logistic map. Its hump falls away unevenly, its answer has the shoulder between -3 and 0 described
above, and a stable fixed point coexists with the cascade. Whether its ratios go on to 4.669 is a question
about doublings past 64, where the float32 roughness of the model's answer swamps the reading: at
period 64 the readings already scatter by 0.0008. The next section follows them past there, on a
smooth fit to the model's own answers.

### Past period 64: the cascade on a smooth fit of the model's answer

The model's answer is read once, finely, and kept:

    uv run uni response --template rewrite --knob formality \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-19:10:2901 --layer 23

`uni response` writes every curve it reads to `curves/`, with what it was read from and the
direction's squared length, in a file named by the hash of its own bytes. So this one is
`curves/1bb8e39470dc1a00.json`, committed, and 2901 forward passes are not needed again. It is
named by content and not by the command because the readings are float32 on one device: the same
command elsewhere reads a curve that differs in its last bits, and writes it beside this one
rather than over it.

![the model's answer at layer 23, sampled every 0.01 from -19 to 10](figures/response-a6166ccd174afb26.png)

`uni smooth` fits a Chebyshev series of each degree to the curve by least squares and says how far
the readings lie from it:

    uv run uni smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 60 --degree 90 --degree 150 --degree 220

    curve 1bb8e39470dc1a00 at layer 23: 2901 readings from -19 to 10, jitter 1.35e-05
    degree   60: rms residual 5.52e-04, largest 2.18e-03
    degree   90: rms residual 2.04e-04, largest 6.96e-04
    degree  150: rms residual 1.81e-05, largest 8.94e-05
    degree  220: rms residual 1.41e-05, largest 8.52e-05

The jitter is the readings' own scatter about whatever smooth curve lies under them, read from
their fourth differences: a fourth difference of a smooth curve at a spacing of 0.01 is nothing,
and of independent scatter s it is a number of variance 70 s^2. So a series whose residual comes
down to the jitter has fitted the curve and not yet its noise. Degrees 150 and 220 are there, at
1.3 and 1.05 times it. Degrees 60 and 90 are 41 and 15 times above it, and are here to show what
a coarser fit changes. The fit is numpy's, in Chebyshev polynomials rather than the powers `uni
cascade` fits its parabolas in: on these pushes the powers' normal equations have a condition
number of 3e14 at degree 20 and are singular in a float by 60, where the Chebyshev least-squares
problem has 11 at degree 150. A fit numpy reports as rank deficient is refused.

`--map smooth --curve FILE --layer L --degree D` is the response map with the model's answer
replaced by that series C: a push x goes to `gain * C(x) / |v|^2`, the same division by the same
squared length, so a gain here is a gain there. The state is the push with every bit a float
keeps, since the series has no roughness to lose them in, and a push outside -19 to 10, where the
series was fitted to nothing, is refused rather than extrapolated. The map is recorded by the
curve's name, the layer and the degree; the coefficients are recomputed from those.

Its top is measured as the model's was, on a narrower grid:

    uv run uni critical --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 150 --value 4.5 --grid=-8.616:-8.612:51

    the map at 4.5 turns at -8.6140274 +- 2.0e-12 (a cubic through 51 states, scatter 2.6e-15), written -8.614027438063532

The series has no noise, so a cubic's scatter about it is the cubic's failure to follow it, which
shrinks with the grid; on a grid of ±0.025 the top came out 1.8e-9 away, and on grids from ±0.005
to ±0.0005 within 3e-12 of this. That matters far down the cascade. An error e in the top offsets
F^p(x_c) - x_c by e, which moves the superstable gain of period 2^n by e over a slope that grows as
(delta / alpha)^n, while the spacings shrink as delta^-n: relative to its spacing, the gain moves
by about e alpha^n, with alpha = 2.50. From a top 2e-9 off, the ratios over periods 1024 to 4096
and 2048 to 8192 read 4.6691 and 4.6693; from this one, 4.6692 and 4.6692. The same grid gives the
other degrees' tops: -8.61449199320805 (60), -8.613308370166342 (90), -8.61400587604972 (220).
The model's own top, -8.61403 +- 0.00002, is 2.6e-6 from degree 150's and 2.4e-5 from degree 220's.

Each grid below is 21 gains, a fiftieth of the spacing either side of its superstable gain: placed
around the gain a scan found, and past period 1024 around where the spacings extrapolated by delta
put it. A grid that missed its gain would be refused, for crossing zero no times.

    uv run uni cascade --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 150 --critical=-8.614027438063532 --period 2 \
        --grid 3.06727:3.07527:21 --grid 4.1438:4.1876:21 --grid 4.46037:4.47239:21 \
        --grid 4.53008:4.53268:21 --grid 4.545166:4.545728:21 --grid 4.5484020:4.5485226:21 \
        --grid 4.5490954:4.5491212:21 --grid 4.54924390:4.54924943:21 \
        --grid 4.549275708:4.549276894:21 --grid 4.549282521:4.549282775:21 \
        --grid 4.5492839804:4.5492840348:21 --grid 4.54928429290:4.54928430454:21 \
        --grid 4.54928435982:4.54928436232:21

<details>
<summary>The same for degrees 60, 90 and 220</summary>

    uv run uni cascade --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 60 --critical=-8.61449199320805 --period 2 \
        --grid 3.06732:3.07532:21 --grid 4.1436:4.1874:21 --grid 4.46009:4.47212:21 \
        --grid 4.52974:4.53233:21 --grid 4.544921:4.545487:21 --grid 4.5481901:4.5483120:21 \
        --grid 4.5488910:4.5489172:21 --grid 4.54904120:4.54904679:21 \
        --grid 4.549073356:4.549074554:21 --grid 4.549080244:4.549080500:21 \
        --grid 4.5490817188:4.5490817737:21 --grid 4.54908203471:4.54908204648:21 \
        --grid 4.54908210237:4.54908210489:21

    uv run uni cascade --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 90 --critical=-8.613308370166342 --period 2 \
        --grid 3.06712:3.07512:21 --grid 4.1440:4.1878:21 --grid 4.46021:4.47222:21 \
        --grid 4.53010:4.53270:21 --grid 4.545087:4.545646:21 --grid 4.5483203:4.5484408:21 \
        --grid 4.5490142:4.5490400:21 --grid 4.54916282:4.54916836:21 \
        --grid 4.549194666:4.549195853:21 --grid 4.549201487:4.549201741:21 \
        --grid 4.5492029473:4.5492030017:21 --grid 4.54920326013:4.54920327179:21 \
        --grid 4.54920332713:4.54920332963:21

    uv run uni cascade --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 220 --critical=-8.61400587604972 --period 2 \
        --grid 3.06727:3.07527:21 --grid 4.1439:4.1876:21 --grid 4.46037:4.47239:21 \
        --grid 4.53008:4.53268:21 --grid 4.545158:4.545720:21 --grid 4.5483975:4.5485182:21 \
        --grid 4.5490916:4.5491174:21 --grid 4.54924023:4.54924577:21 \
        --grid 4.549272073:4.549273259:21 --grid 4.549278892:4.549279146:21 \
        --grid 4.5492803526:4.5492804070:21 --grid 4.54928066538:4.54928067704:21 \
        --grid 4.54928073237:4.54928073487:21

</details>

Each runs in about two seconds. Where both reach, the fitted maps' superstable gains beside the
model's, with the model's error:

| period | model, 6 decimals | error | degree 150 | degree 220 | degree 60 |
| -----: | ----------------: | ----: | ---------: | ---------: | --------: |
| 2 | 3.071274395 | 8.4e-08 | 3.071274394 | 3.071270336 | 3.071324269 |
| 4 | 4.165735261 | 2.1e-06 | 4.165734784 | 4.165741152 | 4.165491775 |
| 8 | 4.466379481 | 2.5e-06 | 4.466381728 | 4.466379377 | 4.466103116 |
| 16 | 4.531386572 | 4.2e-06 | 4.531382973 | 4.531382442 | 4.531034549 |
| 32 | 4.545438504 | 5.0e-06 | 4.545446815 | 4.545439132 | 4.545204042 |
| 64 | 4.548472873 | 4.9e-06 | 4.548462265 | 4.548457824 | 4.548251058 |

Degree 150 lands within 2.2 of the model's errors of every one of them, and within 1.1e-5. Degree
220 is within 1.5e-5 and 3.1 errors from period 4 on, but 4.1e-6 off at period 2, where the model's
error is 8e-8: at that precision the fit's own error shows, and no fit to readings with a jitter
of 1.35e-5 is exact. Degree 60, fitted to 41 times the jitter, is 5e-5 to 3.5e-4 off everywhere. And past period
64, where the model can no longer be read, the series goes on:

| spacing ratio over periods | model | degree 60 | degree 90 | degree 150 | degree 220 |
| :------------------------- | ----: | --------: | --------: | ---------: | ---------: |
| 2, 4, 8 | 3.64039 | 3.63981 | 3.64529 | 3.64035 | 3.64049 |
| 4, 8, 16 | 4.6248 | 4.62967 | 4.60757 | 4.62525 | 4.62499 |
| 8, 16, 32 | 4.6262 | 4.58248 | 4.66649 | 4.62187 | 4.62435 |
| 16, 32, 64 | 4.631 | 4.65029 | 4.63459 | 4.66393 | 4.65655 |
| 32, 64, 128 | | 4.66585 | 4.66153 | 4.66768 | 4.66806 |
| 64, 128, 256 | | 4.66829 | 4.66773 | 4.66875 | 4.66882 |
| 128, 256, 512 | | 4.66904 | 4.66888 | 4.66913 | 4.66913 |
| 256, 512, 1024 | | 4.66916 | 4.66913 | 4.66918 | 4.66919 |
| 512, 1024, 2048 | | 4.66919 | 4.66919 | 4.66920 | 4.66920 |
| 1024, 2048, 4096 | | 4.66920 | 4.66920 | 4.66920 | 4.66920 |
| 2048, 4096, 8192 | | 4.66920 | 4.66920 | 4.66920 | 4.66920 |

Every fitted ratio carries an error of 2e-6 to 1.5e-5, from its parabolas. The model's are
4.5e-5, 3.7e-4, 2.4e-3 and 1.2e-2, rounded here to the digits they leave.

Two things are in this table. The first ratios are the shape's. Over periods 8, 16, 32 the four
fits spread from 4.582 to 4.666, and even degrees 150 and 220, both at the jitter, differ by 0.007
over periods 16, 32, 64. The fits at the jitter sit low and flat where the model did, at 4.625
over 4, 8, 16 and 4.622 to 4.624 over 8, 16, 32, so the 0.9% the model's ratios fell short by
there is its hump's own shape and not its roughness. Over 16, 32, 64 the model's 4.6309 +- 0.0120 lies 2.1 and 2.7 errors
below those two fits.

The last ratios are not the shape's. The loosest fit and the closest, whose superstable gains
differ by 2e-4, both reach 4.66920 by period 4096, as the other two do, and over 2048, 4096, 8192
the four read 4.6692031, 4.6692017, 4.6692012 and 4.6692026, within 2e-6 of each other and inside
their errors of 7.5e-6 to 8e-6. Feigenbaum's delta is 4.6692016. On the way there each gap to delta is
about 4.9 times the next (for degree 150: -1.9e-5, -3.9e-6, -8e-7), a steady geometric approach.
This is what universality claims: a constant the map's shape does not set, and here four maps
whose early ratios differ by 2% share it.

What this rests on is that the series is the model's answer. Degree 150 fits the curve to 1.3
times the curve's own jitter and places the model's six measurable superstable gains within 2.2
of their errors, and every degree, however loosely fitted, carries the cascade to the same delta.

### Alpha: the cycles shrink by the same factor

Delta is how the superstable gains close in along the knob. Alpha, the second of Feigenbaum's
constants, is how the cycles themselves close in on the top. At the superstable gain of period p the
orbit of the top x_c is a cycle, and its point nearest the top is the one half a period round:

    d = F^(p/2)(x_c) - x_c

Each doubling brings that point nearer by a factor of 2.5029 and puts it on the other side,
so d_n / d_(n+1) runs to -2.5029, whatever the shape of a quadratic hump.

`uni cascade` reads it from the grids it already holds, since half a period divides the period. It
fits a parabola through the half-period's return on each grid and evaluates it at the superstable
gain. The error is that parabola's own and the gain's error carried along its slope. Each row gains
the nearest point and its error, and a line follows the spacing ratios for each pair of periods:

    nearest-point ratio over periods 2, 4: -2.1673930 +- 4.6e-06

On the logistic map,

    uv run uni cascade --map logistic --critical 0.5 --period 2 --grid 3.2355:3.2365:11 \
        --grid 3.4981:3.4991:11 --grid 3.5544:3.5549:11 --grid 3.56655:3.56675:11 --grid 3.5692:3.5693:11

prints the ratios -2.6547448, -2.5318377, -2.5087183 and -2.5041118. The same distances solved to 40
digits with mpmath, in `tests/test_cascade.py`, give -2.6547448, -2.5318377, -2.5087182 and
-2.5041128: they agree to 1e-6. The commands of
the two sections above print these for the model and its four fits:

| periods | model | degree 60 | degree 90 | degree 150 | degree 220 |
| :------ | ----: | --------: | --------: | ---------: | ---------: |
| 2, 4 | -2.16739 +- 4.6e-06 | -2.16809 | -2.16662 | -2.16739 | -2.16737 |
| 4, 8 | -3.82764 +- 4.2e-05 | -3.82671 | -3.83220 | -3.82763 | -3.82773 |
| 8, 16 | -2.21061 +- 2.0e-04 | -2.21656 | -2.20362 | -2.21064 | -2.21060 |
| 16, 32 | -2.64443 +- 1.4e-03 | -2.63794 | -2.66437 | -2.64529 | -2.64605 |
| 32, 64 | -2.42986 +- 5.8e-03 | -2.45031 | -2.44271 | -2.44933 | -2.44861 |
| 64, 128 | | -2.52445 | -2.52691 | -2.52511 | -2.52510 |
| 128, 256 | | -2.49431 | -2.49322 | -2.49411 | -2.49408 |
| 256, 512 | | -2.50634 | -2.50676 | -2.50644 | -2.50643 |
| 512, 1024 | | -2.50153 | -2.50136 | -2.50150 | -2.50150 |
| 1024, 2048 | | -2.50346 | -2.50352 | -2.50347 | -2.50347 |
| 2048, 4096 | | -2.50269 | -2.50266 | -2.50268 | -2.50268 |
| 4096, 8192 | | -2.50300 | -2.50301 | -2.50300 | -2.50300 |

The fitted ratios carry errors of 2e-7 to 2.7e-6.

The early ratios belong to the shape, as delta's did. They swing from -2.17 to -3.83 and back, where
the logistic map's fall steadily from -2.65. The late ratios belong to no fit in particular. From
period 64 on they alternate about -2.5029, and each one's distance from it is -1/2.47 to -1/2.52
times the one before, for all four fits. Over 2048, 4096 and 4096, 8192 the fits read -2.50266 to
-2.50269 and -2.50300 to -2.50301, which bracket alpha = 2.5029079. Aitken's extrapolation of a
geometric approach, taken on the last three ratios printed (1024, 2048 through 4096, 8192), gives
-2.5029084, -2.5029076, -2.5029077 and -2.5029082 for degrees 60, 90, 150 and 220. That is alpha to
within 5e-7, from ratios printed with errors of about 1e-6, and the extrapolation assumes only that
the approach is geometric, which the steady ratio of the distances shows it is.

The model's own distances and those of the degree-150 fit, from the same commands:

| period | model | error | degree 150 | model - fit |
| -----: | ----: | ----: | ---------: | ----------: |
| 2 | 14.1050843 | 1.8e-07 | 14.1050807 | 3.6e-06 |
| 4 | -6.5078572 | 1.4e-05 | -6.5078541 | -3.1e-06 |
| 8 | 1.7002267 | 1.8e-05 | 1.7002313 | -4.6e-06 |
| 16 | -0.7691226 | 7.0e-05 | -0.7691136 | -9.0e-06 |
| 32 | 0.2908460 | 1.5e-04 | 0.2907482 | 9.8e-05 |
| 64 | -0.1196967 | 2.8e-04 | -0.1187053 | -9.9e-04 |

From period 4 to 32 they agree within 0.65 of the model's errors, and so do the ratios through 16,
32. At period 2 the gap is 20 errors, but d there is a single step from the one push x_c, and
nothing averages the model's roughness at that push. The 3.6e-6 is 1.05e-5 in the model's answer
once the gain and squared length are divided out, which is the size of its jitter.

At period 64 the model's distance is 3.5 of its errors from the fit's, and its ratio over 32, 64 is
3.4 errors from the fit's. The gap is not the grid. The model read on the grid above, on grids twice
and half as wide, and on one twice as dense, gives -0.119697, -0.119346, -0.119618 and -0.119780,
each with errors of 2.3e-4 to 4.5e-4, while every fit at the jitter reads -0.11871. Nor is it the
top: moving the fit's top by the model top's error, 2e-5, either way moves its distance by 4e-5. The
gap grows about tenfold a doubling from period 16, through 9e-6, 9.8e-5 and 9.9e-4. It has two
candidate causes, and these measurements do not tell them apart. One is the model's float32
roughness, about 7e-6 of push a step, which the direct reading carries and the fits remove. The
other is shape the fits miss. The next section measures how noise is carried along these cycles, and finds the roughness
large enough.

<details>
<summary>The period-64 grids, and the fit from moved tops</summary>

    for grid in 4.5478:4.5491:21 4.54717:4.54977:21 4.54815:4.54880:21 4.5478:4.5491:41; do
        uv run uni cascade --map response --template rewrite --knob formality \
            --text "The meeting moved to Thursday because the room was booked." \
            --layer 23 --decimals 6 --critical=-8.614030 --period 64 --grid $grid
    done

    for top in -8.61401 -8.61405; do
        uv run uni cascade --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
            --degree 150 --critical=$top --period 2 \
            --grid 3.06727:3.07527:21 --grid 4.1438:4.1876:21 --grid 4.46037:4.47239:21 \
            --grid 4.53008:4.53268:21 --grid 4.545166:4.545728:21 --grid 4.5484020:4.5485226:21
    done

</details>

### Kappa: how much noise each doubling needs removed

Noise truncates a cascade. With noise in every step, the cycles that are smaller than the noise can
no longer be told apart, and the doublings stop being seen at the period where that happens. The
theory's third number, kappa = 6.619, says where: each doubling more needs the noise smaller by
kappa.

It is read off the same superstable cycles. Put noise of unit size into every step of the cycle from
the top. Noise landing at x_k is carried to the cycle's return by the slopes at x_k to x_(p-1), so
the return moves by

    G_p = sqrt(sum over k of (F'(x_k) F'(x_(k+1)) ... F'(x_(p-1)))^2)

its noise gain. From one doubling to the next the gain grows and the cycle shrinks. Measured against
the cycle's own size, the distance d of the last section, the noise's reach grows by (G_2p / G_p)
|d_p / d_2p|, and that runs to kappa. `uni cascade` prints G_p beside the nearest point, through a
parabola across the grid as the distances are, and the growth for each pair of periods:

    noise growth over periods 2, 4: 4.4768900 +- 6.3e-06

It does so for the maps whose slope is exact: the logistic map's, from its formula, and a smooth
map's, from the derivative of its series. The model's own answer has no slope to give. It is float32
and rough at 1e-5, and a difference across that roughness is the roughness's slope.

On the logistic map, on grids a hundredth of a spacing wide and taken to period 8192,

    uv run uni cascade --map logistic --critical 0.5 --period 2 \
        --grid 3.2360494775:3.2360994775:21 --grid 3.49759047256:3.50021540978:21 \
        --grid 3.55443336986:3.5549941615:21 --grid 3.56662288174:3.56674314691:21 \
        --grid 3.56923399988:3.56925976139:21 --grid 3.56979325223:3.56979876985:21 \
        --grid 3.56991302819:3.5699142099:21 --grid 3.56993868059:3.56993893368:21 \
        --grid 3.56994417455:3.56994422876:21 --grid 3.56994535119:3.5699453628:21 \
        --grid 3.56994560319:3.56994560568:21 --grid 3.56994565716:3.56994565769:21 \
        --grid 3.56994566872:3.56994566883:21

the growths are 6.8839698, 6.6606340, 6.6277960, 6.6207460 and 6.6194037 over periods 2 to 64. They
settle at 6.6190372 +- 4.0e-8 over 512, 1024 and 6.6190367 +- 5.4e-7 over 2048, 4096. The same gains
solved to 40 digits with mpmath, in `tests/test_cascade.py`, give 6.8839697, 6.660634, 6.627796 and
6.620746. So kappa, read this way, is 6.619037.

On the four fits of the model's answer, from the commands of the section before last:

| periods | logistic | degree 60 | degree 90 | degree 150 | degree 220 |
| :------ | -------: | --------: | --------: | ---------: | ---------: |
| 2, 4 | 6.88397 | 4.47906 | 4.47391 | 4.47689 | 4.47687 |
| 4, 8 | 6.66063 | 10.05310 | 10.07673 | 10.04598 | 10.04550 |
| 8, 16 | 6.62780 | 5.80916 | 5.79721 | 5.82400 | 5.82383 |
| 16, 32 | 6.62075 | 6.96005 | 7.03265 | 6.99047 | 6.99144 |
| 32, 64 | 6.61940 | 6.47864 | 6.45611 | 6.47688 | 6.47401 |
| 64, 128 | 6.61911 | 6.67539 | 6.68181 | 6.67743 | 6.67717 |
| 128, 256 | 6.61905 | 6.59621 | 6.59325 | 6.59573 | 6.59557 |
| 256, 512 | 6.61904 | 6.62810 | 6.62918 | 6.62836 | 6.62832 |
| 512, 1024 | 6.61904 | 6.61540 | 6.61495 | 6.61531 | 6.61531 |
| 1024, 2048 | 6.61904 | 6.62049 | 6.62066 | 6.62053 | 6.62052 |
| 2048, 4096 | 6.61904 | 6.61846 | 6.61839 | 6.61844 | 6.61844 |
| 4096, 8192 | 6.61904 | 6.61927 | 6.61930 | 6.61927 | 6.61927 |

The fitted growths carry errors of 2.2e-6 to 5.3e-5.

The pattern is alpha's. The early growths belong to the hump: they go 4.48, 10.05, 5.82, 6.99, 6.48
on degree 150, where the logistic map's fall steadily from 6.88. From period 64 on they alternate
about 6.619, and each one's distance from 6.619037 is -1/2.43 to -1/2.54 times the one before, for
all four fits. Over 2048, 4096 and 4096, 8192 they read 6.61839 to 6.61846 and 6.61927 to 6.61930,
either side of the logistic map's 6.619037. Aitken's extrapolation on the last three growths printed
gives 6.6190360, 6.6190362, 6.6190367 and 6.6190361 for degrees 60, 90, 150 and 220: the logistic
map's kappa to within 1.2e-6, from growths printed with errors of about 3e-6.

The noise gain also accounts for what the model's direct readings could not resolve. The curve's
jitter, 1.35e-5 in the answer, is 6.9e-6 of push a step at a gain of 4.55, and the direct cascade's
returns scatter about their parabolas by that jitter carried by the gain:

| period | jitter x gain / squared length x G_p | direct scatter |
| -----: | -----------------------------------: | -------------: |
| 2 | 8.2e-06 | 1.3e-06 |
| 4 | 2.3e-05 | 2.9e-05 |
| 8 | 6.4e-05 | 6.0e-05 |
| 16 | 1.7e-04 | 1.9e-04 |
| 32 | 4.6e-04 | 4.2e-04 |
| 64 | 1.2e-03 | 8.0e-04 |

G_p is the degree-150 fit's, at the model's own superstable gains. From period 4 to 64 the scatter
is 0.66 to 1.27 times the prediction. At period 2 it is 0.16 of it, but there the grid's 21 orbits
all start from the one push x_c and take their second step from pushes within 0.014 of each other,
so most of their roughness is shared and shifts the return instead of scattering it. So the model's
float32 roughness is noise the cascade carries, and at period 64 it moves the return by 1e-3. That
is the size of the gap between the model's d_64 and the fits'. Whether that roughness is what makes
the gap is a question about a bias rather than a scatter. On the noisy map of the next section,
noise independent from one step to the next leaves the mean half-period point where the cycle's is
until its reach is past one, and the roughness reaches 1.2e-3 in a distance of 0.119. So a roughness
that biases d_64 would have to be shared between neighbouring pushes, which is not read here.

### Temperature: how many doublings a drawn token lets through

Every map above reads the model's answer with no token written. A loop that generates draws its
tokens, and the least it can draw is one: the token after the prompt, under the push, drawn at a
temperature T with probability proportional to p^(1/T). The answer is then read over the prompt
and that token together, as `uni response` reads it over the prompt, so it depends on which token
came up. Over the draw it has a mean, which is a map, and a spread, which is noise that map is read
with at every step.

    uv run uni temperature --template rewrite --knob formality \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-19:10:291 --layer 23 \
        --temperature 1 --temperature 0.7 --temperature 0.5 --temperature 0.3 \
        --temperature 0.2 --temperature 0.1 --temperature 0.05 --temperature 0.02

reads, at each push, the answer with each of the 151,936 tokens in the vocabulary appended, and
weighs them at each temperature. So the mean and the spread carry no sampling error. That is a
forward pass per token, and it took three hours here. It writes a curve of means and one of
spreads for each temperature:

| T | means | spreads |
| -: | :---- | :------ |
| 1 | `curves/2a1a49bcd3238c74.json` | `curves/16428756a7f5c3a6.json` |
| 0.7 | `curves/0509861b3e600267.json` | `curves/87f60bd9cf796196.json` |
| 0.5 | `curves/d0ad9e1cb35460f9.json` | `curves/36247784fe751616.json` |
| 0.3 | `curves/507a6f83ebd66ffd.json` | `curves/007dac775ef90de2.json` |
| 0.2 | `curves/3b9549a3116a9bf1.json` | `curves/f2932e16848f3c1a.json` |
| 0.1 | `curves/9586920052d4640b.json` | `curves/dd49428b0fab9882.json` |
| 0.05 | `curves/30c1551d63d2bc85.json` | `curves/6b23efc1a2322b3f.json` |
| 0.02 | `curves/edbaf45599bbc1f2.json` | `curves/9387ab2dbfe7bbfd.json` |

The same command with `--remote` read the other sixteen curves in `curves/` on the host, and its
table agrees with this one to every digit printed.

The spread, from the table it prints:

| push | T = 1 | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 | 0.02 |
| ---: | ----: | --: | --: | --: | --: | --: | ---: | ---: |
| -19 | 8.6e-02 | 8.9e-02 | 8.4e-02 | 7.3e-02 | 6.6e-02 | 5.2e-02 | 3.0e-02 | 6.5e-03 |
| -8.6 | 9.9e-02 | 9.4e-02 | 8.4e-02 | 6.6e-02 | 4.7e-02 | 1.1e-02 | 6.6e-04 | 3.4e-07 |
| -3 | 6.5e-02 | 2.7e-02 | 8.8e-03 | 8.4e-04 | 5.2e-05 | 1.4e-08 | 1.1e-15 | 4.8e-37 |
| 0 | 2.7e-02 | 7.9e-03 | 1.6e-03 | 5.0e-05 | 7.8e-07 | 5.4e-12 | 3.7e-22 | 1.3e-52 |
| 5.5 | 5.1e-02 | 1.5e-02 | 5.8e-03 | 1.7e-03 | 3.9e-04 | 4.7e-06 | 6.8e-10 | 2.1e-21 |
| 10 | 9.2e-02 | 4.3e-02 | 1.5e-02 | 1.8e-03 | 1.4e-04 | 6.9e-08 | 2.0e-14 | 6.4e-34 |

It is largest at the top, -8.6, which every superstable cycle passes through. There the next token
is spread thin enough that cooling to 0.2 halves the spread and no more. Where one token dominates,
at 0 and -3, cooling removes it within a few steps of T. And the draw moves the map as well as
spreading it: the mean at the top is 15.9966 at T = 1, 16.0179 at 0.5 and 16.0866 at 0.05, where the
answer with no token is 16.0178.

#### The prediction

A spread s in the answer is noise gain * s / |v|^2 in the next push, and the Kappa section says
how noise at every step is carried to the return. Here its size depends on where it is put in, so
each step's noise is taken at its own push and carried by the slopes after it:

    S_p = sqrt(sum over k of (gain * s(x_k) / |v|^2)^2 (F'(x_(k+1)) ... F'(x_(p-1)))^2)

`uni cascade --noise SPREADS` prints S_p in place of G_p, reading s between the curve's pushes on
the straight line through the two either side, and prints for each period how far it reaches in the
cycle's nearest distance:

    noise over nearest distance at period 8: 2.5053677e-01 +- 9.8e-08

Below one, the noise moves a run's return by less than the cycle's half-period point lies from the
top. Past one it moves it further, and in a single run the doubling can no longer be told from
the noise. Carried along the
cycles of the degree-150 fit, with the grids of the section on it and `--noise` added:

    uv run uni cascade --map smooth --curve curves/1bb8e39470dc1a00.json --layer 23 \
        --degree 150 --critical=-8.614027438063532 --period 2 \
        --grid 3.06727:3.07527:21 --grid 4.1438:4.1876:21 --grid 4.46037:4.47239:21 \
        --grid 4.53008:4.53268:21 --grid 4.545166:4.545728:21 --grid 4.5484020:4.5485226:21 \
        --grid 4.5490954:4.5491212:21 --grid 4.54924390:4.54924943:21 \
        --grid 4.549275708:4.549276894:21 --noise curves/16428756a7f5c3a6.json

and the same with each temperature's spreads:

| T | period 2 | 4 | 8 | 16 | 32 | 64 | 128 | last period below one |
| -: | -------: | -: | -: | -: | -: | -: | --: | --------------------: |
| 1 | 3.7e-03 | 0.023 | 0.25 | 1.48 | 10.4 | 67 | 450 | 8 |
| 0.7 | 3.3e-03 | 0.020 | 0.23 | 1.34 | 9.4 | 61 | 408 | 8 |
| 0.5 | 3.0e-03 | 0.018 | 0.20 | 1.20 | 8.5 | 55 | 367 | 8 |
| 0.3 | 2.3e-03 | 0.015 | 0.17 | 0.97 | 6.9 | 44 | 297 | 16 |
| 0.2 | 1.6e-03 | 0.012 | 0.13 | 0.75 | 5.4 | 35 | 234 | 16 |
| 0.1 | 3.9e-04 | 8.7e-03 | 0.052 | 0.35 | 2.6 | 17 | 115 | 16 |
| 0.05 | 2.2e-05 | 7.9e-03 | 9.0e-03 | 0.078 | 0.64 | 4.3 | 29 | 32 |
| 0.02 | 1.1e-08 | 5.8e-03 | 1.8e-03 | 5.0e-03 | 0.10 | 0.77 | 5.3 | 64 |

Past the period where the reach passes one it grows by about 6.6 a doubling, as kappa says it must,
so cooling lets through a doubling for each factor of 6.6 it takes off the noise the cycle collects.
From T = 1 to 0.02 that is three doublings more, from period 8 to period 64. The early columns
are not monotone at the coldest temperatures. The period-4 cycle's point half a period round is
at -15.1, toward the lowest pushes, where the spread survives cooling longest, and its reach stays
near 6e-3 while period 2's, whose other point is at 5.5, collapses.

#### The check

The prediction is carried along the cycles of a map, and the loop that draws a token runs on its
mean, which is not the map with no token: at T = 1 its superstable gain of period 64 is 4.5098,
against 4.5485 with none. So each temperature is checked on its own mean. `uni smooth` fits the
curves of means as it fits the answer with no token:

| T | jitter | degree 140: rms residual | largest |
| -: | -----: | ----------------------: | ------: |
| 1 | 2.93e-05 | 1.37e-05 | 5.09e-05 |
| 0.7 | 2.96e-05 | 1.41e-05 | 5.09e-05 |
| 0.5 | 3.18e-05 | 1.68e-05 | 6.39e-05 |
| 0.3 | 6.12e-05 | 6.24e-05 | 4.88e-04 |
| 0.2 | 1.48e-04 | 1.54e-04 | 1.43e-03 |
| 0.1 | 4.62e-04 | 3.91e-04 | 4.07e-03 |
| 0.05 | 7.53e-04 | 6.32e-04 | 6.06e-03 |
| 0.02 | 1.05e-03 | 9.15e-04 | 7.01e-03 |

Degree 150 is refused on 291 readings, and a curve read every 0.1 has fourth differences that see
some of its shape, so this jitter overstates its scatter. From T = 1 to 0.5 the fit comes down to
1.4e-5 to 1.7e-5, about the answer's own roughness read every 0.01. Colder, the curve itself
roughens: its jitter grows from 6.1e-5 at 0.3 to 1.05e-3 at 0.02, and the largest miss to 7e-3.
Down to 0.2 the largest miss, 1.4e-3, is 3% of the spread at the top it is checked against. At 0.1
it is 4.1e-3 against a spread there of 1.1e-2, and a map fitted that loosely would test the fit
rather than the noise. So the check is made from T = 1 down to 0.2.

Each mean map's top and superstable gains are found as the smooth map's were:

    uv run uni critical --map smooth --curve curves/2a1a49bcd3238c74.json --layer 23 \
        --degree 140 --value 4.5 --grid=-8.608:-8.588:51

    the map at 4.5 turns at -8.5979469 +- 4.9e-08 (a cubic through 51 states, scatter 3.2e-10), written -8.59794690133104

    uv run uni cascade --map smooth --curve curves/2a1a49bcd3238c74.json --layer 23 \
        --degree 140 --critical=-8.59794690133104 --period 2 \
        --grid 3.06363:3.07163:21 --grid 4.1094:4.1519:21 --grid 4.42299:4.43492:21 \
        --grid 4.49174:4.49430:21 --grid 4.506542:4.507094:21 --grid 4.5097277:4.5098464:21 \
        --noise curves/16428756a7f5c3a6.json

and the others with the grids below. That command is the prediction on the mean map. Its
superstable values are where the check is read:

<details>
<summary>The tops and grids at 0.7, 0.5, 0.3 and 0.2</summary>

| T | means | top grid | top | cascade grids |
| -: | :---- | :------- | :-- | :------------ |
| 0.7 | `0509861b3e600267` | -8.61:-8.59 | -8.599686762143579 | 3.06223:3.07023 4.1049:4.1473 4.41700:4.42888 4.48559:4.48815 4.500339:4.500888 4.5035104:4.5036287 |
| 0.5 | `d0ad9e1cb35460f9` | -8.621:-8.601 | -8.610080566092575 | 3.06292:3.07092 4.1013:4.1435 4.41529:4.42724 4.48352:4.48606 4.498166:4.498712 4.5013150:4.5014324 |
| 0.3 | `507a6f83ebd66ffd` | -8.653:-8.633 | -8.642651123903905 | 3.06643:3.07443 4.0912:4.1329 4.41247:4.42473 4.47807:4.48050 4.492471:4.493009 4.4955928:4.4957093 |
| 0.2 | `3b9549a3116a9bf1` | -8.657:-8.637 | -8.646836247852228 | 3.06499:3.07299 4.0873:4.1288 4.40656:4.41875 4.47017:4.47252 4.484394:4.484927 4.4875143:4.4876308 |

Each top grid is 51 states and each cascade grid 21 gains, with the temperature's spreads as
`--noise`.

</details>

Two loops are run at those gains. `--map noisy` is the smooth mean map with a normal draw of the
measured spread added to its answer at each step: noise of the size the prediction assumes and
of no other shape. `--map sampled` is the model itself: the response map with the token after the
prompt drawn at T and the answer read with it, the push written to six decimals. Both take
`--seed`, and their draws are fixed by it, by the gain and by the state, so a run is the same run
when it is repeated. `uni spread` runs the orbit of the top many times at each gain, from seeds
counted up from `--seed`, and prints where the draws land half a period round and after a period,
and how far the spread of the return reaches in the distance of the mean half-period point:

    uv run uni spread --map noisy --curve curves/2a1a49bcd3238c74.json --layer 23 \
        --degree 140 --spreads curves/16428756a7f5c3a6.json --seed 0 \
        --critical=-8.59794690133104 --period 2 --draws 2000 \
        --value 3.067628676 --value 4.130639374 --value 4.428954633 \
        --value 4.49301896 --value 4.506817811 --value 4.509787049

    uv run uni spread --map sampled --template rewrite --knob formality \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 23 --decimals 6 --temperature 1 --seed 0 \
        --critical=-8.597947 --period 2 --draws 300 \
        --value 3.067628676 --value 4.130639374 --value 4.428954633 \
        --value 4.49301896 --value 4.506817811 --value 4.509787049

The noisy map runs in seconds. The sampled one is two forward passes a step, the prompt's and the
token's, for 37,800 steps at 300 draws, and took from 25 minutes to under an hour a temperature on the machines it ran on.
The other temperatures are the same commands with their curves, tops and superstable values.

The reach, predicted on each mean map by `uni cascade --noise` and measured by `uni spread` on the
two loops, with the measured errors:

| T | period | predicted | noisy, 2000 draws | sampled, 300 draws |
| -: | -----: | --------: | ----------------: | -----------------: |
| 1 | 2 | 3.77e-03 | 3.76e-03 +- 5.9e-05 | 3.41e-03 +- 2.1e-04 |
| | 4 | 0.0234 | 0.0233 +- 3.6e-04 | 0.0235 +- 1.1e-03 |
| | 8 | 0.250 | 0.248 +- 4.2e-03 | 0.251 +- 1.1e-02 |
| | 16 | 1.485 | 1.218 +- 0.027 | 1.116 +- 0.059 |
| 0.7 | 2 | 3.39e-03 | 3.31e-03 +- 5.3e-05 | 3.29e-03 +- 1.9e-04 |
| | 4 | 0.0209 | 0.0206 +- 3.3e-04 | 0.0201 +- 1.1e-03 |
| | 8 | 0.225 | 0.227 +- 3.8e-03 | 0.219 +- 9.5e-03 |
| | 16 | 1.340 | 1.137 +- 0.023 | 1.154 +- 0.063 |
| 0.5 | 2 | 3.00e-03 | 2.97e-03 +- 4.5e-05 | 2.98e-03 +- 1.8e-04 |
| | 4 | 0.0188 | 0.0185 +- 3.0e-04 | 0.0184 +- 1.0e-03 |
| | 8 | 0.200 | 0.196 +- 3.2e-03 | 0.211 +- 1.0e-02 |
| | 16 | 1.207 | 1.027 +- 0.020 | 0.950 +- 0.051 |
| 0.3 | 2 | 2.34e-03 | 2.37e-03 +- 3.9e-05 | 2.24e-03 +- 1.3e-04 |
| | 4 | 0.0160 | 0.0164 +- 2.6e-04 | 0.0151 +- 6.4e-04 |
| | 8 | 0.160 | 0.159 +- 2.5e-03 | 0.149 +- 6.4e-03 |
| | 16 | 1.009 | 0.900 +- 0.017 | 0.884 +- 0.044 |
| 0.2 | 2 | 1.60e-03 | 1.57e-03 +- 2.4e-05 | 1.51e-03 +- 1.4e-04 |
| | 4 | 0.0128 | 0.0127 +- 2.0e-04 | 0.0122 +- 5.6e-04 |
| | 8 | 0.122 | 0.124 +- 1.9e-03 | 0.115 +- 5.8e-03 |
| | 16 | 0.790 | 0.720 +- 0.013 | 0.753 +- 0.040 |

Through period 8, where the reach is at most a quarter, every measurement on either loop lies
within 1.7 of its errors of the prediction. Twelve of the fifteen on the sampled loop lie below it,
by up to 10%. So the token's noise at these temperatures is what its spread says, carried by the
mean map's slopes, and its shape changes that by no more than a few percent the draws can resolve.

At period 16 the prediction runs from 0.79 to 1.49, and both loops fall short of it by up to a
quarter, more the larger it is. A reach near one carries the orbit far enough from the cycle that the
cycle's slopes no longer carry its noise. But where the reach passes one is where it was predicted.
It is past one at T = 1 and 0.7 on both loops, below it at 0.3 and 0.2 on both, and at 0.5, predicted
at 1.21, the two loops read 1.03 and 0.95. Cooling from 1 to 0.2 lets period 16 through, and it
does so between 0.5 and 0.3, where the prediction puts it.

Past one, the reach stops growing. The return's spread settles at 0.8 to 1.3 on every loop, the
width of the band the orbit now wanders in. What is left of the cycle is where the draws land on
average, half a period round, beside the distance the mean map's own cycle puts there:

| T | period | cycle | noisy | sampled |
| -: | -----: | ----: | ----: | ------: |
| 1 | 16 | -0.764 | -0.780 +- 0.013 | -0.803 +- 0.031 |
| | 32 | 0.289 | 0.377 +- 0.024 | 0.302 +- 0.060 |
| | 64 | -0.118 | 0.486 +- 0.028 | 0.482 +- 0.073 |
| 0.7 | 16 | -0.765 | -0.776 +- 0.012 | -0.764 +- 0.032 |
| | 32 | 0.290 | 0.347 +- 0.023 | 0.419 +- 0.057 |
| | 64 | -0.118 | 0.377 +- 0.028 | 0.325 +- 0.072 |
| 0.5 | 16 | -0.766 | -0.787 +- 0.011 | -0.810 +- 0.026 |
| | 32 | 0.291 | 0.378 +- 0.021 | 0.388 +- 0.053 |
| | 64 | -0.119 | 0.317 +- 0.027 | 0.278 +- 0.070 |
| 0.3 | 16 | -0.739 | -0.759 +- 0.009 | -0.721 +- 0.022 |
| | 32 | 0.289 | 0.324 +- 0.019 | 0.342 +- 0.048 |
| | 64 | -0.117 | 0.216 +- 0.025 | 0.291 +- 0.061 |
| 0.2 | 16 | -0.708 | -0.715 +- 0.006 | -0.708 +- 0.019 |
| | 32 | 0.272 | 0.305 +- 0.014 | 0.280 +- 0.036 |
| | 64 | -0.111 | -0.002 +- 0.020 | 0.023 +- 0.054 |

Averaged over hundreds of runs, a cycle survives a doubling past the one a single run can see: at
period 32 the draws still land on the cycle's side, pushed outward. At period 64 no temperature
checked keeps it. From T = 1 to 0.3 the draws land on the far side of the top, and at 0.2, whose predicted reach
there is the smallest, 35 against 45 to 67, on the top itself.

So a drawn token truncates the model's cascade as noise of its spread does, and where the prediction
said: the temperature sets the spread, kappa sets how fast the cycles outgrow it, and the two
together say which doubling is the last. What the table of the prediction says about colder
temperatures, down to period 64 at T = 0.02, rests on the spreads and the fit with no token, since
the colder mean maps are not smooth enough to check it on.

### Rung 5, a second knob: brevity

Everything above turns one knob, `formality`. PROJECT.md's Rung 5 asks whether the constants move
when the knob does. Four more contrasts in `uni/directions/` use the formality contrast's
template, layer and texts, and change the quality (some positivity and past pairs change a detail of
the text with it): `positivity` (upbeat minus gloomy),
`certainty` (confident minus hedged), `brevity` (terse minus wordy) and `past` (past tense minus
future). Each is derived as formality was:

    uv run uni direction brevity

    uni/directions/Qwen--Qwen2.5-0.5B-Instruct/brevity.json  layer 12  length 5.023152  sha256 ca56e8d1547df1f4ed6111b9ab6f7e6cdae7374130cade73c1bf0dac4564bdcc

and read coarsely, at four layers, with the same command for each:

    uv run uni response --template rewrite --knob brevity \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-60:60:241 --layer 16 --layer 18 --layer 20 --layer 23

| knob | layer 16 | layer 18 | layer 20 | layer 23 |
| :--- | :------- | :------- | :------- | :------- |
| positivity | -32.0, 10.5 | -27.5, 12.0 | -28.0, 14.5 | 24.0 |
| certainty | -8.5, 10.0 | -8.0, 11.0 | -9.0, 13.0 | -18.5, 15.0, 21.0, 27.0, 31.0 |
| brevity | -4.0, 5.5 | -3.0, 6.0 | -3.5, 7.5 | -11.0, 11.0, 41.0 |
| past | none | none | none | none |

The cells are where the command says the slope changes sign. `past` never turns: pushed either
way, the layers after it only follow. The other three turn, and the pictures show how:

![positivity](figures/response-6c3efc59883ca4da.png)
![certainty](figures/response-7f58b36bb40f9119.png)
![brevity](figures/response-71d0ef0d608dfe20.png)
![past](figures/response-ba8b9f8f646e88b6.png)

A turn is not yet a cascade. The loop has to fold its orbit back over the turn, and where a curve
turns into a long flat stretch, as positivity and certainty do at layer 23, a cycle that passes
over the flat stretch has a multiplier near zero there, which holds it stable rather than letting it
double. `brevity` at layers 18 and 20 turns twice with steep sides between, and it is the one
followed here. The others are not claimed to have no cascade, only not to have been followed.

The loop is the response map of the formality sections with this direction and read at layer 20.
Its gain is negative: the answer is fed back reversed, so the top of the answer is the bottom of
the map, which universality does not distinguish from a top. The model's answer is read finely over
-30 to 10, and every orbit below stays inside that, since the smooth map refuses a push outside the
pushes its curve was read at:

    uv run uni response --template rewrite --knob brevity \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-30:10:4001 --layer 18 --layer 20

![the model's answer along brevity at layers 18 and 20, every 0.01 from -30 to 10](figures/response-ea76c33c719c2e31.png)

    uv run uni smooth --curve curves/de7ea4e2adf647e2.json --layer 20 \
        --degree 150 --degree 220 --degree 300 --degree 400

    curve de7ea4e2adf647e2 at layer 20: 4001 readings from -30 to 10, jitter 1.11e-05
    degree  150: rms residual 3.00e-03, largest 2.08e-02
    degree  220: rms residual 3.42e-04, largest 1.79e-03
    degree  300: rms residual 5.08e-05, largest 3.38e-04
    degree  400: rms residual 1.62e-05, largest 1.30e-04

The fall from the top is sharper than formality's hump, so a series needs more terms to follow it:
degree 400 comes down to 1.46 times the jitter, where formality's reached it by 150. Degrees 300
and 220, at 4.6 and 31 times the jitter, are here to show what a looser fit changes, as 60 and
90 were there.

The tops, on grids of ±0.002 (±0.1 for the model, whose readings are rough at 1e-5):

    uv run uni critical --map smooth --curve curves/de7ea4e2adf647e2.json --layer 20 \
        --degree 400 --value=-9.36 --grid=-3.3367:-3.3327:51

    the map at -9.36 turns at -3.3346724 +- 1.7e-11 (a cubic through 51 states, scatter 1.4e-13), written -3.3346724231195863

    uv run uni critical --map response --template rewrite --knob brevity \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 20 --decimals 6 --value=-9.36 --grid=-3.435:-3.235:101

    the map at -9.36 turns at -3.3347469 +- 6.8e-06 (a cubic through 101 states, scatter 3.9e-06), written -3.334747

Degree 300's top is -3.335001328786538, on the grid -3.3370:-3.3330:51, and degree 220's
-3.3356095904168246, on -3.3376:-3.3336:51. Degree 400's lies 7.5e-5 from the model's, which is 11
of the model's errors, where formality's best fit lay inside its model's error. So the fit does not
place this top where the model does. The superstable gains below are each read from their own top,
the model's from its own and each fit's from its own.

The cascade, on grids a fiftieth of a spacing either side of each superstable gain, placed as the
formality fits' were:

    uv run uni cascade --map smooth --curve curves/de7ea4e2adf647e2.json --layer 20 \
        --degree 400 --critical=-3.3346724231195863 --period 2 \
        --grid=-6.3214:-6.2192:21 --grid=-8.8751:-8.7729:21 --grid=-9.23569:-9.21954:21 \
        --grid=-9.33181:-9.32772:21 --grid=-9.353283:-9.352361:21 --grid=-9.3579083:-9.3577089:21 \
        --grid=-9.3589002:-9.3588574:21 --grid=-9.35911270:-9.35910353:21 \
        --grid=-9.359158217:-9.359156253:21 --grid=-9.359167965:-9.359167545:21 \
        --grid=-9.3591700532:-9.3591699631:21 --grid=-9.35917050034:-9.35917048104:21 \
        --grid=-9.35917059610:-9.35917059197:21

<details>
<summary>Degrees 300 and 220, and the model read directly</summary>

    uv run uni cascade --map smooth --curve curves/de7ea4e2adf647e2.json --layer 20 \
        --degree 300 --critical=-3.335001328786538 --period 2 \
        --grid=-6.3210:-6.2188:21 --grid=-8.8753:-8.7732:21 --grid=-9.23556:-9.21943:21 \
        --grid=-9.33193:-9.32783:21 --grid=-9.353295:-9.352377:21 --grid=-9.3579222:-9.3577227:21 \
        --grid=-9.3589144:-9.3588716:21 --grid=-9.35912694:-9.35911777:21 \
        --grid=-9.359172470:-9.359170504:21 --grid=-9.359182220:-9.359181799:21 \
        --grid=-9.3591843081:-9.3591842180:21 --grid=-9.35918475535:-9.35918473605:21 \
        --grid=-9.35918485114:-9.35918484700:21

    uv run uni cascade --map smooth --curve curves/de7ea4e2adf647e2.json --layer 20 \
        --degree 220 --critical=-3.3356095904168246 --period 2 \
        --grid=-6.3202:-6.2180:21 --grid=-8.8757:-8.7735:21 --grid=-9.23508:-9.21898:21 \
        --grid=-9.33156:-9.32746:21 --grid=-9.353268:-9.352336:21 --grid=-9.357927:-9.357726:21 \
        --grid=-9.3589246:-9.3588816:21 --grid=-9.35913832:-9.35912910:21 \
        --grid=-9.359184089:-9.359182113:21 --grid=-9.359193891:-9.359193468:21 \
        --grid=-9.3591959901:-9.3591958995:21 --grid=-9.35919643971:-9.35919642030:21 \
        --grid=-9.35919653601:-9.35919653185:21

    uv run uni cascade --map response --template rewrite --knob brevity \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 20 --decimals 6 --critical=-3.334747 --period 2 \
        --grid=-6.7810:-5.7595:21 --grid=-9.3348:-8.3133:21 --grid=-9.3083:-9.1469:21 \
        --grid=-9.3502:-9.3093:21 --grid=-9.35743:-9.34821:21 --grid=-9.358806:-9.356811:21

The model's grids are a fifth of a spacing either side, wider than the fits', since its superstable
gains need not be theirs. The model's six grids take two minutes.

</details>

The model's superstable gains beside the fits':

| period | model, 6 decimals | error | degree 400 | degree 300 | degree 220 |
| -----: | ----------------: | ----: | ---------: | ---------: | ---------: |
| 2 | -6.270211332 | 8.4e-05 | -6.270284949 | -6.269933493 | -6.269139342 |
| 4 | -8.824350545 | 3.5e-04 | -8.824020215 | -8.824251546 | -8.824585393 |
| 8 | -9.227566482 | 1.4e-05 | -9.227613842 | -9.227496039 | -9.227029834 |
| 16 | -9.329784613 | 1.6e-05 | -9.329763495 | -9.329881378 | -9.329510753 |
| 32 | -9.352832754 | 6.8e-06 | -9.352822179 | -9.352835654 | -9.352802309 |
| 64 | -9.357817658 | 5.6e-06 | -9.357808593 | -9.357822454 | -9.357826283 |

Degree 400 lands within 3.4 of the model's errors of every one, and within 3.3e-4. The ratios:

| spacing ratio over periods | model | degree 400 | degree 300 | degree 220 |
| :------------------------- | ----: | ---------: | ---------: | ---------: |
| 2, 4, 8 | 6.3344 +- 6.3e-03 | 6.32749 | 6.33442 | 6.34981 |
| 4, 8, 16 | 3.9447 +- 3.5e-03 | 3.95100 | 3.93850 | 3.92702 |
| 8, 16, 32 | 4.4350 +- 4.0e-03 | 4.42999 | 4.46040 | 4.39992 |
| 16, 32, 64 | 4.6236 +- 9.8e-03 | 4.62430 | 4.60301 | 4.63608 |
| 32, 64, 128 | | 4.65932 | 4.65827 | 4.66559 |
| 64, 128, 256 | | 4.66682 | 4.66712 | 4.66938 |
| 128, 256, 512 | | 4.66872 | 4.66870 | 4.66898 |
| 256, 512, 1024 | | 4.66909 | 4.66910 | 4.66919 |
| 512, 1024, 2048 | | 4.66918 | 4.66918 | 4.66919 |
| 1024, 2048, 4096 | | 4.66920 | 4.66920 | 4.66920 |
| 2048, 4096, 8192 | | 4.66920 | 4.66920 | 4.66920 |

The fitted ratios carry errors of 3.4e-6 to 1.3e-4.

| nearest-point ratio over periods | model | degree 400 | degree 300 | degree 220 |
| :------------------------------- | ----: | ---------: | ---------: | ---------: |
| 2, 4 | -5.80777 +- 8.5e-04 | -5.80886 | -5.80703 | -5.80365 |
| 4, 8 | -3.31808 +- 5.0e-04 | -3.31673 | -3.32021 | -3.32737 |
| 8, 16 | -2.45724 +- 4.9e-04 | -2.45894 | -2.45140 | -2.44649 |
| 16, 32 | -2.47575 +- 1.1e-03 | -2.47379 | -2.48379 | -2.49767 |
| 32, 64 | -2.50429 +- 4.0e-03 | -2.50467 | -2.49386 | -2.48998 |
| 64, 128 | | -2.50030 | -2.50362 | -2.50597 |
| 128, 256 | | -2.50357 | -2.50225 | -2.50206 |
| 256, 512 | | -2.50256 | -2.50308 | -2.50324 |
| 512, 1024 | | -2.50303 | -2.50282 | -2.50277 |
| 1024, 2048 | | -2.50286 | -2.50294 | -2.50296 |
| 2048, 4096 | | -2.50293 | -2.50289 | -2.50289 |
| 4096, 8192 | | -2.50290 | -2.50291 | -2.50292 |

| noise growth over periods | degree 400 | degree 300 | degree 220 |
| :------------------------ | ---------: | ---------: | ---------: |
| 2, 4 | 14.45651 | 14.45264 | 14.43593 |
| 4, 8 | 8.24975 | 8.25578 | 8.28635 |
| 8, 16 | 6.35701 | 6.35300 | 6.27919 |
| 16, 32 | 6.50656 | 6.53433 | 6.59296 |
| 32, 64 | 6.61765 | 6.58677 | 6.57400 |
| 64, 128 | 6.61056 | 6.61967 | 6.62674 |
| 128, 256 | 6.62047 | 6.61695 | 6.61633 |
| 256, 512 | 6.61805 | 6.61943 | 6.61987 |
| 512, 1024 | 6.61934 | 6.61879 | 6.61867 |
| 1024, 2048 | 6.61890 | 6.61912 | 6.61917 |
| 2048, 4096 | 6.61909 | 6.61900 | 6.61898 |
| 4096, 8192 | 6.61901 | 6.61905 | 6.61906 |

The fitted nearest-point ratios carry errors of 4.7e-7 to 2e-5, and the growths 1.9e-6 to 2.5e-4.

This knob's hump is not formality's. Its first spacing ratios are 6.33 and 3.94 where formality's
were 3.64 and 4.62, its first nearest-point ratio is -5.81 where formality's was -2.17, and its
first noise growth is 14.5 where formality's was 4.5. Through period 64, the model's own superstable
gains lie within 3.4 of their errors of the degree-400 fit's, its spacing ratios within 1.8 and its
nearest-point ratios within 3.5, despite the tops' 11-error gap.

And the late numbers are formality's. Over 2048, 4096, 8192 the three fits read 4.6691985, 4.6692018
and 4.6692014, where formality's four read 4.6692012 to 4.6692031 and Feigenbaum's delta is 4.6692016.
Aitken's extrapolation on the last three nearest-point ratios gives -2.5029074, -2.5029071 and
-2.5029076 for degrees 400, 300 and 220, within 8e-7 of alpha = 2.5029079, and on the last three
growths 6.6190350, 6.6190354 and 6.6190363, within 2e-6 of the logistic map's kappa, 6.619037. A
second knob, a different quality pushed along a different direction, with a hump of its own shape,
gives fits whose deep ratios reach the same three constants. The model itself is read to period 64,
where its ratios are 4.624 and -2.504, and it follows the best fit there. The limits are the fits'.

### Rung 5, a second loop: summarize

The loops above all rewrite one sentence. This one summarizes another, under the knob they share,
formality, so the loop is what changes and the knob is not: the `summarize` template of
`uni/templates.toml`, and the text "The committee postponed the vote after two members raised
concerns about the budget."

    uv run uni response --template summarize --knob formality \
        --start "The committee postponed the vote after two members raised concerns about the budget." \
        --grid=-60:60:241 --layer 16 --layer 18 --layer 20 --layer 23

    layer 16: maximum 3.3395 at -9; the slope changes sign at [-9.0, 23.0]
    layer 18: maximum 4.2142 at -8.5; the slope changes sign at [-8.5, -1.5, 0.0, 11.5]
    layer 20: maximum -3.4797 at -8; the slope changes sign at [-8.0, -3.0, 0.5, 17.5]
    layer 23: maximum 15.1554 at -9; the slope changes sign at [-48.0, -9.0]

![the summarize loop's answer along formality, at four layers](figures/response-b59ed490bc564e86.png)

At layer 23 it has a hump like the rewrite loop's, with a top near -9 and a steep fall to the right,
and a shoulder near 0 as that loop had between -3 and 0. It is read at layer 23 as the rewrite loop
was, finely over the pushes its cascade visits:

    uv run uni response --template summarize --knob formality \
        --start "The committee postponed the vote after two members raised concerns about the budget." \
        --grid=-20:12:3201 --layer 23

![the summarize loop's answer at layer 23, every 0.01 from -20 to 12](figures/response-fba06596ba37787a.png)

    uv run uni smooth --curve curves/b0b78d1f63093ba4.json --layer 23 \
        --degree 60 --degree 90 --degree 150 --degree 220 --degree 300

    curve b0b78d1f63093ba4 at layer 23: 3201 readings from -20 to 12, jitter 1.36e-05
    degree   60: rms residual 1.82e-03, largest 6.93e-03
    degree   90: rms residual 2.56e-04, largest 1.08e-03
    degree  150: rms residual 2.62e-05, largest 1.17e-04
    degree  220: rms residual 1.47e-05, largest 8.31e-05
    degree  300: rms residual 1.42e-05, largest 7.79e-05

Degree 220 is at 1.08 times the jitter and degree 300 adds nothing to it; 150 is at 1.9 and 90 at
19. The cascade is read on 220, 150 and 90. Their tops, and the model's:

    uv run uni critical --map smooth --curve curves/b0b78d1f63093ba4.json --layer 23 \
        --degree 220 --value 4.8 --grid=-8.7486:-8.7446:51

    the map at 4.8 turns at -8.7466417 +- 1.4e-12 (a cubic through 51 states, scatter 1.9e-15), written -8.74664174487155

    uv run uni critical --map response --template summarize --knob formality \
        --text "The committee postponed the vote after two members raised concerns about the budget." \
        --layer 23 --decimals 6 --value 4.8 --grid=-9.25:-8.25:101

    the map at 4.8 turns at -8.7466491 +- 1.6e-05 (a cubic through 101 states, scatter 8.0e-06), written -8.746649

Degree 150's top is -8.746668511874857, on the grid -8.7487:-8.7447:51, and degree 90's
-8.747041615800935, on -8.7490:-8.7450:51. Degree 220's lies 7.4e-6 from the model's, inside the
model's error.

    uv run uni cascade --map smooth --curve curves/b0b78d1f63093ba4.json --layer 23 \
        --degree 220 --critical=-8.74664174487155 --period 2 \
        --grid=3.2099:3.2529:21 --grid=4.2834:4.3263:21 --grid=4.59689:4.60881:21 \
        --grid=4.66539:4.66794:21 --grid=4.680285:4.680841:21 --grid=4.6834891:4.6836085:21 \
        --grid=4.6841759:4.6842015:21 --grid=4.68432303:4.68432851:21 \
        --grid=4.684354536:4.684355710:21 --grid=4.684361285:4.684361536:21 \
        --grid=4.6843627300:4.6843627838:21 --grid=4.68436303953:4.68436305106:21 \
        --grid=4.68436310582:4.68436310829:21

<details>
<summary>Degrees 150 and 90, and the model read directly</summary>

    uv run uni cascade --map smooth --curve curves/b0b78d1f63093ba4.json --layer 23 \
        --degree 150 --critical=-8.746668511874857 --period 2 \
        --grid=3.2099:3.2529:21 --grid=4.2834:4.3263:21 --grid=4.59690:4.60882:21 \
        --grid=4.66539:4.66794:21 --grid=4.680281:4.680837:21 --grid=4.6834857:4.6836052:21 \
        --grid=4.6841727:4.6841983:21 --grid=4.68431990:4.68432538:21 \
        --grid=4.684351415:4.684352590:21 --grid=4.684358166:4.684358417:21 \
        --grid=4.6843596114:4.6843596653:21 --grid=4.68435992108:4.68435993262:21 \
        --grid=4.68435998739:4.68435998987:21

    uv run uni cascade --map smooth --curve curves/b0b78d1f63093ba4.json --layer 23 \
        --degree 90 --critical=-8.747041615800935 --period 2 \
        --grid=3.2100:3.2529:21 --grid=4.2833:4.3262:21 --grid=4.59689:4.60882:21 \
        --grid=4.66541:4.66796:21 --grid=4.680355:4.680913:21 --grid=4.6835641:4.6836837:21 \
        --grid=4.6842514:4.6842770:21 --grid=4.68439860:4.68440409:21 \
        --grid=4.684430131:4.684431305:21 --grid=4.684436883:4.684437135:21 \
        --grid=4.6844383291:4.6844383830:21 --grid=4.68443863888:4.68443865042:21 \
        --grid=4.68443870521:4.68443870768:21

    uv run uni cascade --map response --template summarize --knob formality \
        --text "The committee postponed the vote after two members raised concerns about the budget." \
        --layer 23 --decimals 6 --critical=-8.746649 --period 2 \
        --grid=3.0167:3.4461:21 --grid=4.0902:4.5196:21 --grid=4.5433:4.6624:21 \
        --grid=4.6539:4.6794:21 --grid=4.67778:4.68334:21 --grid=4.682952:4.684146:21

</details>

| period | model, 6 decimals | error | degree 220 | degree 150 | degree 90 |
| -----: | ----------------: | ----: | ---------: | ---------: | --------: |
| 2 | 3.23138673 | 9.3e-06 | 3.231384275 | 3.23139137 | 3.231449606 |
| 4 | 4.304935607 | 1.1e-04 | 4.304863114 | 4.304864354 | 4.304737815 |
| 8 | 4.602851863 | 1.0e-04 | 4.602851324 | 4.602861767 | 4.602854195 |
| 16 | 4.666669552 | 9.7e-06 | 4.666666244 | 4.666661523 | 4.666685684 |
| 32 | 4.680540868 | 7.4e-06 | 4.680563331 | 4.680558968 | 4.680634078 |
| 64 | 4.683522055 | 5.2e-06 | 4.683548836 | 4.683545472 | 4.683623859 |

Through period 16 the model's gains lie within 0.7 of its errors of degree 220's. At periods 32 and
64 they lie 3.0 and 5.2 errors below, 2.2e-5 and 2.7e-5, so by period 64 the model's gain is not the
best fit's.

| spacing ratio over periods | model | degree 220 | degree 150 | degree 90 |
| :------------------------- | ----: | ---------: | ---------: | --------: |
| 2, 4, 8 | 3.6035 +- 2.2e-03 | 3.60242 | 3.60229 | 3.60023 |
| 4, 8, 16 | 4.6682 +- 9.4e-03 | 4.66957 | 4.67082 | 4.67037 |
| 8, 16, 32 | 4.6007 +- 8.8e-03 | 4.59196 | 4.59075 | 4.57626 |
| 16, 32, 64 | 4.6530 +- 1.7e-02 | 4.65485 | 4.65342 | 4.66536 |
| 32, 64, 128 | | 4.66579 | 4.66594 | 4.66907 |
| 64, 128, 256 | | 4.66850 | 4.66851 | 4.66894 |
| 128, 256, 512 | | 4.66906 | 4.66906 | 4.66918 |
| 256, 512, 1024 | | 4.66917 | 4.66917 | 4.66919 |
| 512, 1024, 2048 | | 4.66920 | 4.66920 | 4.66920 |
| 1024, 2048, 4096 | | 4.66920 | 4.66920 | 4.66920 |
| 2048, 4096, 8192 | | 4.66920 | 4.66920 | 4.66920 |

The fitted ratios carry errors of 2.6e-6 to 1.1e-5.

| nearest-point ratio over periods | model | degree 220 | degree 150 | degree 90 |
| :------------------------------- | ----: | ---------: | ---------: | --------: |
| 2, 4 | -2.30778 +- 2.7e-04 | -2.30795 | -2.30796 | -2.30848 |
| 4, 8 | -3.83537 +- 1.6e-03 | -3.83515 | -3.83492 | -3.83272 |
| 8, 16 | -2.20133 +- 1.0e-03 | -2.20141 | -2.20201 | -2.20408 |
| 16, 32 | -2.65070 +- 2.0e-03 | -2.64752 | -2.64707 | -2.63412 |
| 32, 64 | -2.46381 +- 6.3e-03 | -2.44829 | -2.44881 | -2.45201 |
| 64, 128 | | -2.52498 | -2.52498 | -2.52458 |
| 128, 256 | | -2.49407 | -2.49413 | -2.49444 |
| 256, 512 | | -2.50644 | -2.50642 | -2.50633 |
| 512, 1024 | | -2.50150 | -2.50150 | -2.50155 |
| 1024, 2048 | | -2.50347 | -2.50347 | -2.50345 |
| 2048, 4096 | | -2.50268 | -2.50268 | -2.50269 |
| 4096, 8192 | | -2.50300 | -2.50300 | -2.50300 |

| noise growth over periods | degree 220 | degree 150 | degree 90 |
| :------------------------ | ---------: | ---------: | --------: |
| 2, 4 | 4.68648 | 4.68643 | 4.68761 |
| 4, 8 | 10.17848 | 10.17635 | 10.15513 |
| 8, 16 | 5.79115 | 5.79176 | 5.80572 |
| 16, 32 | 6.99716 | 6.99333 | 6.96497 |
| 32, 64 | 6.47350 | 6.47454 | 6.48675 |
| 64, 128 | 6.67711 | 6.67702 | 6.67655 |
| 128, 256 | 6.59563 | 6.59575 | 6.59673 |
| 256, 512 | 6.62835 | 6.62831 | 6.62811 |
| 512, 1024 | 6.61530 | 6.61532 | 6.61544 |
| 1024, 2048 | 6.62053 | 6.62052 | 6.62048 |
| 2048, 4096 | 6.61844 | 6.61844 | 6.61846 |
| 4096, 8192 | 6.61927 | 6.61927 | 6.61927 |

The fitted nearest-point ratios carry errors of 3.5e-7 to 2.1e-6, and the growths 8e-7 to 3.4e-5.

Its early numbers are close to the rewrite loop's, 3.60 and 4.67 against 3.64 and 4.62, -2.31
against -2.17, and growths of 4.69 and 10.18 against 4.48 and 10.05: the two loops share their
knob, and their humps share most of their shape. Through period 64 the model's spacing ratios lie within 1.0 of
their errors of degree 220's and its nearest-point ratios within 2.5.

Its late numbers are the constants. Over 2048, 4096, 8192 the three fits read 4.6692018, 4.6692001
and 4.6692033. Aitken's extrapolation on the last three nearest-point ratios gives -2.5029077,
-2.5029076 and -2.5029081 for degrees 220, 150 and 90, within 3e-7 of alpha = 2.5029079, and on the
last three growths 6.6190364, 6.6190364 and 6.6190368, within 7e-7 of kappa = 6.619037. The fits of a
second loop reach the three constants. The model itself is read to period 64, where its ratios follow
the best fit's within 1.0 and 2.5 of their errors while its gains drift 5.2 errors from the fit's. The
limits are the fits'.

### Rung 5, a second model: SmolLM2-360M

Every model above is Qwen2.5-0.5B-Instruct. `uni/pinned.toml` now pins a second, of another family
and size, HuggingFaceTB/SmolLM2-360M-Instruct at a fixed revision, under the name `smollm2-360m`: a
llama-architecture model with 32 layers where Qwen has 24. Every command that loads a model takes
`--model smollm2-360m`.

The formality contrast is words, so it holds for this model too, and derived on it the direction
lands in the model's own directory:

    uv run uni direction formality --model smollm2-360m

    uni/directions/HuggingFaceTB--SmolLM2-360M-Instruct/formality.json  layer 12  length 33.991301  sha256 4e7e429d36ed74cf242683643442fdc93629bbaa411b564906db7b976b364bf3

Its pushes are held to a narrower range than Qwen's:

    uv run uni response --model smollm2-360m --template rewrite --knob formality         --start "The meeting moved to Thursday because the room was booked."         --grid=-10:10:81 --layer 28

    uni: float32 rounding could move the answer to a push of -10 at layer 28 by 0.0234, past the 0.01 a reading is trusted to; a push this large drowns out what the model writes

The answers of this model run to hundreds and thousands, so the fixed 0.01 a reading is trusted to
binds sooner. Read from -3 to 3:

    uv run uni response --model smollm2-360m --template rewrite --knob formality \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-3:3:241 --layer 20 --layer 24 --layer 28 --layer 31

    layer 20: maximum 2624.9009 at -3; the slope changes sign at []
    layer 24: maximum 2987.2190 at 0.525; the slope changes sign at [-2.075, 0.5250000000000004]
    layer 28: maximum 3875.6792 at 0.825; the slope changes sign at [-2.05, 0.8250000000000002]
    layer 31: maximum 2489.2493 at -3; the slope changes sign at [-1.65, 0.2250000000000001]

![SmolLM2's answer along formality, at four layers](figures/response-306fdda17132fc54.png)

The answer turns, but this push was not followed further. Layer 12 is halfway through Qwen's 24
layers and not yet halfway through this model's 32, so a second contrast pushes the same pairs at
layer 16, halfway through: `uni/directions/formality-16.toml` is formality's template and pairs, word for
word, with `layer = 16`.

    uv run uni direction formality-16 --model smollm2-360m

    uni/directions/HuggingFaceTB--SmolLM2-360M-Instruct/formality-16.json  layer 16  length 40.282390  sha256 69aef4feeeb61e026ee681b8336fbc74c83c1d94459eef4cf081eb5b6a354447

    uv run uni response --model smollm2-360m --template rewrite --knob formality-16 \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-3:3:241 --layer 24 --layer 28 --layer 31

    layer 24: maximum 501.4469 at 0.525; the slope changes sign at [-1.6749999999999998, 0.5250000000000004]
    layer 28: maximum 663.1817 at 0.75; the slope changes sign at [-1.75, 0.75]
    layer 31: maximum 1184.4930 at 0.275; the slope changes sign at [-2.325, 0.27500000000000036]

![SmolLM2's answer along formality-16, at three layers](figures/response-4c6cc8ec5aaa5534.png)

At layer 28 the answer has a top of 663.18 at 0.75, and the loop is followed around it with a positive
gain, as Qwen's was. At the gains where it cascades the orbit stays above -0.5 and below 3, so the
answer is read there every 0.001:

    uv run uni response --model smollm2-360m --template rewrite --knob formality-16 \
        --start "The meeting moved to Thursday because the room was booked." \
        --grid=-0.5:3:3501 --layer 28

![SmolLM2's answer at layer 28, every 0.001 from -0.5 to 3](figures/response-13bff396ae931fc6.png)

    uv run uni smooth --curve curves/b8fd2a0a13062bfd.json --layer 28 \
        --degree 30 --degree 60 --degree 90 --degree 150 --degree 220

    curve b8fd2a0a13062bfd at layer 28: 3501 readings from -0.5 to 3, jitter 3.64e-04
    degree   30: rms residual 4.98e-04, largest 2.02e-03
    degree   60: rms residual 4.85e-04, largest 2.25e-03
    degree   90: rms residual 4.77e-04, largest 2.35e-03
    degree  150: rms residual 4.62e-04, largest 2.49e-03
    degree  220: rms residual 4.43e-04, largest 2.42e-03

The jitter is 27 times Qwen's, on answers some 40 times larger. Every degree from 30 on fits the
curve to within 1.37 times it, and adding terms takes the residual no lower than 1.22 times it, so the
cascade is read on degrees 30, 60 and 90. Their tops, on grids of ±0.002, and the model's:

    uv run uni critical --map smooth --curve curves/b8fd2a0a13062bfd.json --layer 28 \
        --degree 30 --value 6.8 --grid=0.7493:0.7533:51

    the map at 6.8 turns at 0.75126951 +- 1.4e-11 (a cubic through 51 states, scatter 8.7e-14), written 0.7512695071127926

    uv run uni critical --map response --model smollm2-360m --template rewrite --knob formality-16 \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 28 --decimals 6 --value 6.8 --grid=0.651:0.851:101

    the map at 6.8 turns at 0.75126744 +- 4.5e-06 (a cubic through 101 states, scatter 2.0e-06), written 0.751267

Degree 60's top is 0.7512697573580202 and degree 90's 0.7512602554143141, on the same grid.
Degree 30's lies 2.1e-6 from the model's, 0.5 of its error, degree 60's 2.3e-6 and degree 90's 7.2e-6.

    uv run uni cascade --map smooth --curve curves/b8fd2a0a13062bfd.json --layer 28 \
        --degree 30 --critical=0.7512695071127926 --period 2 \
        --grid=5.8095:5.8407:21 --grid=6.5881:6.6193:21 --grid=6.77604:6.78307:21 \
        --grid=6.816918:6.818443:21 --grid=6.825695:6.826022:21 --grid=6.8275758:6.8276459:21 \
        --grid=6.82797864:6.82799365:21 --grid=6.82806492:6.82806814:21 \
        --grid=6.828083405:6.828084093:21 --grid=6.8280873625:6.8280875100:21 \
        --grid=6.8280882102:6.8280882418:21 --grid=6.82808839170:6.82808839847:21 \
        --grid=6.828088430582:6.828088432031:21

<details>
<summary>Degrees 60 and 90, and the model read directly</summary>

    uv run uni cascade --map smooth --curve curves/b8fd2a0a13062bfd.json --layer 28 \
        --degree 60 --critical=0.7512697573580202 --period 2 \
        --grid=5.8095:5.8407:21 --grid=6.5881:6.6193:21 --grid=6.77604:6.78307:21 \
        --grid=6.816916:6.818441:21 --grid=6.825693:6.826021:21 --grid=6.8275743:6.8276444:21 \
        --grid=6.82797719:6.82799221:21 --grid=6.82806348:6.82806670:21 \
        --grid=6.828081960:6.828082648:21 --grid=6.8280859176:6.8280860651:21 \
        --grid=6.8280867652:6.8280867968:21 --grid=6.82808694675:6.82808695352:21 \
        --grid=6.828086985633:6.828086987082:21

    uv run uni cascade --map smooth --curve curves/b8fd2a0a13062bfd.json --layer 28 \
        --degree 90 --critical=0.7512602554143141 --period 2 \
        --grid=5.8096:5.8407:21 --grid=6.5881:6.6193:21 --grid=6.77604:6.78308:21 \
        --grid=6.816910:6.818434:21 --grid=6.825692:6.826019:21 --grid=6.8275736:6.8276437:21 \
        --grid=6.82797660:6.82799162:21 --grid=6.82806291:6.82806613:21 \
        --grid=6.828081401:6.828082090:21 --grid=6.8280853602:6.8280855077:21 \
        --grid=6.8280862081:6.8280862397:21 --grid=6.82808638975:6.82808639651:21 \
        --grid=6.828086428641:6.828086430090:21

    uv run uni cascade --map response --model smollm2-360m --template rewrite --knob formality-16 \
        --text "The meeting moved to Thursday because the room was booked." \
        --layer 28 --decimals 6 --critical=0.751267 --period 2 \
        --grid=5.6694:5.9808:21 --grid=6.4480:6.7594:21 --grid=6.7444:6.8147:21 \
        --grid=6.81005:6.82530:21 --grid=6.82422:6.82749:21 --grid=6.827259:6.827960:21

</details>

| period | model, 6 decimals | error | degree 30 | degree 60 | degree 90 |
| -----: | ----------------: | ----: | --------: | --------: | --------: |
| 2 | 5.825116441 | 2.3e-06 | 5.8251152 | 5.825113385 | 5.825128752 |
| 4 | 6.603670736 | 1.1e-04 | 6.603704226 | 6.603703703 | 6.603691157 |
| 8 | 6.779547947 | 3.0e-05 | 6.779555045 | 6.779556385 | 6.779560213 |
| 16 | 6.81767306 | 8.5e-06 | 6.817680027 | 6.817678666 | 6.817672034 |
| 32 | 6.825872066 | 5.2e-06 | 6.825858445 | 6.825856991 | 6.825855527 |
| 64 | 6.827615 | 7.4e-06 | 6.827610812 | 6.827609364 | 6.827608632 |

Degree 30 lands within 2.6 of the model's errors of every one, and within 3.4e-5; degree 60 within 2.9,
and degree 90 within 5.4, at period 2.

| spacing ratio over periods | model | degree 30 | degree 60 | degree 90 |
| :------------------------- | ----: | --------: | --------: | --------: |
| 2, 4, 8 | 4.4267 +- 3.5e-03 | 4.42755 | 4.42751 | 4.42694 |
| 4, 8, 16 | 4.6132 +- 5.4e-03 | 4.61248 | 4.61286 | 4.61455 |
| 8, 16, 32 | 4.6500 +- 7.5e-03 | 4.66166 | 4.66138 | 4.65716 |
| 16, 32, 64 | 4.7041 +- 2.7e-02 | 4.66707 | 4.66700 | 4.66800 |
| 32, 64, 128 | | 4.66882 | 4.66880 | 4.66904 |
| 64, 128, 256 | | 4.66911 | 4.66911 | 4.66913 |
| 128, 256, 512 | | 4.66918 | 4.66918 | 4.66919 |
| 256, 512, 1024 | | 4.66920 | 4.66920 | 4.66920 |
| 512, 1024, 2048 | | 4.66920 | 4.66920 | 4.66920 |
| 1024, 2048, 4096 | | 4.66920 | 4.66920 | 4.66920 |
| 2048, 4096, 8192 | | 4.66920 | 4.66920 | 4.66919 |

The fitted ratios carry errors of 3.6e-6 to 8.5e-6.

| nearest-point ratio over periods | model | degree 30 | degree 60 | degree 90 |
| :------------------------------- | ----: | --------: | --------: | --------: |
| 2, 4 | -3.10671 +- 5.2e-04 | -3.10654 | -3.10654 | -3.10668 |
| 4, 8 | -2.39486 +- 6.5e-04 | -2.39494 | -2.39492 | -2.39464 |
| 8, 16 | -2.55309 +- 9.3e-04 | -2.55241 | -2.55258 | -2.55350 |
| 16, 32 | -2.48358 +- 2.4e-03 | -2.48540 | -2.48545 | -2.48406 |
| 32, 64 | -2.50292 +- 1.5e-02 | -2.51038 | -2.51035 | -2.51098 |
| 64, 128 | | -2.50003 | -2.50004 | -2.49986 |
| 128, 256 | | -2.50408 | -2.50408 | -2.50416 |
| 256, 512 | | -2.50244 | -2.50245 | -2.50242 |
| 512, 1024 | | -2.50309 | -2.50309 | -2.50311 |
| 1024, 2048 | | -2.50283 | -2.50283 | -2.50283 |
| 2048, 4096 | | -2.50294 | -2.50294 | -2.50294 |
| 4096, 8192 | | -2.50289 | -2.50289 | -2.50289 |

| noise growth over periods | degree 30 | degree 60 | degree 90 |
| :------------------------ | --------: | --------: | --------: |
| 2, 4 | 7.77337 | 7.77345 | 7.77344 |
| 4, 8 | 6.22786 | 6.22784 | 6.22742 |
| 8, 16 | 6.73132 | 6.73155 | 6.73247 |
| 16, 32 | 6.56811 | 6.56815 | 6.56424 |
| 32, 64 | 6.63785 | 6.63775 | 6.63959 |
| 64, 128 | 6.61123 | 6.61124 | 6.61076 |
| 128, 256 | 6.62209 | 6.62208 | 6.62229 |
| 256, 512 | 6.61780 | 6.61781 | 6.61773 |
| 512, 1024 | 6.61953 | 6.61953 | 6.61956 |
| 1024, 2048 | 6.61884 | 6.61884 | 6.61883 |
| 2048, 4096 | 6.61912 | 6.61912 | 6.61912 |
| 4096, 8192 | 6.61900 | 6.61900 | 6.61900 |

The fitted nearest-point ratios carry errors of 5.4e-7 to 2.2e-6, and the growths 2.1e-6 to 3.6e-6.

This hump's early numbers are its own: a first spacing ratio of 4.43 where Qwen's formality hump gave
3.64, a first nearest-point ratio of -3.11 where Qwen's gave -2.17, and a first noise growth of 7.77
where Qwen's gave 4.48. The model's own spacing ratios through period 64 lie within 1.6 of their
errors of every fit's, and its nearest-point ratios within 0.8.

Over 2048, 4096, 8192 the three fits read 4.6691962, 4.6691971 and 4.6691919, 0.5 to 1.3 of their
errors below delta = 4.6692016; over 1024, 2048, 4096 they read 4.6692031, 4.6692037 and 4.6692043.
Aitken's extrapolation on the last three nearest-point ratios gives -2.5029064, -2.5029070 and
-2.5029064 for degrees 30, 60 and 90, within 1.5e-6 of alpha = 2.5029079, from ratios printed with
errors of about 2e-6, and on the last three growths 6.6190359, 6.6190346 and 6.6190350, within 2.4e-6
of kappa = 6.619037. A second model, with other weights, another architecture and another size, gives
fits whose deep ratios reach the same three constants. The model itself is read to period 64, where
its ratios follow the fits' within their errors. The limits are the fits'.

## Running on the experiment host

Every `uni` command accepts `--remote`. With it, this working tree, uncommitted edits
included, is mirrored to a host with rsync and the same command runs there over ssh,
streaming its output back. Without it, the command runs here.

`uni temperature` writes its curves into this checkout, so on the host the directory it wrote
them into comes back once it succeeds, beside the curves already here: each file is named by its
own content, so nothing here is overwritten. `uni plot` and `uni response` write figures, which do
not come back, and are refused with `--remote`.

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
