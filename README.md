# universality

Does Feigenbaum universality apply to LLM feedback loops? The founding document is
[PROJECT.md](PROJECT.md); this file is how to run the code.

## Setup

Everything runs with [uv](https://docs.astral.sh/uv/).

    uv run uni gen "hello"

That one command installs everything and generates from the pinned model. The first run
downloads the checkpoint, about a gigabyte, into the Hugging Face cache. It prints the
text, its sha256, and each generated token with its log-probability.

The model, its revision, dtype, and generation limit are pinned in
[uni/pinned.toml](uni/pinned.toml), and nothing else in the code names them. Generation
runs on Metal, greedy at batch size one; the checkpoint's own sampling settings are
ignored. CPU is too slow for this work, so it is not an option.

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
the whole prompt, and `rewrite` asks for a rewrite. The start may be empty.

`--map` chooses which map is iterated, and defaults to the model. Whichever it is, the runner
only ever calls `step(state)` on it, so nothing below the command line knows there is more than
one. `--value` is that map's parameter: the knob's setting for the model, and r for the logistic
map below. Each map reads the flags that describe it and refuses the other's, so a `--knob`
carried over from a model run does not ride along unapplied on an orbit of numbers.

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

This writes `uni/directions/formality.json`, which holds the vector, a copy of the
contrast that produced it, and the checkpoint it was read from, and is committed. Every
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

A sweep runs each cell exactly once, so a grid or a set of starts that names one twice is refused
rather than run twice into one file, and so is a grid the map cannot take — every value is offered
to the map and to the knob before the first cell runs, because a sweep that dies two hundred cells
in dies again on every resume.

One failure those checks cannot cover: a model's states are its own replies, so they can grow
until the rendered state leaves no room to generate, and whether step 300 still fits is knowable
only by running to step 300. So a cell the map refuses **partway through its orbit** does not stop
the sweep. The run says so on that cell's line, carries on to the cells after it — which are
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
some cells and was then stopped outright exits `78`, because what ended it outranks how far it
got, and its refusals are on stderr rather than in the code. And the states it did produce are
thrown away rather than written under their own shorter name, which would put a file in the
directory this sweep never named and no rerun would ever look for. Both are the price of a sweep
directory that holds finished orbits and nothing else; `universality-sweep-81g` carries the
question of whether a sweep should be able to say "tried and cannot" somewhere.

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
    rsync --archive "$UNI_REMOTE_USER@$UNI_REMOTE_HOST:$UNI_REMOTE_DIR/sweeps/1a8fce648065056f/" \
        sweeps/1a8fce648065056f/
    uv run uni plot sweeps/1a8fce648065056f --observable along:formality --burn-in 10

The middle line is not decoration. The sweep ran on the host and its cells stay there - `sweeps/`
is excluded from the sync, and the sync has no leg coming back - so the directory has to be
brought home before anything here can draw it. Plotting is a local command by design: a figure
is an output this repo commits, and drawing one on the host puts it where no commit can reach
it, which is also why `figures/` is excluded from the sync rather than deleted by it. `uni plot
--remote` is refused for the same reason rather than left to draw somewhere unreachable and exit
0 - it is the one command that says where its answer lands. The sweep's name is the same on both
machines, because it is the hash of what the sweep is. There is no `uni fetch` doing the middle
line for you yet; it is filed as `universality-remote-lhh`.

![orbit diagram of the rewrite loop along the formality direction](figures/1a8fce648065056f-along-formality-burn10-orbit.png)

The knob works, and monotonically: where the settled state sits along the formality direction
rises steadily from about -4 at a coefficient of -6 to about +5 at +3.5, and then stops rising.
That is the knob doing what a knob should.

What the picture does not show is a cascade. Over most of the range each coefficient carries a
single dot, which is an orbit that has reached a fixed point: the model rewrites a text into
itself. The periods are measured rather than eyeballed — `uni observe` reports each one — and
across the 25 cells they are 19 fixed points, a period 2 at -0.5, 1.0 and 1.5, a period 3 at 0.5,
a period 4 at -6, and one orbit at +6 that had not repeated within its 31 states. The cycles
longer than one sit around the unsteered point and at the far ends, not in a doubling sequence,
and 0.5 apart on the knob is far too coarse a grid to call any of it a bifurcation.

![return map of the rewrite loop along the formality direction](figures/1a8fce648065056f-along-formality-burn10-return.png)

The return map says the same thing in one line: the points lie on the diagonal. Rung 1 of
PROJECT.md asks whether this map has one smooth hump, because that is the shape the whole theory
rests on. This is not that shape — it is the identity, which is what a return map of fixed points
looks like. A hump needs states that move.

![orbit diagram of the rewrite loop read for length](figures/1a8fce648065056f-length-burn10-orbit.png)

Read for the character length of the state instead, the same sweep shows the knob's real effect
on this loop: near zero the fixed point is a single tidy sentence of about 60 characters, and
steering in either direction inflates it by more than twenty times.

**And that is the caveat this sweep has to carry.** The pinned `generation.max_new_tokens` is
256, and **446 of the 750 states in this sweep are at or over that ceiling**. So over much of the
range the map being iterated is not the rewrite loop but the rewrite loop truncated, and a fixed
point reached by filling the budget every step is a fixed point of the ceiling as much as of the
model. Read the middle of the picture, where the states are short, and treat the wings as
measuring the cap. Filed as `universality-sweep-zjh`.

What this sweep settles is therefore narrow and worth stating plainly: at this template, this
direction, this start, this step count and this resolution, the rewrite loop does not period
double. It converges. Whether a finer grid, a longer run, more starts, or a knob that does not
drive the model into the token ceiling would show anything else is the next question, and it is
the question Rung 1 exists to ask.

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
