"""Observables read back off a written trajectory, and what a trajectory has to carry for them."""

import dataclasses
import statistics
from pathlib import Path

import pytest
import torch

from uni.cli import main
from uni.loop import Trajectory, write_trajectory
from uni.maps import model_spec
from uni.observe import (
    Length,
    Logprob,
    ObserveError,
    Projection,
    Step,
    Value,
    Weights,
    identities,
    observables,
    readings,
    steering_additions,
    steering_directions,
    steps,
    template_of,
    written_by,
)
from uni.pinned import load_pinned
from uni.steer import Steer, read_direction
from uni.template import load_templates

TEXT = "The store will open late tomorrow because of the storm, so plan your trip around noon."


def spec(template="rewrite", knob=None, kind="model", pinned=None):
    # What ModelMap.spec writes, so a test file is a real one; `kind` stands in for maps this build cannot read.
    return {**model_spec(pinned or load_pinned(), load_templates()[template], knob), "kind": kind}


@pytest.fixture(scope="module")
def formality():
    return read_direction("formality", load_pinned())


def test_each_step_carries_the_state_it_came_from():
    assert steps(Trajectory(spec(), 0.0, "a", ("b", "c"))) == (Step(1, "a", "b"), Step(2, "b", "c"))


def test_a_trajectory_of_no_steps_has_none():
    assert steps(Trajectory(spec(), 0.0, "a", ())) == ()


def test_states_are_numbered_by_where_they_first_appeared():
    assert identities(["a", "b", "a", "c", "b"]) == (1, 2, 1, 3, 2)


def test_the_template_comes_back_off_the_trajectory():
    assert template_of(Trajectory(spec(), 0.0, "a", ())) == load_templates()["rewrite"]


def test_a_trajectory_from_another_kind_of_map_has_no_model_observables(weights):
    # And costs no checkpoint to find that out: what an orbit can be read for is decided by the
    # kind its file records, before anything that needs weights is built.
    assert observables(Trajectory({"kind": "logistic"}, 0.0, "a", ()), Weights()) == (Length(), Value())


def test_an_orbit_of_a_kind_this_build_cannot_read_is_refused():
    # Answering with the length alone would read as a full reading of a file nothing here knows.
    with pytest.raises(ObserveError, match="'henon' orbit is not one this build can read"):
        observables(Trajectory({"kind": "henon"}, 0.0, "a", ()), Weights())


def test_a_trajectory_that_recorded_no_template_is_refused():
    with pytest.raises(ObserveError, match="template is missing"):
        template_of(Trajectory({"kind": "model"}, 0.0, "a", ()))


def test_the_pinned_model_is_handed_back_when_it_is_the_one_that_wrote_the_orbit():
    assert written_by(Trajectory(spec(), 0.0, "a", ()), load_pinned()) == load_pinned()


def test_an_orbit_written_on_another_checkpoint_is_refused():
    # Every reading would be a real number about a model that never saw this orbit.
    other = dataclasses.replace(load_pinned(), revision="0" * 40)
    with pytest.raises(ObserveError, match="written on"):
        written_by(Trajectory(spec(pinned=other), 0.0, "a", ()), load_pinned())


def test_a_longer_generation_limit_leaves_an_orbit_readable(weights):
    # The limit bounds how long a state may be; it is not part of what the weights score it as.
    longer = dataclasses.replace(load_pinned(), max_new_tokens=load_pinned().max_new_tokens * 2)
    assert written_by(Trajectory(spec(pinned=longer), 0.0, "a", ()), load_pinned()) == load_pinned()


def test_an_unsteered_run_has_no_direction_to_project_onto():
    assert steering_directions(Trajectory(spec(), 0.0, "a", ()), load_pinned()) == ()


def test_a_trajectory_that_has_lost_its_knob_is_not_read_as_unsteered(formality):
    # Recorded null, an unsteered run says so. With the field gone the file says nothing, and
    # reading that as nothing-was-turned scores a steered orbit on a model that was not turned.
    lost = {key: value for key, value in spec().items() if key != "knob"}
    with pytest.raises(ObserveError, match="knob is missing"):
        steering_directions(Trajectory(lost, 0.0, "a", ()), load_pinned())


def test_a_steered_run_hands_back_the_direction_that_steered_it(formality):
    knob = Steer(formality).turn(2.0).spec
    assert steering_directions(Trajectory(spec(knob=knob), 2.0, "a", ()), load_pinned()) == (formality,)


def test_a_direction_that_has_changed_since_the_run_is_refused(formality):
    knob = dict(Steer(formality).turn(2.0).spec, sha256="0" * 64)
    with pytest.raises(ObserveError, match="has changed since it steered this run"):
        steering_directions(Trajectory(spec(knob=knob), 2.0, "a", ()), load_pinned())


def test_an_unsteered_run_adds_nothing_to_the_model_it_is_read_under():
    assert steering_additions((), 0.0) == ()


def test_the_additions_are_the_ones_the_knob_made_during_the_run(formality):
    # Rebuilt by Steer itself, so the value the file records reaches the model as it did then.
    (addition,) = steering_additions((formality,), 2.0)
    assert addition.layer == formality.contrast.layer
    assert torch.equal(addition.vector, 2.0 * torch.tensor(formality.vector))


def test_length_counts_the_characters_of_the_state():
    assert Length().read(Step(1, "anything", "hello")) == 5.0


def test_the_logprob_is_the_one_the_model_reported_as_it_generated(weights, model):
    template = load_templates()["rewrite"]
    generation = model.generate(template.render(TEXT))
    assert generation.token_ids[-1] in model.stop_ids  # so dropping the last logprob drops the stop token
    reading = Logprob(weights, template, ()).read(Step(1, TEXT, generation.text))
    # Close, not equal: generation scores each token behind a growing cache and this scores them
    # in one pass, which the README already records as a sixth-decimal difference.
    assert reading == pytest.approx(statistics.fmean(generation.logprobs[:-1]), abs=1e-4)


def test_an_observable_is_periodic_when_the_orbit_is(weights, model):
    # At a fixed point every step re-sends the same prompt for the same reply, so every reading
    # after the onset is the same number. A sweep locates a bifurcation by watching that fact
    # break, so an observable carrying anything from one call to the next would ruin it.
    trajectory = Trajectory(spec("identity"), 0.0, "hello", ("hello",) * 4)
    numbers = [Logprob(weights, load_templates()["identity"], ()).read(step) for step in steps(trajectory)]
    assert len(set(numbers)) == 1


def test_a_steered_reply_is_scored_by_the_model_the_knob_turned(weights, model, formality):
    # The knob is part of the map: read without it, the number belongs to a model that did not
    # write this reply, and it is a plausible number either way.
    template = load_templates()["rewrite"]
    turned = Steer(formality).turn(2.0)
    generation = model.generate(template.render(TEXT), turned.additions)
    # Steered this far the reply runs to the token budget instead of stopping, so unlike the
    # unsteered case above there is no stop token's log-probability to leave out.
    reported = generation.logprobs[:-1] if generation.token_ids[-1] in model.stop_ids else generation.logprobs
    step = Step(1, TEXT, generation.text)
    reading = Logprob(weights, template, turned.additions).read(step)
    assert reading == pytest.approx(statistics.fmean(reported), abs=1e-4)
    assert reading != Logprob(weights, template, ()).read(step)


def test_a_state_that_cannot_be_read_says_which_step_it_is_at(weights, model):
    # An empty state encodes to no tokens, and a half-printed table needs an address.
    step = Step(2, "hello", "")
    with pytest.raises(ObserveError, match="step 2: the reply encodes to no tokens"):
        readings((Logprob(weights, load_templates()["identity"], ()),), step)


def test_the_projection_separates_the_contrast_the_direction_was_built_from(weights, model, formality):
    pair = formality.contrast.pairs[0]
    projection = Projection(weights, formality.contrast.template, formality)
    assert projection.read(Step(1, pair.text, pair.toward)) > projection.read(Step(1, pair.text, pair.away))


def test_the_projection_names_the_direction_it_reads_along(weights, model, formality):
    assert Projection(weights, formality.contrast.template, formality).name == "along:formality"


def test_the_command_reads_a_trajectory_back(tmp_path, capsys):
    path = write_trajectory(Trajectory(spec("identity"), 0.0, "hello", ("hello", "hello")), tmp_path)
    assert main(["observe", str(path)], {}, Path.cwd()) == 0
    printed = capsys.readouterr().out.splitlines()
    # The verdict is read off the states alone, so it is printed before the model is loaded.
    assert printed[0] == "period 1, entered at step 0"
    assert "length" in printed[2] and "logprob" in printed[2]
    assert printed[3].split() == ["0", "1", "-", "-"]  # the start was given, not stepped into
    assert printed[4].split()[:2] == ["1", "1"]


def test_the_command_refuses_a_file_that_is_not_a_trajectory(tmp_path, capsys):
    path = tmp_path / "nope.json"
    path.write_text("{}")
    assert main(["observe", str(path)], {}, Path.cwd()) != 0
