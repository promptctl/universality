"""Observables read back off a written trajectory, and what a trajectory has to carry for them."""

import statistics
from pathlib import Path

import pytest

from uni.cli import main
from uni.loop import Trajectory, write_trajectory
from uni.observe import Length, Logprob, ObserveError, Projection, Step, identities, steering_directions, steps, template_of
from uni.pinned import load_pinned
from uni.steer import Steer, read_direction
from uni.template import load_templates

TEXT = "The store will open late tomorrow because of the storm, so plan your trip around noon."


def spec(template="rewrite", knob=None, kind="model"):
    parsed = load_templates()[template]
    return {"kind": kind, "template": {"name": parsed.name, "text": parsed.text}, "pinned": {}, "knob": knob}


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


def test_a_trajectory_from_another_kind_of_map_has_no_model_observables():
    with pytest.raises(ObserveError, match="no prompts behind its states"):
        template_of(Trajectory({"kind": "logistic"}, 0.0, "a", ()))


def test_a_trajectory_that_recorded_no_template_is_refused():
    with pytest.raises(ObserveError, match="template is missing"):
        template_of(Trajectory({"kind": "model"}, 0.0, "a", ()))


def test_an_unsteered_run_has_no_direction_to_project_onto():
    assert steering_directions(Trajectory(spec(), 0.0, "a", ()), load_pinned()) == ()


def test_a_steered_run_hands_back_the_direction_that_steered_it(formality):
    knob = Steer(formality).turn(2.0).spec
    assert steering_directions(Trajectory(spec(knob=knob), 2.0, "a", ()), load_pinned()) == (formality,)


def test_a_direction_that_has_changed_since_the_run_is_refused(formality):
    knob = dict(Steer(formality).turn(2.0).spec, sha256="0" * 64)
    with pytest.raises(ObserveError, match="has changed since it steered this run"):
        steering_directions(Trajectory(spec(knob=knob), 2.0, "a", ()), load_pinned())


def test_length_counts_the_characters_of_the_state():
    assert Length().read(Step(1, "anything", "hello")) == 5.0


def test_the_logprob_is_the_one_the_model_reported_as_it_generated(model):
    template = load_templates()["rewrite"]
    generation = model.generate(template.render(TEXT))
    assert generation.token_ids[-1] in model.stop_ids  # so dropping the last logprob drops the stop token
    reading = Logprob(model, template).read(Step(1, TEXT, generation.text))
    # Close, not equal: generation scores each token behind a growing cache and this scores them
    # in one pass, which the README already records as a sixth-decimal difference.
    assert reading == pytest.approx(statistics.fmean(generation.logprobs[:-1]), abs=1e-4)


def test_an_observable_is_periodic_when_the_orbit_is(model):
    # At a fixed point every step re-sends the same prompt for the same reply, so every reading
    # after the onset is the same number. A sweep locates a bifurcation by watching that fact
    # break, so an observable carrying anything from one call to the next would ruin it.
    trajectory = Trajectory(spec("identity"), 0.0, "hello", ("hello",) * 4)
    readings = [Logprob(model, load_templates()["identity"]).read(step) for step in steps(trajectory)]
    assert len(set(readings)) == 1


def test_the_projection_separates_the_contrast_the_direction_was_built_from(model, formality):
    pair = formality.contrast.pairs[0]
    projection = Projection(model, formality.contrast.template, formality)
    assert projection.read(Step(1, pair.text, pair.toward)) > projection.read(Step(1, pair.text, pair.away))


def test_the_projection_names_the_direction_it_reads_along(model, formality):
    assert Projection(model, formality.contrast.template, formality).name == "along:formality"


def test_the_command_reads_a_trajectory_back(tmp_path, capsys):
    path = write_trajectory(Trajectory(spec("identity"), 0.0, "hello", ("hello", "hello")), tmp_path)
    assert main(["observe", str(path)], {}, Path.cwd()) == 0
    printed = capsys.readouterr().out
    assert "period 1, entered at step 0" in printed
    assert "length" in printed and "logprob" in printed


def test_the_command_refuses_a_file_that_is_not_a_trajectory(tmp_path, capsys):
    path = tmp_path / "nope.json"
    path.write_text("{}")
    assert main(["observe", str(path)], {}, Path.cwd()) != 0
