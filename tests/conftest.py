import pytest


@pytest.fixture(scope="session")
def weights():
    # What the observables that need a checkpoint hold. Made once, so the whole session reads the
    # weights once however many tests ask for them.
    from uni.observe import Weights
    from uni.pinned import load_pinned

    return Weights(load_pinned())


@pytest.fixture(scope="session")
def model(weights):
    # The loaded model itself, for the tests that call it directly rather than through an
    # observable. Taken from `weights` so it is the same one, loaded once.
    return weights.model


@pytest.fixture(scope="session")
def budgeted(model):
    # The session's checkpoint under a smaller pinned budget. generate reads the budget from the
    # pin, so a shallow copy is the same weights on the same device, not a second checkpoint.
    from copy import copy
    from dataclasses import replace

    def under(tokens):
        small = copy(model)
        small.pinned = replace(model.pinned, max_new_tokens=tokens)
        return small

    return under
