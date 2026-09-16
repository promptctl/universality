import pytest


@pytest.fixture(scope="session")
def weights():
    # What the observables that need a checkpoint hold. Made once, so the whole session reads the
    # weights once however many tests ask for them.
    from uni.observe import Weights

    return Weights()


@pytest.fixture(scope="session")
def model(weights):
    # The loaded model itself, for the tests that call it directly rather than through an
    # observable. Taken from `weights` so it is the same one, loaded once.
    return weights.model
