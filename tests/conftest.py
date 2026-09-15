import pytest

from uni.pinned import load_pinned


@pytest.fixture(scope="session")
def model():
    # Imported here so the tests that never touch the model never load torch.
    from uni.model import Model

    return Model(load_pinned())
