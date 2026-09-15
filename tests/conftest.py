from dataclasses import replace
from typing import get_args

import pytest

from uni.pinned import Device, load_pinned


def pytest_addoption(parser):
    parser.addoption("--device", choices=get_args(Device), default=load_pinned().device, help="device the model tests run on (default: the pinned device)")


@pytest.fixture(scope="session")
def model(request):
    # Imported here so the tests that never touch the model never load torch.
    from uni.model import Model

    return Model(replace(load_pinned(), device=request.config.getoption("--device")))
