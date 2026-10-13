import pytest

from cavalier.scenario import load_scenario


@pytest.fixture(scope="session")
def bundle():
    return load_scenario("cuba1962")
