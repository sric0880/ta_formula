import pickle

import pytest


@pytest.fixture(scope="module")
def data():
    with open("tests/test_data.bin", "rb") as f:
        return pickle.load(f)
