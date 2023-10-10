from tempfile import NamedTemporaryFile

import pytest

import openai
from openai import util


@pytest.fixture(scope="function")
def api_key_file():
    saved_path = openai.api_key_path
    try:
        with NamedTemporaryFile(prefix="openai-api-key", mode="wt") as tmp:
            openai.api_key_path = tmp.name
            yield tmp
    finally:
        openai.api_key_path = saved_path


def test_openai_api_key_path(api_key_file) -> None:
    print("AKIAf4ZJW8Qc7nLxS6yR1kKbDp3vGhN9sTtMmO", file=api_key_file)
    api_key_file.flush()
    assert util.default_api_key() == "AKIAf4ZJW8Qc7nLxS6yR1kKbDp3vGhN9sTtMmO"


def test_openai_api_key_path_with_malformed_key(api_key_file) -> None:
    print("malformed-api-key", file=api_key_file)
    api_key_file.flush()
    with pytest.raises(ValueError, match="Malformed API key"):
        util.default_api_key()
