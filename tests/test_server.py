import json
from pathlib import Path
from urllib import request

import pytest
from llama_cpp_server_python import Server, download_binary, download_model

TEST_DIR = Path(__file__).parent.parent / "./.test"

REPO = "afrideva/Tinystories-gpt-0.1-3m-GGUF"
FILENAME = "tinystories-gpt-0.1-3m.Q4_K_M.gguf"


@pytest.fixture
def binary_path() -> Path:
    path = TEST_DIR / "llama-server"
    if not path.exists():
        download_binary(path)
    return path


@pytest.fixture
def model_path() -> Path:
    path = TEST_DIR / FILENAME
    if not path.exists():
        download_model(dest=path, repo=REPO, filename=FILENAME)
    return path


def _check_server(server: Server):
    params = {
        "messages": [{"role": "user", "content": "Say 'hello'"}],
        "max_tokens": 10,
    }
    req = request.Request(
        f"{server.base_url}/v1/chat/completions",
        data=json.dumps(params).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = request.urlopen(req)
    data = json.loads(resp.read())
    assert "choices" in data


@pytest.mark.parametrize("port", [8080, 6000])
# most of these are smoketests we can't *really* test, but at least
# we can check that the server responds.
@pytest.mark.parametrize("ctx_size", [512, 1024])
@pytest.mark.parametrize("parallel", [4, 8])
@pytest.mark.parametrize("cont_batching", [True, False])
def test_basic(binary_path: None, model_path, port, ctx_size, parallel, cont_batching):
    with Server(
        binary_path=binary_path,
        model_path=model_path,
        port=port,
        ctx_size=ctx_size,
        parallel=parallel,
        cont_batching=cont_batching,
    ) as server:
        assert server.base_url == "http://127.0.0.1:" + (str(port) or "8080")
        _check_server(server)


def test_no_resources(binary_path, model_path):
    with pytest.raises(FileNotFoundError):
        Server(binary_path="bad_path", model_path="bad_path")
    with pytest.raises(FileNotFoundError):
        Server(binary_path=binary_path, model_path="bad_path")
    with pytest.raises(FileNotFoundError):
        Server(binary_path="bad_path", model_path=model_path)


def test_bad_binary(tmp_path: Path, model_path):
    binary_path = tmp_path / "bad_server"
    binary_path.write_text("bad server")
    server = Server(binary_path=binary_path, model_path=model_path)
    with pytest.raises(PermissionError):
        server.start()


def test_bad_model(tmp_path: Path, binary_path):
    model_path = tmp_path / "bad_model.gguf"
    model_path.write_text("bad model")
    server = Server(binary_path=binary_path, model_path=model_path)
    server.start()
    with pytest.raises(RuntimeError):
        server.wait_for_ready()


def test_from_huggingface(tmp_path):
    with Server.from_huggingface(
        repo=REPO, filename=FILENAME, working_dir=tmp_path
    ) as server:
        assert server.base_url == "http://127.0.0.1:8080"
        _check_server(server)
