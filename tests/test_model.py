from pathlib import Path

from llama_cpp_server_python import download_model


def test_download_model(tmp_path: Path):
    repo = "Qwen/Qwen2-7B"
    filename = "README.md"
    dest = tmp_path / filename
    assert not dest.exists()
    download_model(dest=dest, repo=repo, filename=filename)
    assert dest.exists()
    assert "MMLU" in dest.read_text()
