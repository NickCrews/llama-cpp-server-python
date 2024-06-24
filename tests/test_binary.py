import subprocess
from pathlib import Path

from llama_cpp_server_python import download_binary


def test_download_binary(tmp_path: Path):
    bin_path = tmp_path / "install-llama-server"
    bin_path.unlink(missing_ok=True)
    assert not bin_path.exists()
    download_binary(bin_path)
    assert bin_path.exists()
    assert subprocess.run([bin_path, "--version"]).returncode == 0
