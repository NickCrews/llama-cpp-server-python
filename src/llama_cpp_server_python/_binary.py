from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

_SCRIPT = """
curl -s https://raw.githubusercontent.com/NickCrews/llama.cpp/dcd044206696c4c8ec0620f2376f80daa608179b/download-release.sh | bash -s -- {dest_dir} {options}
"""

logger = logging.getLogger(__name__)


def download_binary(
    dest: str | Path,
    *,
    tag: str = "latest",
    filename: str | None = None,
    os: str | None = None,
    arch: str | None = None,
    backend: str | None = None,
):
    """
    Download the llama.cpp server binary.

    Uses https://raw.githubusercontent.com/NickCrews/llama.cpp/dcd044206696c4c8ec0620f2376f80daa608179b/download-release.sh
    to download the binary.

    Parameters
    ----------
    dest :
        The destination path of the binary.
    tag :
        The tag of the release to download.
    filename :
        The filename of the release to download.
    os :
        The operating system of the release to download.
    arch :
        The architecture of the release to download.
    """
    options = [f"--tag={tag}"]
    if filename is not None:
        options.append(f"--filename={filename}")
    if os is not None:
        options.append(f"--os={os}")
    if arch is not None:
        options.append(f"--arch={arch}")
    if backend is not None:
        options.append(f"--backend={backend}")
    options_str = " ".join(options)
    with tempfile.TemporaryDirectory() as tmpdir:
        script = _SCRIPT.format(dest_dir=tmpdir, options=options_str)
        logger.info(f"Downloading llama.cpp server binary using '{script}'")
        completed = subprocess.run(script, shell=True, check=True, capture_output=True)
        logger.info(completed.stdout)
        logger.info(completed.stderr)
        raw_path = tmpdir + "/llama-server"
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(raw_path, dest)
