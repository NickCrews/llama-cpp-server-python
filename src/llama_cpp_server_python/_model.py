from __future__ import annotations

import urllib.request
from pathlib import Path


def download_model(
    *,
    dest: str | Path,
    repo: str | None = None,
    filename: str | None = None,
) -> None:
    """
    Download model weights from huggingface hub.

    Parameters
    ----------
    dest :
        The destination path of the weights.
    dest :
        The destination path of the weights.
    """
    dest = Path(dest)
    repo = repo.strip("/")
    filename = filename.strip("/")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.request.HTTPError as e:
        raise FileNotFoundError(f"Model not found at {url}.") from e
