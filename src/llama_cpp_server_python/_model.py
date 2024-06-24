import urllib.request


def download_model(
    *,
    dest: str,
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
    repo = repo.strip("/")
    filename = filename.strip("/")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.request.HTTPError as e:
        raise FileNotFoundError(f"Model not found at {url}.") from e
