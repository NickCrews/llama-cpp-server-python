from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from llama_cpp_server_python._binary import download_binary
from llama_cpp_server_python._model import download_model

logger = logging.getLogger(__name__)


class Server:
    """
    Wrapper around the llama-server binary.

    Examples
    --------
    >>> from openai import OpenAI
    >>> from llama_cpp_server_python import Server
    >>> repo = "Qwen/Qwen2-0.5B-Instruct-GGUF"
    >>> filename = "qwen2-0_5b-instruct-q4_0.gguf"
    >>> with Server.from_huggingface(repo=repo, filename=filename) as server:
    ...     client = OpenAI(base_url=server.base_url)
    ...     pass  # interact with the client

    For more control over the server, you can download the model and binary
    separately, and pass in other parameters:

    >>> binary_path = "path/to/llama-server"
    >>> model_path = "path/to/model.gguf"
    >>> server = Server(binary_path=binary_path, model_path=model_path, port=6000, ctx_size=1024)
    >>> server.start()
    >>> client = OpenAI(base_url=server.base_url)
    >>> pass  # interact with the client
    >>> server.stop() # or use a context manager as above
    """

    def __init__(
        self,
        *,
        binary_path: str | Path,
        model_path: str | Path,
        port: int = 8080,
        ctx_size: int | None = None,
        parallel: int | None = None,
        cont_batching: bool = True,
    ) -> None:
        """
        Create a server instance (but don't start it yet).

        For details on most parameters see
        https://github.com/ggerganov/llama.cpp/tree/master/examples/server

        Parameters
        ----------
        binary_path :
            The path to the llama-server binary.
        model_path :
            The path to the model weights file.
            Must be a .gguf file, per the llama.cpp model format.
        port :
            The port to run the server on.
        ctx_size :
            The context size of the model.
        parallel :
            The number of parallel requests to handle.
        cont_batching :
            Whether to use continuous batching.
        """
        self.binary_path = Path(binary_path)
        self.model_path = Path(model_path)
        self.port = port
        self.ctx_size = ctx_size
        self.parallel = parallel
        self.cont_batching = cont_batching

        self.process = None
        self._check_resources()

    @classmethod
    def from_huggingface(
        cls, *, repo: str, filename: str, working_dir: str | Path = "./llama"
    ) -> "Server":
        """Create a server from a HuggingFace model repository.

        If you need more control, download the model and binary separately,
        and then call the constructor directly.

        Parameters
        ----------
        repo :
            The HuggingFace model repository, eg "Qwen/Qwen2-0.5B-Instruct-GGUF".
        filename :
            The filename of the model weights, eg "qwen2-0_5b-instruct-q4_0.gguf".
        working_dir :
            The working directory to download the model and server binary to.

        Returns
        -------
        Server
        """
        working_dir = Path(working_dir)
        binary_path = working_dir / "llama-server"
        model_path = working_dir / filename
        if not binary_path.exists():
            download_binary(binary_path)
        if not model_path.exists():
            download_model(dest=model_path, repo=repo, filename=filename)
        return cls(binary_path=binary_path, model_path=model_path)

    @property
    def base_url(self) -> str:
        """The base URL of the server, e.g. 'http://localhost:8080'."""
        return f"http://localhost:{self.port}"

    def start(self) -> None:
        """Start the server in a subprocess. Blocks until the server is ready.

        Pair this with a .stop() call when you are done.
        Or, use a context manager with 'with Server(...) as server: ...'
        to automatically start and stop the server.

        You can start and stop the server multiple times in a row.
        """
        if self.process is not None:
            raise RuntimeError("Server is already running.")
        self._check_resources()
        logger.info(f"Starting server with command: '{' '.join(self._command)}'...")
        self.process = subprocess.Popen(
            self._command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._wait_for_start()

    def stop(self) -> None:
        """Terminate the server subprocess. No-op if there is no active subprocess."""
        if self.process is None:
            return
        self.process.kill()
        self.process = None

    def __enter__(self):
        """Start the server when entering a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the server when exiting a context manager."""
        self.stop()

    @property
    def _command(self) -> list[str]:
        cmd = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--port",
            f"{self.port}",
        ]
        if self.ctx_size is not None:
            cmd.extend(["--ctx_size", f"{self.ctx_size}"])
        if self.parallel is not None:
            cmd.extend(["--parallel", f"{self.parallel}"])
        if self.cont_batching:
            cmd.append("--cont_batching")
        return cmd

    def _check_resources(self) -> None:
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Server binary not found at {self.binary_path}.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}.")

    def _wait_for_start(self, timeout: int = 5) -> None:
        if self.process is None:
            return
        start = time.time()
        outs, errs = b"", b""
        while time.time() - start < timeout:
            try:
                outs, errs = self.process.communicate(timeout=0.01)
            except subprocess.TimeoutExpired:
                if "HTTP server listening" in errs.decode():
                    return
            else:
                raise RuntimeError(
                    f"Server exited unexpectedly with code {self.process.returncode}."
                    f" stdout: {outs.decode()}"
                    f" stderr: {errs.decode()}"
                )
