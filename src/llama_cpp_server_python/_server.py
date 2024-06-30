from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from llama_cpp_server_python._binary import download_binary
from llama_cpp_server_python._model import download_model

if TYPE_CHECKING:
    import io


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
    >>> server.wait_for_ready()
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
        logger: logging.Logger | None = None,
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
        logger :
            A logger to use for logging server output.
            If None, a new logger is created.
            You can configure the logger with handlers, formatters, etc. after
            creating the server as needed.
        """
        self.binary_path = Path(binary_path)
        self.model_path = Path(model_path)
        self.port = port
        self.ctx_size = ctx_size
        self.parallel = parallel
        self.cont_batching = cont_batching

        self._logging_threads = []
        self._status = "stopped"
        self.process = None
        self._check_resources()

        if logger is None:
            logger = logging.getLogger(__name__ + ".Server" + str(self.port))
        self.logger = logger

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

    @property
    def status(self) -> str:
        """The status of the server: 'stopped', 'starting', or 'running'."""
        return self._status

    def start(self) -> None:
        """Start the server in a subprocess.

        This returns immediately. If you want to wait for the server to be ready,
        call 'wait_for_start()' after this.

        Pair this with a .stop() call when you are done.
        Or, use a context manager with 'with Server(...) as server: ...'
        to automatically start and stop the server.

        You can start and stop the server multiple times in a row.
        """
        if self.process is not None:
            raise RuntimeError("Server is already running.")
        self._check_resources()
        self.logger.info(
            f"Starting server with command: '{' '.join(self._command)}'..."
        )
        self.process = subprocess.Popen(
            self._command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self._status = "starting"
        self._logging_threads = self._watch_outputs()

    def stop(self) -> None:
        """Terminate the server subprocess. No-op if there is no active subprocess."""
        if self.process is None:
            return
        self.process.kill()
        for thread in self._logging_threads:
            thread.join()
        self._status = "stopped"
        self.process = None

    def __enter__(self):
        """Start the server when entering a context manager."""
        self.start()
        self.wait_for_ready()
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

    def wait_for_ready(self, *, timeout: int = 5) -> None:
        """Wait until the server is ready to receive requests."""
        if self._status == "running":
            return
        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Server exited unexpectedly with code {self.process.returncode}."
                )
            if self._status == "running":
                self.logger.info("Server started.")
                return
            time.sleep(0.1)
        raise TimeoutError(f"Server did not start within {timeout} seconds.")

    def _watch_outputs(self) -> list[threading.Thread]:
        def watch(file: io.StringIO):
            for line in file:
                line = line.strip()
                if "HTTP server listening" in line:
                    self._status = "running"
                self.logger.info(line)

        std_out_thread = threading.Thread(target=watch, args=(self.process.stdout,))
        std_err_thread = threading.Thread(target=watch, args=(self.process.stderr,))
        std_out_thread.start()
        std_err_thread.start()
        return [std_out_thread, std_err_thread]
