"""Bootstrap a server from llama-cpp in a few lines of python.

This package provides a simple way to install and run the llama-cpp server
binary from python.
"""

from ._binary import download_binary as download_binary
from ._model import download_model as download_model
from ._server import Server as Server
