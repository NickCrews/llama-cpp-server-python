# llama-cpp-server-python

**Bootstrap a [server from llama-cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) in a few lines of python.**

```python
from openai import OpenAI
from llama_cpp_server_python import Server
repo = "Qwen/Qwen2-0.5B-Instruct-GGUF"
filename = "qwen2-0_5b-instruct-q4_0.gguf"
with Server.from_huggingface(repo=repo, filename=filename) as server:
    client = OpenAI(base_url=server.base_url)
    # interact with the client
```

For more control, you can download the model and binary separately,
and pass in other parameters:

```python
binary_path = "path/to/llama-server"
model_path = "path/to/model.gguf"
from llama_cpp_server_python import download_binary, download_model
download_binary(binary_path)
download_model(dest=model_path, repo=repo, filename=filename)
server = Server(binary_path=binary_path, model_path=model_path, port=6000, ctx_size=1024)
server.start()
client = OpenAI(base_url=server.base_url)
# interact with the client
server.stop() # or use a context manager as above
```

For detailed API, read the source code.

## Install

This only currently works on Linux and Mac. File an issue if you want a pointer on
what needs to happen to make Windows work.

For now, install directly from source:

`python -m pip install git+https://github.com/NickCrews/llama-cpp-server-python@00cc5ece8783848139d41fb7f9c5e5c9b7a62686`

I recommend using a static SHA for stability, but you could also do `@main` to be lazy.

## Motivation

I has a few requirements:

- use a local LLM (free)
- support batched inference (I was doing bulk processing, ie with pandas)
- support structured output (ie limit output to valid json)

I found https://github.com/abetlen/llama-cpp-python, but as of this writing,
[it did not support batched inference](https://github.com/abetlen/llama-cpp-python/issues/771),
and it didn't support structured output.

However, the [server from the upstream llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
project supports all of these requirements:
See `--cont-batching` argument during server startup,
and `json_schema` param of the `/completion` endpoint.

So I wanted a quick and easy way to

- download and install the server binary
- download some model weights from huggingface hub
- get a server running and then use a "http://localhost:8080" url in a client.

This is NOT a client. You can either use an OpenAI library as above,
or send http POST requests directly.

## License

MIT