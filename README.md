# llama2-server-docker-gpu

This repository contains scripts allowing easily run a GPU accelerated Llama 2 REST server in a Docker container.
This server will run only models that are stored in the HuggingFace repository and are compatible
with [llama.cpp](https://github.com/ggerganov/llama.cpp).

For the GPU support https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX/cu122 version of the llama-cpp library
is
used.

## Pre-requisites

To actually run the server on the GPU, you need to have the following installed:

- [NVIDIA and CUDA 12.2](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.htm)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

`scripts/rhel7-install-nvidia-runtime.sh` script shows how those dependencies can be installed on RHEL 7.

To see if you have the drivers installed, run the following command:

```bash
docker run --rm --runtime=nvidia -it -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.2.0-devel-ubuntu20.04 nvidia-smi
```

## Run llama server

First, build the Docker image:

```bash
cd docker && ./build.sh
```

Then you can run the server with the `run-server.sh` script which takes following parameters:

- `--hg-repo-id` - ID of the HuggingFace repository containing the model
- `--hg-filename` - Name of the file containing the model

I've tested that with a following repositories:

- https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
- https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML
- https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML

Chosen model will be cached in the `models` directory, so it will be downloaded only once.

### Examples

**Run Llama Llama-2-7B-Chat-GGML with the q2 quantization on GPU**

```bash
./run-server.sh --hg-repo-id TheBloke/Llama-2-7B-Chat-GGML --hg-filename llama-2-7b-chat.ggmlv3.q2_K.bin --n_gpu_layers 2048
```

**Run Llama Llama-2-70B-Chat-GGML with the q5 quantization on GPU (tested on g5.12xlarge)**

```bash
./run-server.sh --hg-repo-id TheBloke/Llama-2-70B-Chat-GGML --hg-filename llama-2-70b-chat.ggmlv3.q5_K_S.bin --n_gpu_layers 2048 --n_gqa 8
```

### Server parameters

This `run-server.sh` a wrapper around the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) server, so it
will accept the same parameters as the original server. You can see the list of parameters by running:

```bash
./run-server.sh --help
```

You will need to play with them a bit to find the optimal configuration for your use case.

## Llama client

`llama_client.py` contains a simple Python REST client for the Llama server.
It can be used as follows:

```python
from llama2_server.llama_client import (
    LlamaClient,
    CreateCompletionRequest,
    ChatCompletionRequestMessage, CreateEmbeddingRequest, CreateChatCompletionRequest,
)

client = LlamaClient("https://localhost:8080")

# completions
print(client.create_completions(CreateCompletionRequest(prompt="Name all planets in the solar system")))

# chat completions
print(client.create_chat_completion(CreateChatCompletionRequest(messages=[
    ChatCompletionRequestMessage(role="system", content="You are a well-known astronomer"),
    ChatCompletionRequestMessage(role="user", content="List all planets in the solar system"),
])))

# embeddings
print(client.create_embeddings(CreateEmbeddingRequest(input=["Hello world!"])))
```

### CLI

After running `poetry install && poetry shell` you should be able to call LlamaClient using `llama-cli` CLI from your
terminal:

**Completions**

```shell
llama-cli completion --llama-url 'http://localhost:8080' --prompt 'List all planets in the solar system'
```

**Chat completions**

```shell
llama-cli chat-completion --max_tokens 1024 --llama-url 'https://localhost:8080' --message 'system|You are a well-known astronomer' --message 'user|List all planets in the solar system'
```

**Embeddings**

```shell
llama-cli embeddings --llama-url 'http://localhost:8080' --text 'List all planets in the solar system'
```
