import json
from typing import Literal, Optional, Union

import requests
from pydantic import BaseModel, Field


# Response models


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    index: int
    object: str
    embedding: list[float]


class Embedding(BaseModel):
    object: Literal["list"]
    model: str
    data: list[EmbeddingData]
    usage: EmbeddingUsage


class CompletionLogprobs(BaseModel):
    text_offset: list[int]
    token_logprobs: list[Optional[float]]
    tokens: list[str]
    top_logprobs: list[Optional[dict[str, float]]]


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChoice]


class Completion(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: str
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str]


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class ChatCompletionChunkDeltaEmpty(BaseModel):
    pass


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Union[ChatCompletionChunkDelta, ChatCompletionChunkDeltaEmpty]
    finish_reason: Optional[str]


class ChatCompletionChunk(BaseModel):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: list[ChatCompletionChunkChoice]


# Requests models
class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="The role of the message."
    )
    content: str = Field(default="", description="The content of the message.")


class CreateChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    max_tokens: int = Field(
        default=16, ge=1, description="The maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Adjust the randomness of the generated text.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Limit the next token selection to a subset of tokens with a cumulative"
            " probability above a threshold P."
        ),
    )
    mirostat_mode: int = Field(
        default=0,
        ge=0,
        le=2,
        description=(
            "Enable Mirostat constant-perplexity algorithm of the specified version (1"
            " or 2; 0 = disabled)"
        ),
    )
    mirostat_tau: float = Field(
        default=5.0, ge=0.0, le=10.0, description="Mirostat target entropy."
    )
    mirostat_eta: float = Field(
        default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
    )
    stop: Optional[list[str]] = Field(
        default=None,
        description=(
            "A list of tokens at which to stop generation. If None, no stop tokens are"
            " used."
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "Whether to stream the results as they are generated. Useful for chatbots."
        ),
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on whether they appear in the"
            " text so far."
        ),
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on their existing frequency in"
            " the text so far."
        ),
    )
    logit_bias: Optional[dict[str, float]] = Field(None)
    model: Optional[str] = Field(
        description="The model to use for generating completions.", default=None
    )
    n: Optional[int] = 1
    user: Optional[str] = Field(None)
    top_k: int = Field(
        default=40,
        ge=0,
        description="Limit the next token selection to the K most probable tokens.",
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        description=(
            "A penalty applied to each token that is already generated. This helps"
            " prevent the model from repeating itself."
        ),
    )
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, list[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description=(
            "A suffix to append to the generated text. If None, no suffix is appended."
            " Useful for chatbots."
        ),
    )
    max_tokens: int = Field(
        default=16, ge=1, description="The maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Adjust the randomness of the generated text.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Limit the next token selection to a subset of tokens with a cumulative"
            " probability above a threshold P."
        ),
    )
    mirostat_mode: int = Field(
        default=0,
        ge=0,
        le=2,
        description=(
            "Enable Mirostat constant-perplexity algorithm of the specified version (1"
            " or 2; 0 = disabled)"
        ),
    )
    mirostat_tau: float = Field(
        default=5.0, ge=0.0, le=10.0, description="Mirostat target entropy."
    )
    mirostat_eta: float = Field(
        default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
    )
    echo: bool = Field(
        default=False,
        description=(
            "Whether to echo the prompt in the generated text. Useful for chatbots."
        ),
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "A list of tokens at which to stop generation. If None, no stop tokens are"
            " used."
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "Whether to stream the results as they are generated. Useful for chatbots."
        ),
    )
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The number of logprobs to generate. If None, no logprobs are generated."
        ),
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on whether they appear in the"
            " text so far."
        ),
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on their existing frequency in"
            " the text so far."
        ),
    )
    logit_bias: Optional[dict[str, float]] = Field(None)
    model: Optional[str] = Field(
        description="The model to use for generating completions.", default=None
    )
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(default=None)
    top_k: int = Field(
        default=40,
        ge=0,
        description="Limit the next token selection to the K most probable tokens.",
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        description=(
            "A penalty applied to each token that is already generated. This helps"
            " prevent the model from repeating itself."
        ),
    )
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = Field(
        description="The model to use for generating completions.", default=None
    )
    input: Union[str, list[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None)


# Llama API client


class LlamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def create_chat_completion(
        self, request: CreateChatCompletionRequest
    ) -> ChatCompletion:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(request.dict()))
        return ChatCompletion(**response.json())

    def create_completions(self, request: CreateCompletionRequest) -> Completion:
        url = f"{self.base_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(request.dict()))
        return Completion(**response.json())

    def create_embeddings(self, request: CreateEmbeddingRequest) -> Embedding:
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(request.dict()))
        return Embedding(**response.json())
