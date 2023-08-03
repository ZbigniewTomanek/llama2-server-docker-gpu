#!/usr/bin/env python3
import json
from typing import Type, TypeVar

import typer
from pydantic import ValidationError

from llama2_server.llama_client import (
    LlamaClient,
    CreateCompletionRequest,
    ChatCompletionRequestMessage, CreateEmbeddingRequest, CreateChatCompletionRequest,
)

app = typer.Typer()

T = TypeVar("T", bound="BaseModel")


def parse_request(model_cls: Type[T], **kwargs) -> T:
    try:
        return model_cls(**kwargs)
    except ValidationError as e:
        typer.echo(f"Error parsing model: {e}")
        typer.echo(f"Allowed request fields:")
        typer.echo(json.dumps(sorted(list(model_cls.model_fields.keys())), indent=2))
        raise typer.Exit(code=1)


def parse_extra_args(args: list[str]) -> dict[str, str]:
    if len(args) % 2 != 0:
        raise ValueError("Extra args must be in the form --key value")
    parsed_args = {}
    for i in range(len(args)):
        arg = args[i]
        if arg.startswith("--"):
            parsed_args[arg[2:]] = args[i + 1]
    return parsed_args


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def completion(
        ctx: typer.Context,
        llama_url: str = typer.Option(..., help="The host of the llama server"),
        prompt: str = typer.Option(..., help="The prompt to complete"),
) -> None:
    client = LlamaClient(llama_url)
    print(ctx.args)
    request = parse_request(CreateCompletionRequest, prompt=prompt, **parse_extra_args(ctx.args))
    response = client.create_completions(request)
    typer.echo(response.model_dump_json(indent=4))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def chat_completion(
        ctx: typer.Context,
        llama_url: str = typer.Option(..., help="The host of the llama server"),
        message: list[str] = typer.Option(
            ...,
            help=(
                    "The messages to complete in the form [user,system,assistant]|message, eg."
                    " 'user|Hello, how are you?'"
            ),
        ),
) -> None:
    client = LlamaClient(llama_url)
    parsed_messages = []
    for msg in message:
        if "|" not in msg:
            raise ValueError(
                f"Message {msg} must be in the form [user,system,assistant]|message"
            )
        user, msg = msg.split("|")
        parsed_messages.append(ChatCompletionRequestMessage(role=user, content=msg))
    request = parse_request(CreateChatCompletionRequest, messages=parsed_messages, **parse_extra_args(ctx.args))
    response = client.create_chat_completion(request)
    typer.echo(response.model_dump_json(indent=4))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def embeddings(
        ctx: typer.Context,
        llama_url: str = typer.Option(..., help="The host of the llama server"),
        text: str = typer.Option(..., help="Text to embed"),
) -> None:
    client = LlamaClient(llama_url)
    request = parse_request(CreateEmbeddingRequest, input=text, **parse_extra_args(ctx.args))
    response = client.create_embeddings(request)
    typer.echo(response.model_dump_json(indent=4))


if __name__ == "__main__":
    app()
