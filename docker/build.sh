#!/bin/bash

if [ "$(basename "$(pwd)")" != "docker" ]; then
  echo >&2 "Please run this script from the docker directory."
  exit 1
fi

docker build --platform linux/amd64 -t llama-2-server-gpu:latest -f ./Dockerfile .. "$@"
