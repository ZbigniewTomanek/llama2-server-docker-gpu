#!/bin/bash

docker inspect llama-2-server-gpu:latest >/dev/null 2>&1 || {
  echo >&2 "llama-2-server-gpu:latest image does not exist. Run docker/build.sh first."
  exit 1
}

mkdir -p models
docker run -it \
  --name llama-2-server-gpu \
  --rm \
  --runtime=nvidia \
  -v "$(pwd)/models:/cache" \
  -p 8080:8080 \
  llama-2-server-gpu:latest sh -c 'cd /app && python3.9 llama2_server/main.py "$@"' -- "$@"
