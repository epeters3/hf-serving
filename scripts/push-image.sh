#!/bin/bash

set -eo pipefail

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker push epeters3/hf-serving:latest
