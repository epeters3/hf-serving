#!/bin/bash

set -eo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."  # go to root

docker build --tag epeters3/hf-serving:latest --file Dockerfile .
