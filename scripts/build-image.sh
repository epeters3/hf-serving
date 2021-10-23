#!/bin/bash

set -eo pipefail

docker build --tag epeters3/hf-serving:latest --file Dockerfile .
