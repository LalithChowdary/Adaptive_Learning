#!/bin/zsh
set -eu
cd "$(dirname "$0")"
exec ../model/.venv/bin/python server.py "$@"
