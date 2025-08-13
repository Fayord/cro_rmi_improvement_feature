#!/usr/bin/env sh
# load .env relative to this script (POSIX sh compatible)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    # export all variables defined in .env
    set -a
    . "$SCRIPT_DIR/.env"
    set +a
fi

echo "USE_CACHED_LLM: $USE_CACHED_LLM"

uvicorn main:app --host 0.0.0.0 --port 7900 --reload --log-level info
