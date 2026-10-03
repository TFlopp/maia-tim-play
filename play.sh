#!/bin/sh
exec python3 "$(dirname "$0")/maia_tim/server.py" "$@"
