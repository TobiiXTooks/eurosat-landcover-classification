#!/bin/bash
# Live LaTeX preview: rebuilds main.pdf on every save and serves an
# auto-reloading viewer in the browser. No VS Code required.
#
#   ./preview.sh        # start watching + serving, opens the browser
#
# Stop with Ctrl-C (cleans up the background web server too).

set -e
cd "$(dirname "$0")"

export PATH="/Library/TeX/texbin:$PATH"
PORT="${PORT:-8800}"

# Static file server (background) so the browser can fetch the PDF.
python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

URL="http://127.0.0.1:$PORT/preview.html"
echo "Live preview at: $URL"
open "$URL" 2>/dev/null || true

# Continuous compile: rebuilds preview.pdf whenever any source file changes.
latexmk -pvc -pdf -synctex=1 -interaction=nonstopmode -file-line-error \
        -view=none -jobname=preview main.tex
