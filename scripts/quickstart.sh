#!/usr/bin/env bash
# Build Neuron, fetch the small SmolLM2 135M Instruct model and start a chat.
#
#   scripts/quickstart.sh            terminal chat
#   scripts/quickstart.sh --serve    web chat and API on http://127.0.0.1:8080
#
# Other arguments go straight to neuron chat or neuron serve, for example
# scripts/quickstart.sh --serve --port 9000
# NEURON_MODELS_DIR picks the model cache (default ~/.cache/neuron/models)
# and NEURON_BUILD_DIR the build directory (default build).
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$root"

mode=chat
if [[ ${1:-} == --serve ]]; then
    mode=serve
    shift
elif [[ ${1:-} == -h || ${1:-} == --help ]]; then
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

for tool in cmake curl sha256sum; do
    if ! command -v "$tool" >/dev/null; then
        printf 'quickstart: %s is required but not installed\n' "$tool" >&2
        exit 1
    fi
done
if [[ $(uname -m) != x86_64 ]]; then
    printf 'quickstart: Neuron needs an x86-64 CPU (this is %s)\n' "$(uname -m)" >&2
    exit 1
fi

# Use Ninja when make is missing, as on some minimal systems.
if [[ -z ${CMAKE_GENERATOR:-} ]] && ! command -v make >/dev/null && command -v ninja >/dev/null; then
    export CMAKE_GENERATOR=Ninja
fi

build_dir=${NEURON_BUILD_DIR:-build}
jobs=$(nproc 2>/dev/null || echo 2)
threads=$(( jobs > 4 ? 4 : jobs ))

printf '==> Building Neuron in %s\n' "$build_dir"
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$build_dir" -j "$jobs" --target neuron_cli

export NEURON_MODELS_DIR=${NEURON_MODELS_DIR:-"$HOME/.cache/neuron/models"}
printf '==> Fetching SmolLM2 135M Instruct into %s\n' "$NEURON_MODELS_DIR"
scripts/fetch_models.sh smollm2

model="$NEURON_MODELS_DIR/SmolLM2-135M-Instruct-f16.gguf"
if [[ $mode == serve ]]; then
    printf '==> Starting the server. Open the address it prints; Ctrl-C stops it\n'
else
    printf '==> Starting chat. /reset clears history, /exit quits\n'
fi
exec "$build_dir/neuron" "$mode" -m "$model" --threads "$threads" "$@"
