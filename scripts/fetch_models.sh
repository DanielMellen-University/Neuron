#!/usr/bin/env bash
# Pinned unquantized models for the M1 reference. Requires curl and sha256sum.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
models_dir=${NEURON_MODELS_DIR:-"$root/models"}
mkdir -p -- "$models_dir"

fetch() (
    local name=$1 sha=$2 url=$3
    local dest="$models_dir/$name" tmp
    if [[ -f "$dest" ]] && [[ $(sha256sum -- "$dest" | cut -d ' ' -f 1) == "$sha" ]]; then
        printf 'Verified %s\n' "$dest"
        return
    fi
    # Keep an existing file until its replacement passes verification.
    tmp=$(mktemp "$models_dir/.${name}.XXXXXX")
    trap 'rm -f -- "$tmp"' EXIT
    curl --fail --location --retry 3 --connect-timeout 30 --output "$tmp" "$url"
    if [[ $(sha256sum -- "$tmp" | cut -d ' ' -f 1) != "$sha" ]]; then
        printf 'SHA-256 mismatch for %s\n' "$name" >&2
        return 1
    fi
    mv -f -- "$tmp" "$dest"
    printf 'Downloaded and verified %s\n' "$dest"
)

# Hashes computed from the actual downloads, not just the remote LFS pointers.
fetch stories15M.gguf \
    61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a \
    https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M.gguf
fetch SmolLM2-135M-Instruct-f16.gguf \
    f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57 \
    https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-f16.gguf
