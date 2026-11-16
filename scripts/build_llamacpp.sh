#!/usr/bin/env bash
# CPU-only, host-native Release reference. No llama.cpp files are vendored.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${NEURON_LLAMACPP_DIR:-"$root/third_party/llama.cpp"}
# Upstream tag b12284 (2026-11-11). Always fetch by the full immutable SHA.
revision=9e9619d030bbe6b584d826a0a6cf06e38251f52b
if [[ ! -e "$source_dir/.git" ]]; then
    mkdir -p -- "$source_dir"
    if [[ -n $(ls -A -- "$source_dir") ]]; then
        printf 'Expected an empty directory or llama.cpp checkout: %s\n' "$source_dir" >&2
        exit 1
    fi
    git -C "$source_dir" init
    git -C "$source_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [[ $(git -C "$source_dir" rev-parse --show-toplevel) != $(cd -- "$source_dir" && pwd -P) ]]; then
    printf 'Expected a git checkout at %s\n' "$source_dir" >&2
    exit 1
fi
if [[ -n $(git -C "$source_dir" status --porcelain) ]]; then
    printf 'Refusing to change a dirty llama.cpp checkout: %s\n' "$source_dir" >&2
    exit 1
fi
if ! git -C "$source_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch --depth 1 origin "$revision"
fi
git -C "$source_dir" checkout --detach "$revision"
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]]

cmake -S "$source_dir" -B "$source_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="${CC:-gcc}" \
    -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_RPC=OFF \
    -DGGML_WEBGPU=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build "$source_dir/build" --config Release --target llama-server \
    --parallel "${NEURON_BUILD_JOBS:-4}"
printf 'Built %s at %s\n' "$source_dir/build/bin/llama-server" "$revision"
