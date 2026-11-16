#!/usr/bin/env python3
"""Local model checks. Requires the pinned TinyStories file; never runs in CI."""

import argparse
import array
import json
import math
from pathlib import Path
import subprocess
import tempfile

from harness import ROOT, read_json, sha256, write_json


def token_flags(case):
    return (["--no-add-special"] if not case.get("add_special", True) else []) + (
        ["--no-parse-special"] if not case.get("parse_special", True) else [])


def floats(path):
    result = array.array("f")
    result.frombytes(path.read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-tokenizer", action="store_true",
                        help="replace fixture IDs with results from the reference tool")
    args = parser.parse_args()
    binary, model, reference = map(lambda p: str(p.resolve()),
                                   (args.binary, args.model, args.reference_tool))
    fixture_path = ROOT / "bench/reference/tinystories-tokenizer.json"
    fixture = read_json(fixture_path)
    if sha256(model) != fixture["model_sha256"]:
        raise ValueError("model hash differs from the tokenizer fixture")
    # Include exactly the committed prompts, even if someone edits prompts.json.
    prompts = read_json(ROOT / "bench/prompts.json")
    for prompt in prompts:
        case = next(c for c in fixture["cases"] if c["id"] == prompt["id"])
        if case["text"] != prompt["text"]:
            raise ValueError("prompt differs from the tokenizer fixture")
    for case in fixture["cases"]:
        command = [reference, model, case["text"], *token_flags(case)]
        expected = json.loads(subprocess.check_output(command, stderr=subprocess.DEVNULL))
        if args.refresh_tokenizer:
            case["tokens"] = expected
        if expected != case["tokens"]:
            raise ValueError(f"reference tokenizer changed: {case['id']}")
        actual = json.loads(subprocess.check_output(
            [binary, "tokenize", "-m", model, "-p", case["text"], *token_flags(case)]))
        if actual != expected:
            raise ValueError(f"tokenizer mismatch: {case['id']}")
    if args.refresh_tokenizer:
        write_json(fixture_path, fixture)
    print(f"Tokenizer: {len(fixture['cases'])}/{len(fixture['cases'])} exact matches", flush=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix="neuron-m2-logits-") as directory:
        for prompt in prompts:
            rpath, npath = Path(directory) / "llama.f32", Path(directory) / "neuron.f32"
            subprocess.run([reference, model, prompt["text"], "--logits", str(rpath)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run([binary, "logits", "-m", model, "-p", prompt["text"],
                            "--output", str(npath), "--ctx-size", "2048"], check=True)
            expected, actual = floats(rpath), floats(npath)
            if len(expected) != 32000 or len(actual) != len(expected):
                raise ValueError("logit vector length differs from the vocabulary")
            if not all(math.isfinite(x) for x in [*expected, *actual]):
                raise ValueError("nonfinite logits")
            difference = max(abs(a - b) for a, b in zip(actual, expected))
            checks.append({"prompt_id": prompt["id"], "logits": len(actual),
                           "max_absolute_difference": difference})
            print(f"{prompt['id']}: max abs diff {difference:.12g}", flush=True)
            if difference > 1e-3:
                raise ValueError("logit tolerance exceeded")
    write_json(args.output, {"model_sha256": fixture["model_sha256"],
                             "llama_cpp_commit": fixture["llama_cpp_commit"],
                             "neuron_binary_sha256": sha256(binary), "reference_tool_sha256": sha256(reference),
                             "context": 2048, "reference_kv_type": "F32", "threads": 1,
                             "tokenizer_cases": len(fixture["cases"]), "logit_checks": checks})


if __name__ == "__main__":
    main()
