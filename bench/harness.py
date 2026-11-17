#!/usr/bin/env python3
"""Run llama.cpp or Neuron and compare token sequences. Stdlib only."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "tinystories-15m": "stories15M.gguf",
    "smollm2-135m-instruct": "SmolLM2-135M-Instruct-f16.gguf",
}
SEED = 42
N_PREDICT = 128
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete result, even if the process is interrupted while writing.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink()
            raise
    temporary.replace(path)


def token_comparison(reference, candidate):
    """Indices are zero based; an unmatched tail diverges at the shorter length."""
    total = max(len(reference), len(candidate))
    matches = sum(a == b for a, b in zip(reference, candidate))
    first = next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b), None)
    if first is None and len(reference) != len(candidate):
        first = min(len(reference), len(candidate))
    return {"first_divergent_token_index": first, "matching_tokens": matches,
            "total_tokens": total, "agreement_rate": matches / total if total else 1.0}


def result_rows(result):
    if result["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    model = result["model"]["id"]
    if not isinstance(model, str) or not model:
        raise ValueError("model.id must be a nonempty string")
    rows = {}
    for row in result["runs"]:
        key = (model, row["threads"], row["prompt_id"], row["repetition"])
        if (type(row["threads"]) is not int or row["threads"] <= 0
                or type(row["repetition"]) is not int or row["repetition"] < 0
                or not isinstance(row["prompt_id"], str) or not row["prompt_id"]):
            raise ValueError("invalid run key")
        if key in rows:
            raise ValueError(f"duplicate run: {key}")
        for name in ("prompt_tokens", "generated_tokens"):
            tokens = row[name]
            if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError(f"{name} must contain nonnegative integer token ids")
        for name in ("prompt_per_second", "predicted_per_second"):
            rate = row["timings"][name]
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"invalid timing: {name}")
        rows[key] = row
    if not rows:
        raise ValueError("result has no runs")
    return rows


def compare_results(reference, candidate):
    left, right = result_rows(reference), result_rows(candidate)
    comparisons = []
    exact = True
    matched = total = 0
    for key in sorted(left.keys() | right.keys()):
        item = dict(zip(("model", "threads", "prompt_id", "repetition"), key))
        if key not in left or key not in right:
            item["status"] = "missing_reference" if key not in left else "missing_candidate"
            row = left.get(key, right.get(key))
            total += len(row["generated_tokens"])
            exact = False
        else:
            a, b = left[key], right[key]
            item["status"] = "compared"
            item["prompt"] = token_comparison(a["prompt_tokens"], b["prompt_tokens"])
            item["generation"] = token_comparison(a["generated_tokens"], b["generated_tokens"])
            item["speed_ratio_candidate_over_reference"] = {
                "pp": b["timings"]["prompt_per_second"] / a["timings"]["prompt_per_second"],
                "tg": b["timings"]["predicted_per_second"] / a["timings"]["predicted_per_second"],
            }
            exact &= (item["prompt"]["first_divergent_token_index"] is None
                      and item["generation"]["first_divergent_token_index"] is None)
            matched += item["generation"]["matching_tokens"]
            total += item["generation"]["total_tokens"]
        comparisons.append(item)
    return {"exact_match": bool(exact), "generated_token_agreement_rate": matched / total if total else 1.0,
            "matching_generated_tokens": matched, "total_generated_tokens": total,
            "runs": comparisons}


def request(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + route, data=data,
                                 headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=120) as response:
        return json.load(response)


@contextlib.contextmanager
def server(binary, model, threads, context=2048):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--threads", str(threads), "--threads-batch", str(threads),
               "--ctx-size", str(context), "--batch-size", "512", "--ubatch-size", "512",
               "--parallel", "1", "--n-gpu-layers", "0", "--flash-attn", "off",
               "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-webui"]
    if model.name == MODELS["tinystories-15m"]:
        # This older GGUF declares only 128 context tokens. The current server
        # caps slots to that metadata, too short for prompt + 128 output tokens.
        command += ["--override-kv", "llama.context_length=int:2048"]
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        # Explicit flags above define the run, without inherited llama argument overrides.
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError("llama-server exited before becoming ready")
                try:
                    if request(base, "/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("llama-server startup timed out")
                time.sleep(0.1)
            yield base, command
        except BaseException:
            log.seek(0)
            print(log.read()[-8000:], file=sys.stderr)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def machine_state():
    paths = [Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
             Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
             Path("/sys/devices/system/cpu/intel_pstate/no_turbo")]
    paths += sorted(Path("/sys/class/power_supply").glob("*/online"))
    paths += sorted(Path("/sys/class/power_supply").glob("*/status"))
    paths += sorted(Path("/sys/class/power_supply").glob("*/capacity"))
    values = {}
    for path in paths:
        try:
            values[str(path)] = path.read_text().strip()
        except OSError:
            values[str(path)] = "unavailable"
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "load_average": list(os.getloadavg()), "power": values}


def completion(base, tokens):
    response = request(base, "/completion", {
        "prompt": tokens, "n_predict": N_PREDICT, "seed": SEED,
        "temperature": 0, "samplers": ["temperature"], "repeat_penalty": 1.0,
        "cache_prompt": False, "return_tokens": True, "stream": False,
        "ignore_eos": False,
    })
    timings = response["timings"]
    if (response["truncated"] or timings["prompt_n"] != len(tokens)
            or timings.get("cache_n", 0) != 0):
        raise RuntimeError("prompt was truncated or cached; invalid timing sample")
    if not response["tokens"] or len(response["tokens"]) != timings["predicted_n"]:
        raise RuntimeError("missing generated token ids or unexpected token count")
    return response


def summarize(runs):
    summary = []
    for threads in sorted({row["threads"] for row in runs}):
        rows = [row for row in runs if row["threads"] == threads]
        counts = {name: sum(row["timings"][name] for row in rows)
                  for name in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}
        # llama-server times the generation steps after the first sampled token,
        # which comes from prompt evaluation. Match its n_gen_steps() convention.
        counts["generation_steps"] = sum(max(row["timings"]["predicted_n"] - 1, 0) for row in rows)
        summary.append({"threads": threads, "samples": len(rows), **counts,
                        "pp_tokens_per_second": 1000 * counts["prompt_n"] / counts["prompt_ms"],
                        "tg_tokens_per_second": 1000 * counts["generation_steps"] / counts["predicted_ms"]})
    return summary


def run(args):
    source = args.llamacpp_dir.resolve()
    binary = source / "build/bin/llama-server"
    model = args.models_dir.resolve() / MODELS[args.model]
    prompts = read_json(ROOT / "bench/prompts.json")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("llama.cpp checkout must be clean")
    result = {
        "schema_version": 1,
        "model": {"id": args.model, "file": model.name, "sha256": sha256(model),
                  "size_bytes": model.stat().st_size},
        "engine": {"name": "llama.cpp", "commit": revision, "server_sha256": sha256(binary),
                   "version": subprocess.check_output([str(binary), "--version"],
                                                       stderr=subprocess.STDOUT, text=True).strip()},
        "machine": {"name": args.machine, "platform": platform.platform(),
                    "logical_cpus": os.cpu_count(), "notes": args.notes,
                    "before": machine_state()},
        "settings": {"seed": SEED, "n_predict": N_PREDICT, "temperature": 0,
                     "samplers": ["temperature"], "repeat_penalty": 1.0,
                     "cache_prompt": False, "ignore_eos": False, "add_special": True,
                     "parse_special": True, "prompt_format": "raw completion; no chat template",
                     "repetitions": args.repetitions, "warmups_per_thread_count": 1,
                     "prompts_sha256": sha256(ROOT / "bench/prompts.json")},
        "commands": [], "runs": [],
    }
    for threads in (1, 4, 8):
        print(f"{args.model}: {threads} threads", file=sys.stderr, flush=True)
        with server(binary, model, threads) as (base, command):
            result["commands"].append(command)
            tokenized = [request(base, "/tokenize", {"content": p["text"], "add_special": True,
                                                    "parse_special": True})["tokens"] for p in prompts]
            completion(base, tokenized[0])  # Untimed model and context warmup.
            expected = {}
            for repetition in range(args.repetitions):
                for prompt, tokens in zip(prompts, tokenized):
                    response = completion(base, tokens)
                    ids = response["tokens"]
                    if prompt["id"] in expected and expected[prompt["id"]] != ids:
                        raise RuntimeError("greedy tokens changed between repetitions")
                    expected[prompt["id"]] = ids
                    result["runs"].append({"threads": threads, "prompt_id": prompt["id"],
                                           "repetition": repetition, "prompt": prompt["text"],
                                           "prompt_tokens": tokens, "generated_tokens": ids,
                                           "generated_text": response["content"],
                                           "stop_type": response["stop_type"],
                                           "timings": response["timings"]})
    result["machine"]["after"] = machine_state()
    result["summary"] = summarize(result["runs"])
    result_rows(result)
    write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    return 0


def select_threads(result, threads):
    if threads is None:
        return result
    if threads <= 0:
        raise ValueError("threads must be positive")
    return dict(result, runs=[r for r in result["runs"] if r["threads"] == threads])


def run_neuron(args):
    """Collect Neuron runs with the M1 prompts and explicit execution settings."""
    if args.threads < 1:
        raise ValueError("threads must be positive")
    model = args.model_file.resolve()
    binary = args.binary.resolve()
    prompts = read_json(ROOT / "bench/prompts.json")
    model_hash = sha256(model)
    expected_hash = read_json(ROOT / "bench/reference/tinystories-15m.json")["model"]["sha256"]
    if model_hash != expected_hash:
        raise ValueError("run-neuron requires the pinned TinyStories model")
    result = {
        "schema_version": 1,
        "model": {"id": "tinystories-15m", "file": model.name, "sha256": model_hash},
        "engine": {"name": "neuron", "binary_sha256": sha256(binary),
                   "version": subprocess.check_output([str(binary), "--version"], text=True).strip()},
        "machine": {"name": "Ghost", "notes": args.notes, "before": machine_state()},
        "settings": {"n_predict": N_PREDICT, "greedy": True, "context": 2048,
                     "threads": args.threads, "isa": args.isa, "repetitions": args.repetitions,
                     "prompts_sha256": sha256(ROOT / "bench/prompts.json")},
        "runs": [], "commands": [],
    }
    for repetition in range(args.repetitions):
        for prompt in prompts:
            command = [str(binary), "run", "-m", str(model), "-p", prompt["text"],
                       "-n", str(N_PREDICT), "--greedy", "--json", "--ctx-size", "2048",
                       "--threads", str(args.threads), "--isa", args.isa]
            row = json.loads(subprocess.check_output(command, text=True, timeout=120))
            row.update(threads=args.threads, prompt_id=prompt["id"], repetition=repetition, prompt=prompt["text"])
            result["runs"].append(row)
            result["commands"].append(command)
            print(f"neuron: {prompt['id']} repetition {repetition}: {len(row['generated_tokens'])} tokens",
                  file=sys.stderr, flush=True)
    result["machine"]["after"] = machine_state()
    result["summary"] = summarize(result["runs"])
    result_rows(result)
    write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    collect = commands.add_parser("run", help="collect one model at 1, 4 and 8 threads")
    collect.add_argument("--model", choices=MODELS, required=True)
    collect.add_argument("--models-dir", type=Path,
                         default=Path(os.environ.get("NEURON_MODELS_DIR", ROOT / "models")))
    collect.add_argument("--llamacpp-dir", type=Path,
                         default=Path(os.environ.get("NEURON_LLAMACPP_DIR", ROOT / "third_party/llama.cpp")))
    collect.add_argument("--repetitions", type=int, default=3)
    collect.add_argument("--machine", default="Ghost")
    collect.add_argument("--notes", required=True, help="other activity and benchmark conditions")
    collect.add_argument("--output", type=Path, required=True)
    neuron = commands.add_parser("run-neuron", help="collect TinyStories runs")
    neuron.add_argument("--binary", type=Path, required=True)
    neuron.add_argument("--model-file", type=Path, required=True)
    neuron.add_argument("--output", type=Path, required=True)
    neuron.add_argument("--repetitions", type=int, default=3)
    neuron.add_argument("--notes", required=True)
    neuron.add_argument("--threads", type=int, default=1)
    neuron.add_argument("--isa", choices=["auto", "scalar", "avx2", "avx512", "vnni"], default="auto")
    compare = commands.add_parser("compare", help="compare candidate tokens and speed to a reference")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--exact-match", action="store_true", help="exit 1 on any prompt or generated token mismatch")
    compare.add_argument("--threads", type=int, help="compare only this thread count in both inputs")
    args = parser.parse_args(argv)
    try:
        if args.mode in ("run", "run-neuron"):
            if args.repetitions < 1:
                raise ValueError("repetitions must be positive")
            return run(args) if args.mode == "run" else run_neuron(args)
        comparison = compare_results(select_threads(read_json(args.reference), args.threads),
                                     select_threads(read_json(args.candidate), args.threads))
        print(json.dumps(comparison, indent=2, allow_nan=False))
        return 1 if args.exact_match and not comparison["exact_match"] else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"harness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
