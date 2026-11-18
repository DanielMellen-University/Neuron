#!/usr/bin/env python3
"""End-to-end CLI checks on a tiny generated GGUF model (no real weights)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

NEURON, WRITER = sys.argv.pop(1), sys.argv.pop(1)


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.model = os.path.join(cls.directory.name, "tiny.gguf")
        subprocess.run([WRITER, cls.model], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def neuron(self, *args, check=True, stdin=None):
        result = subprocess.run([NEURON, *args], capture_output=True, input=stdin, timeout=60)
        if check and result.returncode != 0:
            self.fail(f"{args} exited {result.returncode}: {result.stderr.decode()}")
        return result

    def run_json(self, *args):
        return json.loads(self.neuron("run", "-m", self.model, "-p", "a", "--json", *args).stdout)

    def test_greedy_follows_successor_chain(self):
        out = self.neuron("run", "-m", self.model, "-p", "a", "--greedy", "-n", "12").stdout
        self.assertEqual(out, b" b cabcde\n")
        record = self.run_json("--greedy", "-n", "3")
        self.assertEqual(record["generated_tokens"], [10, 11, 4])
        self.assertNotIn("sampling", record)

    def test_temperature_zero_matches_greedy(self):
        self.assertEqual(self.run_json("--temp", "0", "-n", "5")["generated_tokens"],
                         self.run_json("--greedy", "-n", "5")["generated_tokens"])

    def test_seed_repeats_and_varies(self):
        runs = {}
        for seed in range(1, 9):
            first = self.run_json("--temp", "1.5", "--seed", str(seed), "-n", "8")
            again = self.run_json("--temp", "1.5", "--seed", str(seed), "-n", "8")
            self.assertEqual(first["generated_tokens"], again["generated_tokens"])
            self.assertEqual(first["sampling"]["seed"], seed)
            runs[seed] = tuple(first["generated_tokens"])
        self.assertGreater(len(set(runs.values())), 1)

    def test_top_k_one_is_greedy(self):
        greedy = self.run_json("--greedy", "-n", "6")["generated_tokens"]
        for seed in ("1", "2", "3"):
            sampled = self.run_json("--temp", "2", "--top-k", "1", "--seed", seed, "-n", "6")
            self.assertEqual(sampled["generated_tokens"], greedy)

    def test_repeat_penalty_is_reported(self):
        record = self.run_json("--greedy", "--repeat-penalty", "1.3", "--repeat-last-n", "8", "-n", "3")
        self.assertEqual(record["sampling"]["repeat_penalty"], 1.3)
        self.assertEqual(record["sampling"]["repeat_last_n"], 8)

    def test_rejects_bad_sampling_flags(self):
        for flags in (["--temp", "-1"], ["--temp", "x"], ["--temp", "1", "--top-p", "0"],
                      ["--temp", "1", "--min-p", "1"], ["--greedy", "--temp", "0.5"],
                      ["--greedy", "--repeat-penalty", "0"], ["--temp", "1", "--top-k", "-2"], []):
            result = self.neuron("run", "-m", self.model, "-p", "a", "-n", "2", *flags, check=False)
            self.assertEqual(result.returncode, 2, flags)
            self.assertTrue(result.stderr.startswith(b"neuron: "), result.stderr)

    def test_stop_strings_cut_text(self):
        def text(*flags):
            return self.neuron("run", "-m", self.model, "-p", "a", "--greedy", "-n", "12", *flags).stdout
        # " b cabcde": "ab" spans the " ca" and "b" pieces.
        self.assertEqual(text("--stop", "ab"), b" b c\n")
        self.assertEqual(text("--stop", "zz", "--stop", " c"), b" b\n")
        # A held prefix that never completes is printed at the end.
        self.assertEqual(text("--stop", "ex"), b" b cabcde\n")
        record = self.run_json("--greedy", "-n", "12", "--stop", "cd")
        self.assertEqual(record["stop"], "cd")
        self.assertEqual(record["generated_tokens"], [10, 11, 4, 5, 6, 7])
        self.assertIsNone(self.run_json("--greedy", "-n", "2", "--stop", "q")["stop"])
        self.assertNotIn("stop", self.run_json("--greedy", "-n", "2"))
        bad = self.neuron("run", "-m", self.model, "-p", "a", "--greedy", "--stop", "", check=False)
        self.assertEqual(bad.returncode, 2)

    def test_chat_validates_sampling_before_loading(self):
        for flags in (["--temp", "-1"], ["--top-p", "2"], ["--stop", ""], ["--greedy"]):
            result = self.neuron("chat", "-m", self.model, *flags, check=False)
            self.assertEqual(result.returncode, 2, flags)
            self.assertNotIn(b"chat supports", result.stderr, flags)
        # Valid flags reach the model check, which rejects this llama fixture.
        result = self.neuron("chat", "-m", self.model, "--temp", "0.7", "--top-k", "20", "--stop", "x", check=False)
        self.assertIn(b"chat supports Qwen2 and SmolLM2", result.stderr)

    def test_serve_validates_defaults_before_loading(self):
        for flags in (["--temp", "3x"], ["--min-p", "1"], ["--stop", ""]):
            result = self.neuron("serve", "-m", self.model, "--port", "1", *flags, check=False)
            self.assertEqual(result.returncode, 2, flags)
            self.assertNotIn(b"chat supports", result.stderr, flags)
    def test_inspect_lists_metadata_and_tensors(self):
        out = self.neuron("inspect", "-m", self.model).stdout.decode()
        self.assertTrue(out.startswith("GGUF v3, 16 metadata keys, 12 tensors, alignment 32"), out)
        self.assertRegex(out, r"general\.name +string \"neuron cli fixture\"")
        self.assertRegex(out, r"tokenizer\.ggml\.tokens +\[string x 12\] \"<unk>\", \"<s>\"")
        self.assertRegex(out, r"token_embd\.weight +F32 +12 x 12 +576 bytes at 0")
        self.assertIn("\n900 parameters in 3.5 KiB (32.00 bits per weight)\n  F32  12 tensors, 900 parameters", out)
        short = self.neuron("inspect", "-m", self.model, "--no-tensors").stdout.decode()
        self.assertNotIn("tensors:", short)
        self.assertEqual(self.neuron("inspect", check=False).returncode, 2)

    def test_inspect_json(self):
        record = json.loads(self.neuron("inspect", "-m", self.model, "--json").stdout)
        self.assertEqual(record["version"], 3)
        self.assertEqual(record["parameters"], 900)
        self.assertEqual(record["metadata"]["llama.context_length"], {"type": "u32", "value": 64})
        tokens = record["metadata"]["tokenizer.ggml.tokens"]["value"]
        self.assertEqual((tokens["element_type"], tokens["count"], len(tokens["items"])), ("string", 12, 8))
        self.assertEqual(record["tensors"][0], {"name": "token_embd.weight", "type": "F32", "shape": [12, 12],
                                                "parameters": 144, "bytes": 576, "offset": 0})
        full = json.loads(self.neuron("inspect", "-m", self.model, "--json", "--array-items", "0", "--no-tensors").stdout)
        self.assertEqual(full["metadata"]["tokenizer.ggml.tokens"]["value"]["items"][-1], "\u2581c")
        self.assertNotIn("tensors", full)
        text = self.neuron("inspect", "-m", self.model, "--array-items", "2").stdout.decode()
        self.assertIn('[string x 12] "<unk>", "<s>", ... 10 more', text)

    def test_inspect_key_prints_raw_value(self):
        def key(name):
            return self.neuron("inspect", "-m", self.model, "--key", name).stdout
        self.assertEqual(key("general.name"), b"neuron cli fixture")
        self.assertEqual(key("llama.block_count"), b"1")
        self.assertEqual(key("tokenizer.ggml.tokens").split(b"\n")[:3], [b"<unk>", b"<s>", b"</s>"])
        missing = self.neuron("inspect", "-m", self.model, "--key", "nope", check=False)
        self.assertEqual(missing.returncode, 2)
        self.assertIn(b"no metadata key nope", missing.stderr)
        both = self.neuron("inspect", "-m", self.model, "--key", "general.name", "--json", check=False)
        self.assertEqual(both.returncode, 2)

    def test_prompt_file_and_standard_input(self):
        path = os.path.join(self.directory.name, "prompt.txt")
        with open(path, "wb") as f:
            f.write(b"a b")
        from_file = self.neuron("tokenize", "-m", self.model, "--prompt-file", path).stdout
        from_stdin = self.neuron("tokenize", "-m", self.model, "--prompt-file", "-", stdin=b"a b").stdout
        self.assertEqual(from_file, b"[1,9,10]\n")
        self.assertEqual(from_stdin, from_file)
        out = self.neuron("run", "-m", self.model, "--prompt-file", "-", "--greedy", "-n", "2", stdin=b"a").stdout
        self.assertEqual(out, b" b c\n")
        empty = os.path.join(self.directory.name, "empty.txt")
        open(empty, "wb").close()
        self.assertEqual(self.neuron("tokenize", "-m", self.model, "--prompt-file", empty).stdout, b"[1]\n")
        self.assertEqual(self.neuron("tokenize", "-m", self.model, "--prompt-file", "-", stdin=b"").stdout, b"[1]\n")
        missing = self.neuron("tokenize", "-m", self.model, "--prompt-file", path + ".missing", check=False)
        self.assertEqual(missing.returncode, 2)
        self.assertIn(b"cannot read prompt file", missing.stderr)


if __name__ == "__main__":
    unittest.main()
