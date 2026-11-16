"""Model-free tests for token comparison and the command-line contract."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HARNESS = Path(__file__).resolve().parents[1] / "bench/harness.py"
SPEC = importlib.util.spec_from_file_location("harness", HARNESS)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def fixture():
    return {"schema_version": 1, "model": {"id": "test-model"}, "runs": [
        {"threads": 1, "prompt_id": "one", "repetition": 0,
         "prompt_tokens": [1, 2], "generated_tokens": [3, 4, 5],
         "timings": {"prompt_per_second": 100.0, "predicted_per_second": 20.0}}
    ]}


class TokenComparisonTests(unittest.TestCase):
    def test_first_divergence_and_positional_agreement(self):
        result = harness.token_comparison([1, 2, 3, 4], [1, 9, 3, 8])
        self.assertEqual(result["first_divergent_token_index"], 1)
        self.assertEqual(result["agreement_rate"], 0.5)

    def test_unequal_lengths_in_both_directions(self):
        for a, b in [([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 2])]:
            with self.subTest(a=a):
                result = harness.token_comparison(a, b)
                self.assertEqual(result["first_divergent_token_index"], 2)
                self.assertAlmostEqual(result["agreement_rate"], 2 / 3)

    def test_empty_sequences(self):
        result = harness.token_comparison([], [])
        self.assertIsNone(result["first_divergent_token_index"])
        self.assertEqual(result["agreement_rate"], 1)
        self.assertEqual(harness.token_comparison([], [7])["first_divergent_token_index"], 0)
        self.assertEqual(harness.token_comparison([7], [])["agreement_rate"], 0)


class ResultComparisonTests(unittest.TestCase):
    def test_summary_pools_time_and_excludes_first_sample_from_generation_steps(self):
        runs = [{"threads": 1, "timings": {"prompt_n": 10, "prompt_ms": 100,
                                          "predicted_n": 4, "predicted_ms": 200}},
                {"threads": 1, "timings": {"prompt_n": 20, "prompt_ms": 200,
                                          "predicted_n": 2, "predicted_ms": 600}}]
        summary = harness.summarize(runs)[0]
        self.assertEqual(summary["pp_tokens_per_second"], 100)
        self.assertEqual(summary["generation_steps"], 4)
        self.assertEqual(summary["tg_tokens_per_second"], 5)

    def test_identical_tokens_with_faster_candidate(self):
        a, b = fixture(), fixture()
        b["runs"][0]["timings"] = {"prompt_per_second": 200, "predicted_per_second": 10}
        result = harness.compare_results(a, b)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["generated_token_agreement_rate"], 1)
        self.assertEqual(result["runs"][0]["speed_ratio_candidate_over_reference"], {"pp": 2, "tg": 0.5})

    def test_prompt_mismatch_fails_exact_even_when_generation_matches(self):
        a, b = fixture(), fixture()
        b["runs"][0]["prompt_tokens"][1] = 8
        result = harness.compare_results(a, b)
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["runs"][0]["prompt"]["first_divergent_token_index"], 1)
        self.assertEqual(result["generated_token_agreement_rate"], 1)

    def test_rows_are_matched_by_key_and_agreement_is_weighted_by_length(self):
        a = fixture()
        row = copy.deepcopy(a["runs"][0])
        row.update(prompt_id="two", generated_tokens=[6])
        a["runs"].append(row)
        b = copy.deepcopy(a)
        b["runs"].reverse()
        self.assertTrue(harness.compare_results(a, b)["exact_match"])
        b["runs"][0]["generated_tokens"] = [9]
        result = harness.compare_results(a, b)
        self.assertEqual(result["generated_token_agreement_rate"], 0.75)

    def test_missing_extra_or_different_keys_are_mismatches(self):
        a = fixture()
        for field, value in [("threads", 4), ("prompt_id", "other"), ("repetition", 1)]:
            with self.subTest(field=field):
                b = fixture()
                b["runs"][0][field] = value
                result = harness.compare_results(a, b)
                self.assertFalse(result["exact_match"])
                self.assertEqual(result["generated_token_agreement_rate"], 0)
                self.assertEqual({r["status"] for r in result["runs"]},
                                 {"missing_reference", "missing_candidate"})

    def test_duplicate_keys_and_invalid_tokens_are_rejected(self):
        bad = fixture()
        bad["runs"].append(copy.deepcopy(bad["runs"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            harness.compare_results(fixture(), bad)
        for tokens in [[-1], [True], [1.5], "123"]:
            with self.subTest(tokens=tokens):
                bad = fixture()
                bad["runs"][0]["generated_tokens"] = tokens
                with self.assertRaises(ValueError):
                    harness.compare_results(fixture(), bad)

    def test_invalid_timings_and_schema_are_rejected(self):
        for rate in [0, -1, float("nan"), float("inf"), "fast", True]:
            with self.subTest(rate=rate):
                bad = fixture()
                bad["runs"][0]["timings"]["prompt_per_second"] = rate
                with self.assertRaises(ValueError):
                    harness.compare_results(fixture(), bad)
        for update in [{"runs": []}, {"schema_version": 2}]:
            bad = fixture()
            bad.update(update)
            with self.assertRaises(ValueError):
                harness.compare_results(fixture(), bad)


class CommandLineTests(unittest.TestCase):
    def invoke(self, reference, candidate, *flags):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "reference.json", Path(directory) / "candidate.json"
            a.write_text(json.dumps(reference), encoding="utf-8")
            b.write_text(json.dumps(candidate), encoding="utf-8")
            return subprocess.run([sys.executable, str(HARNESS), "compare", str(a), str(b), *flags],
                                  capture_output=True, text=True, check=False)

    def test_exact_match_exit_codes_and_json_report(self):
        a, b = fixture(), fixture()
        result = self.invoke(a, b, "--exact-match")
        self.assertEqual(result.returncode, 0, result.stderr)
        b["runs"][0]["generated_tokens"][1] = 99
        result = self.invoke(a, b)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["runs"][0]["generation"]["first_divergent_token_index"], 1)
        self.assertAlmostEqual(report["generated_token_agreement_rate"], 2 / 3)
        self.assertEqual(self.invoke(a, b, "--exact-match").returncode, 1)
        b = fixture()
        b["runs"][0]["prompt_tokens"] = [9]
        self.assertEqual(self.invoke(a, b, "--exact-match").returncode, 1)

    def test_bad_input_is_an_error_even_without_exact_flag(self):
        result = self.invoke(fixture(), {})
        self.assertEqual(result.returncode, 2)
        self.assertIn("harness:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
