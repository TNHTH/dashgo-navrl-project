from __future__ import annotations

import unittest

from tools.benchmark_train_modes import (
    OFFICIAL_DECISION_CANDIDATES,
    aggregate_candidate_results,
    choose_best_official,
    decision_ready_reason,
)


def make_result(
    name: str,
    *,
    repeat_index: int,
    num_envs: int,
    fps: float,
    success: bool = True,
    has_nan: bool = False,
    timed_out: bool = False,
    map_source: str = "dashgo_official",
    enable_cameras: bool = False,
) -> dict:
    return {
        "name": name,
        "profile": "pilot",
        "map_source": map_source,
        "num_envs": num_envs,
        "enable_cameras": enable_cameras,
        "repeat_index": repeat_index,
        "success": success,
        "finite": not has_nan,
        "has_nan": has_nan,
        "timed_out": timed_out,
        "frames_per_second": fps,
        "peak_memory_ratio": 0.5,
        "peak_utilization_gpu": 60.0,
        "return_code": 0 if success else 1,
    }


class BenchmarkTrainModesTest(unittest.TestCase):
    def test_decision_ready_false_when_repeats_too_low(self) -> None:
        results = []
        for name, num_envs, fps in [
            ("official_e8_camoff", 8, 220.0),
            ("official_e24_camoff", 24, 440.0),
            ("official_e64_camoff", 64, 610.0),
            ("official_e96_camoff", 96, 670.0),
            ("official_e128_camoff", 128, 720.0),
        ]:
            results.append(make_result(name, repeat_index=1, num_envs=num_envs, fps=fps))
        aggregates = aggregate_candidate_results(results, required_repeats=1)
        ready, reason = decision_ready_reason(aggregates, repeats=1)
        self.assertFalse(ready)
        self.assertIn("repeats_too_low", reason)

    def test_choose_best_official_uses_stable_median_results(self) -> None:
        results = []
        for repeat_index in (1, 2):
            for name, num_envs, fps in [
                ("official_e8_camoff", 8, 220.0),
                ("official_e24_camoff", 24, 445.0),
                ("official_e64_camoff", 64, 617.0),
                ("official_e96_camoff", 96, 670.0),
                ("official_e128_camoff", 128, 665.0),
            ]:
                results.append(make_result(name, repeat_index=repeat_index, num_envs=num_envs, fps=fps))
        aggregates = aggregate_candidate_results(results, required_repeats=2)
        ready, reason = decision_ready_reason(aggregates, repeats=2)
        self.assertTrue(ready, msg=reason)
        selected, selection_reason = choose_best_official(aggregates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["name"], "official_e96_camoff")
        self.assertIn("selected_by_median_fps_with_stability", selection_reason)

    def test_decision_ready_false_when_any_formal_candidate_unstable(self) -> None:
        results = []
        for repeat_index in (1, 2):
            for name, num_envs, fps in [
                ("official_e8_camoff", 8, 220.0),
                ("official_e24_camoff", 24, 445.0),
                ("official_e64_camoff", 64, 617.0),
                ("official_e96_camoff", 96, 670.0),
                ("official_e128_camoff", 128, 725.0),
            ]:
                has_nan = name == "official_e128_camoff" and repeat_index == 2
                success = not has_nan
                results.append(
                    make_result(
                        name,
                        repeat_index=repeat_index,
                        num_envs=num_envs,
                        fps=fps,
                        success=success,
                        has_nan=has_nan,
                    )
                )
        aggregates = aggregate_candidate_results(results, required_repeats=2)
        aggregate_names = {item["name"] for item in aggregates}
        self.assertEqual(set(OFFICIAL_DECISION_CANDIDATES), aggregate_names)
        ready, reason = decision_ready_reason(aggregates, repeats=2)
        self.assertFalse(ready)
        self.assertIn("official_candidates_not_stable", reason)


if __name__ == "__main__":
    unittest.main()
