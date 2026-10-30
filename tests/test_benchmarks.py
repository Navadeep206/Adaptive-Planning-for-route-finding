"""Unit and regression tests for Day 8 Performance Evaluation and Benchmarking Suite.

Validates:
- Deterministic grid generator properties and obstacle density adherence.
- Path validity invariants (bounds, obstacle avoidance, 4-connected step checks).
- A* benchmark execution, statistics aggregation, and edge-case handling.
- Dynamic obstacle rerouting and corridor closure fail-safe execution.
- YOLO latency decomposition and metric formatting.
- CSV serialization and Markdown report generation.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from tools.benchmark_suite import (
    AStarBenchmarker,
    AStarBenchmarkRecord,
    AStarBenchmarkSummary,
    YOLOBenchmarker,
    YOLOLatencyDecomposition,
    DynamicObstacleBenchmarker,
    DynamicObstacleBenchmarkResult,
    BenchmarkReportGenerator
)
from perception.object_detector import Detection


class TestBenchmarkSuite(unittest.TestCase):
    """Test suite verifying benchmarking tools and metrics aggregation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.benchmarker = AStarBenchmarker()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_grid_generator_invariants(self):
        """Verify grid generator adheres to dimensions, density bounds, and start/goal validity."""
        rows, cols = 15, 20
        density = 0.25
        grid, start, goal = AStarBenchmarker.generate_random_grid(rows, cols, density, seed=123)

        self.assertEqual(grid.shape, (rows, cols))
        self.assertEqual(start, (0, 0))
        self.assertEqual(goal, (rows - 1, cols - 1))
        self.assertEqual(grid[start], 1, "Start cell must always be drivable")
        self.assertEqual(grid[goal], 1, "Goal cell must always be drivable")

        # Check approximate density (excluding start & goal)
        total_cells = rows * cols
        obstacle_count = np.count_nonzero(grid == 2)
        empirical_density = obstacle_count / float(total_cells)
        self.assertAlmostEqual(empirical_density, density, delta=0.08)

    def test_path_validator_accepts_valid_orthogonal_path(self):
        """Verify path validator accepts legitimate, obstacle-free, orthogonal paths."""
        grid = np.ones((5, 5), dtype=np.int32)
        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        valid, err = AStarBenchmarker.validate_path(grid, path, start=(0, 0), goal=(2, 2))
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_path_validator_rejects_empty_or_mismatched_path(self):
        """Verify validator rejects empty paths or paths with incorrect endpoints."""
        grid = np.ones((5, 5), dtype=np.int32)
        valid, err = AStarBenchmarker.validate_path(grid, [], start=(0, 0), goal=(4, 4))
        self.assertFalse(valid)
        self.assertIn("empty", err.lower())

        path_wrong_start = [(1, 1), (2, 2)]
        valid, err = AStarBenchmarker.validate_path(grid, path_wrong_start, start=(0, 0), goal=(2, 2))
        self.assertFalse(valid)
        self.assertIn("start", err.lower())

    def test_path_validator_rejects_obstacle_collisions_and_diagonal_steps(self):
        """Verify validator flags obstacle collisions and non-orthogonal diagonal leaps."""
        grid = np.ones((5, 5), dtype=np.int32)
        grid[1, 1] = 2  # Obstacle

        # Path walking into obstacle
        path_obstacle = [(0, 1), (1, 1), (2, 1)]
        valid, err = AStarBenchmarker.validate_path(grid, path_obstacle, start=(0, 1), goal=(2, 1))
        self.assertFalse(valid)
        self.assertIn("obstacle", err.lower())

        # Path taking diagonal step (0,0) -> (1,1) on all-drivable grid
        grid_clear = np.ones((5, 5), dtype=np.int32)
        path_diagonal = [(0, 0), (1, 1), (2, 2)]
        valid, err = AStarBenchmarker.validate_path(grid_clear, path_diagonal, start=(0, 0), goal=(2, 2))
        self.assertFalse(valid)
        self.assertIn("non-orthogonal", err.lower())

    def test_astar_benchmark_suite_execution(self):
        """Verify A* benchmark suite runs on multiple configurations and generates summaries."""
        records, summaries = self.benchmarker.run_benchmark_suite(
            grid_sizes=[(6, 6), (10, 10)],
            densities=[0.0, 0.20],
            trials_per_config=2,
            base_seed=999
        )

        self.assertGreater(len(records), 0)
        self.assertGreater(len(summaries), 0)

        for summary in summaries:
            self.assertGreaterEqual(summary.trials_count, 1)
            self.assertGreaterEqual(summary.median_time_ms, 0.0)
            self.assertGreaterEqual(summary.success_rate_pct, 0.0)
            self.assertLessEqual(summary.success_rate_pct, 100.0)

    def test_dynamic_replanning_benchmarking_scenarios(self):
        """Verify all 4 controlled dynamic obstacle scenarios execute and record proper fail-safes."""
        results = DynamicObstacleBenchmarker.run_scenarios()
        self.assertEqual(len(results), 4)

        scenarios = {r.scenario_name: r for r in results}

        # Scenario 1: single hazard reroute
        s1 = scenarios["scenario_1_single_hazard_reroute"]
        self.assertTrue(s1.replanning_occurred)
        self.assertTrue(s1.alternate_route_valid)
        self.assertFalse(s1.collision_occurred)

        # Scenario 2: autonomous navigation detour
        s2 = scenarios["scenario_2_dynamic_hazard_during_autonomous_nav"]
        self.assertTrue(s2.replanning_occurred)
        self.assertTrue(s2.alternate_route_valid)
        self.assertFalse(s2.collision_occurred)

        # Scenario 3: impassable corridor closure
        s3 = scenarios["scenario_3_corridor_closure_fail_safe"]
        self.assertTrue(s3.replanning_occurred)
        self.assertFalse(s3.alternate_route_valid, "Corridor was blocked; route should be invalid")
        self.assertTrue(s3.vehicle_stopped_safely)
        self.assertFalse(s3.collision_occurred)

        # Scenario 4: safety guard on vehicle cell
        s4 = scenarios["scenario_4_vehicle_position_safety_guard"]
        self.assertTrue(s4.alternate_route_valid)
        self.assertFalse(s4.collision_occurred)

    def test_yolo_benchmark_latency_decomposition_with_mock(self):
        """Verify YOLO latency breakdown calculates median, drawing, and total pipeline times."""
        mock_detector = MagicMock()
        mock_detector.is_loaded.return_value = True
        mock_detector.model_name = "mock_yolo.pt"
        mock_detector._model.device = "mock_cpu"

        dummy_det = Detection(class_id=0, class_name="obstacle_box", confidence=0.88, bbox=(10, 10, 50, 50))
        mock_detector.detect.return_value = ([dummy_det], 12.5)
        mock_detector.draw_detections.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        benchmarker = YOLOBenchmarker(detector=mock_detector)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = benchmarker.benchmark_inference(
            dummy_frame,
            source_label="unit_test_synthetic",
            warmup_runs=1,
            benchmark_runs=3
        )

        self.assertEqual(result.model_name, "mock_yolo.pt")
        self.assertEqual(result.detections_count, 1)
        self.assertIn("obstacle_box", result.detected_classes)
        self.assertGreater(result.inference_median_ms, 0.0)
        self.assertGreater(result.total_pipeline_ms, 0.0)
        self.assertGreater(result.effective_fps, 0.0)

    def test_csv_and_markdown_report_generation(self):
        """Verify report generator produces valid CSV tables and Markdown report."""
        csv_path = os.path.join(self.test_dir, "test_metrics.csv")
        sample_rows = [
            {"name": "test_1", "time_ms": 1.5, "success": True},
            {"name": "test_2", "time_ms": 2.8, "success": False}
        ]
        BenchmarkReportGenerator.export_csv(csv_path, sample_rows)
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("name,time_ms,success", content)
            self.assertIn("test_1,1.5,True", content)

        # Markdown report generation
        md_path = os.path.join(self.test_dir, "BENCHMARK_REPORT.md")
        system_specs = {"OS": "Darwin", "Arch": "arm64"}

        sample_summary = AStarBenchmarkSummary(
            test_case="test_10x10",
            grid_rows=10,
            grid_cols=10,
            density=0.10,
            trials_count=5,
            success_rate_pct=100.0,
            valid_paths_pct=100.0,
            median_time_ms=0.45,
            mean_time_ms=0.48,
            min_time_ms=0.40,
            max_time_ms=0.60,
            std_time_ms=0.05,
            median_explored_nodes=18.0,
            median_path_length=14.0
        )

        sample_yolo = YOLOLatencyDecomposition(
            frame_source="test_frame",
            image_shape=(480, 640),
            model_name="yolo11n.pt",
            device_name="cpu",
            warmup_runs=1,
            benchmark_runs=2,
            capture_time_ms=1.2,
            inference_median_ms=35.0,
            inference_mean_ms=35.5,
            inference_min_ms=34.0,
            inference_max_ms=38.0,
            inference_std_ms=1.2,
            drawing_time_ms=0.8,
            total_pipeline_ms=37.0,
            effective_fps=27.0,
            detections_count=1,
            detected_classes=["car"]
        )

        sample_dynamic = DynamicObstacleBenchmarkResult(
            scenario_name="test_scenario",
            initial_path_length=10,
            obstacle_spawned_coord=(3, 3),
            replanning_occurred=True,
            replan_execution_time_ms=0.35,
            new_path_length=12,
            alternate_route_valid=True,
            vehicle_stopped_safely=True,
            collision_occurred=False,
            notes="Passed"
        )

        report = BenchmarkReportGenerator.generate_markdown_report(
            output_path=md_path,
            system_specs=system_specs,
            astar_summaries=[sample_summary],
            yolo_results=[sample_yolo],
            dynamic_results=[sample_dynamic]
        )

        self.assertTrue(os.path.exists(md_path))
        self.assertIn("# Day 8: Performance Evaluation and Benchmarking Report", report)
        self.assertIn("A* Path Planning Benchmark", report)
        self.assertIn("YOLO Vision & Perception Pipeline Benchmark", report)
        self.assertIn("Dynamic Obstacle Avoidance & Fail-Safe Verification", report)


if __name__ == "__main__":
    unittest.main()
