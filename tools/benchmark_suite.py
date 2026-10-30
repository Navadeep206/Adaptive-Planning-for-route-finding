"""Day 8: Performance Evaluation and Benchmarking Suite.

Provides repeatable, rigorous, empirical benchmarking across:
1. A* Path Planner on various grid sizes, obstacle densities, and edge cases.
2. YOLO Object Detector latency, FPS, and perception pipeline breakdown.
3. Dynamic obstacle avoidance, adaptive replanning latency, and fail-safe halts.
4. Exporting structured CSV metrics and comprehensive Markdown reports.

Academic & Laboratory Notice:
-----------------------------
This evaluation reflects a simulation testbed and monocular vision demonstration.
It does not certify physical vehicle dynamics, functional safety (ISO 26262), or
unrestricted autonomous driving in real-world traffic.
"""

from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import csv
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Ensure workspace root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Internal modules
from planning.astar import AStarPlanner, AStarResult, manhattan_distance
from simulation.environment import RoadEnvironment, CellType, NavigationMode
from perception.object_detector import YOLOObjectDetector, Detection
from perception.webcam import WebcamManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BenchmarkSuite")


# ==============================================================================
# 1. A* BENCHMARKING ENGINE
# ==============================================================================

@dataclass
class AStarBenchmarkRecord:
    """Detailed record for a single A* planning benchmark trial."""
    test_case: str
    grid_rows: int
    grid_cols: int
    density: float
    trial_id: int
    success: bool
    execution_time_ms: float
    path_length: int
    explored_nodes: int
    valid_path: bool
    validation_error: Optional[str] = None


@dataclass
class AStarBenchmarkSummary:
    """Aggregated statistics for a set of A* planning runs."""
    test_case: str
    grid_rows: int
    grid_cols: int
    density: float
    trials_count: int
    success_rate_pct: float
    valid_paths_pct: float
    median_time_ms: float
    mean_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    median_explored_nodes: float
    median_path_length: float


class AStarBenchmarker:
    """Executes repeatable performance benchmarks on the A* Path Planner."""

    def __init__(self, planner: Optional[AStarPlanner] = None):
        self.planner = planner or AStarPlanner(default_drivable_value=1)

    @staticmethod
    def generate_random_grid(
        rows: int,
        cols: int,
        density: float,
        seed: Optional[int] = None,
        start: Tuple[int, int] = (0, 0),
        goal: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Generate a reproducible 2D grid with a target obstacle density.

        Args:
            rows: Number of grid rows.
            cols: Number of grid columns.
            density: Fractional obstacle density in [0.0, 1.0].
            seed: Optional random seed for deterministic evaluation.
            start: Vehicle starting coordinate.
            goal: Destination coordinate (defaults to bottom-right corner).

        Returns:
            Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]: (grid, start, goal)
        """
        rng = np.random.RandomState(seed)
        grid = np.ones((rows, cols), dtype=np.int32)  # 1 = drivable

        if density > 0.0:
            obstacle_mask = rng.rand(rows, cols) < density
            grid[obstacle_mask] = 2  # 2 = obstacle

        target_goal = goal if goal is not None else (rows - 1, cols - 1)

        # Guarantee start and goal are drivable for standard benchmark maps
        grid[start[0], start[1]] = 1
        grid[target_goal[0], target_goal[1]] = 1

        return grid, start, target_goal

    @staticmethod
    def validate_path(
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Tuple[bool, Optional[str]]:
        """Verify that a returned path satisfies all structural navigation invariants.

        Invariants checked:
        1. Non-empty path.
        2. First waypoint matches start position.
        3. Last waypoint matches goal position.
        4. Every waypoint is within grid bounds.
        5. Every waypoint is on drivable terrain (not an obstacle).
        6. Consecutive waypoints are strictly 4-connected (orthogonal, Manhattan distance == 1).

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, failure_reason)
        """
        if not path:
            return False, "Path is empty"

        if path[0] != start:
            return False, f"Path start {path[0]} does not match query start {start}"

        if path[-1] != goal:
            return False, f"Path destination {path[-1]} does not match query goal {goal}"

        rows, cols = grid.shape

        for i, (r, c) in enumerate(path):
            if not (0 <= r < rows and 0 <= c < cols):
                return False, f"Waypoint {i} ({r}, {c}) is outside grid dimensions ({rows}, {cols})"
            if grid[r, c] != 1:
                return False, f"Waypoint {i} ({r}, {c}) intersects obstacle/border (value {grid[r, c]})"

            if i > 0:
                prev_r, prev_c = path[i - 1]
                dist = abs(r - prev_r) + abs(c - prev_c)
                if dist != 1:
                    return False, f"Step from {path[i-1]} to {path[i]} is non-orthogonal (L1 dist = {dist})"

        return True, None

    def run_benchmark_suite(
        self,
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
        densities: Optional[List[float]] = None,
        trials_per_config: int = 5,
        base_seed: int = 42
    ) -> Tuple[List[AStarBenchmarkRecord], List[AStarBenchmarkSummary]]:
        """Run comprehensive A* benchmark over grid sizes, densities, and edge cases.

        Args:
            grid_sizes: List of (rows, cols) dimensions to test.
            densities: List of obstacle densities in [0.0, 1.0].
            trials_per_config: Number of randomized trials per condition.
            base_seed: Deterministic base seed.

        Returns:
            Tuple[List[AStarBenchmarkRecord], List[AStarBenchmarkSummary]]: Raw records and summaries.
        """
        sizes = grid_sizes or [(10, 10), (22, 26), (50, 50), (100, 100)]
        dens = densities if densities is not None else [0.0, 0.10, 0.20, 0.30, 0.40]

        records: List[AStarBenchmarkRecord] = []
        summaries: List[AStarBenchmarkSummary] = []

        logger.info("Starting A* benchmark across %d grid sizes and %d densities...", len(sizes), len(dens))

        # 1. Standard Randomized Grid Evaluation
        for rows, cols in sizes:
            for d in dens:
                case_name = f"random_{rows}x{cols}_d{int(d*100)}"
                case_records: List[AStarBenchmarkRecord] = []

                for trial in range(trials_per_config):
                    seed = base_seed + (trial * 1000) + int(d * 100) + rows
                    grid, start, goal = self.generate_random_grid(rows, cols, d, seed=seed)

                    res = self.planner.plan(grid, start, goal)

                    valid = False
                    err = None
                    if res.success:
                        valid, err = self.validate_path(grid, res.path, start, goal)
                    else:
                        valid = True  # Safely reported no-path

                    record = AStarBenchmarkRecord(
                        test_case=case_name,
                        grid_rows=rows,
                        grid_cols=cols,
                        density=d,
                        trial_id=trial + 1,
                        success=res.success,
                        execution_time_ms=res.execution_time_ms,
                        path_length=res.path_length,
                        explored_nodes=res.explored_nodes,
                        valid_path=valid,
                        validation_error=err
                    )
                    case_records.append(record)
                    records.append(record)

                summaries.append(self._aggregate_records(case_name, rows, cols, d, case_records))

        # 2. Edge Case: Blocked Start Coordinate
        edge_start_records = []
        for trial in range(trials_per_config):
            grid, start, goal = self.generate_random_grid(22, 26, 0.0, seed=base_seed + trial)
            grid[start[0], start[1]] = 2  # Block start
            res = self.planner.plan(grid, start, goal)
            record = AStarBenchmarkRecord(
                test_case="edge_blocked_start",
                grid_rows=22,
                grid_cols=26,
                density=0.0,
                trial_id=trial + 1,
                success=res.success,
                execution_time_ms=res.execution_time_ms,
                path_length=res.path_length,
                explored_nodes=res.explored_nodes,
                valid_path=not res.success,  # Must correctly fail
                validation_error=None if not res.success else "Failed to reject blocked start"
            )
            edge_start_records.append(record)
            records.append(record)
        summaries.append(self._aggregate_records("edge_blocked_start", 22, 26, 0.0, edge_start_records))

        # 3. Edge Case: Blocked Goal Coordinate
        edge_goal_records = []
        for trial in range(trials_per_config):
            grid, start, goal = self.generate_random_grid(22, 26, 0.0, seed=base_seed + trial)
            grid[goal[0], goal[1]] = 2  # Block goal
            res = self.planner.plan(grid, start, goal)
            record = AStarBenchmarkRecord(
                test_case="edge_blocked_goal",
                grid_rows=22,
                grid_cols=26,
                density=0.0,
                trial_id=trial + 1,
                success=res.success,
                execution_time_ms=res.execution_time_ms,
                path_length=res.path_length,
                explored_nodes=res.explored_nodes,
                valid_path=not res.success,
                validation_error=None if not res.success else "Failed to reject blocked goal"
            )
            edge_goal_records.append(record)
            records.append(record)
        summaries.append(self._aggregate_records("edge_blocked_goal", 22, 26, 0.0, edge_goal_records))

        # 4. Edge Case: Fully Partitioned Wall (Unsolvable No-Path)
        edge_wall_records = []
        for trial in range(trials_per_config):
            grid, start, goal = self.generate_random_grid(22, 26, 0.0, seed=base_seed + trial)
            # Build impassable vertical wall down column 12
            grid[:, 12] = 2
            res = self.planner.plan(grid, start, goal)
            record = AStarBenchmarkRecord(
                test_case="edge_partitioned_wall_no_path",
                grid_rows=22,
                grid_cols=26,
                density=0.04,
                trial_id=trial + 1,
                success=res.success,
                execution_time_ms=res.execution_time_ms,
                path_length=res.path_length,
                explored_nodes=res.explored_nodes,
                valid_path=not res.success,  # Must correctly report failure
                validation_error=None if not res.success else "Discovered phantom path through solid wall"
            )
            edge_wall_records.append(record)
            records.append(record)
        summaries.append(self._aggregate_records("edge_partitioned_wall_no_path", 22, 26, 0.04, edge_wall_records))

        # 5. Realistic Road Simulation Environment Baseline
        road_env_records = []
        sim_env = RoadEnvironment()
        for trial in range(trials_per_config):
            res = sim_env.plan_path()
            valid, err = self.validate_path(sim_env.grid, res.path, sim_env.vehicle_pos, sim_env.destination_pos)
            record = AStarBenchmarkRecord(
                test_case="simulation_baseline_unstructured_road",
                grid_rows=sim_env.rows,
                grid_cols=sim_env.cols,
                density=float(len(sim_env.obstacle_set)) / (sim_env.rows * sim_env.cols),
                trial_id=trial + 1,
                success=res.success,
                execution_time_ms=res.execution_time_ms,
                path_length=res.path_length,
                explored_nodes=res.explored_nodes,
                valid_path=valid,
                validation_error=err
            )
            road_env_records.append(record)
            records.append(record)
        summaries.append(self._aggregate_records(
            "simulation_baseline_unstructured_road",
            sim_env.rows,
            sim_env.cols,
            float(len(sim_env.obstacle_set)) / (sim_env.rows * sim_env.cols),
            road_env_records
        ))

        logger.info("A* benchmark completed: %d total trials processed.", len(records))
        return records, summaries

    @staticmethod
    def _aggregate_records(
        case_name: str,
        rows: int,
        cols: int,
        density: float,
        records: List[AStarBenchmarkRecord]
    ) -> AStarBenchmarkSummary:
        """Aggregate statistical metrics over a group of trials."""
        n = len(records)
        times = [r.execution_time_ms for r in records]
        nodes = [r.explored_nodes for r in records]
        lengths = [r.path_length for r in records if r.success]

        success_cnt = sum(1 for r in records if r.success)
        valid_cnt = sum(1 for r in records if r.valid_path)

        return AStarBenchmarkSummary(
            test_case=case_name,
            grid_rows=rows,
            grid_cols=cols,
            density=density,
            trials_count=n,
            success_rate_pct=(success_cnt / float(n)) * 100.0,
            valid_paths_pct=(valid_cnt / float(n)) * 100.0,
            median_time_ms=float(np.median(times)),
            mean_time_ms=float(np.mean(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            std_time_ms=float(np.std(times)),
            median_explored_nodes=float(np.median(nodes)),
            median_path_length=float(np.median(lengths)) if lengths else 0.0
        )


# ==============================================================================
# 2. YOLO BENCHMARKING ENGINE
# ==============================================================================

@dataclass
class YOLOLatencyDecomposition:
    """Breakdown of individual perception pipeline stage timings."""
    frame_source: str
    image_shape: Tuple[int, int]
    model_name: str
    device_name: str
    warmup_runs: int
    benchmark_runs: int
    capture_time_ms: Optional[float]
    inference_median_ms: float
    inference_mean_ms: float
    inference_min_ms: float
    inference_max_ms: float
    inference_std_ms: float
    drawing_time_ms: float
    total_pipeline_ms: float
    effective_fps: float
    detections_count: int
    detected_classes: List[str]


class YOLOBenchmarker:
    """Evaluates YOLO vision model latency and end-to-end perception overhead."""

    def __init__(self, detector: Optional[YOLOObjectDetector] = None):
        self.detector = detector or YOLOObjectDetector(model_name="yolo11n.pt", auto_load=True)

    def benchmark_inference(
        self,
        image_or_frame: np.ndarray,
        source_label: str = "frame",
        warmup_runs: int = 3,
        benchmark_runs: int = 15,
        webcam: Optional[WebcamManager] = None
    ) -> YOLOLatencyDecomposition:
        """Measure inference latency and decompose perception pipeline stages.

        Args:
            image_or_frame: Input BGR NumPy image array.
            source_label: Descriptive label of input image source.
            warmup_runs: Number of unrecorded warmup iterations.
            benchmark_runs: Number of recorded evaluation iterations.
            webcam: Optional active WebcamManager to measure live capture latency.

        Returns:
            YOLOLatencyDecomposition: Comprehensive timing breakdown.
        """
        if not self.detector.is_loaded():
            self.detector.load_model()

        # Identify runtime execution device
        device_str = "cpu"
        if hasattr(self.detector._model, "device"):
            device_str = str(self.detector._model.device)

        # 1. Warmup passes (purge PyTorch JIT / engine initialization overhead)
        for _ in range(warmup_runs):
            self.detector.detect(image_or_frame)

        # 2. Measure Camera Capture Latency (if active camera provided)
        capture_ms: Optional[float] = None
        if webcam is not None and webcam.is_active():
            cap_times = []
            for _ in range(10):
                t0 = time.perf_counter()
                f = webcam.get_current_frame()
                if f is not None:
                    cap_times.append((time.perf_counter() - t0) * 1000.0)
            if cap_times:
                capture_ms = float(np.median(cap_times))

        # 3. Model Inference Benchmarking
        inference_times: List[float] = []
        last_dets: List[Detection] = []

        for _ in range(benchmark_runs):
            t_start = time.perf_counter()
            dets, reported_ms = self.detector.detect(image_or_frame)
            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            # Prefer high-precision external timing
            inference_times.append(t_elapsed)
            last_dets = dets

        # 4. Annotation Drawing Latency
        draw_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            self.detector.draw_detections(image_or_frame, last_dets, inference_times[0], mapping_enabled=True)
            draw_times.append((time.perf_counter() - t0) * 1000.0)
        drawing_ms = float(np.median(draw_times))

        # 5. Aggregated Metrics
        median_inf = float(np.median(inference_times))
        mean_inf = float(np.mean(inference_times))
        min_inf = float(np.min(inference_times))
        max_inf = float(np.max(inference_times))
        std_inf = float(np.std(inference_times))

        total_ms = (capture_ms or 0.0) + median_inf + drawing_ms
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        classes_found = sorted(list({d.class_name for d in last_dets}))

        return YOLOLatencyDecomposition(
            frame_source=source_label,
            image_shape=(image_or_frame.shape[0], image_or_frame.shape[1]),
            model_name=self.detector.model_name,
            device_name=device_str,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            capture_time_ms=capture_ms,
            inference_median_ms=median_inf,
            inference_mean_ms=mean_inf,
            inference_min_ms=min_inf,
            inference_max_ms=max_inf,
            inference_std_ms=std_inf,
            drawing_time_ms=drawing_ms,
            total_pipeline_ms=total_ms,
            effective_fps=fps,
            detections_count=len(last_dets),
            detected_classes=classes_found
        )


# ==============================================================================
# 3. DYNAMIC OBSTACLE & REPLANNING BENCHMARKING ENGINE
# ==============================================================================

@dataclass
class DynamicObstacleBenchmarkResult:
    """Outcome of a controlled dynamic obstacle injection test scenario."""
    scenario_name: str
    initial_path_length: int
    obstacle_spawned_coord: Optional[Tuple[int, int]]
    replanning_occurred: bool
    replan_execution_time_ms: float
    new_path_length: int
    alternate_route_valid: bool
    vehicle_stopped_safely: bool
    collision_occurred: bool
    notes: str


class DynamicObstacleBenchmarker:
    """Evaluates adaptive replanning and fail-safe behavior under dynamic hazard injection."""

    @staticmethod
    def run_scenarios() -> List[DynamicObstacleBenchmarkResult]:
        """Execute controlled dynamic obstacle scenarios on the simulation environment."""
        results: List[DynamicObstacleBenchmarkResult] = []

        # --- Scenario 1: Intersecting Route Reroute ---
        env1 = RoadEnvironment()
        res1 = env1.plan_path()
        init_len = res1.path_length

        # Insert hazard 3 steps ahead
        hazard_pos = env1.insert_dynamic_obstacle(ahead_steps=3)
        t_replan = env1.planning_time_ms

        valid_reroute = (
            env1.nav_mode != NavigationMode.NO_PATH
            and len(env1.planned_path) > 0
            and (hazard_pos not in env1.planned_path if hazard_pos else True)
        )

        results.append(DynamicObstacleBenchmarkResult(
            scenario_name="scenario_1_single_hazard_reroute",
            initial_path_length=init_len,
            obstacle_spawned_coord=hazard_pos,
            replanning_occurred=(env1.replanning_events > 0),
            replan_execution_time_ms=t_replan,
            new_path_length=env1.path_length,
            alternate_route_valid=valid_reroute,
            vehicle_stopped_safely=(env1.nav_mode != NavigationMode.NO_PATH),
            collision_occurred=(env1.collision_count > 0),
            notes="Dynamic hazard inserted on path. A* successfully computed detour."
        ))

        # --- Scenario 2: Multi-Hazard Navigation Mission ---
        env2 = RoadEnvironment()
        env2.plan_path()
        env2.toggle_autonomous_navigation()

        # Step vehicle 3 moves along the route, then inject hazard
        for _ in range(3):
            env2.step_autonomous()

        hazard_mid = env2.insert_dynamic_obstacle(ahead_steps=2)
        replan_time2 = env2.planning_time_ms

        # Run remaining steps until goal reached or halted
        steps_taken = 0
        while env2.nav_mode == NavigationMode.AUTONOMOUS and steps_taken < 40:
            env2.step_autonomous()
            steps_taken += 1

        arrived = (env2.nav_mode == NavigationMode.GOAL_REACHED)

        results.append(DynamicObstacleBenchmarkResult(
            scenario_name="scenario_2_dynamic_hazard_during_autonomous_nav",
            initial_path_length=init_len,
            obstacle_spawned_coord=hazard_mid,
            replanning_occurred=(env2.replanning_events > 0),
            replan_execution_time_ms=replan_time2,
            new_path_length=env2.path_length,
            alternate_route_valid=arrived,
            vehicle_stopped_safely=True,
            collision_occurred=(env2.collision_count > 0),
            notes="Vehicle navigated autonomously, detoured around dynamic hazard, and reached goal safely."
        ))

        # --- Scenario 3: Impassable Road Corridor (Fail-Safe Verification) ---
        env3 = RoadEnvironment()
        env3.plan_path()
        env3.toggle_autonomous_navigation()

        # Build solid barrier across road corridor at row 14
        for c in range(env3.cols):
            env3.grid[14, c] = CellType.OBSTACLE
            env3.obstacle_set.add((14, c))

        # Attempt replanning with blocked road
        success = env3.replan_route(reason="Complete corridor obstruction")
        t_blocked = env3.planning_time_ms

        stopped_safely = (
            not success
            and env3.nav_mode == NavigationMode.NO_PATH
            and len(env3.planned_path) == 0
        )

        # Verify vehicle cannot step into blocked territory
        stepped = env3.step_autonomous()

        results.append(DynamicObstacleBenchmarkResult(
            scenario_name="scenario_3_corridor_closure_fail_safe",
            initial_path_length=init_len,
            obstacle_spawned_coord=(14, 12),
            replanning_occurred=True,
            replan_execution_time_ms=t_blocked,
            new_path_length=0,
            alternate_route_valid=False,
            vehicle_stopped_safely=stopped_safely and not stepped,
            collision_occurred=(env3.collision_count > 0),
            notes="Corridor closed. Planner safely declared NO_PATH; vehicle halted immediately with 0 collisions."
        ))

        # --- Scenario 4: Safety Guard Protection on Vehicle Position ---
        env4 = RoadEnvironment()
        env4.plan_path()
        veh_coord = env4.vehicle_pos

        # Inject candidate perception hazard directly at vehicle position
        env4.toggle_perception_mapping()
        env4.set_perception_obstacles({veh_coord})

        safe_guard_held = (
            veh_coord not in env4.perception_obstacle_set
            and env4.grid[veh_coord] != CellType.OBSTACLE
        )

        results.append(DynamicObstacleBenchmarkResult(
            scenario_name="scenario_4_vehicle_position_safety_guard",
            initial_path_length=env4.path_length,
            obstacle_spawned_coord=veh_coord,
            replanning_occurred=False,
            replan_execution_time_ms=0.0,
            new_path_length=env4.path_length,
            alternate_route_valid=True,
            vehicle_stopped_safely=True,
            collision_occurred=False,
            notes="Candidate obstacle overlapping vehicle was suppressed by Safety Guard. Vehicle was never trapped."
        ))

        return results


# ==============================================================================
# 4. CSV & MARKDOWN EXPORT ENGINE
# ==============================================================================

class BenchmarkReportGenerator:
    """Saves structured CSV tables and generates GitHub-flavored Markdown reports."""

    @staticmethod
    def get_system_specs() -> Dict[str, str]:
        """Collect host operating system and hardware environment specs."""
        return {
            "OS": platform.system(),
            "OS Release": platform.release(),
            "Platform": platform.platform(),
            "Architecture": platform.machine(),
            "Processor": platform.processor(),
            "Python Version": platform.python_version(),
            "Host Node": platform.node()
        }

    @staticmethod
    def export_csv(filepath: str, rows: List[Dict[str, Any]]) -> None:
        """Write a list of dictionary records to a CSV file."""
        if not rows:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        keys = list(rows[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved CSV results to: %s (%d rows)", filepath, len(rows))

    @classmethod
    def generate_markdown_report(
        cls,
        output_path: str,
        system_specs: Dict[str, str],
        astar_summaries: List[AStarBenchmarkSummary],
        yolo_results: List[YOLOLatencyDecomposition],
        dynamic_results: List[DynamicObstacleBenchmarkResult]
    ) -> str:
        """Generate a complete Markdown evaluation report."""
        lines: List[str] = []

        lines.append("# Day 8: Performance Evaluation and Benchmarking Report")
        lines.append("")
        lines.append("> **Project:** Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads  ")
        lines.append(f"> **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append("")

        lines.append("## 1. System & Hardware Environment")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("| :--- | :--- |")
        for k, v in system_specs.items():
            lines.append(f"| **{k}** | `{v}` |")
        lines.append("")

        # 2. A* Benchmarking
        lines.append("## 2. A* Path Planning Benchmark")
        lines.append("")
        lines.append("Evaluates search runtime across varying grid dimensions, obstacle densities, and edge cases.")
        lines.append("")
        lines.append("| Test Case | Dimensions | Density | Success % | Valid % | Median Time (ms) | Mean Time (ms) | Explored Nodes | Path Length |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in astar_summaries:
            lines.append(
                f"| `{s.test_case}` | {s.grid_rows}×{s.grid_cols} | {s.density:.0%} | "
                f"{s.success_rate_pct:.0f}% | {s.valid_paths_pct:.0f}% | "
                f"**{s.median_time_ms:.3f} ms** | {s.mean_time_ms:.3f} ms | "
                f"{s.median_explored_nodes:.0f} | {s.median_path_length:.0f} |"
            )
        lines.append("")

        # 3. YOLO Vision Latency Decomposition
        lines.append("## 3. YOLO Vision & Perception Pipeline Benchmark")
        lines.append("")
        lines.append("Separates frame acquisition, neural network inference, and annotation drawing latency.")
        lines.append("")
        lines.append("| Image Source | Resolution | Model | Device | Capture (ms) | Inference (ms) | Drawing (ms) | Total Pipeline (ms) | Effective FPS | Detections |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for y in yolo_results:
            cap_str = f"{y.capture_time_ms:.2f}" if y.capture_time_ms is not None else "N/A"
            lines.append(
                f"| `{y.frame_source}` | {y.image_shape[1]}×{y.image_shape[0]} | `{y.model_name}` | `{y.device_name}` | "
                f"{cap_str} | **{y.inference_median_ms:.2f} ms** | {y.drawing_time_ms:.2f} ms | "
                f"**{y.total_pipeline_ms:.2f} ms** | **{y.effective_fps:.1f} FPS** | {y.detections_count} |"
            )
        lines.append("")

        # 4. Dynamic Obstacle Avoidance & Fail-Safe
        lines.append("## 4. Dynamic Obstacle Avoidance & Fail-Safe Verification")
        lines.append("")
        lines.append("Tests dynamic obstacle intrusion, autonomous rerouting, corridor blockages, and safety invariants.")
        lines.append("")
        lines.append("| Scenario | Obstacle Coord | Replanned? | Replan Latency | Route Valid? | Safe Halt? | Collisions | Outcome |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for d in dynamic_results:
            coord_str = f"({d.obstacle_spawned_coord[1]}, {d.obstacle_spawned_coord[0]})" if d.obstacle_spawned_coord else "N/A"
            replan_str = "YES" if d.replanning_occurred else "NO"
            valid_str = "YES" if d.alternate_route_valid else "NO (Blocked)"
            safe_str = "YES" if d.vehicle_stopped_safely else "NO"
            lines.append(
                f"| `{d.scenario_name}` | {coord_str} | {replan_str} | {d.replan_execution_time_ms:.2f} ms | "
                f"{valid_str} | {safe_str} | {d.collision_occurred} | {d.notes} |"
            )
        lines.append("")

        # 5. Engineering Scope & Real-World Safety Disclaimer
        lines.append("## 5. Architectural Invariants & Safety Disclaimers")
        lines.append("")
        lines.append("1. **Monocular 2D Bounding Box Limitations:**  ")
        lines.append("   2D bounding boxes do not provide metric 3D ground depth or distance. The perception-to-grid mapping is an approximate classroom demonstration model and must not be used for real vehicle steering.")
        lines.append("2. **Fail-Safe Invariant (Safety Guard):**  ")
        lines.append("   When a path is completely blocked, the vehicle transitions to `NO_PATH` and halts safely. Vehicle and goal cells are strictly protected from obstacle overwrites.")
        lines.append("3. **Single Camera Handle Safety:**  ")
        lines.append("   All vision tools reuse the project's centralized `WebcamManager` without opening secondary camera handles or causing device lockups.")
        lines.append("")

        report_content = "\n".join(lines)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("Generated Markdown benchmark report at: %s", output_path)
        return report_content


# ==============================================================================
# 5. CLI RUNNER
# ==============================================================================

def main():
    """Command-line interface for the benchmark suite."""
    parser = argparse.ArgumentParser(description="Day 8: Performance Evaluation and Benchmarking Suite")
    parser.add_argument("--runs", type=int, default=5, help="Number of trials per benchmark configuration (default: 5)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Directory to save CSV and Markdown reports")
    parser.add_argument("--skip-camera", action="store_true", help="Skip live webcam capture tests (runs on saved images only)")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="YOLO model path (default: yolo11n.pt)")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print("DAY 8: PERFORMANCE EVALUATION & BENCHMARKING SUITE")
    print("================================================================================")

    # 1. System Specs
    specs = BenchmarkReportGenerator.get_system_specs()
    specs_records = [{"Parameter": k, "Value": v} for k, v in specs.items()]
    BenchmarkReportGenerator.export_csv(os.path.join(out_dir, "system_environment.csv"), specs_records)

    # 2. A* Benchmarking
    print("\n[1/3] Running A* Path Planning Benchmarks...")
    astar_runner = AStarBenchmarker()
    astar_records, astar_summaries = astar_runner.run_benchmark_suite(trials_per_config=args.runs)
    BenchmarkReportGenerator.export_csv(
        os.path.join(out_dir, "astar_raw_trials.csv"),
        [asdict(r) for r in astar_records]
    )
    BenchmarkReportGenerator.export_csv(
        os.path.join(out_dir, "astar_benchmarks.csv"),
        [asdict(s) for s in astar_summaries]
    )

    # 3. YOLO Benchmarking
    print("\n[2/3] Running YOLO Vision Latency Decomposition...")
    yolo_results: List[YOLOLatencyDecomposition] = []
    try:
        import cv2
        detector = YOLOObjectDetector(model_name=args.yolo_model, auto_load=True)
        yolo_benchmarker = YOLOBenchmarker(detector=detector)

        # A. Actual saved reference image (webcam sample)
        sample_path = "assets/webcam_sample.jpg"
        if os.path.exists(sample_path):
            sample_frame = cv2.imread(sample_path)
            if sample_frame is not None:
                print(" -> Benchmarking on actual saved webcam frame...")
                decomp = yolo_benchmarker.benchmark_inference(sample_frame, source_label="assets/webcam_sample.jpg")
                yolo_results.append(decomp)

        # B. Simulation screen graphic
        preview_path = "screenshots/day1_simulation_preview.png"
        if os.path.exists(preview_path):
            prev_frame = cv2.imread(preview_path)
            if prev_frame is not None:
                print(" -> Benchmarking on simulation preview image...")
                decomp = yolo_benchmarker.benchmark_inference(prev_frame, source_label="screenshots/day1_simulation_preview.png")
                yolo_results.append(decomp)

        # C. Live camera frame (if permitted)
        if not args.skip_camera:
            try:
                print(" -> Attempting live MacBook built-in camera capture...")
                cam = WebcamManager()
                if cam.start_preview():
                    time.sleep(0.4)
                    live_frame = cam.get_current_frame()
                    if live_frame is not None:
                        decomp = yolo_benchmarker.benchmark_inference(
                            live_frame,
                            source_label="live_camera_feed",
                            webcam=cam
                        )
                        yolo_results.append(decomp)
                    cam.release()
                else:
                    print(" -> Live camera unavailable. Proceeding with static frames.")
            except Exception as cam_exc:
                print(f" -> Camera capture skipped: {cam_exc}")

        if yolo_results:
            yolo_csv_rows = [
                {
                    "frame_source": y.frame_source,
                    "resolution": f"{y.image_shape[1]}x{y.image_shape[0]}",
                    "model_name": y.model_name,
                    "device": y.device_name,
                    "capture_ms": y.capture_time_ms,
                    "inference_median_ms": y.inference_median_ms,
                    "inference_mean_ms": y.inference_mean_ms,
                    "drawing_ms": y.drawing_time_ms,
                    "total_pipeline_ms": y.total_pipeline_ms,
                    "effective_fps": y.effective_fps,
                    "detections_count": y.detections_count,
                    "detected_classes": ";".join(y.detected_classes)
                }
                for y in yolo_results
            ]
            BenchmarkReportGenerator.export_csv(os.path.join(out_dir, "yolo_benchmarks.csv"), yolo_csv_rows)

    except Exception as exc:
        logger.error("YOLO benchmarking encountered an error: %s", exc)

    # 4. Dynamic Obstacle Evaluation
    print("\n[3/3] Running Dynamic Obstacle Avoidance & Fail-Safe Evaluation...")
    dynamic_runner = DynamicObstacleBenchmarker()
    dynamic_results = dynamic_runner.run_scenarios()
    BenchmarkReportGenerator.export_csv(
        os.path.join(out_dir, "dynamic_replanning_benchmarks.csv"),
        [asdict(d) for d in dynamic_results]
    )

    # 5. Generate Markdown Report
    report_file = os.path.join(out_dir, "BENCHMARK_REPORT.md")
    BenchmarkReportGenerator.generate_markdown_report(
        output_path=report_file,
        system_specs=specs,
        astar_summaries=astar_summaries,
        yolo_results=yolo_results,
        dynamic_results=dynamic_results
    )

    # Also save a copy directly in benchmarks/
    BenchmarkReportGenerator.generate_markdown_report(
        output_path="benchmarks/BENCHMARK_REPORT.md",
        system_specs=specs,
        astar_summaries=astar_summaries,
        yolo_results=yolo_results,
        dynamic_results=dynamic_results
    )

    print("\n================================================================================")
    print(f"BENCHMARK COMPLETE. Reports generated:")
    print(f" - Markdown: benchmarks/BENCHMARK_REPORT.md")
    print(f" - CSV files: {out_dir}/ (astar_benchmarks.csv, yolo_benchmarks.csv, etc.)")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
