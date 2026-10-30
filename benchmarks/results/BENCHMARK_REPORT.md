# Day 8: Performance Evaluation and Benchmarking Report

> **Project:** Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads  
> **Date:** 2026-10-30 11:43:08  

## 1. System & Hardware Environment

| Parameter | Value |
| :--- | :--- |
| **OS** | `Darwin` |
| **OS Release** | `27.0.0` |
| **Platform** | `macOS-27.0.1-arm64-arm-64bit` |
| **Architecture** | `arm64` |
| **Processor** | `arm` |
| **Python Version** | `3.9.6` |
| **Host Node** | `Navadeeps-MacBook-Air.local` |

## 2. A* Path Planning Benchmark

Evaluates search runtime across varying grid dimensions, obstacle densities, and edge cases.

| Test Case | Dimensions | Density | Success % | Valid % | Median Time (ms) | Mean Time (ms) | Explored Nodes | Path Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random_10x10_d0` | 10×10 | 0% | 100% | 100% | **0.059 ms** | 0.072 ms | 19 | 18 |
| `random_10x10_d10` | 10×10 | 10% | 100% | 100% | **0.080 ms** | 0.085 ms | 24 | 18 |
| `random_10x10_d20` | 10×10 | 20% | 33% | 100% | **0.066 ms** | 0.102 ms | 20 | 18 |
| `random_10x10_d30` | 10×10 | 30% | 33% | 100% | **0.067 ms** | 0.056 ms | 24 | 18 |
| `random_10x10_d40` | 10×10 | 40% | 33% | 100% | **0.088 ms** | 0.119 ms | 31 | 18 |
| `random_22x26_d0` | 22×26 | 0% | 100% | 100% | **0.142 ms** | 0.143 ms | 47 | 46 |
| `random_22x26_d10` | 22×26 | 10% | 100% | 100% | **0.221 ms** | 0.252 ms | 73 | 46 |
| `random_22x26_d20` | 22×26 | 20% | 100% | 100% | **0.250 ms** | 0.263 ms | 86 | 46 |
| `random_22x26_d30` | 22×26 | 30% | 100% | 100% | **0.399 ms** | 0.406 ms | 145 | 48 |
| `random_22x26_d40` | 22×26 | 40% | 0% | 100% | **0.016 ms** | 0.019 ms | 6 | 0 |
| `random_50x50_d0` | 50×50 | 0% | 100% | 100% | **0.287 ms** | 0.293 ms | 99 | 98 |
| `random_50x50_d10` | 50×50 | 10% | 100% | 100% | **0.665 ms** | 0.725 ms | 201 | 98 |
| `random_50x50_d20` | 50×50 | 20% | 100% | 100% | **0.627 ms** | 0.641 ms | 207 | 98 |
| `random_50x50_d30` | 50×50 | 30% | 67% | 100% | **2.597 ms** | 3.149 ms | 924 | 108 |
| `random_50x50_d40` | 50×50 | 40% | 0% | 100% | **0.007 ms** | 0.013 ms | 2 | 0 |
| `random_100x100_d0` | 100×100 | 0% | 100% | 100% | **0.592 ms** | 0.592 ms | 199 | 198 |
| `random_100x100_d10` | 100×100 | 10% | 100% | 100% | **2.012 ms** | 2.224 ms | 629 | 198 |
| `random_100x100_d20` | 100×100 | 20% | 67% | 100% | **2.147 ms** | 10.252 ms | 670 | 198 |
| `random_100x100_d30` | 100×100 | 30% | 100% | 100% | **5.707 ms** | 4.564 ms | 1833 | 200 |
| `random_100x100_d40` | 100×100 | 40% | 0% | 100% | **9.753 ms** | 7.073 ms | 3486 | 0 |
| `edge_blocked_start` | 22×26 | 0% | 0% | 100% | **0.001 ms** | 0.001 ms | 0 | 0 |
| `edge_blocked_goal` | 22×26 | 0% | 0% | 100% | **0.001 ms** | 0.001 ms | 0 | 0 |
| `edge_partitioned_wall_no_path` | 22×26 | 4% | 0% | 100% | **0.873 ms** | 0.877 ms | 264 | 0 |
| `simulation_baseline_unstructured_road` | 22×26 | 2% | 100% | 100% | **0.245 ms** | 0.250 ms | 81 | 27 |

## 3. YOLO Vision & Perception Pipeline Benchmark

Separates frame acquisition, neural network inference, and annotation drawing latency.

| Image Source | Resolution | Model | Device | Capture (ms) | Inference (ms) | Drawing (ms) | Total Pipeline (ms) | Effective FPS | Detections |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `assets/webcam_sample.jpg` | 1280×720 | `yolo11n.pt` | `cpu` | N/A | **33.83 ms** | 0.14 ms | **33.96 ms** | **29.4 FPS** | 0 |
| `screenshots/day1_simulation_preview.png` | 1000×700 | `yolo11n.pt` | `cpu` | N/A | **38.74 ms** | 0.11 ms | **38.84 ms** | **25.7 FPS** | 0 |

## 4. Dynamic Obstacle Avoidance & Fail-Safe Verification

Tests dynamic obstacle intrusion, autonomous rerouting, corridor blockages, and safety invariants.

| Scenario | Obstacle Coord | Replanned? | Replan Latency | Route Valid? | Safe Halt? | Collisions | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `scenario_1_single_hazard_reroute` | (13, 4) | YES | 0.28 ms | YES | YES | False | Dynamic hazard inserted on path. A* successfully computed detour. |
| `scenario_2_dynamic_hazard_during_autonomous_nav` | (13, 6) | YES | 0.24 ms | YES | YES | False | Vehicle navigated autonomously, detoured around dynamic hazard, and reached goal safely. |
| `scenario_3_corridor_closure_fail_safe` | (12, 14) | YES | 0.34 ms | NO (Blocked) | YES | False | Corridor closed. Planner safely declared NO_PATH; vehicle halted immediately with 0 collisions. |
| `scenario_4_vehicle_position_safety_guard` | (12, 2) | NO | 0.00 ms | YES | YES | False | Candidate obstacle overlapping vehicle was suppressed by Safety Guard. Vehicle was never trapped. |

## 5. Architectural Invariants & Safety Disclaimers

1. **Monocular 2D Bounding Box Limitations:**  
   2D bounding boxes do not provide metric 3D ground depth or distance. The perception-to-grid mapping is an approximate classroom demonstration model and must not be used for real vehicle steering.
2. **Fail-Safe Invariant (Safety Guard):**  
   When a path is completely blocked, the vehicle transitions to `NO_PATH` and halts safely. Vehicle and goal cells are strictly protected from obstacle overwrites.
3. **Single Camera Handle Safety:**  
   All vision tools reuse the project's centralized `WebcamManager` without opening secondary camera handles or causing device lockups.
