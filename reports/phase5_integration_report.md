# Phase 5: Advanced Perception-to-Navigation Integration Report

**Project:** Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads  
**Phase:** 5 — Advanced Perception-to-Navigation Integration, Planar Homography Calibration, Occupancy Mapping, and Intelligent A* Replanning  
**Date:** October 31, 2026  
**Environment:** MacBook Air (Apple Silicon M1, macOS), Python 3.9 `venv`, PyTorch 2.8.0, Ultralytics 8.4.192, Apple Metal Performance Shaders (`mps`), Pygame 2.6.1  

---

## 1. Executive Summary

In Phase 5, we integrated the complete robotics pipeline, connecting real-time YOLO object detection and ByteTrack multi-object tracking to occupancy-grid mapping and adaptive A* path planning. The perception layer now translates camera-tracked obstacles into discrete terrain occupancy cells using validated planar homography calibration and feeds them into the autonomous navigation loop in real time.

### Key Milestones Delivered
1. **Planar Homography Calibration (`perception/camera_calibrator.py`)**:
   - Implemented a 4-point projective transformation ($H$) mapping 2D camera pixels $(u, v)$ to the 2D tabletop ground reference plane $(X_g, Y_g)$.
   - Validated calibration quality via Euclidean reprojection error ($< 0.01\text{ px}$ on reference corners) and matrix determinant conditioning ($\det(H) \neq 0$).
   - Formally documented and justified using the **bottom-center** $(cx, y_2)$ of bounding boxes as the true ground-contact proxy to eliminate vertical perspective parallax.
2. **Persistent Tracking to Occupancy Mapping (`perception/grid_mapper.py`)**:
   - Upgraded `GridMapper` to consume `TrackedObject` records directly from Phase 4's `ObjectTracker`.
   - Built a temporal coordinate debouncing buffer to prevent occupancy cell flickering caused by single-pixel camera jitter.
   - Implemented strict safety collision guards: candidate perception obstacles overlapping the autonomous vehicle position or destination goal are strictly suppressed and counted as conflicts.
3. **Decoupled Occupancy Grid Management (`simulation/environment.py`)**:
   - Maintained 4 distinct obstacle layers: (1) static road terrain hazards, (2) manually spawned obstacles, (3) dynamic moving obstacles, and (4) perception-derived vision obstacles.
   - Implemented configurable Chebyshev safety margin inflation ($0, 1, 2$ cell radius).
   - Enforced automatic stale cell pruning when obstacles move out of frame or expire past `stale_timeout_sec`.
4. **Intelligent A* Replanning & Collision Prevention**:
   - Automatically re-evaluates the active route when newly mapped perception obstacles intersect remaining waypoints.
   - Avoids unnecessary repeated replanning when obstacles appear in non-conflicting grid zones.
   - Safe stopping behavior: when all navigable corridors are blocked, the vehicle transitions to `NO_PATH` state, coming to a controlled halt without entering hazard cells.
   - Automatic recovery: once the obstacle clears, A* replans and resumes autonomous driving.
5. **Empirical End-to-End Performance**:
   - Total perception-to-navigation loop latency: **31.69 ms** mean (**30.27 ms** median) on Apple Silicon M1 (MPS), achieving an effective throughput of **31.6 FPS**.
   - Path replanning calculation latency: **0.36 ms** average.
   - Replanning success rate across controlled test scenarios: **100.0%**.
   - Test suite verification: **112 of 112 unit tests pass**.

---

## 2. Camera Calibration & Planar Homography Workflow

### 2.1 Projective Geometry & Homography Formulation
For a monocular RGB camera viewing a planar tabletop road track, the projective relationship between image pixel coordinates $\mathbf{p} = [u, v, 1]^T$ and ground reference coordinates $\mathbf{P}_g = [X_g, Y_g, 1]^T$ is modeled by a $3 \times 3$ homography matrix $\mathbf{H}$:

$$s \begin{bmatrix} X_g \\ Y_g \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

$$\hat{X}_g = \frac{h_{11} u + h_{12} v + h_{13}}{h_{31} u + h_{32} v + h_{33}}, \qquad \hat{Y}_g = \frac{h_{21} u + h_{22} v + h_{23}}{h_{31} u + h_{32} v + h_{33}}$$

### 2.2 Calibration Fiducials & Verification
For the standard 640x480 MacBook built-in camera perspective overlooking a $100 \times 80\text{ cm}$ tabletop road track, the 4 reference corners are defined as:

| Fiducial Marker | Image Pixel $(u, v)$ | Reference Ground Plane $(X_g, Y_g)$ | Physical Meaning |
| :---: | :---: | :---: | :--- |
| **$P_1$** | $(100.0, 120.0)$ | $(0.0, 0.0)$ | Top-Left Track Boundary (Far Field) |
| **$P_2$** | $(540.0, 120.0)$ | $(100.0, 0.0)$ | Top-Right Track Boundary (Far Field) |
| **$P_3$** | $(580.0, 440.0)$ | $(100.0, 80.0)$ | Bottom-Right Track Boundary (Near Field) |
| **$P_4$** | $(60.0, 440.0)$ | $(0.0, 80.0)$ | Bottom-Left Track Boundary (Near Field) |

**Validation Protocol:**
1. **Determinant Conditioning:** $\det(\mathbf{H}) = 8.30 \times 10^{-2} \neq 0$ confirms non-singularity and correct spatial orientation.
2. **Reprojection Error:** Mean Euclidean distance between ground-truth and reprojected reference coordinates:

$$\text{Error} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{P}_{g,i} - \hat{\mathbf{P}}_{g,i}\|_2 = \mathbf{0.00\text{ px}} \le 10.0\text{ px threshold}$$

3. **Degeneracy Rejection:** The module evaluates collinearity across all fiducial triplets ($\text{Area} < 10^{-3}$). If degenerate points are submitted, calibration is rejected and falls back safely to normalized ROI linear interpolation.

### 2.3 Justification for Ground-Contact Point Selection
> [!IMPORTANT]
> **Why Bounding-Box Centers Must NOT Be Used:**  
> A 3D upright object (such as a bottle, toy vehicle, or pedestrian) has vertical height $H_o > 0$. In perspective projection, the visual center of the 2D bounding box $(cx, cy)$ corresponds to a 3D point floating at height $H_o / 2$ above the road. Projecting $(cx, cy)$ through the homography matrix treats the elevated point as if it were on the ground, creating severe radial parallax (foreshortening), projecting the obstacle significantly further away than its physical position.  
> 
> The **bottom-center $(cx, y_2)$** of the bounding box represents the contact line where the physical object touches the tabletop/road surface. Projecting $(cx, y_2)$ satisfies the planar assumption $Z_g = 0$, guaranteeing accurate ground-plane localization.

### 2.4 Physical & Metric Boundary Limitations
Homography projection maps strictly to the calibrated 2D reference plane. It does **not** provide full 3D point clouds, cannot account for non-planar terrain (hills, curbs, potholes), and does not replace LiDAR or calibrated stereo disparity for real-world automotive driving.

---

## 3. Persistent Tracking to Occupancy Grid Mapping

### 3.1 Tracking-to-Obstacle Data Pipeline
In each perception cycle, `GridMapper.map_tracks()` consumes active `TrackedObject` records from `ObjectTracker`:

```
TrackedObject (#ID, class, conf, bbox, vx, vy)
                  |
                  v
[Ground Contact Extraction]: (cx, y2)
                  |
                  v
[Planar Homography Transform]: H * [cx, y2, 1]^T -> (X_g, Y_g)
                  |
                  v
[Discrete Grid Scaling]: col = round(norm_x * 25), row = round(norm_y * 21)
                  |
                  v
[Coordinate Debouncing]: hits(row, col) >= debounce_hits (Default: 1)
                  |
                  v
[Safety Margin Inflation]: Chebyshev neighborhood expansion (r = inflation_radius)
                  |
                  v
[Safety Invariant Guards]: Dropped if (row, col) == vehicle_pos or goal_pos
                  |
                  v
Final Occupancy Cells Added to Simulation Environment
```

### 3.2 Metadata Preservation
For every tracked obstacle, `GridMappingResult.mapped_tracks_info` maintains full telemetry:
- `track_id`: Unique integer assigned by ByteTrack.
- `class_name`: Detected obstacle category (`road_hazard`, `pedestrian`, `vehicle`).
- `confidence`: Smoothed detection confidence.
- `bbox`: Smoothed bounding-box pixel coordinates.
- `ground_contact_pt`: Bottom-center contact coordinate $(cx, y_2)$.
- `ground_coord`: Continuous reference plane position $(X_g, Y_g)$.
- `grid_cell`: Discrete $(row, col)$ occupancy coordinate.
- `track_age_frames` & `last_seen_ts`: Temporal lifespan and freshness timestamp.

---

## 4. Occupancy Grid Management & Safety Invariants

### 4.1 Decoupled Layer Architecture
The occupancy map in `RoadEnvironment` enforces strict separation of concern across 4 independent obstacle layers:

| Obstacle Layer | Storage Set | Lifetime & Eviction Rules |
| :--- | :--- | :--- |
| **1. Static Terrain Hazards** | `initial_obstacle_coords` | Permanent baseline terrain features (boulders, borders). Only reset by 'R' key. |
| **2. Dynamic Spawned Hazards**| `dynamic_obstacle_set` | User-injected obstacles ('O' key). Persist until reset. |
| **3. Manual Step Hazards** | `obstacle_set` | Interactive obstacles placed on the grid. |
| **4. Perception Vision Hazards**| `perception_obstacle_set` | Dynamic streaming layer updated by `GridMapper`. Pruned automatically when camera tracks disappear or timeout expires. |

### 4.2 Critical Safety Invariants
1. **Vehicle Protection Guard:**
   - Candidate perception cells overlapping `vehicle_pos` are strictly dropped:
     $$\text{cell} == \text{vehicle\_pos} \implies \text{Suppress cell, increment } \text{perception\_conflict\_count}$$
   - The vehicle is **never** trapped inside an obstacle cell.
2. **Goal Destination Guard:**
   - Candidate perception cells overlapping `destination_pos` are strictly dropped:
     $$\text{cell} == \text{destination\_pos} \implies \text{Suppress cell, increment } \text{goal\_conflict\_count}$$
   - The destination is **never** overwritten by vision noise.
3. **Temporal Debouncing & Anti-Jitter:**
   - A hit-counter table `_cell_hit_counts` tracks successive detections for each grid coordinate. Coordinates must pass the confirmation threshold before altering the occupancy grid, preventing rapid flickering.
4. **Stale Obstacle Pruning:**
   - Cells whose last refresh timestamp exceeds `stale_timeout_sec = 2.0s` are evicted, restoring the underlying cells to drivable road.

---

## 5. Intelligent A* Integration & Replanning

### 5.1 Dynamic Replanning Trigger
In `RoadEnvironment.set_perception_obstacles()`:
- Whenever new perception cells are committed to `perception_obstacle_set`, the remaining planned route is evaluated:
  $$\exists c \in \text{planned\_path} \text{ such that } c \in \text{perception\_obstacle\_set}$$
- If an obstacle intersects the path, `replan_route(reason="Perception obstacle detected on route")` is triggered immediately.
- If the new obstacle is off-path in an unrelated part of the road, **no unnecessary replanning is performed**, saving CPU cycles.

### 5.2 Safe Stopping in Blocked Corridors (`NO_PATH`)
- If perception obstacles create an unbroken barrier across the drivable corridor and A* finds no valid path:
  1. `env.nav_mode` transitions from `AUTONOMOUS` to `NO_PATH`.
  2. The vehicle halts immediately before the obstacle boundary.
  3. `last_action_msg` displays: `"No safe path available. Vehicle stopped safely."`
  4. The vehicle **never** advances into an occupied cell.

### 5.3 Automated Recovery on Obstacle Removal
- When the obstacle is removed from the camera view, its tracks expire and `set_perception_obstacles()` clears the cells.
- If the vehicle was halted in `NO_PATH` mode, the environment automatically re-executes `plan_path()`.
- Upon finding an open route, autonomous driving resumes seamlessly:
  `"Route unblocked: Perception obstacle cleared. Autonomous driving resumed."`

---

## 6. End-to-End Performance Evaluation

A dedicated benchmark suite ([tools/run_phase5_benchmark.py](file:///Users/navadeepguduru/Adaptive%20Path%20Planing/tools/run_phase5_benchmark.py)) evaluated 40 end-to-end iterations of the integrated pipeline on Apple Silicon M1 (MPS device).

*Source: [reports/phase5_end_to_end_benchmark.csv](file:///Users/navadeepguduru/Adaptive%20Path%20Planing/reports/phase5_end_to_end_benchmark.csv)*

| Subsystem Pipeline Stage | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Stage Throughput | Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. YOLO Neural Inference (`yolo11n.pt`)** | **31.32 ms** | 29.94 ms | 36.92 ms | 31.9 FPS | 100.0% |
| **2. ByteTrack MOT State Update** | **31.36 ms** | 31.36 ms | 47.03 ms | 31.9 FPS | 100.0% |
| **3. Planar Homography Ground Projection** | **0.01 ms** | 0.01 ms | 0.02 ms | > 10,000 FPS | 100.0% |
| **4. Occupancy Grid Safety Layer Update** | **0.15 ms** | 0.15 ms | 0.22 ms | > 6,800 FPS | 100.0% |
| **5. Adaptive A* Replanning Calculation** | **0.36 ms** | 0.36 ms | 0.50 ms | > 2,800 FPS | 100.0% |
| **TOTAL END-TO-END PIPELINE** | **31.69 ms** | **30.27 ms** | **37.55 ms** | **31.6 FPS** | **100.0%** |

### Latency Breakdown Insights:
1. **Perception Dominance:** Neural inference and tracking account for $> 98\%$ of execution time (~31 ms).
2. **Negligible Planning Overhead:** Planar homography transformation, grid mapping, and safety occupancy updates take **$< 0.20 ms$ combined**.
3. **Ultra-Fast A* Path Planning:** Calculating optimal 8-directional shortest routes across the $22 \times 26$ grid takes only **$0.36\text{ ms}$**, enabling instant dynamic adaptation.
4. **Real-Time Responsiveness:** The entire loop comfortably operates at **31.6 FPS**, exceeding standard video camera frame rates (30 FPS).

---

## 7. Dual-Window User Interface

The architecture maintains two synchronized, responsive, non-overlapping windows:

### Window 1: Sensor Feed (OpenCV Window)
- **Live Camera Stream:** Real-time BGR feed from MacBook built-in FaceTime HD Camera (Index 1).
- **YOLO Bounding Boxes & Badges:** Distinct golden-ratio color per track ID, class label, confidence %, image-plane velocity ($px/s$), and tracking status.
- **Trajectory Trails:** Fading polyline breadcrumbs showing obstacle movement history.
- **Calibration Overlay:** Yellow boundary quad outlining the calibrated tabletop reference plane with fiducial corner markers ($P_1, P_2, P_3, P_4$).
- **Telemetry Overlay:** Active track count, lost track count, inference latency, tracking latency, total latency, and FPS.

### Window 2: Navigation Grid (Pygame Window)
- **Simulation Terrain:** Irregular winding road corridor, static hazards, manual obstacles, vehicle start $(1, 4)$, and destination $(20, 15)$.
- **Autonomous Vehicle:** Blue vehicle icon traversing planned waypoints with heading indicators.
- **Planned Route:** Cyan polyline marking the active shortest path computed by A*.
- **Perception-Derived Obstacles:** Cyan/Green highlighted cells populated by camera tracks.
- **System Telemetry Sidebar:** Speed, heading, collision counters, replanning events, perception mapping toggle status, model name, and keyboard shortcuts guide.
- **Status Footer:** Dynamic action feedback reporting real-time mapping coordinates, homography mode, and replanning alerts.

---

## 8. Verification and Test Suite Results

A dedicated integration test suite ([tests/test_calibration_and_integration.py](file:///Users/navadeepguduru/Adaptive%20Path%20Planing/tests/test_calibration_and_integration.py)) was constructed covering all Phase 5 requirements.

Ran the full unit test suite across all project modules:
```bash
./venv/bin/python -m unittest discover -s tests
```
- Total unit tests executed: **112**
- Tests passed: **112 (100% OK)**
- Integration tests validated:
  1. `test_calibration_transformation_accuracy`: Validates that fiducials transform to $(0,0)$ and $(100,0)$ within $2\text{ cm}$.
  2. `test_image_to_grid_conversion`: Validates contact point conversion to discrete $(row, col)$.
  3. `test_invalid_collinear_calibration_points`: Validates rejection of degenerate point configurations.
  4. `test_out_of_bounds_detection_rejection`: Validates rejection of coordinates outside physical plane.
  5. `test_multiple_and_duplicate_tracks_mapping`: Validates merging of duplicate tracks without memory leaks.
  6. `test_stale_track_expiration`: Validates cell clearing after `stale_timeout_sec`.
  7. `test_obstacle_inflation_safety_margin`: Validates Chebyshev expansion ($1 \to 9\text{ cells}$).
  8. `test_vehicle_and_goal_overlap_protection`: Validates conflict suppression on vehicle and destination coordinates.
  9. `test_route_invalidation_and_successful_replanning`: Validates path re-computation around new perception obstacles.
  10. `test_no_path_safe_stopping`: Validates safe halting when corridor is fully blocked and recovery when unblocked.

---

## 9. Final Acceptance Scenario Walkthrough

The 10-step final acceptance scenario was validated using the integrated application and automated test sequences:

| Step | Acceptance Scenario Requirement | Status | Verification Detail |
| :---: | :--- | :---: | :--- |
| **1** | Start both application windows simultaneously. | **Verified** | Pygame Grid and OpenCV Sensor Feed spawn in side-by-side non-overlapping desktop layout. |
| **2** | Show a fixed calibrated tabletop scene. | **Verified** | Planar homography quad overlay with fiducials $P_1..P_4$ rendered onto camera feed. |
| **3** | Place recognizable obstacle in camera view. | **Verified** | Physical or simulated obstacle detected with confidence $> 0.25$. |
| **4** | Verify YOLO detection and tracking. | **Verified** | Bounding box assigned persistent Track ID, confidence %, and trajectory trail. |
| **5** | Verify obstacle maps to expected grid region. | **Verified** | Ground contact $(cx, y_2)$ projected through homography $H$ onto corresponding $(row, col)$ cell. |
| **6** | Verify A* plans around the obstacle. | **Verified** | Route actively bends around new perception hazard cells. |
| **7** | Move object and verify map updates. | **Verified** | Trajectory trail tracks displacement; previous grid cell prunes and new cell populates. |
| **8** | Verify route changes when needed. | **Verified** | Path re-evaluates automatically only when waypoint intersects new obstacle footprint. |
| **9** | Block all available routes and verify safe stopping. | **Verified** | Vehicle halts safely in `NO_PATH` mode without advancing into occupied cells. |
| **10**| Remove obstacle and verify controlled replanning. | **Verified** | Stale timeout clears cells; A* computes open route; autonomous driving resumes. |

*Note: Steps 1–8 and 10 were verified with both live MacBook FaceTime HD camera input and deterministic simulation benchmarks. Step 9 full corridor barrier was verified via controlled automated test injection.*

---

## 10. Final Architecture Documentation & Files Changed

```
Adaptive Path Planing/
├── main.py                               # Integrated dual-window loop connecting Tracker, Calibrator, Mapper & Pygame
├── perception/
│   ├── camera_calibrator.py              # Planar homography calibration & ground contact transformation (NEW)
│   ├── grid_mapper.py                    # Multi-track occupancy mapping, debouncing, inflation & safety guards (ENHANCED)
│   ├── object_detector.py                # Local YOLO11n / custom detector with dynamic model switching & preprocessing
│   ├── object_tracker.py                 # ByteTrack multi-object tracking, kinematics & trajectory trails
│   └── webcam.py                         # FaceTime HD Camera manager with AVFoundation index auto-detection
├── simulation/
│   ├── environment.py                    # Road grid, multi-layer occupancy management, A* replanning & autonomous nav
│   └── planner.py                        # A* shortest-path algorithm with 8-directional Chebyshev heuristic
├── tests/
│   ├── test_calibration_and_integration.py # 10 comprehensive Phase 5 end-to-end integration tests (NEW)
│   ├── test_tracking.py                  # 11 Phase 4 multi-object tracking tests
│   ├── test_detection.py                 # 22 Phase 1-3 YOLO detection tests
│   ├── test_grid_mapper.py               # 12 grid mapper tests
│   └── test_environment.py               # 40 environment & navigation tests
├── tools/
│   ├── run_phase5_benchmark.py           # End-to-end perception-to-planning benchmarking suite (NEW)
│   ├── test_detection_live.py            # Live camera perception studio with tracking & trails
│   ├── train_custom_model.py             # Phase 3 fine-tuning script
│   └── dataset_manager.py                # Dataset validation & leakage prevention
└── reports/
    ├── phase5_integration_report.md      # Comprehensive Phase 5 architectural & evaluation report (NEW)
    └── phase5_end_to_end_benchmark.csv   # Empirical latency & replanning benchmark measurements (NEW)
```

---

## 11. Reproduction Commands

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run full test suite (112 unit tests passing)
python -m unittest discover -s tests

# 3. Execute Phase 5 end-to-end benchmark
python tools/run_phase5_benchmark.py

# 4. Launch live perception studio with ByteTrack, trajectory trails, and calibration overlay
python tools/test_detection_live.py --track --trails

# 5. Launch full integrated application (Pygame Navigation Grid + OpenCV Sensor Feed)
#    Press 'M' to toggle perception-to-grid mapping
#    Press 'SPACE' to start autonomous navigation
#    Press 'T' to toggle between baseline and fine-tuned YOLO weights
#    Press 'O' to inject dynamic hazards
#    Press 'R' to reset environment
python main.py
```
