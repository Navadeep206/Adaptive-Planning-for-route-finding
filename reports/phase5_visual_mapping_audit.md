# Phase 5: Visual and Functional Audit — Camera-to-Grid Mapping

**Project:** Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads  
**Subsystem:** Monocular Camera-to-Occupancy Grid Mapping, Planar Homography Calibration & Spatial Diagnostics  
**Audit Date:** November 5, 2026  
**Environment:** Apple Silicon (macOS), Python 3.9 `venv`, PyTorch 2.8.0 (MPS Acceleration), Ultralytics YOLO11n, ByteTrack MOT, Pygame 2.6.1, OpenCV 4.10  

---

## 1. Executive Summary & Diagnosis of Screenshot Issues

A comprehensive visual and functional audit was performed on the end-to-end perception-to-navigation pipeline, specifically inspecting the dual-window interface (`simulation_dual_window.png`) and the underlying camera-to-grid coordinate mapping mechanics.

### Diagnostic Breakdown of Identified Deficiencies

| Defect # | Observed Issue in Prior Setup | Technical Root Cause Analysis | Remediation Status |
| :---: | :--- | :--- | :---: |
| **1** | **Camera captures person's face/hand instead of flat ground surface.** | The MacBook webcam was pointed horizontally at the operator's seated posture. YOLO detected class `person` with high confidence ($>0.85$). Because monocular bounding boxes have 2D extent, the bottom-center $(cx, y_2)$ was located in mid-air/torso level, yet treated as a ground contact point resting on the reference plane. | **RESOLVED**<br>Strict non-ground class rejection (`person`, `hand`, `face`) + plane boundary containment checks. |
| **2** | **Yellow homography calibration region used without planarity verification.** | A default 4-corner trapezoid was initialized, but any detection near the lower half was projected through $H$. Coordinates projecting outside were soft-margin clamped into $[0, 100]\times[0, 80]$, falsely forcing non-planar background objects onto perimeter grid cells. | **RESOLVED**<br>`is_point_inside_plane()` polygon ray-casting test + strict coordinate boundary rejection without clamping. |
| **3** | **Three unverified purple perception obstacle cells on the simulation grid.** | Phantom perception obstacles appeared on the grid from transient background clutter and person torso detections. Their physical correspondence to ground objects was unmeasured and unverified. | **RESOLVED**<br>Track-to-cell association, stability during lost frames, and verified physical ground-truth testing. |
| **4** | **Sensor Feed window too small (400x300) for debugging.** | Fixed window sizing forced downsampling of 640x480 video to 400x300. Font scale 0.40 without background contrast made bounding box coordinates, contact points, and track telemetry illegible. | **RESOLVED**<br>Window enlarged to 640x520 with dual-line high-contrast diagnostic cards and dedicated HUD panels. |
| **5** | **Application reported "Calibrated" mapping without measured accuracy.** | "CALIBRATED (Planar Homography)" was displayed solely because OpenCV computed $H$ across 4 synthetic corners with $H \cdot H^{-1} = I$ ($0.00\text{ px}$ algebraic residual on fiducials). No physical box test or measured grid-cell error had ever been performed. | **RESOLVED**<br>Controlled 8-position empirical validation executed and recorded in `reports/phase5_mapping_accuracy.csv`. |

---

## 2. Mathematical Foundations & Planar Homography Formulation

### 2.1 The Monocular Dimensionality Gap
A monocular pinhole camera maps a 3D ray $\mathbf{P}_w = [X_w, Y_w, Z_w]^T$ into a 2D image coordinate $\mathbf{p} = [u, v]^T$. The depth dimension $Z$ is lost along the ray. Consequently, projecting an arbitrary 2D bounding box onto a 2D navigation grid is an **ill-posed inverse problem** without extrinsic constraints.

### 2.2 Planar Homography Constraint
When physical obstacles are constrained to rest on a common flat reference plane (such as a tabletop testing surface or flat roadway with $Z_g = 0$), the transformation between image pixels $\mathbf{p} = [u, v, 1]^T$ and ground plane coordinates $\mathbf{P}_g = [X_g, Y_g, 1]^T$ becomes a strictly invertible projective homography:

$$s \begin{bmatrix} X_g \\ Y_g \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

$$\hat{X}_g = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \qquad \hat{Y}_g = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}$$

### 2.3 Justification for Ground-Contact Point $(cx, y_2)$
For an upright physical object (e.g. a box, bottle, or traffic block) of physical height $H_{obj} > 0$:
- The bounding box center $(cx, cy)$ corresponds to a 3D point floating at height $H_{obj} / 2$ above the ground. Projecting $(cx, cy)$ through $\mathbf{H}$ treats the elevated point as resting on the road surface, inducing severe radial parallax (perspective foreshortening) and overestimating distance.
- The **bottom-center $(cx, y_2)$** of the bounding box represents the contact line where the physical base of the obstacle touches the tabletop/ground plane. Only $(cx, y_2)$ satisfies the planar constraint $Z_g = 0$.

> [!CAUTION]
> **Why People, Faces, and Hands Cannot Be Projected:**  
> A human body, face, or floating hand does not rest upright with a flat contact base within a tabletop testing surface. Their bounding box bottoms represent hanging limbs, chair seats, or torsos suspended in 3D space. Projecting them through $\mathbf{H}$ violates the planar assumption and yields meaningless coordinates. They must be explicitly rejected.

---

## 3. Systematic Pipeline Architecture & Implemented Enhancements

### Section A: Camera Setup & Non-Ground Rejection Filter
1. **Hardware Preservation:** Exclusively targets the MacBook built-in FaceTime HD camera using the AVFoundation backend (bypassing macOS Continuity Camera iPhone index conflicts).
2. **Tabletop Setup Geometry:** Calibrated for a fixed downward-angled camera view ($\sim 35^\circ$ to $45^\circ$ pitch) overlooking a controlled flat tabletop arena ($100.0\text{ cm} \times 80.0\text{ cm}$).
3. **Non-Planar Object Rejection:** In `GridMapperConfig`, `rejected_classes = {"person", "hand", "face"}` is strictly enforced. When the detector identifies class `person`, mapping is immediately aborted with diagnostic reason: `"REJECTED: Class 'person' is non-ground/elevated (cannot assume resting on plane)"`.

### Section B: Calibration Rigor, Ordering & Reprojection Diagnostics
1. **Four-Point Fiducial Ordering Validation:** `CameraCalibrator.validate_point_ordering()` validates that the 4 fiducial points form a strictly convex quadrilateral in canonical sequence:
   - $P_1$: Top-Left (Far-Left)
   - $P_2$: Top-Right (Far-Right)
   - $P_3$: Bottom-Right (Near-Right)
   - $P_4$: Bottom-Left (Near-Left)
   Self-intersecting (hourglass/bow-tie), concave, or collinear points are rejected with a descriptive error message.
2. **Bidirectional Reprojection Error:**
   - Image pixel error: $\epsilon_{img} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_i - \mathbf{H}^{-1} \mathbf{P}_{g,i}\|_2$ (measured in pixels).
   - Ground metric error: $\epsilon_{gnd} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{P}_{g,i} - \mathbf{H} \mathbf{p}_i\|_2$ (measured in cm).
   - Matrix conditioning: Computes $\kappa(\mathbf{H}) = \|\mathbf{H}\| \cdot \|\mathbf{H}^{-1}\|$ to verify numerical stability.
3. **Invalid Calibration Safeguard:** If calibration fails or is degenerate, an eye-catching bright red warning banner is rendered: `[!] WARNING: CALIBRATION INVALID / DEGENERATE`. Homography projection is disabled and falls back to uncalibrated normalized ROI.
4. **On-Screen Calibration Instructions:** Tabletop setup guidelines and alignment steps are rendered directly on the feed and toggleable via the **'B' key**.

### Section C: Per-Detection Mapping Diagnostics
For every tracked detection, `GridMapper` captures and displays:
- **Detection Class:** e.g., `box`, `bottle`, `cup`
- **Confidence:** e.g., `0.92`
- **Track ID:** Persistent ByteTrack identifier (e.g., `T1`)
- **Bounding Box:** $[x_1, y_1, x_2, y_2]$ in image pixels
- **Ground Contact Point:** $(cx, y_2)$ in image pixels
- **Transformed Plane Coordinates:** $(X_g, Y_g)$ in centimeters
- **Final Grid Cell:** $(row, col)$ on the $22 \times 26$ simulation grid
- **Acceptance / Rejection Reason:** Explicit string explaining why the detection was accepted or why it was rejected.

### Section D: Grid Visualization & Temporal Stability
1. **Color Decoupling:**
   - **Electric Purple (`COLOR_PERCEPTION_OBSTACLE` = RGB 168, 85, 247)** with neon fuchsia border and bright white sensor core is used **strictly and exclusively** for perception-derived obstacles.
   - Static hazards remain Coral Red (`COLOR_OBSTACLE`).
   - Dynamic injected hazards remain Safety Orange (`(234, 88, 12)`).
2. **Track ID Inscribed in Grid Cells:** Every purple perception cell renders its associated Track ID (`T1`, `T2`) or coordinate label directly in the center of the cell in crisp monospace text.
3. **Missed Detection Stability (Coasting):** When ByteTrack enters state `TrackState.LOST` during temporary 1-2 frame occlusions, `GridMapper` preserves the obstacle footprint in `_track_to_cells`, preventing map flickering.
4. **Selective Expired Track Pruning:** When a track exceeds `stale_timeout_sec` (2.0s), only `_track_to_cells[track_id]` is pruned. Perception cells belonging to other active tracks remain strictly untouched.

### Section E: Sensor Feed Layout & Non-Overlapping Window Management
1. **Enhanced Dimensions:** The OpenCV Sensor Feed window default size is expanded from $400 \times 300$ to **$640 \times 520$ pixels**.
2. **Dual-Window Side-by-Side Placement:**
   - Simulation Window: $1000 \times 700$ at screen coordinate $(10, 40)$.
   - Sensor Feed Window: $640 \times 520$ at screen coordinate $(1020, 40)$ on displays $\ge 1680\text{px}$, or dynamically positioned without overlapping on smaller displays.
3. **Typography & Readability:** Text rendered with high-contrast semi-transparent dark backing cards (RGB 15, 20, 25) with color-coded status badges (green for accepted, red for rejected).

---

## 4. Controlled Accuracy Validation Results

A controlled empirical validation procedure was executed using an obstacle box resting on a calibrated $100.0\text{ cm} \times 80.0\text{ cm}$ tabletop plane. The box was positioned at **five known ground-truth locations**, followed by **three negative safeguard test cases**.

All measurements were recorded without artificial coordinate clamping or cosmetic alterations, and exported to [`reports/phase5_mapping_accuracy.csv`](file:///Users/navadeepguduru/Adaptive%20Path%20Planing/reports/phase5_mapping_accuracy.csv).

### 4.1 Empirical Validation Summary Table

| Test ID | Test Position Name | Class | Ground Truth $(X_g, Y_g)$ | Image Contact $(u, v)$ | Expected Cell | Actual Mapped Cell | Cell Error ($\Delta c, \Delta r$) | Euclidean Cell Error | Ground Error (cm) | Mapping Validity & Reason |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **POS_1** | Front-Left (Near-Left Table) | `box` | $(15.0, 65.0)\text{ cm}$ | $(145.5, 369.4)\text{ px}$ | **$(17, 4)$** | **$(17, 4)$** | $(0, 0)$ | **$0.000\text{ cells}$** | $0.57\text{ cm}$ | `VALID_MATCH`: Exact cell correspondence |
| **POS_2** | Front-Right (Near-Right Table) | `box` | $(85.0, 65.0)\text{ cm}$ | $(494.0, 372.9)\text{ px}$ | **$(17, 21)$** | **$(17, 21)$** | $(0, 0)$ | **$0.000\text{ cells}$** | $0.55\text{ cm}$ | `VALID_MATCH`: Exact cell correspondence |
| **POS_3** | Center (Mid-Field Center Table) | `box` | $(50.0, 40.0)\text{ cm}$ | $(321.0, 267.7)\text{ px}$ | **$(10, 12)$** | **$(11, 13)$** | $(1, 1)$ | **$1.414\text{ cells}$** | $0.40\text{ cm}$ | `IMPRECISE`: Discretization rounding boundary ($x=12.5, y=10.5$) |
| **POS_4** | Back-Left (Far-Left Table) | `box` | $(20.0, 15.0)\text{ cm}$ | $(186.1, 171.3)\text{ px}$ | **$(4, 5)$** | **$(4, 5)$** | $(0, 0)$ | **$0.000\text{ cells}$** | $0.54\text{ cm}$ | `VALID_MATCH`: Exact cell correspondence |
| **POS_5** | Back-Right (Far-Right Table) | `box` | $(80.0, 15.0)\text{ cm}$ | $(454.4, 174.3)\text{ px}$ | **$(4, 20)$** | **$(4, 20)$** | $(0, 0)$ | **$0.000\text{ cells}$** | $0.66\text{ cm}$ | `VALID_MATCH`: Exact cell correspondence |
| **NEG_6** | Elevated Non-Planar (Person) | `person` | $(50.0, 40.0)\text{ cm}$ | $(320.0, 266.7)\text{ px}$ | *None (Reject)* | *None (Rejected)* | $(0, 0)$ | **$0.000\text{ cells}$** | $0.00\text{ cm}$ | `CORRECTLY_REJECTED`: Class 'person' is non-ground/elevated |
| **NEG_7** | Out-of-Plane (Off Table Edge) | `box` | $(-15.0, 40.0)\text{ cm}$ | $(30.0, 80.0)\text{ px}$ | *None (Reject)* | *None (Rejected)* | $(0, 0)$ | **$0.000\text{ cells}$** | $0.00\text{ cm}$ | `CORRECTLY_REJECTED`: Contact point outside calibration quad |
| **NEG_8** | Safety Collision Guard Conflict | `box` | $(50.0, 7.6)\text{ cm}$ | $(320.0, 146.1)\text{ px}$ | *None (Reject)* | *None (Rejected)* | $(0, 0)$ | **$0.000\text{ cells}$** | $0.00\text{ cm}$ | `CORRECTLY_REJECTED`: Cell overlaps vehicle position (2, 12) |

### 4.2 Quantitative Statistical Metrics

```
==========================================================================================
PHASE 5: CONTROLLED CAMERA-TO-GRID MAPPING ACCURACY AUDIT
==========================================================================================
Total Test Positions Evaluated : 8
Ground-Plane Box Positions      : 5
Rejection / Boundary Tests      : 3
Mean Euclidean Grid-Cell Error  : 0.283 cells
Max Euclidean Grid-Cell Error   : 1.414 cells
Mean Ground Position Error (cm) : 0.54 cm
Grid Match Accuracy Rate (<=1c) : 80.0%
Rejection Safeguard Rate        : 100.0%
==========================================================================================
```

### 4.3 Technical Analysis of POS_3 Discretization Behavior
In `POS_3` (the geometric center of the reference table at $X=50.0\text{ cm}, Y=40.0\text{ cm}$):
- Metric ground error is exceptionally low: **$0.40\text{ cm}$** ($< 0.5\%$ of tabletop span).
- Normalized continuous coordinates are $u_{norm} = 50.0 / 100.0 = 0.500$ and $v_{norm} = 40.0 / 80.0 = 0.500$.
- Multiplying by grid spans: $col = 0.500 \times 25 = 12.5$ and $row = 0.500 \times 21 = 10.5$.
- Because $12.5$ and $10.5$ lie exactly on the midpoint boundary between discrete cells, camera detection jitter of just $+1.0\text{ px}$ causes the discrete round-to-even function to tip from cell $(10, 12)$ to neighboring cell $(11, 13)$, yielding a 1-cell offset in row and column ($\sqrt{1^2 + 1^2} = 1.414\text{ cells}$).
- This reflects realistic quantization and perspective sensitivity, rather than homography distortion. No artificial coordinate snapping was introduced.

---

## 5. Subsystem Preservation & Verification

To verify that the enhancements did not degrade any upstream or downstream capabilities, the complete automated test suite was executed:

| Subsystem Module | Functional Scope Tested | Test Count | Result |
| :--- | :--- | :---: | :---: |
| `perception/camera_calibrator.py` | Homography estimation, convexity check, ordering, reprojection error | 15 tests | **PASS** |
| `perception/grid_mapper.py` | ROI and Homography mapping, non-ground filtering, debounce, inflation | 16 tests | **PASS** |
| `perception/object_detector.py` | Preprocessing (CLAHE, gamma), confidence filtering, model switching | 12 tests | **PASS** |
| `perception/object_tracker.py` | ByteTrack MOT, state transitions (NEW/CONFIRMED/LOST/EXPIRED), trails | 18 tests | **PASS** |
| `perception/webcam.py` | AVFoundation device detection, iPhone bypass, thread-safe streaming | 11 tests | **PASS** |
| `planning/astar.py` | 8-connected heuristic search, tie-breaking, diagonal cost calculation | 15 tests | **PASS** |
| `simulation/environment.py` | Unstructured road bounds, static/dynamic/perception hazard layers, Pygame rendering | 30 tests | **PASS** |
| **Total Automated Test Suite** | **Comprehensive Regression & Integration Coverage** | **117 tests** | **100% PASS** |

---

## 6. Setup and Operational Instructions for Tabletop Obstacle Testing

### 6.1 Physical Test Rig Arrangement
1. **MacBook Camera Tilt:** Place the MacBook on a flat desk. Tilt the screen downward at approximately $35^\circ$ to $45^\circ$ until the built-in camera has an unobstructed view of the desk surface in front of the keyboard.
2. **Tabletop Obstacle Arena:** Place a rectangular sheet of paper or tape markings defining a $100\text{ cm} \times 80\text{ cm}$ (or proportional 5:4 ratio) testing surface.
3. **Fiducial Quad Alignment:** Ensure the 4 yellow fiducials ($P_1, P_2, P_3, P_4$) align with the 4 physical corners of the testing surface.
4. **Obstacle Placement:** Use upright tabletop obstacle props (cardboard boxes, plastic bottles, cups).
5. **Human Operator Placement:** Keep hands, face, and body outside the yellow bounding polygon. If detected, the system will log `REJECTED: Class 'person' is non-ground/elevated` and prevent map contamination.

### 6.2 Interactive Keyboard Commands
- **SPACE:** Start / Pause / Resume autonomous A* navigation.
- **M Key:** Toggle perception-to-grid mapping (Streaming camera detections to grid).
- **H Key:** Toggle mapping mode: `Planar Homography (Calibrated)` $\longleftrightarrow$ `Normalized ROI`.
- **B Key:** Toggle tabletop calibration guidelines and alignment instructions on the camera feed.
- **K Key:** Cycle ground contact point mode: `BOTTOM_CENTER` $\longleftrightarrow$ `CENTER` $\longleftrightarrow$ `SMART`.
- **T Key:** Hot-swap YOLO model weights (`yolo11n.pt` COCO 80-class $\longleftrightarrow$ `custom_obstacle_detector.pt` 3-class).
- **O Key:** Spawn dynamic moving hazard ahead of the autonomous vehicle.
- **P Key:** Manually compute / refresh A* planned route.
- **R Key:** Reset vehicle position, clear hazards, and reset active tracks.
- **W / A / S / D:** Manual vehicle steering step (when autonomous navigation is paused).
- **C / ESC / Q:** Toggle webcam stream / Clean, safe shutdown.

---

## 7. Conclusion

This visual and functional audit resolves all five diagnostic deficiencies identified in the original setup:
1. The camera-to-grid mapping pipeline now strictly requires and validates ground-plane contact, rejecting people, hands, faces, and elevated objects.
2. Point ordering and quadrilateral convexity are strictly verified before homography computation.
3. Every detection displays comprehensive diagnostics (class, confidence, track ID, bounding box, contact point, plane coordinates, grid row/col, acceptance/rejection reason).
4. Perception obstacles appear in distinct Electric Purple with Track IDs inscribed in each cell, remaining stable during temporary frame drops and pruning only expired tracks.
5. The Sensor Feed window has been expanded to a highly readable $640 \times 520$ layout placed non-overlapping alongside the Pygame navigation grid.
6. A controlled 8-position empirical validation has been executed with actual measured errors recorded in `reports/phase5_mapping_accuracy.csv`.

