# Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads

An academic and engineering project developing an adaptive path planning and collision avoidance architecture for autonomous ground vehicles (AGVs) operating on unstructured, unpaved, and irregular roads.

---

### 📌 Project Roadmap Overview
This 10-day project incrementally constructs an autonomous navigation system designed for off-road and rural environments lacking painted lane markings:
- **Day 1:** Modular Architecture, Simulation Grid Baseline & Perception Hooks (Completed)
- **Day 2:** Classical Path Planning (A* Search on Grid Maps) (Completed)
- **Day 3:** Autonomous Movement, Dynamic Hazard Insertion & Adaptive Replanning (Completed)
- **Day 4:** Reliable MacBook Webcam Integration (Completed)
- **Day 5 (Current): Local YOLO Object Detection Module (Completed)**
- **Day 6:** Unstructured Road Boundary Cost Fields & Dynamic Weighting
- **Day 7:** Sensor-Grid Fusion (Mapping Vision Detections to Costmap)
- **Day 8:** Dynamic Moving Obstacle Avoidance
- **Day 9:** Closed-Loop Autonomous Navigation Simulation
- **Day 10:** Performance Benchmarking, Robustness Testing & Final Documentation

---

## 🎯 Day 5 Scope & Objectives: Local YOLO Object Detection
Day 5 introduces local deep-learning perception by adding an Ultralytics YOLO object detection module to process video frames from the MacBook built-in camera:
1. **Lightweight Pretrained YOLO Model (`yolo11n.pt`):**
   - Employs the Ultralytics YOLOv11 nano model (`yolo11n.pt`, ~5.4 MB) running 100% locally on CPU/Apple Silicon Neural Engine with zero paid APIs.
   - Recognizes 80 COCO classes (vehicles, pedestrians, animals, traffic signs, obstacles).
2. **Structured Detection Data Model:**
   - Exposes strongly typed `Detection` records containing class ID, class name, confidence score, bounding box `(x1, y1, x2, y2)`, dimensions, and center coordinates.
3. **Robust Fallbacks & Error Handling:**
   - Safely handles empty frames, corrupt images, missing model weights, and inference exceptions without crashing.
   - Configurable confidence threshold (`conf_threshold`, default 0.35) and input image size (`imgsz`, default 640).
4. **Standalone Live Detection Demo Tool (`tools/test_detection_live.py`):**
   - Connects to the verified MacBook camera (bypassing iPhone Continuity Camera).
   - Annotates live frames with bounding boxes, labels, confidence scores, stream FPS, and neural network inference latency ($ms$).
   - Respects the single camera owner rule: ensures clean, exclusive hardware access.
5. **Decoupled Architecture & Scope Safety:**
   - Object detection is strictly isolated from the A* planner and Pygame grid. Detections do not automatically alter the simulation map yet (scheduled for Day 7 Sensor-Grid Fusion).
   - No assumptions of real-world 3D depth, metric distance, or drivable area are made from 2D bounding boxes.
6. **Automated Unit Testing Without Heavy Dependencies:**
   - Comprehensive unit tests in `tests/test_detection.py` using mock frames and mock model outputs, enabling deterministic CI execution without downloading weights or requiring a camera.

---

## 📂 Project Directory Structure

```text
adaptive-path-planning/
├── main.py                     # Application entry point, event loop & key handling
├── requirements.txt            # Dependencies (Pygame, OpenCV, NumPy, Ultralytics)
├── README.md                   # Comprehensive documentation and runbook
├── .gitignore                  # Git exclusions for Python bytecode and venvs
├── yolo11n.pt                  # Local Ultralytics YOLOv11 nano model weights (~5.4 MB)
├── planning/                   # Classical path planning package
│   ├── __init__.py             # Planning package initialization
│   └── astar.py                # Modular A* planner with Manhattan heuristic & priority queue
├── perception/
│   ├── __init__.py             # Perception package initialization & exports
│   ├── webcam.py               # Modular OpenCV webcam manager with AVFoundation & thread capture
│   └── object_detector.py      # Modular Ultralytics YOLO object detector & Detection dataclass
├── simulation/
│   ├── __init__.py             # Simulation package initialization & exports
│   └── environment.py          # 2D road environment, autonomous controller & HUD
├── tools/
│   ├── __init__.py             # Tools package initialization
│   ├── test_webcam_live.py     # Standalone manual live test utility for MacBook camera
│   └── test_detection_live.py  # Standalone live YOLO detection demo on MacBook camera
├── tests/
│   ├── test_day1.py            # Automated tests: Baseline environment & sensors (8 tests)
│   ├── test_astar.py           # Automated tests: Classical A* algorithm (15 tests)
│   ├── test_day3.py            # Automated tests: Autonomous navigation & replanning (11 tests)
│   ├── test_webcam.py          # Automated tests: MacBook webcam integration & mock streaming (9 tests)
│   └── test_detection.py       # Automated tests: YOLO detection parsing, errors & decoupling (7 tests)
├── assets/                     # Sprites, textures, and map assets (.gitkeep)
└── screenshots/                # Rendered UI screenshots and visual validation (.gitkeep)
```

---

## 💻 Prerequisites
- **Python:** Version `3.8+` (Tested on Python `3.9.6` and `3.11+`)
- **OS Support:** macOS (Apple Silicon & Intel), Linux (Ubuntu/Debian), Windows 10/11
- **Hardware (Optional):** USB or integrated webcam for sensor preview

---

## 🚀 Virtual Environment Setup & Installation

### On macOS / Linux
```bash
# 1. Clone or navigate to the project directory
cd "Adaptive Path Planing"

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### On Windows (Command Prompt / PowerShell)
```cmd
:: 1. Navigate to the project directory
cd "Adaptive Path Planing"

:: 2. Create a virtual environment
python -m venv venv

:: 3. Activate the virtual environment
:: In Command Prompt (cmd):
venv\Scripts\activate.bat
:: In PowerShell:
venv\Scripts\Activate.ps1

:: 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧠 Day 2 A* Path Planning Architecture & Theory

### 1. The A* Search Algorithm
A* is an informed heuristic search algorithm that computes the mathematically optimal (shortest) collision-free path on a graph or grid. It evaluates nodes by combining the exact historical cost to reach a cell with an optimistic estimate of the remaining distance to the goal.

### 2. The Evaluation Function: $f(n) = g(n) + h(n)$
At each step of the search loop, A* expands the node $n$ from the open priority queue that minimizes:
$$f(n) = g(n) + h(n)$$

Where:
- **$g(n)$ (Path Cost):** The exact cumulative movement cost incurred from the vehicle's starting cell to the current node $n$. Each orthogonal grid step has a uniform cost of $1$.
- **$h(n)$ (Heuristic Function):** The estimated cost from the current node $n$ to the navigation destination cell.
- **$f(n)$ (Total Estimated Cost):** The estimated total cost of the complete trajectory passing through node $n$.

### 3. Why Manhattan Distance is Appropriate for 4-Direction Movement
In grid-based robotics, the choice of heuristic must match the kinematic movement constraints:
- In our unstructured road environment, the vehicle is restricted to **4 orthogonal directions** only: North $(-1, 0)$, South $(+1, 0)$, West $(0, -1)$, and East $(0, 1)$, with no diagonal cutting.
- The **L1 Manhattan Distance** between cells $(r_1, c_1)$ and $(r_2, c_2)$ is defined as:
  $$h(n) = |r_1 - r_2| + |c_1 - c_2|$$
- **Admissibility ($h(n) \le h^*(n)$):** In an unconstrained grid free of hazards, Manhattan distance represents the exact minimal number of orthogonal steps required to reach the destination. Since obstacles, potholes, and off-road borders can only force detours (increasing the true cost $h^*(n)$), the heuristic never overestimates the true remaining cost. Admissibility guarantees that A* will return the optimal path.
- **Consistency / Monotonicity ($h(n) \le c(n, n') + h(n')$):** For any two adjacent nodes $n$ and $n'$, the change in Manhattan distance cannot exceed the step cost $c(n, n') = 1$. Consistency ensures that when a node is removed from the priority queue and added to the closed set, the found path to that node is already optimal, eliminating the need to re-open closed nodes.

### 4. Implementation Structure (`planning/astar.py`)
- **Priority Queue:** Implemented using Python's native `heapq` module storing tuples `(f_score, h_score, counter, current_node)`. The inclusion of `h_score` breaks ties in favor of nodes closest to the goal, and `counter` guarantees comparison safety.
- **Closed Set:** A fast `set` tracking all expanded grid cells.
- **Parent Pointers:** A hash map reconstructing the final ordered path from start to goal upon reaching the target.
- **Edge Cases Handled:**
  - Start identical to destination: Returns a valid 0-move path `[start]`.
  - Walled-off / enclosed destination: Terminates cleanly, returning an empty path with `success=False` and reporting all explored nodes.
  - Out-of-bounds or non-drivable start/goal: Promptly rejects search before expanding the open queue.
  - Obstacle traversal immunity: Blocked cells and non-drivable shoulders are strictly prohibited from entering the open queue.

---


## 🤖 Day 3 Autonomous Navigation & Dynamic Replanning Architecture

### 1. Autonomous Waypoint Following & Timed Pacing
The Day 3 controller implements closed-loop waypoint tracking along the optimal A* trajectory:
- **Decoupled Update Loop:** The Pygame display loop renders at 60 FPS (`clock.tick(60)`), preserving high responsiveness for user inputs (`SPACE`, `O`, `P`, `R`, etc.).
- **Stepping Pacing (250 ms):** Vehicle motion is governed by an internal delta-time accumulator (`env.update(dt_ms)`). The vehicle steps exactly one grid cell every 250 milliseconds (4 cells per second), matching realistic low-speed autonomous crawl speeds on rugged terrain.
- **State Machine Modes (`NavigationMode`):**
  - `MANUAL`: Vehicle is controlled manually via keyboard (`W`/`A`/`S`/`D` or arrow keys).
  - `AUTONOMOUS`: Vehicle autonomously traverses the planned path.
  - `PAUSED`: Autonomous traversal paused via `SPACE`, preserving current progress.
  - `REPLANNING`: Brief transient state during path recalculation upon encountering dynamic obstacles.
  - `GOAL_REACHED`: Vehicle has safely arrived at the destination cell `(12, 22)`.
  - `NO_PATH`: Destination is unreachable or completely blocked; vehicle halts safely.

### 2. Dynamic Obstacle Insertion (`O` Key)
Simulates unpredictable hazards (fallen rocks, fallen trees, potholes) emerging dynamically:
- **Waypoint-Targeted Placement:** Inspects the vehicle's remaining planned route and prefers placing the new obstacle directly on the path 3 cells ahead (`ahead_steps=3`).
- **Safe Fallback:** If the remaining route is shorter than 3 cells or no route exists, it randomly selects a safe drivable road cell.
- **Integrity Constraints:**
  - Never overwrites the current vehicle position.
  - Never overwrites the target destination.
  - Never overwrites off-road boundaries or existing obstacles.
- **Visual Highlighting:** Dynamically inserted obstacles are highlighted in the grid with an energetic amber border to distinguish them from static hazards.

### 3. Collision Checking & Adaptive Replanning
Safety-critical autonomous vehicles cannot rely on static pre-departure paths. Before each autonomous step:
1. **Lookahead Route Validation (`is_route_blocked()`):** Checks every remaining cell in `self.planned_route` against the grid environment.
2. **Blocked Cell Detection:** If an obstacle is detected in the remaining path:
   - Vehicle movement into the blocked cell is **strictly prohibited**.
   - The controller triggers `replan_route()`.
   - The A* planner recomputes the shortest collision-free detour from the vehicle's *current position* to the destination.
   - If a valid detour is found, `self.planned_route` is updated and autonomous motion resumes seamlessly.
   - If the destination is completely blocked, the vehicle transitions to `NavigationMode.NO_PATH` and halts safely.
3. **Zero-Collision Safety Guarantee:** Collision count remains strictly zero across all test cases and simulation runs.
4. **Selective Replanning:** Hazards placed off the active route corridor do not trigger replanning, saving compute cycles.

---

## ▶️ Running the Application

Ensure the virtual environment is activated, then run:

```bash
python3 main.py
```
*(On Windows: `python main.py`)*

#### MacBook Built-in Camera Exclusive Configuration:
The application is configured to connect to your **MacBook's built-in FaceTime HD Camera**. It inspects macOS system profiler device lists non-intrusively to locate the MacBook camera index without opening or probing any iPhone handles, preventing accidental iPhone Continuity Camera notifications.

### Interactive Controls
| Key | Action | State Constraint | Details |
| :--- | :--- | :--- | :--- |
| **SPACE** | **Start / Pause / Resume Autonomous Mode** | Any | Automatically plans A* route if missing, starts motion, pauses, or resumes traversal |
| **O** | **Insert Dynamic Obstacle** | Any | Spawns a hazard ahead on the path or on drivable terrain, challenging the planner |
| **P** | **Calculate A* Route** | Manual / Paused | Computes optimal route from current vehicle position to destination |
| **R** | **Reset Simulation** | Any | Resets vehicle to `(12, 2)`, clears dynamic hazards, restores static map, and sets mode to `MANUAL` |
| **W** / **Up Arrow** | Move Vehicle North | Manual / Paused | Moves vehicle 1 step up (locked during autonomous movement; invalidates path) |
| **S** / **Down Arrow** | Move Vehicle South | Manual / Paused | Moves vehicle 1 step down (locked during autonomous movement; invalidates path) |
| **A** / **Left Arrow** | Move Vehicle West | Manual / Paused | Moves vehicle 1 step left (locked during autonomous movement; invalidates path) |
| **D** / **Right Arrow** | Move Vehicle East | Manual / Paused | Moves vehicle 1 step right (locked during autonomous movement; invalidates path) |
| **C** | Toggle Camera Stream | Any | Opens or closes the auxiliary MacBook camera preview window |
| **ESC** | Safe Shutdown | Any | Cleanly releases camera, graphics handles, and exits |

---

## 📊 Telemetry & Display Metrics

The sidebar HUD provides real-time planning, state, and safety metrics:
- **Navigation Mode:** Current state badge (`MANUAL`, `AUTONOMOUS`, `PAUSED`, `REPLANNING`, `GOAL REACHED`, `NO PATH`).
- **Vehicle Pos (X, Y):** Current vehicle grid coordinate.
- **Destination (X, Y):** Target destination grid coordinate.
- **Remaining Moves:** Number of steps remaining to reach the destination.
- **Replanning Events:** Total number of dynamic replanning events triggered.
- **Collision Count:** Verified safety count (guaranteed strictly 0).
- **Execution Time:** High-resolution A* algorithm execution time in milliseconds ($ms$).
- **Explored Nodes:** Total unique grid cells expanded during the last search.
- **Hazards:** Active hazard breakdown: `Total (Static + Dynamic)`.

---

## ⚠️ Limitations of the Grid Simulation

While this discrete simulation effectively validates path planning and real-time replanning algorithms, several physical assumptions differ from real-world autonomous vehicles:
1. **Discrete Grid vs. Continuous Kinematics:** Vehicles in the real world have minimum turning radii (Ackermann steering geometry) and continuous acceleration/braking limits, whereas grid cells represent discrete 1-meter step hops.
2. **Idealized Sensor Perception:** Dynamic obstacles in Day 3 are injected directly into the digital map representation. In real-world deployment (scheduled for Days 5–7), obstacles must be detected through noisy camera streams and projected onto a costmap via inverse perspective mapping (IPM).
3. **Static Friction & Terrain Slip:** Unstructured dirt and gravel roads present variable wheel slip and non-uniform traction, which are simplified as uniform travel cost on drivable cells in this phase.

---

## 📷 Day 4: Reliable MacBook Webcam Integration Architecture

### 1. Device Selection & iPhone Continuity Camera Bypass
On macOS Sonoma and later, Apple's Continuity Camera often registers a paired iPhone as an available video input device. When opening camera index 0 without device inspection, OpenCV can accidentally connect to the iPhone instead of the MacBook's physical camera, causing unwanted connection alerts and chime sounds.

Our detection architecture in `perception/webcam.py` resolves this reliably:
- **Priority 1: Environment Override:** Inspects `MACBOOK_CAMERA_INDEX` (e.g. `export MACBOOK_CAMERA_INDEX=0`).
- **Priority 2: System Profiler Discovery:** Parses `system_profiler -json SPCameraDataType` non-intrusively without opening any camera handles. It inspects device model IDs and names, explicitly filtering out any device containing `"iPhone"` while targeting `"FaceTime HD Camera"` or `"built-in"`. On MacBook Air, this maps index `0` to the built-in camera and index `1` to the iPhone.
- **Priority 3: Hardware Fallback:** Defaults safely to index `0`.
- **Command-Line Configuration:** Allows explicit override via `python main.py --camera-index <int>`.

### 2. Multi-Threaded Capture & Sensor Warmup Handling
Hardware cameras on macOS AVFoundation require a brief sensor warmup period (~100–200 ms) before the first valid frame is returned:
- **Asynchronous Worker Thread:** Frame capture runs inside `_capture_worker` on a dedicated daemon thread, decoupling video I/O from the 60 FPS Pygame simulation loop.
- **Warmup Tolerance:** Tolerates initial empty frames during hardware startup, only declaring an initialization failure if no valid frames arrive within 2.5 seconds.
- **Latency-Free Bounded Buffer:** Stores only the most recent frame (`_latest_frame`) under a thread lock (bounded buffer depth of 1), completely eliminating frame delay and memory buildup.
- **Rolling FPS:** Continuously calculates rolling frames-per-second and image resolution.
- **Disconnection Detection:** Detects sustained frame read errors (>1.5s) and cleanly transitions the system to `CameraState.ERROR` without crashing.

### 3. Simulation & HUD Telemetry States
The HUD sidebar dynamically reports camera operational states:
- `OFF`: Camera is idle (Press `C` to toggle).
- `STARTING`: Establishing AVFoundation session and warming up sensor.
- `ACTIVE`: Streaming verified live frames (displays live FPS and resolution, e.g. `ACTIVE (30.0 FPS • 1920x1080)`).
- `ERROR`: Hardware unavailable or permissions denied with a clear reason.

### 4. Standalone Live Verification Utility
To verify live camera streaming independently of the Pygame simulation, run the standalone test tool:

```bash
python tools/test_webcam_live.py
```
*(Optionally specify an index: `python tools/test_webcam_live.py --camera-index 0`)*

Features:
- Connects **only** to the verified MacBook camera; never probes or switches to an iPhone.
- Displays a live OpenCV preview window with real-time HUD telemetry (resolution, rolling FPS, frame counter, device index).
- Exits cleanly and releases hardware upon pressing **`ESC`** or **`q`**.

### 5. macOS Camera Permissions Troubleshooting
1. When running for the first time, macOS may present a permission dialog: `"[Terminal] would like to access the camera"`. Click **OK**.
2. If access was denied or the status reads `Unavailable (Check connection/permissions)`:
   - Open **System Settings** → **Privacy & Security** → **Camera**.
   - Toggle **ON** the switch next to your terminal application (e.g., Terminal, iTerm2, or VS Code).
   - Restart the terminal and launch `python main.py`.

---

## 👁️ Day 5: Local YOLO Object Detection Architecture

### 1. Modular YOLO Perception Engine (`perception/object_detector.py`)
Day 5 introduces local deep-learning inference to process camera frames in real time:
- **Lightweight Pretrained Model (`yolo11n.pt`):** Uses the state-of-the-art Ultralytics YOLOv11 nano model (~5.4 MB), running 100% locally on CPU or Apple Silicon Neural Engine (MPS) with zero cloud APIs or paid services.
- **Structured Data Contract (`Detection`):**
  Each detection returns a structured dataclass containing:
  - `class_id` (int) and `class_name` (str) from the 80 COCO classes.
  - `confidence` (float between 0.0 and 1.0).
  - `bbox` as integer pixel coordinates `(x1, y1, x2, y2)`.
  - Computed helpers: `width`, `height`, `center` `(cx, cy)`, and `to_dict()`.
- **Configurable Thresholds:** Easily configure confidence filtering (`conf_threshold`, default 0.35) and input image sizing (`imgsz`, default 640).
- **Graceful Error Recovery:** Safely rejects `None` frames, zero-sized arrays, missing weights, or inference exceptions without crashing.

### 2. Standalone Live Detection Demo Tool (`tools/test_detection_live.py`)
To test real-time object detection with live video overlay from your MacBook webcam:

```bash
python tools/test_detection_live.py
```

Optional CLI parameters:
```bash
python tools/test_detection_live.py --model yolo11n.pt --conf 0.40 --imgsz 640
```

> [!IMPORTANT]
> **Single Camera Owner Rule:** On macOS AVFoundation, camera hardware can only be accessed by one application at a time. Ensure that `python main.py` is closed before launching `test_detection_live.py` so the camera handle is available.

Features:
- Connects directly to the verified MacBook built-in camera (bypassing paired iPhones).
- Overlays stylish bounding boxes, class labels, confidence scores, and real-time neural network inference latency ($ms$).
- Displays stream FPS counter.
- Press **`ESC`** or **`q`** to cleanly exit and release camera hardware.

### 3. Decoupling & Scientific Scope Boundaries
- **Strict Decoupling from Path Planning:** Object detection operates purely as an isolated perception layer. Detections do **not** automatically inject obstacles into the A* grid or alter the vehicle route.
- **Sensor-Grid Fusion Scheduled for Day 7:** Transforming camera pixels into road obstacles and projecting them onto the 2D grid costmap via Inverse Perspective Mapping (IPM) is scheduled for **Day 7**.
- **No Distance/Depth Claims:** 2D bounding boxes from a monocular camera do not provide real-world metric distance, depth, or safe drivable-area boundaries without spatial calibration or stereo/depth estimation.

---

## 🧪 Automated Testing & Verification

Automated test suites comprehensively verify environment physics, classical A* planning, autonomous navigation, webcam integration, and YOLO detection:
1. `tests/test_day1.py` (8 tests): Baseline grid geometry, obstacle seeding, vehicle boundaries, collision prevention, and webcam lifecycle.
2. `tests/test_astar.py` (15 tests): Comprehensive A* verification covering Manhattan heuristic admissibility, wall detours, enclosed destinations, start==goal edge cases, invalid coordinate rejection, 4-directional enforcement, move-count accuracy, deterministic repeatability, `RoadEnvironment` integration, route clearing on manual movement, and headless rendering.
3. `tests/test_day3.py` (11 tests): Autonomous navigation controller, `SPACE` pause/resume, goal arrival detection, dynamic obstacle injection ahead of vehicle, adaptive detour replanning, zero-collision safety guarantee, safe halting on complete path blockage, dynamic obstacle non-overwrite constraints, reset restoration, manual control locking during autonomous mode, selective replanning avoidance for off-route hazards, and headless rendering.
4. `tests/test_webcam.py` (9 tests): Successful camera initialization and frame capture, RGB color conversion, camera unavailable error handling, sustained read failure detection, repeated toggle idempotence, duplicate session prevention, resource cleanup, environment variable device configuration, iPhone bypass filtering, and decoupling from A* path planning.
5. `tests/test_detection.py` (8 tests): `Detection` dataclass properties and serialization, empty/invalid frame safety, unloaded model handling, graceful load failure recovery, YOLO result parsing into structured detections, visual annotation rendering, and grid decoupling safety invariant.

Run all tests with:
```bash
source venv/bin/activate
python3 -m unittest discover tests
```

Expected result:
```text
...................................................
----------------------------------------------------------------------
Ran 51 tests in ~2.2s

OK
```

---

## 🔮 Planned Day 6 Work: Road Boundary Cost Fields & Dynamic Weighting
On Day 6, we will enhance the grid planner with:
- **Boundary Cost Fields:** Applying distance-transform cost gradients to heavily penalize proximity to rough road edges, ditch drops, and potholes, encouraging the vehicle to naturally maintain center-lane clearance.
- **Dynamic Risk Weighting:** Modulating traversal costs near terrain hazards dynamically.

