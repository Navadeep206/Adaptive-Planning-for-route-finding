# Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads

An academic and engineering project developing an adaptive path planning and collision avoidance architecture for autonomous ground vehicles (AGVs) operating on unstructured, unpaved, and irregular roads.

---

### 📌 Project Roadmap Overview
This 10-day project incrementally constructs an autonomous navigation system designed for off-road and rural environments lacking painted lane markings:
- **Day 1:** Modular Architecture, Simulation Grid Baseline & Perception Hooks (Completed)
- **Day 2:** Classical Path Planning (A* Search on Grid Maps) (Completed)
- **Day 3 (Current): Autonomous Movement, Dynamic Hazard Insertion & Adaptive Replanning**
- **Day 4:** Unstructured Road Boundary Cost Fields & Dynamic Weighting
- **Day 5:** Computer Vision Preprocessing for Off-Road Terrain
- **Day 6:** Visual Obstacle Detection Baseline
- **Day 7:** Sensor-Grid Fusion (Mapping Vision Detections to Costmap)
- **Day 8:** Dynamic Moving Obstacle Avoidance
- **Day 9:** Closed-Loop Autonomous Navigation Simulation
- **Day 10:** Performance Benchmarking, Robustness Testing & Final Documentation

---

## 🎯 Day 3 Scope & Objectives: Autonomous Movement & Dynamic Replanning
Day 3 brings the simulation to life with closed-loop autonomous waypoint tracking and dynamic obstacle replanning:
1. **Autonomous Waypoint Controller:**
   - **`SPACE` Key Control:** Smooth start, pause, and resume of autonomous movement along the optimal A* route.
   - **Timed Step Pacing:** Moves the vehicle one grid cell at a time every 250 ms (4 cells/sec) while the simulation event loop and renderer maintain a silky-smooth 60 FPS.
   - **Goal Arrival:** Automatically detects arrival at the destination marker, transitions state to `"Goal Reached"`, and stops motion cleanly.
2. **Dynamic Obstacle Injection (`O` Key):**
   - Spawns new hazards during simulation.
   - Prefers open road cells directly along the remaining planned route ahead of the vehicle to challenge the planner.
   - Strictly protects road boundaries, current vehicle position, and destination from being overwritten.
3. **Collision Prevention & Adaptive Replanning:**
   - Evaluates remaining trajectory before every step.
   - When a newly spawned obstacle blocks the route, movement into the blocked cell is strictly prevented, and an alternate collision-free path is recomputed on-the-fly.
   - Avoids unnecessary replanning if hazards spawn off the active route corridor.
   - If a route is completely blocked, safely halts the vehicle in `"No Path"` mode with 0 collisions.
4. **Enhanced Telemetry HUD:**
   - Live **Navigation Mode** tracking: `Manual`, `Autonomous`, `Paused`, `Replanning`, `Goal Reached`, and `No Path`.
   - Dynamic **Replanning Events** counter and verified zero-collision safety metric.
   - High-contrast visual distinctions for dynamically spawned hazards and active navigation states.

---

## 📂 Project Directory Structure

```text
adaptive-path-planning/
├── main.py                     # Application entry point, event loop & key handling
├── requirements.txt            # Locked minimum dependencies (Pygame, OpenCV, NumPy)
├── README.md                   # Comprehensive documentation and runbook
├── .gitignore                  # Git exclusions for Python bytecode and venvs
├── planning/                   # Classical path planning package
│   ├── __init__.py             # Planning package initialization
│   └── astar.py                # Modular A* planner with Manhattan heuristic & priority queue
├── perception/
│   ├── __init__.py             # Perception package initialization
│   └── webcam.py               # Modular OpenCV webcam manager with error handling
├── simulation/
│   ├── __init__.py             # Simulation package initialization & exports
│   └── environment.py          # 2D road environment, autonomous controller & HUD
├── tests/
│   ├── test_day1.py            # Automated tests: Baseline environment & sensors (8 tests)
│   ├── test_astar.py           # Automated tests: Classical A* algorithm (15 tests)
│   └── test_day3.py            # Automated tests: Autonomous navigation & replanning (11 tests)
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

## 📷 Webcam Setup & Permission Troubleshooting

The perception module uses OpenCV with native **AVFoundation** on macOS (`cv2.CAP_AVFOUNDATION`) and asynchronous background frame capture to ensure the Pygame simulation never freezes. Camera availability is completely optional: **if no webcam is present or permission is denied, the simulation continues to run normally**.

### macOS Permissions
1. When running from a terminal for the first time, macOS may present a permission dialog: `"[Terminal] would like to access the camera"`. Click **OK**.
2. If access was denied or the status reads `Unavailable (Check connection/permissions)`:
   - Open **System Settings** → **Privacy & Security** → **Camera**.
   - Toggle **ON** the switch next to your terminal application (e.g., Terminal, iTerm2, or VS Code).
   - Restart the terminal and launch `python3 main.py`.

---

## 🧪 Automated Testing & Verification

Automated test suites comprehensively verify environment physics, classical A* planning, autonomous navigation, and dynamic replanning:
1. `tests/test_day1.py` (8 tests): Baseline grid geometry, obstacle seeding, vehicle boundaries, collision prevention, and webcam lifecycle.
2. `tests/test_astar.py` (15 tests): Comprehensive A* verification covering Manhattan heuristic admissibility, wall detours, enclosed destinations, start==goal edge cases, invalid coordinate rejection, 4-directional enforcement, move-count accuracy, deterministic repeatability, `RoadEnvironment` integration, route clearing on manual movement, and headless rendering.
3. `tests/test_day3.py` (11 tests): Autonomous navigation controller, `SPACE` pause/resume, goal arrival detection, dynamic obstacle injection ahead of vehicle, adaptive detour replanning, zero-collision safety guarantee, safe halting on complete path blockage, dynamic obstacle non-overwrite constraints, reset restoration, manual control locking during autonomous mode, selective replanning avoidance for off-route hazards, and headless rendering.

Run all tests with:
```bash
source venv/bin/activate
python3 -m unittest discover tests
```

Expected result:
```text
..................................
----------------------------------------------------------------------
Ran 34 tests in 1.25s

OK
```

---

## 🔮 Planned Day 4 Work: Road Boundary Cost Fields & Dynamic Weighting
On Day 4, we will enhance the grid planner with:
- **Boundary Cost Fields:** Applying distance-transform cost gradients to heavily penalize proximity to rough road edges, ditch drops, and potholes, encouraging the vehicle to naturally maintain center-lane clearance.
- **Dynamic Risk Weighting:** Modulating traversal costs near terrain hazards dynamically.

