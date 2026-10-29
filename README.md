# Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads

An academic and engineering project developing an adaptive path planning and collision avoidance architecture for autonomous ground vehicles (AGVs) operating on unstructured, unpaved, and irregular roads.

---

## 📌 Project Roadmap Overview
This 10-day project incrementally constructs an autonomous navigation system designed for off-road and rural environments lacking painted lane markings:
- **Day 1 (Current): Modular Architecture, Simulation Grid Baseline & Perception Hooks**
- **Day 2:** Classical Path Planning (A* Search on Grid Maps)
- **Day 3:** Unstructured Road Boundary Cost Fields & Dynamic Weighting
- **Day 4:** Static Collision Avoidance & Local Replanning
- **Day 5:** Computer Vision Preprocessing for Off-Road Terrain
- **Day 6:** Visual Obstacle Detection Baseline
- **Day 7:** Sensor-Grid Fusion (Mapping Vision Detections to Costmap)
- **Day 8:** Dynamic Moving Obstacle Avoidance
- **Day 9:** Closed-Loop Autonomous Navigation Simulation
- **Day 10:** Performance Benchmarking, Robustness Testing & Final Documentation

---

## 🎯 Day 1 Scope & Objectives
Day 1 establishes a modular, reproducible, and verifiable engineering foundation:
1. **Interactive Simulation Window:** A 1000 × 700 Pygame environment rendering a 2D unstructured-road terrain grid.
2. **Organic Road Boundaries:** A visually distinct, winding drivable unpaved roadway with irregular boundaries and off-road non-drivable verges.
3. **Entity Markers:**
   - 🟦 **Vehicle (Agent):** Starts at designated coordinates `(12, 2)`.
   - 🟨 **Destination (Goal):** Positioned at `(15, 20)`.
   - 🟥 **Static Obstacles:** Multiple hazards (boulders, potholes, debris) placed on the road surface.
4. **Collision & Boundary Enforcement:** Manual step verification via W/A/S/D or Arrow keys that enforces strict boundaries—blocking moves outside the drivable road corridor or into obstacles.
5. **Perception Feed Integration:** An OpenCV-powered camera manager with toggleable live preview (`C` key), non-blocking execution, and graceful hardware/permission fallback.
6. **Telemetry & Legend HUD:** Real-time system status card, sensor status, color legend, and controls guide.

> [!NOTE]
> **Day 1 Limitation Notice:** Autonomous path planning (A*) and computer vision obstacle segmentation are intentionally **not** implemented on Day 1. Manual keyboard navigation is provided strictly for environment validation and collision checking.

---

## 📂 Project Directory Structure

```text
adaptive-path-planning/
├── main.py                     # Application entry point and simulation loop
├── requirements.txt            # Locked minimum dependencies (Pygame, OpenCV, NumPy)
├── README.md                   # Comprehensive documentation and runbook
├── .gitignore                  # Git exclusions for Python bytecode and venvs
├── perception/
│   ├── __init__.py             # Perception package initialization
│   └── webcam.py               # Modular OpenCV webcam manager with error handling
├── simulation/
│   ├── __init__.py             # Simulation package initialization
│   └── environment.py          # 2D unstructured road environment and HUD rendering
├── tests/
│   └── test_day1.py            # Automated unit and integration test suite
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

## ▶️ Running the Application

Ensure the virtual environment is activated, then run:

```bash
python3 main.py
```
*(On Windows: `python main.py`)*

#### MacBook Built-in Camera Exclusive Configuration:
The application is configured to exclusively connect to your **MacBook's built-in FaceTime HD Camera**. It inspects macOS system profiler device lists non-intrusively to locate the MacBook camera index without opening or probing any iPhone handles, completely preventing accidental iPhone Continuity Camera connections or notifications.

### Interactive Controls
| Key | Action | Details |
| :--- | :--- | :--- |
| **W** / **Up Arrow** | Move Vehicle North | Moves vehicle 1 grid unit up if cell is drivable |
| **S** / **Down Arrow** | Move Vehicle South | Moves vehicle 1 grid unit down if cell is drivable |
| **A** / **Left Arrow** | Move Vehicle West | Moves vehicle 1 grid unit left if cell is drivable |
| **D** / **Right Arrow** | Move Vehicle East | Moves vehicle 1 grid unit right if cell is drivable |
| **R** | Reset Position | Instantly teleports vehicle back to start coordinate `(12, 2)` |
| **C** | Toggle Camera Stream | Opens or closes the auxiliary MacBook camera preview window |
| **ESC** | Safe Shutdown | Cleanly releases camera, graphics handles, and exits |

---

## 📷 Webcam Setup & Permission Troubleshooting

The perception module uses OpenCV with native **AVFoundation** on macOS (`cv2.CAP_AVFOUNDATION`) and asynchronous background frame capture to ensure the Pygame simulation never freezes. Camera availability is completely optional: **if no webcam is present or permission is denied, the simulation continues to run normally**.

### macOS Permissions
macOS enforces strict Transparency, Consent, and Control (TCC) security for camera devices.
1. When running from a terminal for the first time, macOS may present a permission dialog: `"[Terminal] would like to access the camera"`. Click **OK**.
2. If access was denied or the status reads `Unavailable (Check connection/permissions)`:
   - Open **System Settings** → **Privacy & Security** → **Camera**.
   - Toggle **ON** the switch next to your terminal application (e.g., Terminal, iTerm2, or VS Code).
   - Restart the terminal and launch `python3 main.py`.

### Windows Permissions
1. Open **Settings** → **Privacy & Security** → **Camera**.
2. Ensure **Camera access** is set to **On**.
3. Ensure **Let desktop apps access your camera** is set to **On**.
4. If you have multiple cameras (e.g., OBS Virtual Camera, IR camera), you can adjust the device index with `--camera-index 1`.

---

## 🧪 Automated Testing & Verification

A dedicated verification test suite is included in `tests/test_day1.py`. It tests grid bounds, obstacle placement, collision blocking, reset routines, headless Pygame rendering, and webcam manager lifecycle.

Run the test suite with:
```bash
source venv/bin/activate
python3 -m unittest discover tests
```

Expected result:
```text
Ran 8 tests in ~0.7s
OK
```

---

## 🖼️ Expected Output & Visual Layout

The application renders a dual-panel interface:
1. **Left Panel (Terrain Viewport):**
   - Brown/beige undulating corridor representing the unpaved unstructured road.
   - Dark olive border representing ditches, boulders, and non-drivable verges.
   - Blue vehicle icon with center point marker.
   - Red cross-hatched static hazard markers.
   - Yellow bullseye destination target.
   - Bottom status line reporting collision blocking and movement events.
2. **Right Panel (HUD):**
   - **System Telemetry:** Live vehicle coordinates `(X, Y)`, target coordinates, Manhattan distance to goal, and obstacle count.
   - **Perception & Sensors:** Real-time webcam feed status.
   - **Terrain Legend:** Color key identifying all terrain and entity types.
   - **Keyboard Controls:** Summary of keybindings.

A rendered sample preview is stored at `screenshots/day1_simulation_preview.png`.

---

## 🔮 Planned Day 2 Work: Classical Path Planning
On Day 2, we will implement:
- The classical **A* (A-Star) search algorithm** operating directly on the 2D occupancy grid.
- Admissible heuristic functions (Euclidean and Manhattan distance).
- Continuous path extraction avoiding all static hazards and off-road verges.
- Dynamic visualization of explored nodes (closed set) and the calculated optimal path.
