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
- **Static Obstacles:** Total number of active static road hazards.

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

Automated test suites verify both baseline environment physics and Day 2 A* planning:
1. `tests/test_day1.py`: Baseline grid geometry, obstacle seeding, vehicle boundaries, collision prevention, and webcam lifecycle.
2. `tests/test_astar.py`: Comprehensive A* verification covering Manhattan heuristic admissibility, wall detours, enclosed destinations, start==goal edge cases, invalid coordinate rejection, 4-directional enforcement, move-count accuracy, deterministic repeatability, `RoadEnvironment` integration, route clearing on manual movement, and headless rendering.

Run all tests with:
```bash
source venv/bin/activate
python3 -m unittest discover tests
```

Expected result:
```text
Ran 23 tests in ~0.65s
OK
```

---

## 🔮 Planned Day 3 Work: Boundary Cost Fields & Traversal
On Day 3, we will implement:
- **Boundary Cost Fields:** Penalizing traversal near rough road edges and potholes to keep the vehicle centered on the road corridor.
- **Autonomous Traversal:** Enabling the vehicle to autonomously step along the computed A* trajectory with configurable speed and pacing.
- **Dynamic Replanning:** Recalculating routes on the fly if obstacles appear or shift.

