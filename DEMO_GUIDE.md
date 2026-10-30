# Faculty Demonstration Guide & Viva Preparation

## Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Roads

This guide provides a structured **3–5 minute demonstration walkthrough**, speaking scripts, expected on-screen behaviors, technical boundary explanations, and **ten likely faculty viva questions with rigorous answers**.

---

## 1. Quick Demonstration Overview

| Stage | Duration | Primary Action | Key Concept Demonstrated |
| :--- | :--- | :--- | :--- |
| **Phase 1** | 0:00 – 0:45 | Launch `main.py`, show terrain | Unstructured road modeling, irregular road verges |
| **Phase 2** | 0:45 – 1:30 | Press **`P`** | Classical A* path planning with Manhattan heuristic |
| **Phase 3** | 1:30 – 2:15 | Press **`SPACE`** | Paced autonomous navigation (4 Hz) along trajectory |
| **Phase 4** | 2:15 – 3:00 | Press **`O`** while moving | Dynamic hazard insertion & instantaneous adaptive replanning |
| **Phase 5** | 3:00 – 3:30 | Press **`O`** repeatedly to block road | Fail-safe halt (`NO_PATH`), zero-collision guarantee |
| **Phase 6** | 3:30 – 4:15 | Press **`C`** | MacBook FaceTime HD camera selection (bypassing iPhone) |
| **Phase 7** | 4:15 – 5:00 | Press **`M`** | Local YOLO perception-to-grid mapping with safety guards |

---

## 2. Step-by-Step Demonstration Sequence

### Step 1: Launch the Application (0:00 – 0:45)
1. Open terminal, activate virtual environment, and launch:
   ```bash
   source venv/bin/activate
   python main.py
   ```
2. **What to show on screen**:
   - Two concurrent windows open side-by-side without overlapping:
     - **Left Window**: Pygame 1000×700 simulation canvas showing the winding unpaved road corridor, vehicle at `(12, 2)`, destination at `(15, 20)`, static boulders, and the real-time HUD.
     - **Right Window**: OpenCV 400×300 sensor feed showing live MacBook FaceTime HD camera frames with YOLO status overlay.
   - The sidebar displays `[ ▶ MANUAL CONTROL (WASD) ]` badge, telemetry, and keyboard guide.
3. **What to say**:
   > *"Good morning professors. This project implements an adaptive path planning and collision avoidance architecture for autonomous vehicles operating on unstructured off-road terrain lacking painted lane markings. As you can see, our application launches two synchronized, non-overlapping windows: on the left, our 2D simulation canvas tracking the unpaved road, obstacles, and vehicle kinematics; on the right, our live sensor feed directly streaming from the MacBook's FaceTime camera."*

---

### Step 2: Compute Optimal Path via A* Search (0:45 – 1:30)
1. Press the **`P`** key.
2. **What to show on screen**:
   - An emerald green path lights up connecting the vehicle to the destination.
   - A bright trajectory line with luminous green node beads connects cell centers.
   - The sidebar updates:
     - `Remaining Route: 25 moves` (or exact computed count).
     - `A* Compute Latency: < 2.0 ms`.
     - `Explored Search Cells: ~30-50 cells`.
   - The footer displays `[ SYSTEM STATUS: Path found successfully: 25 steps ]`.
3. **What to say**:
   > *"When I press 'P', our modular A\* planner evaluates the grid using an admissible and consistent Manhattan distance heuristic. Because our autonomous rover is kinematically constrained to 4 orthogonal directions—North, South, East, and West—the Manhattan distance exactly mirrors the minimal unconstrained distance. The priority queue explores only the necessary nodes to find the mathematically optimal detour around all static boulders in less than 2 milliseconds."*

---

### Step 3: Closed-Loop Autonomous Waypoint Traversal (1:30 – 2:15)
1. Press the **`SPACE`** bar.
2. **What to show on screen**:
   - The sidebar state badge changes to a glowing blue `[ ● AUTONOMOUS RUNNING ]`.
   - An electric cyan halo pulses around the vehicle.
   - The vehicle smoothly crawls along the emerald path at exactly 4 cells per second (250 ms step pacing).
   - Remaining route count decrements monotonically with each step.
   - Press **`SPACE`** again: the badge switches to amber `[ ⏸ PAUSED ]`, and vehicle immediately halts. Press **`SPACE`** to resume.
3. **What to say**:
   > *"Pressing SPACE initiates closed-loop autonomous navigation. Notice that the graphics rendering runs smoothly at 60 frames per second, while the vehicle kinematic motion is paced at 4 Hertz—one step every 250 milliseconds. This simulates realistic crawling speeds on rough gravel and allows observers to verify trajectory tracking. We can pause and resume at any instant without losing waypoint state."*

---

### Step 4: Dynamic Hazard Injection & Online Replanning (2:15 – 3:00)
1. While the vehicle is moving autonomously, press the **`O`** key.
2. **What to show on screen**:
   - A vibrant safety-orange dynamic hazard appears 3 cells ahead on the vehicle's path.
   - The sidebar badge flashes magenta `[ ⟳ ADAPTIVE REPLANNING ]`.
   - The vehicle **never enters the blocked cell**.
   - Within 1 millisecond, the emerald trajectory dynamically bends around the new obstacle.
   - The sidebar increments `Replanning Events: 1 dynamic`, and `Collisions (Safety)` remains strictly `0 (Safe)`.
   - The vehicle continues along the detour until it reaches the destination.
3. **What to say**:
   > *"In real-world unstructured environments, unexpected obstacles like fallen rocks or livestock appear suddenly. When I press 'O', a dynamic hazard is injected directly on the planned path. Before each step, our lookahead collision checker evaluates the active route. Detecting the hazard, it halts forward progress and triggers instantaneous replanning. Notice the new green detour: collision count remains strictly zero, guaranteeing safety."*

---

### Step 5: Fail-Safe Safe Stop Behavior (3:00 – 3:30)
1. Reset the vehicle by pressing **`R`**.
2. Press **`SPACE`** to start autonomous movement.
3. Rapidly press **`O`** several times to completely seal the road corridor ahead of the vehicle.
4. **What to show on screen**:
   - When the road is completely partitioned and no collision-free path exists to the destination:
   - The sidebar badge switches to high-visibility crimson: `[ ⚠ NO PATH - SAFE STOP ]`.
   - A red warning halo surrounds the stationary vehicle.
   - The footer displays `[ SYSTEM STATUS: No valid path to destination ]`.
   - The vehicle remains safely stationary with `Collisions (Safety): 0 (Safe)`.
5. **What to say**:
   > *"A critical requirement of functional safety (ISO 26262) is fail-safe halting. If landslides or severe washouts completely partition the road corridor such that no collision-free path exists, the vehicle must never blindly advance or crash into obstacles. Our system detects that the A\* open set has exhausted all reachable nodes, immediately halts the vehicle in a safe stop state, and alerts the operator."*

---

### Step 6: MacBook Webcam & Local YOLO Object Detection (3:30 – 4:15)
1. Press the **`C`** key.
2. **What to show on screen**:
   - The secondary OpenCV camera window opens, streaming live video from the MacBook FaceTime HD camera.
   - The HUD sidebar displays `Webcam Feed: ACTIVE (30.0 FPS • 640x480)`.
   - Hold an object (e.g. coffee mug, water bottle, smartphone, or chair) in front of the camera:
     - Real-time bounding boxes appear with class labels (e.g., `bottle: 84%`).
     - Real-time inference latency is displayed in the HUD and camera overlay (e.g. `~35 ms`).
3. **What to say**:
   > *"When I press 'C', our perception subsystem connects to the MacBook's built-in FaceTime HD camera via AVFoundation. Notice that our code automatically inspected system profiler data to isolate the built-in camera, intentionally bypassing any paired iPhone Continuity Camera. We run a local Ultralytics YOLOv11 nano model on Apple Silicon with zero cloud latency and zero API fees. Inference takes approximately 35 milliseconds."*

---

### Step 7: Perception-to-Grid Mapping & Safety Guard (4:15 – 5:00)
1. Press the **`M`** key to enable perception mapping.
2. **What to show on screen**:
   - The HUD sidebar updates: `Perception Mapping: ON (N cells)`.
   - Hold a water bottle or cup in the camera's active field:
     - The bottom-center of the bounding box is projected into discrete grid cells.
     - Electric purple blocks with neon fuchsia borders and glowing sensor rings appear on the Pygame grid!
     - If the object appears on the vehicle's planned path, the A* planner dynamically routes around the vision-detected object!
     - Safety Guard Telemetry confirms: `Safety Guard: Vehicle & Goal Safe`. Even if an object is held over the vehicle's coordinate, the safety guard strictly suppresses it from trapping the vehicle.
   - Press **`M`** again: all purple perception cells disappear, restoring the original road map.
3. **What to say**:
   > *"Pressing 'M' enables our Day 6 perception-to-grid mapping layer. The bottom contact point of each detected bounding box is mapped to the simulation grid. If the detected object obstructs the vehicle's path, the planner recalculates an adaptive detour in real time. Crucially, our safety invariant guard guarantees that phantom visual noise can never overwrite the vehicle's current position or destination. Pressing 'M' wipes the perception cells, preserving map integrity."*

---

## 3. Key Concepts: What to Emphasize

### 1. A* Path Planning
- **Optimality**: Admissible Manhattan heuristic ($h(n) \le h^*(n)$) guarantees optimal (shortest) path.
- **Efficiency**: Priority queue using Python `heapq` with tie-breaking explored nodes in $< 2\text{ ms}$.
- **Orthogonal Kinematics**: 4-connected grid ensures no illegal diagonal cutting through road verges.

### 2. YOLO Object Detection
- **Local & Offline**: Runs 100% locally on Apple Silicon GPU/MPS or CPU using `yolo11n.pt` (~5.4 MB).
- **Decoupled Architecture**: Perception is isolated from planning. A camera failure or detection drop never crashes or blocks vehicle navigation.

### 3. Collision Avoidance & Dynamic Replanning
- **Lookahead Verification**: Active route is validated before every single 250 ms step.
- **Zero Collisions**: Collision counter remains strictly zero across all benchmarks and live demonstrations.
- **Safe Halting**: If no detour exists, vehicle transitions to `NO_PATH` instead of colliding.

---

## 4. Technical Limitations & Future Work

Be honest and academically rigorous when discussing limitations:

1. **Monocular 2D-to-3D Depth Limitation**:
   - *Limitation*: A single 2D camera image does not provide true 3D metric depth without camera calibration and ground plane homography (Inverse Perspective Mapping / IPM).
   - *Future Work*: Integrate stereo depth cameras (e.g. Intel RealSense) or solid-state LiDAR to generate continuous 3D point clouds.
2. **Kinematic vs. Dynamic Modeling**:
   - *Limitation*: The simulation uses a discrete 2D grid rather than continuous Ackermann steering kinematics (steering angle limits, wheel slip, suspension roll).
   - *Future Work*: Integrate a continuous Dubins or Hybrid A* planner with a Model Predictive Controller (MPC).
3. **Sensor Fusion (Day 7 Roadmap)**:
   - *Limitation*: The current system uses monocular vision. True all-weather off-road navigation requires multi-sensor fusion (camera + LiDAR + ultrasonic + radar) into a probabilistic Bayesian costmap.

---

## 5. Ten Likely Faculty Viva Questions & Answers

### Q1: Why did you choose A* over Dijkstra's algorithm or Greedy Best-First Search?
**Answer**:
> *"Dijkstra's algorithm evaluates nodes purely on cumulative path cost $g(n)$, effectively expanding uniformly in all directions like a circle. This results in unnecessary node expansions in directions opposite the goal. Conversely, Greedy Best-First Search considers only the heuristic $h(n)$, which is fast but prone to suboptimal paths and getting trapped in local minima (cul-de-sacs). A\* combines both via $f(n) = g(n) + h(n)$, achieving the mathematical optimality and completeness of Dijkstra while leveraging the directional pruning speed of best-first search."*

---

### Q2: Why is the Manhattan distance heuristic both admissible and consistent in your environment?
**Answer**:
> *"In our simulation, vehicle motion is strictly constrained to 4 orthogonal directions (North, South, East, West) with uniform step cost $c = 1$. The Manhattan distance $h(n) = |r_1 - r_2| + |c_1 - c_2|$ represents the absolute minimum number of orthogonal moves required to reach the destination on an unobstructed grid. Because physical obstacles and road borders can only force detours—increasing the true remaining cost $h^*(n)$—the heuristic never overestimates the cost ($h(n) \le h^*(n)$), proving admissibility. Furthermore, for any two adjacent nodes $n$ and $n'$, the triangle inequality $|h(n) - h(n')| \le c(n, n') = 1$ holds, proving consistency (monotonicity). Consistency guarantees that the first time a node is expanded from the priority queue, its path is already optimal."*

---

### Q3: How does your system detect dynamic obstacles, and what is the computational cost of replanning?
**Answer**:
> *"Before executing each 250 ms kinematic step, the controller invokes `is_route_blocked()`, which inspects the remaining coordinates in `self.planned_path` against the environment's composite obstacle set in $O(K)$ time, where $K$ is the remaining path length (typically $< 30$ cells). If a hazard intersects the planned trajectory, forward motion into that cell is aborted, and `replan_route()` invokes A\* from the vehicle's current position $(r_v, c_v)$ to the destination $(r_g, c_g)$. Across our empirical benchmarks on the $22 \times 26$ grid, the median replanning latency is under $0.8 \text{ ms}$, which is negligible compared to the $250 \text{ ms}$ motion interval."*

---

### Q4: What happens if an obstacle completely blocks the road and no alternative path exists?
**Answer**:
> *"If the road is completely partitioned, the A\* priority queue expands all reachable cells within the isolated corridor. When the open queue empties without encountering the destination cell, the planner returns `AStarResult(success=False, path=[])`. The environment detects this condition and transitions the vehicle state machine to `NavigationMode.NO_PATH`. The vehicle comes to a complete, fail-safe halt, an emergency warning halo is rendered around the vehicle, and the collision counter remains strictly zero."*

---

### Q5: Why did you choose YOLOv11 nano instead of a larger model like YOLOv11x or a cloud vision API?
**Answer**:
> *"Autonomous ground vehicles operate in remote, rural, and off-road environments where cellular connectivity is intermittent or non-existent, making cloud APIs (Google Cloud Vision, OpenAI) unusable due to latency and network dependency. For onboard processing, safety-critical perception requires real-time inference (at least 15–30 FPS). YOLOv11 nano (`yolo11n.pt`) is optimized for embedded devices, requiring only ~5.4 MB of memory and achieving ~35 ms inference on Apple Silicon, whereas YOLOv11x requires over 100 MB and takes $>300\text{ ms}$ on mobile hardware without dedicated industrial GPUs."*

---

### Q6: Can a monocular camera accurately measure 3D real-world metric distance? How does your project address this?
**Answer**:
> *"No, a single 2D camera image loses the depth dimension due to perspective projection; a small nearby object produces the same bounding box as a large distant object. Claiming true 3D metric localization from monocular 2D bounding boxes without extrinsic camera calibration or active depth sensors is scientifically unsound. In our project, we explicitly define and document our mapping as an **approximate demonstration mapping**. We assume a controlled fixed viewport, use the bottom-center of the bounding box as the physical ground-contact proxy, and project normalized ROI coordinates onto the discrete grid. In future work, we plan to incorporate stereo depth disparity or LiDAR point clouds."*

---

### Q7: How does your system prevent perception-derived obstacles from trapping the vehicle or blocking the goal?
**Answer**:
> *"We implemented a dedicated **Safety Invariant Guard** in `GridMapper.map_detections()`. Before any visual detection is converted into a grid obstacle, its coordinates are checked against the vehicle's current position $(veh_r, veh_c)$ and the destination position $(dest_r, dest_c)$. If a conflict is detected—such as sensor noise or an operator standing directly over the vehicle coordinate—the candidate cell is strictly suppressed and logged. This guarantees that phantom sensor noise can never trap the vehicle in an illegal state or render the goal unreachable at inception."*

---

### Q8: How did you solve the macOS Continuity Camera issue with OpenCV?
**Answer**:
> *"On macOS Sonoma and Sequoia, AVFoundation exposes paired iPhones as available video capture devices, often assigning index 0 to the iPhone. Calling `cv2.VideoCapture(0)` inadvertently wakes up the iPhone, triggers connection chimes, and prompts notifications. In `perception/webcam.py`, our `get_macbook_builtin_camera_index()` function queries the macOS system profiler (`system_profiler -json SPCameraDataType`) non-intrusively without opening any camera handles. It parses device model IDs and names, explicitly discarding any device containing 'iPhone' and specifically targeting 'FaceTime HD Camera' or 'built-in'. This guarantees that the application connects exclusively to the MacBook's physical camera."*

---

### Q9: How is the software threaded, and why doesn't neural network inference freeze the 60 FPS Pygame GUI?
**Answer**:
> *"We employ an asynchronous multi-tier concurrency model:
> 1. The **Pygame rendering loop** runs synchronously on the main thread at a capped 60 FPS (16.6 ms tick) to handle user inputs and smooth graphic rendering.
> 2. The **WebcamManager** runs a dedicated daemon worker thread (`_capture_worker`) that continuously acquires frames from the hardware into a thread-safe, single-slot bounded buffer (`_latest_frame`).
> 3. Neural network inference in `main.py` is throttled by a non-blocking delta-time accumulator to execute every 150 ms (~6.7 Hz). Because the detector reads from the memory buffer rather than waiting on hardware I/O, frame rendering never stutters."*

---

### Q10: How does this simulation reflect functional safety principles (ISO 26262), and what would be required for physical deployment?
**Answer**:
> *"Our architecture incorporates three core principles of automotive functional safety (ISO 26262):
> 1. **Fault Isolation / Decoupling**: Perception errors, camera disconnections, or model exceptions do not crash the vehicle controller or corrupt the static terrain map.
> 2. **Fail-Safe Default State**: In the event of total path blockage or unrecoverable sensor loss, the vehicle transitions to a known safe state (`NO_PATH` halt) rather than executing unpredictable maneuvers.
> 3. **Safety Invariants**: Pre-conditions and post-conditions (e.g., zero collisions, non-overwrite of vehicle/goal cells) are enforced by explicit code guards and verified across 80 automated unit tests.
> 
> For physical vehicle deployment, we would additionally require:
> - Continuous Ackermann vehicle dynamics modeling (steering angle, roll, slip).
> - Redundant multi-sensor hardware (automotive-grade LiDAR and radar).
> - Real-time operating system (RTOS) guarantees (e.g. QNX, ROS 2 on real-time Linux).
> - Hardware watchdog timers and CAN bus drive-by-wire motor actuation."*
