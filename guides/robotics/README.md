# Robotics: From Beginner to Advanced

A comprehensive guide to robotics as a *subject* — not just as the hobbyist Arduino project that blinks an LED and drives a two-wheeled cart in a straight line for thirty seconds before the battery dies. The goal is to take you from "I have soldered a servo" to "I can read a 2026 humanoid-manipulation paper, understand why every algorithmic and mechanical decision was made, and stand up a small version of the system end-to-end in simulation and on real hardware." The guide is biased toward what matters in practice now: the math, the systems engineering, and the learning recipes that actually ship.

> **An honest framing.** Most of the "magic" of modern robotics is not a single breakthrough — it is the compounding of a few simple ideas (rigid-body transforms, feedback control, probabilistic estimation, learned policies) executed against the unforgiving thermodynamics of the real world. The math is mostly undergraduate; the engineering is not. This guide teaches [kinematics](/shared/glossary/#kinematics) and dynamics deeply because you cannot debug a wobbling robot you do not understand, and then teaches perception, planning, learning, and systems work that the textbook chapters either skip or treat in isolation.

---

## Table of Contents

1. [Phase 0: Prerequisites](#phase-0-prerequisites)
2. [Phase 1: Rigid-Body Transforms and Kinematics](#phase-1-rigid-body-transforms-and-kinematics)
3. [Phase 2: Dynamics and Classical Control](#phase-2-dynamics-and-classical-control)
4. [Phase 3: Sensing and Perception](#phase-3-sensing-and-perception)
5. [Phase 4: State Estimation, Localization, and SLAM](#phase-4-state-estimation-localization-and-slam)
6. [Phase 5: Motion Planning and Trajectory Optimization](#phase-5-motion-planning-and-trajectory-optimization)
7. [Phase 6: Manipulation and Grasping](#phase-6-manipulation-and-grasping)
8. [Phase 7: Mobile Robots, Legged Locomotion, and Navigation](#phase-7-mobile-robots-legged-locomotion-and-navigation)
9. [Phase 8: Learning for Robotics](#phase-8-learning-for-robotics)
10. [Phase 9: Simulation, Sim-to-Real, and Robot Systems Engineering](#phase-9-simulation-sim-to-real-and-robot-systems-engineering)
11. [Phase 10: Frontier Topics](#phase-10-frontier-topics)
12. [Suggested Timeline](#suggested-timeline)
13. [Key Advice](#key-advice)
14. [Common Pitfalls](#common-pitfalls)
15. [Additional Resources](#additional-resources)
16. [Glossary](/shared/glossary/)

---

## Phase 0: Prerequisites

Robotics combines linear algebra, classical mechanics, probability, control theory, and a surprising amount of plumbing. You can get started without mastering everything below, but if more than a couple are unfamiliar, slow down before continuing.

### Concepts to Know

- **Linear algebra**: matrices and vectors, eigendecomposition, the SVD, null spaces, rank. You should be able to look at a 4×4 homogeneous transform and read off rotation, translation, and the column conventions without flinching.
- **Calculus**: gradients, Jacobians, the chain rule, partial derivatives — Jacobians are the *single most-used object* in this guide.
- **Probability**: random variables, multivariate Gaussians, conditional probability, Bayes' rule, covariance matrices.
- **Classical mechanics**: Newton's laws, conservation of momentum and energy, rigid-body motion, moments of inertia, the Lagrangian.
- **Differential equations**: first-order ODEs, linear systems, state-space form `ẋ = Ax + Bu`, stability.
- **Programming maturity**: Python and C++ at minimum. You will read other people's drivers and middleware at 3 a.m. trying to figure out which of 14 frames is misplaced.
- **Real-time thinking**: a 1 kHz control loop has a budget of 1 millisecond per iteration. Garbage collection pauses, allocator hiccups, and unaware logging will destroy you.

### The One Equation Everything Comes Back To

```
              ┌────────────────────────────────────────────┐
              │                                            │
              │      x_{t+1}  =  f( x_t, u_t, w_t )        │
              │      z_t      =  h( x_t,      v_t )        │
              │                                            │
              └────────────────────────────────────────────┘

A robot is a dynamical system whose state x evolves under controls u and
process noise w, and which you observe only through noisy measurements z.

Everything in this guide — [kinematics](/shared/glossary/#kinematics) (a special case of f), control
(choosing u), perception (recovering x from z), planning (choosing a
sequence of u), learning (fitting f, h, or the controller from data) —
is some specialization of this one state-space view.
```

If that picture is fuzzy now, it will be sharp by the end of Phase 4.

### What You Need Installed

- **Python 3.10+**, NumPy, SciPy, Matplotlib
- **A simulator** — pick at least one early and live in it:
  - **MuJoCo** (free since 2022) — fastest contact-rich simulator; the de facto choice for manipulation and locomotion research
  - **Isaac Sim / Isaac Lab** (NVIDIA) — GPU-parallel, photorealistic, the heavyweight for learning-based work
  - **PyBullet** — easy install, slower physics, fine for early projects
  - **Gazebo / Gazebo Sim** — ROS-native; preferred for navigation and multi-robot
  - **Drake** (Toyota Research / MIT) — best-in-class for trajectory optimization and contact-aware planning
- **ROS 2** (Humble or newer) — the lingua franca middleware. You will hate it; you will use it anyway.
- **A modeling library** — Pinocchio (rigid-body dynamics, very fast), or KDL, or RBDL.
- **Optimization** — CasADi (symbolic, autodiff, MPC), OSQP (quadratic programs), Drake's MathematicalProgram.
- **Some hardware**, eventually — a hobby arm (e.g. low-cost 6-DoF teleop arms), a small mobile base (e.g. a hobby differential-drive platform), or just a webcam and a turntable. The transition from sim to "this servo is making a smell" teaches things sim cannot.

### Resources

- *Modern Robotics: Mechanics, Planning, and Control* (Lynch & Park) — the modern free textbook and the closest thing to a single canonical reference
- *Probabilistic Robotics* (Thrun, Burgard, Fox) — still the bible for estimation and SLAM
- *Robotics, Vision and Control* (Corke) — practical, MATLAB- and Python-heavy
- *Underactuated Robotics* (Tedrake, MIT 6.832 lecture notes online) — the best free resource on modern control
- *A Mathematical Introduction to Robotic Manipulation* (Murray, Li, Sastry) — the screw-theory bible

---

## Phase 1: Rigid-Body Transforms and Kinematics

Before any robot moves, you must be able to write down where every part of it is, in every coordinate frame you care about. [Kinematics](/shared/glossary/#kinematics) is the geometry of motion without worrying about the forces that cause it. Getting frames wrong is the single most common bug in real robot code; getting them right is most of the battle.

### Concepts to Learn

- **Rigid-body transforms** — a position and an orientation that together describe a frame. Three equivalent representations to fluently switch between:
  - **Rotation matrices** `R ∈ SO(3)` — unambiguous, easy to compose, nine numbers for three degrees of freedom
  - **Quaternions** — four numbers, no gimbal lock, awkward to read, the right choice for storage and interpolation
  - **[Euler angles](/shared/glossary/#euler-angles) ([roll](/shared/glossary/#roll), [pitch](/shared/glossary/#pitch), [yaw](/shared/glossary/#yaw))** — three numbers, intuitive, full of singularities, the right choice for *display only*
  - **Axis-angle and rotation vectors** — three numbers, the right choice for optimization variables
- **Homogeneous transforms** `T ∈ SE(3)` — pack `R` and translation `p` into a 4×4 matrix so composition is matrix multiplication:
  - `T = [[R, p], [0, 1]]`
  - `T_AB · T_BC = T_AC` (frame A from B, then B from C, gives A from C)
- **Frame conventions you must commit to writing** — "robot base frame", "world frame", "end-effector frame", "camera frame", "tool frame", "TCP", "tag frame". Half your bugs will come from `T_world_camera` vs `T_camera_world`.
- **The geometric Jacobian** `J(q)` — the linear map from joint velocities `q̇` to end-effector spatial velocity `(v, ω)`. Every velocity-level, force-level, and many position-level control problems revolve around this matrix.
- **Forward kinematics (FK)** — given joint angles `q`, where is the end-effector? A composition of joint transforms. Always solvable, always cheap.
- **Inverse kinematics (IK)** — given a desired end-effector pose, find joint angles `q`. May have 0, 1, finite, or infinite solutions. Three families to know:
  - **Analytic / closed-form** — for arms with special geometries (spherical wrist), the answer can be written down explicitly. Fast, robust, what industrial controllers ship with.
  - **Iterative Jacobian-based** — Newton-style: linearize, step, repeat. Damped least-squares (Levenberg-Marquardt) handles singularities. The workhorse.
  - **Optimization-based** — pose IK as a constrained optimization with joint limits, collision avoidance, posture preferences. Slower; needed for redundant arms.
- **Singularities** — joint configurations where the Jacobian loses rank and you can no longer move the end-effector in some direction. The wrist singularity (three wrist axes aligned) is the famous one; you will meet it.
- **Redundancy** — arms with more than six DoF have a null space: there's a family of joint configurations producing the same end-effector pose. Exploit this for collision avoidance, joint-limit avoidance, manipulability maximization.
- **DH parameters vs. URDF** — Denavit-Hartenberg parameters are the textbook way to describe arm geometry; URDF (Unified Robot Description Format) is the format everyone actually ships with. Learn both; convert between them once.
- **Screw theory** — a unified language for twists (velocities) and wrenches (forces) on rigid bodies. Pays off massively in advanced control and constraint-based manipulation.

### The Forward Kinematics Pipeline, Annotated

```
joint angles  q = (q1, q2, ..., qn)
       │
       ▼   (1) build per-joint transform T_i(q_i) using DH or URDF
[T_01(q1), T_12(q2), ..., T_{n-1,n}(qn)]
       │
       ▼   (2) compose left-to-right: each link in the previous link's frame
T_0n  =  T_01 · T_12 · ... · T_{n-1,n}     ← end-effector in base frame
       │
       ▼   (3) chain to world if base is moving (mobile manip / floating base)
T_wn  =  T_w0 · T_0n
       │
       ▼   (4) compose tool offset for the actual point you care about
T_w_tool  =  T_wn · T_n_tool

If you ever debug a robot motion and a finger flicks where the wrist should
go, ~80% of the time the bug is in one of these four lines: a swapped frame,
a transposed rotation, an inverted convention, or DH alpha vs theta confused.
```

### Why Quaternions Are Worth The Awkwardness

```
Compose two rotations as matrices:  R_3 = R_1 @ R_2          (27 mults)
Compose two rotations as quats:     q_3 = quat_mul(q_1, q_2)  (16 mults)
Interpolate two orientations:       slerp(q_a, q_b, t)        ← clean and unique
Same with Euler angles:             ambiguous, jumps, wraps, dies near poles

Renormalization to stay on the unit sphere: one divide. Gimbal lock: never.
The cost is that q and -q represent the same rotation, and your IK solver,
loss function, or learned model has to handle the double-cover.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Transform calculator](projects/01-transform-calculator/README.md) | Implement `R`, quaternion, axis-angle conversions; round-trip 10 000 random rotations and verify < 1e-10 error | ⭐⭐ |
| [URDF visualizer](projects/02-urdf-visualizer/README.md) | Load a URDF in Python (e.g. via Pinocchio or yourdfpy); render the kinematic tree with random joint angles | ⭐⭐⭐ |
| [Forward kinematics from scratch](projects/03-forward-kinematics-from-scratch/README.md) | Implement FK for a 6-DoF arm using only NumPy and the URDF; verify against the modeling library | ⭐⭐⭐ |
| [Jacobian from scratch](projects/04-jacobian-from-scratch/README.md) | Compute the geometric Jacobian numerically (finite-difference) and analytically; agree to 6 decimal places | ⭐⭐⭐⭐ |
| [Damped least-squares IK](projects/05-damped-least-squares-ik/README.md) | Solve IK for a 7-DoF arm with damping near singularities; visualize wrist singularity and recovery | ⭐⭐⭐⭐ |
| [Null-space posture control](projects/06-null-space-posture-control/README.md) | Use the Jacobian null-space to maintain a "home" posture while tracking an end-effector trajectory | ⭐⭐⭐⭐⭐ |
| [Hand-eye calibration](projects/07-hand-eye-calibration/README.md) | Compute the transform from camera frame to end-effector using the AX = XB formulation; measure residual | ⭐⭐⭐⭐ |

### Sample Code: A Minimal 6-DoF Forward Kinematics

```python
import numpy as np

def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def T_from_Rp(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T

def fk(q, urdf_chain):
    """urdf_chain: list of (axis, origin_R, origin_p) per joint, revolute."""
    T = np.eye(4)
    for q_i, (axis, R0, p0) in zip(q, urdf_chain):
        T_origin = T_from_Rp(R0, p0)
        T_joint  = T_from_Rp({'x': Rx, 'y': Ry, 'z': Rz}[axis](q_i), np.zeros(3))
        T = T @ T_origin @ T_joint
    return T

# end-effector pose:
T_we = fk(q, chain)
position    = T_we[:3, 3]
orientation = T_we[:3, :3]
```

### Sample Code: Damped Least-Squares IK

```python
def dls_ik(q0, target_T, fk_fn, jac_fn, max_iters=100, damping=1e-2, tol=1e-4):
    q = q0.copy()
    for _ in range(max_iters):
        T = fk_fn(q)
        # Error = (position error, axis-angle of rotation error)
        e_p = target_T[:3, 3] - T[:3, 3]
        R_err = target_T[:3, :3] @ T[:3, :3].T
        e_w = 0.5 * np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
        e = np.concatenate([e_p, e_w])
        if np.linalg.norm(e) < tol:
            return q
        J = jac_fn(q)                              # (6, n)
        # Damped pseudoinverse: J^T (J J^T + λ²I)^-1 e
        JJT = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, e)
        q = q + dq
    return q
```

### Key Insight

Frames are not bureaucracy — they are the source language of robotics. Every line of robot code that crosses a coordinate-frame boundary is a chance for a bug, and the bugs are silent: the robot moves *somewhere*, just not where you meant. The single highest-ROI habit is to name transforms by their endpoints (`T_world_camera`, never just `cam_pose`) and check the chain rule before every product. A discipline of frame hygiene pays for itself in saved hours every week, forever.

### Resources

- *Modern Robotics* (Lynch & Park) — Chapters 2–5
- *A Mathematical Introduction to Robotic Manipulation* (Murray, Li, Sastry) — Chapters 1–3
- Pinocchio documentation (rigid-body algorithms, very fast)
- KDL and PyKDL documentation (the ROS-flavored alternative)
- "Quaternion kinematics for the error-state Kalman filter" (Solà, 2017) — the best free quaternion primer

---

## Phase 2: Dynamics and Classical Control

[Kinematics](/shared/glossary/#kinematics) describes geometry; dynamics describes what happens when forces meet inertia. Classical control is the toolkit for closing the loop: measure, decide, actuate, repeat — fast enough that the laws of physics don't notice you're not omniscient.

### Concepts to Learn

- **The manipulator equation** — the master equation of rigid-body dynamics for a robot arm:
  - `M(q)·q̈ + C(q, q̇)·q̇ + g(q) = τ + J(q)ᵀ · F_ext`
  - `M(q)` — mass matrix (symmetric, positive-definite, configuration-dependent)
  - `C(q, q̇)` — Coriolis and centrifugal terms
  - `g(q)` — gravity torques
  - `τ` — applied joint torques
  - `F_ext` — external wrench (e.g. contact, payload)
- **Recursive Newton-Euler (RNEA)** — `O(n)` algorithm to compute the torques required to follow a given trajectory. The workhorse for inverse dynamics.
- **Articulated-Body Algorithm (ABA)** — `O(n)` forward dynamics: given torques, compute accelerations. The simulator's inner loop.
- **Open-loop vs. closed-loop control**:
  - **Open-loop**: apply a precomputed torque, hope the world cooperates. Fragile.
  - **Closed-loop**: measure, compare to reference, correct. The whole discipline.
- **PID control** — the workhorse linear controller:
  - `u = K_p · e + K_i · ∫e dt + K_d · ė`
  - Proportional pushes toward the target, integral kills steady-state error, derivative damps oscillation.
  - 80% of real robot motion is some flavor of PID with feedforward.
- **PID tuning** — Ziegler-Nichols, manual tuning heuristics, auto-tuners. Most working roboticists tune by ear; learning the *intuition* of how each gain feels matters more than any formula.
- **Feedforward + feedback** — predict the torque needed for the reference trajectory (using inverse dynamics), then add PID correction for the residual. Dramatically better than feedback alone.
- **Computed-torque / inverse-dynamics control** — full feedforward using `M, C, g`; reduces the closed-loop system to a linear one you can pole-place trivially. The right baseline for any torque-controlled arm.
- **Impedance and admittance control** — instead of commanding a position, command a virtual spring-damper between the end-effector and a reference. The right abstraction for contact-rich tasks (insertion, polishing, human collaboration).
- **LQR (Linear-Quadratic Regulator)** — optimal feedback for linear systems with quadratic cost. The single most useful "advanced" controller; works extraordinarily well after linearizing about a setpoint.
- **MPC (Model Predictive Control)** — solve a finite-horizon optimization at every control step, apply the first action, re-solve next step. Handles constraints, nonlinearity, and prediction. The dominant control method in mobile robots and increasingly in manipulation.
- **State-space representation** — `ẋ = Ax + Bu`, `y = Cx + Du`. Controllability, observability, stability via eigenvalues of `A`. The language LQR and Kalman filtering live in.
- **Stability** — Lyapunov functions, BIBO stability, gain and phase margins. You don't need the full theorem; you do need to know when your controller will blow up.
- **Actuator realities**:
  - **Position-controlled servos** — easy to use, no torque control, all dynamics is hidden behind an internal PID
  - **Current/torque-controlled motors** — what serious manipulation and locomotion need; pair with a torque sensor or motor-model
  - **Series-elastic and quasi-direct-drive** — compliant or transparent actuators that make contact safer and torque control accurate
  - **Backlash, friction, stiction, gear-ratio limits** — the real-world terms your simulator pretends don't exist
- **Sample rate** — 100 Hz is the floor for trajectory tracking, 1 kHz the floor for torque control, 10 kHz+ for current loops. Jitter matters as much as rate.

### A Working Robot Control Stack

```
            high-level goal ("move the gripper there")
                          │
                          ▼
              ┌─────────────────────────┐
              │   Motion planner        │  10 Hz, off-line / re-plan
              │   (Phase 5)             │  outputs: trajectory q(t)
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Trajectory tracker    │  100–1000 Hz
              │   FF: inverse dynamics  │  outputs: τ_ref
              │   FB: PID or LQR        │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Motor torque / current│  1–10 kHz
              │   inner loop            │
              └─────────────────────────┘
                          │
                          ▼
                       Physics
                          │
                          ▼
              ┌─────────────────────────┐
              │   State estimator       │  closes every loop
              │   (Phase 4)             │  outputs: x̂
              └─────────────────────────┘

Each layer runs faster than the one above it and treats the layer below
as "just physics that responds to commands." Choose your abstraction
boundary deliberately; bugs love unclear boundaries.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Pendulum PID](projects/08-pendulum-pid/README.md) | Simulate a single pendulum; tune PID to stabilize the up position; plot step response | ⭐⭐ |
| [Cart-pole LQR](projects/09-cart-pole-lqr/README.md) | Linearize about upright; design LQR; observe basin of attraction | ⭐⭐⭐ |
| [Inverse dynamics from scratch](projects/10-inverse-dynamics-from-scratch/README.md) | Implement RNEA for a 2-link arm; verify against Pinocchio | ⭐⭐⭐⭐ |
| [Computed-torque trajectory tracking](projects/11-computed-torque-trajectory-tracking/README.md) | Track a sinusoidal joint trajectory on a 6-DoF arm; compare PID-only vs. FF+PID | ⭐⭐⭐⭐ |
| [Impedance control](projects/12-impedance-control/README.md) | Push against the arm in simulation; tune virtual stiffness; measure compliance | ⭐⭐⭐⭐ |
| [MPC for a unicycle](projects/13-mpc-for-a-unicycle/README.md) | Track a figure-8 with a kinematic-bicycle MPC using CasADi or do-mpc | ⭐⭐⭐⭐ |
| [Real-arm PID tune](projects/14-real-arm-pid-tune/README.md) | On a real hobby arm, tune the joint PIDs and characterize the friction; create a friction-compensation FF | ⭐⭐⭐⭐⭐ |
| [Force-controlled drawing](projects/15-force-controlled-drawing/README.md) | Use impedance control to make the arm draw on a curved surface without breaking the pen | ⭐⭐⭐⭐⭐ |

### Sample Code: An LQR for a Linearized System

```python
import numpy as np
from scipy.linalg import solve_continuous_are

def lqr(A, B, Q, R):
    """Continuous-time infinite-horizon LQR. Returns gain K and cost-to-go P."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P

# Cart-pole linearized about upright (g=9.81, M=1, m=0.1, l=0.5)
g, M, m, l = 9.81, 1.0, 0.1, 0.5
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, 0, (M+m)*g/(M*l), 0],
])
B = np.array([[0], [1/M], [0], [-1/(M*l)]])
Q = np.diag([1.0, 1.0, 10.0, 1.0])
R = np.array([[0.01]])

K, _ = lqr(A, B, Q, R)
# Control law: u = -K @ x  (x = [cart_pos, cart_vel, pole_angle, pole_rate])
```

### Sample Code: A Simple Computed-Torque Controller

```python
def computed_torque(q, qd, q_ref, qd_ref, qdd_ref, dyn):
    """dyn.M(q), dyn.C(q, qd), dyn.g(q) — from Pinocchio / RBDL / your code."""
    # Outer-loop PD on tracking error → virtual acceleration
    Kp = np.diag([100, 100, 100, 50, 50, 30])
    Kd = np.diag([20,  20,  20, 10, 10, 6])
    qdd_cmd = qdd_ref + Kp @ (q_ref - q) + Kd @ (qd_ref - qd)
    # Cancel out the nonlinear dynamics
    tau = dyn.M(q) @ qdd_cmd + dyn.C(q, qd) @ qd + dyn.g(q)
    return tau
```

### Key Insight

Almost every "advanced" controller — LQR, MPC, impedance, computed-torque — is the same idea wearing different clothes: **build a model, predict where things will go, choose actions that minimize the difference between prediction and goal, and close the loop on what you can actually measure.** PID is the same idea with no model. The leap from "I can make a thing move" to "I can make a thing move *with grace under disturbance*" is almost entirely the leap from no model to a *good enough* model, plus the discipline to run the loop fast enough that small corrections suffice. Always reach for the simplest controller that meets the spec; almost no real problem requires the fanciest one.

### Resources

- *Modern Robotics* (Lynch & Park) — Chapters 8–11
- *Underactuated Robotics* (Tedrake) — Chapters 1–4, the best free intro to LQR and MPC
- *Feedback Systems* (Åström & Murray) — the classical-control textbook, free online
- Pinocchio examples (RNEA, CRBA, ABA on real URDFs)
- CasADi examples (NMPC tutorials)

---

## Phase 3: Sensing and Perception

A robot is blind until you give it sensors. Each sensor has its own noise model, latency, failure mode, and coordinate frame, and combining them well is most of what makes a robot robust. Perception is *not* "computer vision"; it is the much broader job of turning every raw signal into actionable belief about the world.

### Concepts to Learn

- **The dominant robot sensors** — know their physics, their noise, their failure modes:
  - **RGB cameras** — high-resolution, cheap, photometric/lighting-sensitive, no depth without inference or pairs
  - **Stereo cameras** — pair of synchronized cameras + disparity; depth quality falls off with distance squared
  - **Time-of-Flight (ToF) / structured-light depth cameras** — direct depth, struggle in sunlight, struggle on shiny/transparent objects
  - **2D [LiDAR](/shared/glossary/#lidar)** — fast horizontal range scans; the backbone of indoor mobile robots
  - **3D [LiDAR](/shared/glossary/#lidar)** — rotating or solid-state, dense point clouds, the backbone of self-driving and outdoor mobile robots
  - **IMU (inertial measurement unit)** — 3-axis gyro + 3-axis accelerometer (optional magnetometer); high rate, drifts in seconds, indispensable
  - **Encoders** — joint position and velocity, the cleanest measurement on a robot
  - **Force/torque sensors** — wrist-mounted F/T or fingertip taxels; the eyes of contact-rich manipulation
  - **GNSS / RTK GPS** — outdoor position; centimeter-class with RTK; fails in cities and indoors
  - **Tactile / vision-based-tactile (e.g. GelSight-style)** — high-resolution contact images; the new frontier in dexterous manipulation
- **The camera pinhole model** — `x_pixel = K · [R | t] · X_world`, with `K` the intrinsic matrix `(fx, fy, cx, cy)`. Distortion (radial, tangential) on top of that. Calibrate before doing anything.
- **Intrinsic calibration** — checkerboard / ChArUco / Kalibr; gives `K` and distortion coefficients per camera.
- **Extrinsic calibration** — the transform between sensor frames (camera ↔ IMU, camera ↔ [lidar](/shared/glossary/#lidar), camera ↔ end-effector). The "hand-eye calibration" of Phase 1 is the manipulator special case.
- **Time synchronization** — sensors drift, clocks drift, USB introduces jitter. Hardware-triggered cameras + PTP-synced clocks > software timestamps. ROS message_filters / ApproximateTime helps but doesn't save you.
- **Classical CV building blocks** (still everywhere):
  - **Edge / corner / blob detection** — Canny, Harris, FAST, ORB, SIFT
  - **Feature matching and descriptors** — for VO, SLAM, panoramas
  - **Optical flow** — Lucas-Kanade, dense flow nets
  - **Stereo and structure-from-motion** — triangulation, bundle adjustment
  - **Fiducials** — ArUco / AprilTag — the cheapest, most reliable known-object pose source. Tape one on everything during development.
- **Modern learned perception**:
  - **Object detection and segmentation** — YOLO family, DETR, SAM-style models
  - **6-DoF pose estimation** — FoundationPose-style models, dense-correspondence approaches
  - **Monocular depth estimation** — DPT, Depth Anything, MiDaS — surprisingly useful as a prior
  - **Optical flow nets** — RAFT and successors
  - **Visual-inertial / visual front-ends for SLAM** — SuperPoint + SuperGlue replacing classical features
  - **Open-vocabulary perception** — CLIP, OWL-ViT, OpenSeg — language-conditioned segmentation, the bridge into Phase 10 VLA models
- **Point cloud processing** — voxel filters, downsampling, [ICP (iterative closest point)](/shared/glossary/#iterative-closest-point-icp) for [registration](/shared/glossary/#point-cloud-registration), normal estimation, segmentation.
- **Sensor noise and outlier rejection** — RANSAC is a hammer; use it. Robust loss functions (Huber, Cauchy) in any optimization touching real data.
- **Latency budgets** — perception adds latency; latency adds phase lag; phase lag eats your control margins. Always characterize end-to-end latency from photon to actuation.

### A Perception Pipeline, Annotated

```
   sensors                              every signal has a frame and a timestamp
      │
      ▼
   drivers   ──── publish raw frames at sensor's native rate
      │
      ▼
   calibration / undistort
      │
      ▼
   detection / feature extraction
      │
      ▼
   per-modality estimation       e.g. depth from stereo, object pose from RGB
      │
      ▼
   fusion        ←───────  this is where Phase 4's filters live
      │
      ▼
   world model: tracked objects, occupancy, semantic map
      │
      ▼
   downstream: planning, control, behavior

Build it backwards. Start by deciding what the planner needs to consume,
then ask what the perception layer must produce, then choose sensors and
algorithms. Far too many systems pick fancy sensors first and discover
the planner only wanted a binary "is there a person nearby?"
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Camera calibration](projects/16-camera-calibration/README.md) | Calibrate your webcam with a checkerboard; report reprojection error < 0.5 px | ⭐⭐ |
| [AprilTag pose](projects/17-apriltag-pose/README.md) | Detect AprilTags, recover 6-DoF pose, project axes back onto the image | ⭐⭐⭐ |
| [Stereo depth](projects/18-stereo-depth/README.md) | Build a stereo rig (two webcams); compute disparity; convert to depth point cloud | ⭐⭐⭐⭐ |
| [ICP registration](projects/19-icp-registration/README.md) | Register two partial point clouds of the same scene; visualize convergence | ⭐⭐⭐ |
| [Visual odometry](projects/20-visual-odometry/README.md) | Implement a monocular VO front-end on a public dataset; report drift over 100 m | ⭐⭐⭐⭐⭐ |
| [IMU integration](projects/21-imu-integration/README.md) | Integrate an IMU on a desk; observe drift in angle (seconds) and position (seconds!) | ⭐⭐⭐ |
| [Hand-eye calibration](projects/07-hand-eye-calibration/README.md) | Solve AX = XB by capturing arm and tag poses; verify by reprojecting tag from arm motion | ⭐⭐⭐⭐ |
| [Object 6-DoF pose](projects/22-object-6-dof-pose/README.md) | Pick a small set of objects, train or fine-tune a 6-DoF pose estimator, evaluate ADD-S | ⭐⭐⭐⭐⭐ |
| [Open-vocab grasping](projects/23-open-vocab-grasping/README.md) | Use SAM + a VLM to segment "the red cup" and propose a top-down grasp | ⭐⭐⭐⭐⭐ |

### Sample Code: AprilTag Pose to Robot Frame

```python
import cv2
import numpy as np

def detect_tag_pose(frame, K, dist, tag_size):
    """Returns T_cam_tag (4x4) for a detected tag, or None."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None: return None

    # 3D corner points in the tag frame (planar, z=0)
    s = tag_size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners[0][0], K, dist)
    if not ok: return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.flatten()
    return T

# Compose into the robot world frame:
T_cam_tag = detect_tag_pose(...)
T_world_tag = T_world_cam @ T_cam_tag        # if the camera is fixed
# or
T_world_tag = T_world_ee @ T_ee_cam @ T_cam_tag   # eye-in-hand
```

### Key Insight

The hardest part of perception is not the algorithm — it is honest accounting of what each signal does and doesn't tell you, and the discipline to model the noise, not just the mean. A 1 cm depth error on a coffee cup at 1 m looks fine on a chart; the same error makes the difference between a grasp and a knocked-over cup. Always pair every estimate with an uncertainty. Filters and planners that ignore uncertainty fail in surprising and unrecoverable ways; the ones that respect it degrade gracefully.

### Resources

- *Multiple View Geometry in Computer Vision* (Hartley & Zisserman) — the classical CV bible
- *Probabilistic Robotics* (Thrun et al.) — Chapters 5–7 on sensor models
- OpenCV documentation and tutorials
- Open3D documentation (point clouds and registration)
- Kalibr (multi-camera, camera-IMU calibration)
- Hands-on labs from Stanford CS231A, CMU 16-720

---

## Phase 4: State Estimation, Localization, and SLAM

A robot's belief about itself and the world is always a probability distribution. The job of state estimation is to take noisy measurements, a noisy motion model, and a prior belief, and produce the best posterior — fast enough that the planner downstream gets it in time. Localization is "where am I in a known map?". SLAM is "where am I, and what's the map, at the same time?".

### Concepts to Learn

- **Bayes' filter** — the abstract recursion every estimator implements:
  - **Predict**: `bel_pred(x_t) = ∫ p(x_t | x_{t-1}, u_t) · bel(x_{t-1}) dx_{t-1}`
  - **Update**:  `bel(x_t) = η · p(z_t | x_t) · bel_pred(x_t)`
- **The Kalman filter (KF)** — the optimal Bayes filter when everything is linear and Gaussian. Closed-form, fast, ubiquitous.
- **Extended Kalman filter (EKF)** — linearize a nonlinear model around the current estimate; the workhorse of legacy robotics. Diverges silently when linearization fails.
- **Unscented Kalman filter (UKF)** — pass a small set of "sigma points" through the nonlinear model; better than EKF for highly nonlinear systems, same `O(n³)` cost.
- **Error-state Kalman filter** — the form that makes IMU integration tractable: state is composed of nominal + error; the error is small and well-modeled as Gaussian. The right tool for INS/visual-inertial.
- **Particle filter (Monte Carlo localization)** — represent the posterior with weighted samples. Handles arbitrary distributions including multi-modal; expensive in high dimensions; the right tool for global localization on a map.
- **Factor graphs and nonlinear least squares** — the modern view: every measurement is a factor; the MAP estimate is the solution to a sparse least-squares problem solved with Gauss-Newton / Levenberg-Marquardt. Implemented in GTSAM, Ceres, g2o.
- **Visual odometry (VO)** — estimate motion from images alone. Drifts without loop closure.
- **Visual-inertial odometry (VIO)** — fuse camera and IMU; the high-rate, low-drift backbone of mobile/aerial robots.
- **[LiDAR](/shared/glossary/#lidar) odometry** — [ICP](/shared/glossary/#iterative-closest-point-icp)-style scan matching; LOAM-family stacks are the standard.
- **[SLAM](/shared/glossary/#slam)** front-end vs. back-end:
  - **Front-end**: detect features, match them, propose constraints
  - **Back-end**: optimize the [pose graph](/shared/glossary/#pose-graph) (smoothing) or run the filter (filtering)
  - Modern [SLAM](/shared/glossary/#slam) is mostly *smoothing*: keep a window of recent poses, optimize jointly. iSAM2 / GTSAM made this real-time.
- **[Loop closure](/shared/glossary/#loop-closure)** — when you revisit a place, recognize it (visual bag-of-words, learned place recognition) and add a constraint that closes the drift. The hardest unsolved part of long-horizon SLAM.
- **Map representations** — occupancy grids (2D), octomaps (3D), TSDFs (dense surfaces), point clouds, ORB-SLAM-style sparse landmarks, neural fields. Choose for the *consumer* of the map (planning, collision, rendering).
- **Robust estimation** — switchable constraints, max-mixtures, graduated non-convexity; do not trust raw measurements in factor graphs.
- **Observability** — not every state is observable from every sensor configuration. The [yaw](/shared/glossary/#yaw) of a stationary IMU is not observable; the scale of a monocular VO is not observable. Know which states float.

### The Kalman Filter in Six Lines

```python
def kalman_step(x, P, u, z, F, B, H, Q, R):
    # Predict
    x_pred = F @ x + B @ u
    P_pred = F @ P @ F.T + Q
    # Update
    y = z - H @ x_pred                            # innovation
    S = H @ P_pred @ H.T + R                      # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)           # Kalman gain
    x = x_pred + K @ y
    P = (np.eye(len(x)) - K @ H) @ P_pred
    return x, P
```

That's it. The entire Kalman filter is six lines. The other 5 000 lines of a real estimator are sensor drivers, time alignment, outlier rejection, and bookkeeping.

### The SLAM Big Picture

```
                  raw images / lidar / IMU
                          │
                          ▼
              ┌─────────────────────────┐
              │  Front-end:             │
              │  features / scan match  │
              │  data association       │
              │  loop closure proposal  │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │  Back-end:              │
              │  factor graph           │
              │  Levenberg-Marquardt    │
              │  marginalize old states │
              └─────────────────────────┘
                          │
                          ▼
                refined trajectory + map

Filters (EKF-SLAM, MSCKF) marginalize aggressively and run fast.
Smoothers (iSAM2, ORB-SLAM3, VINS-Fusion) keep a larger window
and re-optimize, getting better accuracy at higher CPU cost. Smoothers
dominate modern systems.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [1D KF](projects/24-1d-kf/README.md) | Fuse two noisy thermometers; compare to weighted average; verify covariance contraction | ⭐⭐ |
| [2D constant-velocity tracker](projects/25-2d-constant-velocity-tracker/README.md) | Track a moving target with KF; tune Q vs. R; observe steady-state gain | ⭐⭐⭐ |
| [EKF localization](projects/26-ekf-localization/README.md) | Implement EKF on a wheeled robot with wheel odometry + landmark range/bearing | ⭐⭐⭐⭐ |
| [Particle filter](projects/27-particle-filter/README.md) | Monte Carlo localization on a known 2D occupancy map | ⭐⭐⭐ |
| [IMU dead reckoning](projects/21-imu-integration/README.md) | Integrate accelerometer + gyro on a desk; quantify drift; compare error-state vs. direct integration | ⭐⭐⭐⭐ |
| [VIO MVP](projects/28-vio-mvp/README.md) | Fuse a monocular camera + IMU using an error-state EKF; evaluate on a public dataset | ⭐⭐⭐⭐⭐ |
| [2D LiDAR SLAM](projects/29-2d-lidar-slam/README.md) | Reproduce a small [ICP](/shared/glossary/#iterative-closest-point-icp)-based scan-matching [SLAM](/shared/glossary/#slam) with [loop closure](/shared/glossary/#loop-closure) | ⭐⭐⭐⭐⭐ |
| [Factor-graph practice](projects/30-factor-graph-practice/README.md) | Use [GTSAM](/shared/glossary/#gtsam) to build a small [pose-graph](/shared/glossary/#pose-graph) [SLAM](/shared/glossary/#slam), intentionally inject [loop-closure](/shared/glossary/#loop-closure) outliers, fix with robust kernels | ⭐⭐⭐⭐⭐ |

### Sample Code: A Particle Filter for 2D Localization

```python
def particle_filter(particles, weights, u, z, motion, sensor_lik, n_resample_thresh):
    # 1. Predict — propagate every particle through the motion model
    particles = np.array([motion.sample(p, u) for p in particles])

    # 2. Update — reweight by measurement likelihood
    weights *= np.array([sensor_lik(z, p) for p in particles])
    weights += 1e-300                                  # numerical floor
    weights /= weights.sum()

    # 3. Resample if effective sample size is low
    ess = 1.0 / np.sum(weights ** 2)
    if ess < n_resample_thresh:
        idx = np.random.choice(len(particles), size=len(particles), p=weights)
        particles = particles[idx]
        weights = np.ones(len(particles)) / len(particles)

    return particles, weights
```

### Key Insight

State estimation is the place where the philosophy of robotics — "the world is uncertain; act on best beliefs, not facts" — meets concrete linear algebra. The single most useful upgrade to a struggling robot is rarely a better sensor or a smarter planner; it's a more honest noise model. Filters that report tight, confident estimates and are quietly wrong cause crashes. Filters that report wide, honest estimates let downstream planners make graceful decisions. The mantra: **always carry covariance with your mean; always check innovations; never let a filter run open-loop on a sensor that has failed.**

### Resources

- *Probabilistic Robotics* (Thrun, Burgard, Fox) — Chapters 2–4, 7–13, the entire subject
- "Quaternion kinematics for the error-state Kalman filter" (Solà) — the best IMU-fusion primer
- GTSAM tutorials and examples
- Cyrill Stachniss YouTube lectures — outstanding free SLAM course
- The KITTI, EuRoC, and TUM RGB-D datasets — for benchmarking

---

## Phase 5: Motion Planning and Trajectory Optimization

Once you know where the robot is and what it can sense, you have to decide where it should go. Motion planning is the problem of computing a feasible, collision-free, dynamically reasonable trajectory from a start to a goal. It splits naturally into geometric path planning (where) and trajectory optimization (when / how fast).

### Concepts to Learn

- **[Configuration space (C-space)](/shared/glossary/#c-space)** — the abstract space of all joint configurations. A 7-DoF arm lives in a 7-dimensional torus. Obstacles in the workspace become exotic blobs in C-space. Planning is really searching C-space.
- **The C-space distinction matters because**: collisions are defined in workspace, but search is efficient in C-space; the bridge is collision-checking (FK + per-link mesh tests).
- **Discrete graph search**:
  - **[Dijkstra](/shared/glossary/#dijkstras-algorithm)** — shortest path in a weighted graph
  - **[A*](/shared/glossary/#a-star-search)** — Dijkstra + heuristic; the workhorse of grid-based path planning
  - **D* / D* Lite** — incremental replanning when the map changes; standard on outdoor mobile robots
- **Sampling-based motion planning** — for high-DoF arms and continuous spaces:
  - **[PRM (Probabilistic Roadmap)](/shared/glossary/#prm)** — preprocess: sample, connect; query: [A*](/shared/glossary/#a-star-search) on the roadmap. Good for multi-query in a static world.
  - **[RRT (Rapidly-exploring Random Tree)](/shared/glossary/#rrt)** — incremental tree growth toward random samples. Single-query, great for high-DoF arms.
  - **[RRT-Connect](/shared/glossary/#rrt-connect)** — bidirectional, fast in practice, the default starting point.
  - **[RRT*](/shared/glossary/#rrt-star)** — [asymptotically optimal](/shared/glossary/#asymptotic-optimality) version; rewires the tree to improve cost.
  - **BIT*** and **AIT*** — modern any-time, batch-informed planners; combine sampling with heuristic search.
  - **[OMPL](/shared/glossary/#ompl)** — the Open Motion Planning Library packages all of these (and, importantly, a [k-d tree](/shared/glossary/#k-d-tree) for the nearest-neighbour query that otherwise makes these planners quadratic).
- **Trajectory smoothing and shortcutting** — sampling-based plans are jerky; post-process by random [shortcutting](/shared/glossary/#shortcut-smoothing), B-spline fitting, or [trajectory optimization](/shared/glossary/#trajectory-optimization). Note what this cannot do: smoothing never leaves the [homotopy class](/shared/glossary/#homotopy-class) it was handed, so a path around the wrong side of an obstacle stays on the wrong side.
- **[Trajectory optimization](/shared/glossary/#trajectory-optimization)** — formulate path-planning as continuous optimization:
  - **[CHOMP](/shared/glossary/#chomp)** — gradient-based, attracts toward goal, repels from obstacles via a [Signed Distance Field (SDF)](/shared/glossary/#sdf)
  - **[TrajOpt](/shared/glossary/#trajopt)** — sequential convex programming with collision avoidance
  - **[STOMP](/shared/glossary/#stomp)** — gradient-free, stochastic variant
  - **GCS (Graphs of Convex Sets)** — modern: decompose free space into convex regions; optimize over the graph
  - **[Direct collocation](/shared/glossary/#direct-collocation) / [direct shooting](/shared/glossary/#single-shooting)** — for dynamics-aware trajectory optimization (the bridge to MPC)
- **Time-parameterization** — the geometric path is one thing; how fast you traverse it is another. TOPP (time-optimal path parameterization) finds the fastest traversal respecting velocity, acceleration, and torque limits.
- **Kinodynamic planning** — plan in state-space (including velocities, torques), not just configuration. Necessary for under-actuated systems, dynamic motions, drones, legged robots.
- **Constraint-aware planning** — task constraints (keep the cup upright, stay on a surface, maintain contact). Project samples onto the constraint manifold (Atlas RRT, CBiRRT) or formulate as nonlinear programs.
- **Replanning under uncertainty** — POMDPs, belief-space planning. Mostly intractable in full generality; in practice we replan fast.
- **The failure mode to know by name** — the [narrow passage](/shared/glossary/#narrow-passage). The chance of a uniform sample landing in a passage is proportional to its volume, and a long thin one needs several consecutive samples inside it, so difficulty compounds rather than adding up. [Probabilistic completeness](/shared/glossary/#probabilistic-completeness) promises you will get through eventually and says nothing about when.
- **Hybrid systems planning** — planning over modes (in contact vs. not, gear up vs. gear down). Footstep planning, contact-rich manipulation. The cutting edge.

### The Two Worlds of Motion Planning

```
                          GEOMETRIC                          DYNAMIC
                  ─────────────────────────         ─────────────────────────
   variable       configuration q                   state x = (q, q̇)
   constraint     collision-free C_free             also dynamics & torques
   typical alg    RRT / A* / PRM                    direct collocation / iLQR
   timescale      seconds to minutes                fractions of a second
   output         a path                            a trajectory + controls
   downstream     follow with a tracker             apply controls directly

   real systems   sample a path, then optimize a trajectory along it;
                  the path-then-trajectory pipeline is the practical default.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [A* on a grid](projects/31-a-star-on-a-grid/README.md) | Implement A* with the [Manhattan](/shared/glossary/#manhattan-distance) heuristic; verify optimality vs. Dijkstra | ⭐⭐ |
| [RRT in 2D](projects/32-rrt-in-2d/README.md) | Plan around random obstacles; visualize tree growth | ⭐⭐⭐ |
| [RRT-Connect for an arm](projects/33-rrt-connect-for-an-arm/README.md) | Plan a 7-DoF reach around a table in MuJoCo or PyBullet | ⭐⭐⭐⭐ |
| [Shortcut smoothing](projects/34-shortcut-smoothing/README.md) | Post-process an RRT plan; quantify length reduction | ⭐⭐⭐ |
| [CHOMP from scratch](projects/35-chomp-from-scratch/README.md) | Optimize a trajectory against an SDF on a 2D environment | ⭐⭐⭐⭐⭐ |
| [TOPP](projects/36-topp/README.md) | Time-parameterize a geometric path under joint-velocity and acceleration limits | ⭐⭐⭐⭐ |
| [Direct collocation](projects/37-direct-collocation/README.md) | Cart-pole swing-up via collocation in CasADi; compare to LQR for stabilization | ⭐⭐⭐⭐⭐ |
| [Footstep planning](projects/38-footstep-planning/README.md) | A* over a footstep lattice for a planar biped on stepping stones | ⭐⭐⭐⭐⭐ |

### Sample Code: RRT in 2D

```python
import numpy as np

def rrt(start, goal, in_collision, sample_state, step=0.5, max_iters=5000, goal_bias=0.1):
    nodes = [start]
    parents = [-1]
    for _ in range(max_iters):
        target = goal if np.random.rand() < goal_bias else sample_state()
        nearest_idx = np.argmin([np.linalg.norm(target - n) for n in nodes])
        direction = target - nodes[nearest_idx]
        d = np.linalg.norm(direction)
        new = nodes[nearest_idx] + (step / max(d, 1e-9)) * direction
        if not in_collision(nodes[nearest_idx], new):
            nodes.append(new); parents.append(nearest_idx)
            if np.linalg.norm(new - goal) < step:
                # Reconstruct path
                path, i = [], len(nodes) - 1
                while i != -1: path.append(nodes[i]); i = parents[i]
                return path[::-1]
    return None
```

### Sample Code: Direct Collocation Sketch in CasADi

```python
import casadi as ca

N, T = 50, 2.0
dt = T / N

opti = ca.Opti()
X = opti.variable(4, N+1)   # state trajectory
U = opti.variable(1, N)     # control trajectory

def f(x, u):                # cart-pole dynamics
    # (return ẋ as a CasADi expression)
    ...

# Trapezoidal collocation
for k in range(N):
    f_k  = f(X[:, k],   U[:, k])
    f_k1 = f(X[:, k+1], U[:, k])  # zero-order hold on u
    opti.subject_to(X[:, k+1] == X[:, k] + 0.5*dt*(f_k + f_k1))

opti.subject_to(X[:, 0]  == [0, 0, np.pi, 0])     # start: pole down
opti.subject_to(X[:, -1] == [0, 0, 0, 0])         # end:   pole up at origin
opti.subject_to(opti.bounded(-20, U, 20))         # torque limit

opti.minimize(ca.sumsqr(U) * dt)                  # minimum-effort
opti.solver("ipopt")
sol = opti.solve()
```

### Key Insight

There is no single best planner. The right choice depends on dimensionality, replanning frequency, dynamic feasibility, presence of contact, and whether you have an SDF. The right *mental model* is that planning is search-plus-optimization: sampling-based methods explore globally and find any feasible answer fast; optimization-based methods refine locally and find a *good* answer slowly. Real systems chain them. And the planner you ship is rarely the one in your slides — it's the one that survives 10 000 trials in your evaluation harness without a single collision, and replans in under 100 ms on the day the conveyor belt slides 2 cm.

### Resources

- *Planning Algorithms* (LaValle) — the comprehensive textbook, free online
- *Principles of Robot Motion* (Choset et al.) — the other canonical motion-planning text
- OMPL documentation (and reading the code)
- Drake's trajectory optimization tutorials
- "Motion planning around obstacles with convex optimization" (Marcucci et al., 2023) — GCS, the modern frontier

---

## Phase 6: Manipulation and Grasping

Manipulation is the part of robotics where everything you have learned has to work simultaneously, plus contact — the most thermodynamically violent and analytically misbehaved interaction in the field. Picking up a paper clip is harder than walking on Mars.

### Concepts to Learn

- **The manipulation taxonomy** — different problems use different toolkits:
  - **[Pick and place](/shared/glossary/#pick-and-place)** — grasp, move, release. The bread-and-butter; 90% of industrial robots do only this.
  - **Insertion / assembly** — peg-in-hole, connector mating; needs compliance and search.
  - **In-hand manipulation** — re-orient an object within the hand using fingers; the unsolved frontier (much improved by RL recently).
  - **Tool use** — using the gripper as a wedge, hook, scraper.
  - **Deformable manipulation** — cloth, wires, food; mostly unsolved, mostly learned.
- **Grasp synthesis**:
  - **[Analytic grasping](/shared/glossary/#analytic-grasping)** — given object geometry, find contact points that achieve **[force closure](/shared/glossary/#force-closure)** (the grasp resists any external wrench). Classical, geometric, requires accurate model.
  - **[Data-driven grasping](/shared/glossary/#data-driven-grasping)** — learn a grasp-quality function from images / point clouds. [Dex-Net](/shared/glossary/#dex-net), GraspNet, [AnyGrasp](/shared/glossary/#anygrasp) lineage.
  - **Top-down parallel-jaw on tabletop** — the well-solved special case; reduce to a planar problem and pick the best 4-DoF grasp.
  - **[6-DoF grasping](/shared/glossary/#6-dof-grasping)** — full pose; needed for cluttered bins and arbitrary objects.
- **Contact mechanics**:
  - **[Coulomb friction](/shared/glossary/#coulomb-friction)** — `‖f_t‖ ≤ μ · f_n`. The [friction cone](/shared/glossary/#friction-cone) is the model that makes contact-rich planning possible.
  - **Hard vs. soft contact models** — rigid-body assumes infinite stiffness; soft contact (Hunt-Crossley, time-stepping with regularization) is what simulators actually use.
  - **Contact-implicit trajectory optimization** — let the optimizer decide when and where contact happens (CITO, LCS); avoids combinatorial mode enumeration.
- **Force / [impedance control](/shared/glossary/#impedance-control) for contact** — discussed in Phase 2; here is where it earns its keep. Stiffness in unconstrained directions, low stiffness in constrained directions ([hybrid force-position control](/shared/glossary/#hybrid-force-position-control)).
- **[Compliance](/shared/glossary/#compliance)** — mechanical (passive springs, soft fingertips, the [remote centre of compliance](/shared/glossary/#remote-center-of-compliance-rcc)) and software (impedance). Real-world insertion almost always uses both.
- **Grasping with a gripper that isn't parallel-jaw**:
  - **[Suction](/shared/glossary/#suction-gripper)** — fast, robust on flat smooth surfaces (warehouses); fails on porous or curved objects.
  - **Multi-finger hands** — Allegro, Shadow, custom; needed for in-hand manipulation; control complexity explodes.
  - **[Soft grippers](/shared/glossary/#soft-gripper)** — silicone fingers, jamming grippers; passive compliance, hard to model.
- **Vision-based [tactile sensing](/shared/glossary/#tactile-sensing)** — high-resolution local contact images; the new ingredient making contact-rich manipulation tractable.
- **[Bin picking](/shared/glossary/#bin-picking)** — clutter, partial observability, occlusion, the trifecta. Has its own benchmarks (Bin Picking, YCB, GraspNet-1B).
- **Mobile manipulation** — adding the base's DoF to the arm's; collision avoidance across the whole envelope; "where to drive to before reaching".
- **Bimanual manipulation** — two arms, often holding the same object; dual-arm constraint planning, asymmetric and symmetric coordination.
- **Learning vs. classical manipulation**:
  - Classical wins on structured, repeatable industrial tasks
  - Learning wins on unstructured tasks with messy objects and ill-defined geometry
  - The frontier is the *combination*: classical control under learned policies, with priors from foundation models

### Force Closure In One Picture

```
         ┌──────────────────────┐
         │   object             │
         │                      │
         │  contact 1           │
         │  ↓ inside its        │   A grasp is force-closed iff the set
         │  friction cone       │   of wrenches you can apply through
         │                      │   the contacts SPANS R^6 (or R^3 in 2D).
         │       contact 2  ↑   │
         │       inside its     │   Equivalent: any external disturbance
         │       friction cone  │   can be resisted by feasible contact
         │                      │   forces.
         └──────────────────────┘

For two parallel-jaw contacts with friction, that requires the line between
contact points to pass through both friction cones. That's it — that's the
test for whether your gripper holds the object against gravity.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Analytic 2D grasp](projects/39-analytic-2d-grasp/README.md) | For polygonal objects, compute force-closure grasps; visualize friction cones | ⭐⭐⭐ |
| [Top-down learned grasp](projects/40-top-down-learned-grasp/README.md) | Train or fine-tune a grasp-quality net on a small synthetic set; evaluate on real objects | ⭐⭐⭐⭐ |
| [Peg-in-hole with impedance](projects/41-peg-in-hole-with-impedance/README.md) | Implement spiral-search + impedance control to insert a peg; measure success rate vs. tolerance | ⭐⭐⭐⭐ |
| [AnyGrasp pipeline](projects/42-anygrasp-pipeline/README.md) | Run a pretrained 6-DoF grasp net on point clouds from your depth camera; execute on a real arm | ⭐⭐⭐⭐⭐ |
| [Visuomotor pick](projects/43-visuomotor-pick/README.md) | End-to-end policy from camera to grasp on a small object set in MuJoCo | ⭐⭐⭐⭐ |
| [In-hand cube reorientation](projects/44-in-hand-cube-reorientation/README.md) | Train a multi-finger policy in sim to reorient a cube; quantify success/dropout | ⭐⭐⭐⭐⭐ |
| [Cloth folding](projects/45-cloth-folding/README.md) | Pick-place primitive policy for folding a square cloth; measure final-state IoU | ⭐⭐⭐⭐⭐ |

### Sample Code: A Top-Down Grasp Score from a Depth Image

```python
def best_top_down_grasp(depth, mask, gripper_width_px, n_orientations=16):
    """Returns (u, v, theta) maximizing a simple antipodal score."""
    H, W = depth.shape
    ys, xs = np.where(mask)
    best, best_score = None, -np.inf
    for u, v in zip(xs, ys):
        z0 = depth[v, u]
        for theta in np.linspace(0, np.pi, n_orientations, endpoint=False):
            dx, dy = np.cos(theta), np.sin(theta)
            left  = (int(u - 0.5*gripper_width_px*dx), int(v - 0.5*gripper_width_px*dy))
            right = (int(u + 0.5*gripper_width_px*dx), int(v + 0.5*gripper_width_px*dy))
            if not (0 <= left[0] < W and 0 <= right[0] < W
                    and 0 <= left[1] < H and 0 <= right[1] < H):
                continue
            zl, zr = depth[left[1], left[0]], depth[right[1], right[0]]
            # Reward: both finger sides above the object surface, center on object
            score = -(abs(zl - zr)) - 0.1*abs(z0 - 0.5*(zl + zr))
            if score > best_score:
                best, best_score = (u, v, theta), score
    return best
```

### Key Insight

Manipulation breaks the clean "perceive, plan, act" pipeline. The moment you touch an object, your model becomes wrong, your sensors become partly occluded, and your geometry becomes a function of friction and stiffness you didn't measure. The historically successful approach has been to *embrace the imprecision*: control compliantly, plan with margin, design end-effectors that fail gracefully. The recent shift is to embrace the imprecision *with policies that learned to handle it*, from demonstrations or sim-to-real RL. Both work; both fail in different ways; the practitioner who ships chooses ruthlessly between them per-task.

### Resources

- *Modern Robotics* (Lynch & Park) — Chapter 12 (grasping and manipulation)
- *Mechanics of Robotic Manipulation* (Mason) — the geometric/contact foundations
- The Dex-Net papers and dataset (analytic + learned grasping)
- Drake's manipulation course materials (Tedrake), with notebooks
- GraspNet-1Billion, ACRONYM, and related grasp datasets

---

## Phase 7: Mobile Robots, Legged Locomotion, and Navigation

If a manipulator's challenge is the object, a mobile robot's challenge is the world: vast, partially observed, dynamic, occasionally a parking garage with no GPS. Navigation is the integration of mapping, localization, planning, and control, run forever.

### Concepts to Learn

- **Mobile-robot [kinematics](/shared/glossary/#kinematics)** — the under-celebrated chassis math:
  - **Differential drive** — two wheels, fast turning, holonomic-in-place. The most common indoor robot.
  - **Ackermann / car-like** — minimum turning radius, non-holonomic; the cars and many delivery robots.
  - **Omnidirectional** — mecanum / omni wheels; lateral motion at the cost of dirt and friction.
  - **Tracked** — outdoor, rough terrain; slip is severe and must be modeled.
- **Path tracking** — Pure Pursuit, Stanley, MPC-based tracking. Tune for the right speed regime.
- **The navigation stack** — the canonical layered design:
  - **Global planner** — [A*](/shared/glossary/#a-star-search) / [Dijkstra](/shared/glossary/#dijkstras-algorithm) on a static map
  - **Local planner** — DWA, TEB, or MPC over a short horizon, considering dynamic obstacles
  - **Costmap layers** — static (the map), inflation (around obstacles), dynamic (sensed obstacles), social (around humans)
  - **Recovery behaviors** — backup, rotate, clear costmap when stuck
- **Legged locomotion**:
  - **The state of a legged robot** — base pose (6 DoF) + joint angles + contact mode. The contact mode is discrete; the rest is continuous. A hybrid system.
  - **The Zero-Moment Point (ZMP)** — classical biped balance criterion: keep the ground reaction's center of pressure inside the support polygon.
  - **Centroidal dynamics** — track the linear and angular momentum of the body's center of mass; far more tractable than the full multi-body dynamics for online planning.
  - **MPC for locomotion** — every modern quadruped uses some flavor of MPC over a simplified centroidal model.
  - **[Whole-Body Control (WBC)](/shared/glossary/#wbc)** — a fast inner loop solving a [QP](/shared/glossary/#quadratic-program) for joint torques that achieves task-space goals (CoM, foot placement) while respecting friction cones and joint limits.
  - **Gaits** — trot, pace, gallop, walk; emerging from policy or planned explicitly via foot-step sequences.
  - **Learning for locomotion** — RL in massively-parallel simulation (Isaac Gym scale: thousands of robots simultaneously) → policy → real robot. The recipe that made quadruped locomotion go from research demos to commercial reliability over 2018–2024.
- **Aerial robots / drones**:
  - **[Quadrotor](/shared/glossary/#quadrotor) dynamics** — under-actuated (4 inputs, 6 DoF); [differentially flat](/shared/glossary/#differential-flatness) in position and [yaw](/shared/glossary/#yaw), which makes trajectory generation magical.
  - **Polynomial / minimum-snap trajectories** — closed-form smooth trajectories through waypoints.
  - **VIO + control** — the standard stack: VIO localizes, MPC or geometric controller tracks.
- **Self-driving stacks** — the special case at the scale limit:
  - HD maps vs. online perception
  - Behavior / prediction / planning split
  - Safety envelopes, RSS, formal methods, ODD scoping
  - The painful realities of long-tail distribution shift
- **Dynamic obstacles and humans** — predict their trajectories (constant velocity, social-LSTM, learned models), plan around the predictions. Underconfident predictions cause freezing; overconfident ones cause crashes.
- **Multi-robot systems** — decentralized planning, conflict-based search, ORCA / velocity obstacles for reactive collision avoidance.

### The Mobile Robot Stack, Annotated

```
        goal pose
            │
            ▼
   ┌────────────────────────┐
   │ Global planner (A*)    │  static map; runs once per goal or on map change
   └────────────────────────┘
            │
            ▼
   ┌────────────────────────┐
   │ Local planner (DWA/MPC)│  10 Hz; considers dynamic obstacles
   └────────────────────────┘
            │
            ▼
   ┌────────────────────────┐
   │ Path tracker           │  20–100 Hz; computes (v, ω) or (v, δ)
   └────────────────────────┘
            │
            ▼
   ┌────────────────────────┐
   │ Wheel / motor control  │  1 kHz; commands torques or currents
   └────────────────────────┘
            │
            ▼
         Physics
            │
            ▼
   ┌────────────────────────┐
   │ Localization (AMCL/    │  always running; consumes lidar/odom/IMU/cam
   │ VIO / SLAM)            │
   └────────────────────────┘
```

### A Quadruped Control Stack

```
                  Behavior planner (10 Hz): goal vel, gait, posture
                            │
                            ▼
              ┌────────────────────────────┐
              │   Footstep & MPC (50 Hz)   │  centroidal model, friction
              │   horizon ~1 s             │  cones, gait schedule
              └────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │   Whole-body QP (500 Hz)   │  CoM accel, foot Cartesian
              │                            │  goals, contact forces → τ
              └────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │   Joint torque (1–4 kHz)   │  per-joint current loop
              └────────────────────────────┘

OR a learned policy collapses many of these layers into a single
neural network conditioned on body state and a velocity command,
trained with RL in massive parallel simulation. Both flavors win
on different axes today.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Pure pursuit](projects/46-pure-pursuit/README.md) | Follow a smooth path with a [differential-drive](/shared/glossary/#differential-drive) robot; tune look-ahead | ⭐⭐ |
| [DWA local planner](projects/47-dwa-local-planner/README.md) | Implement [Dynamic Window Approach](/shared/glossary/#dynamic-window-approach-dwa); integrate with [A*](/shared/glossary/#a-star-search) global plan | ⭐⭐⭐⭐ |
| [MPC for an Ackermann car](projects/48-mpc-for-an-ackermann-car/README.md) | Tracking a racetrack centerline with [kinematic-bicycle](/shared/glossary/#kinematic-bicycle-model) [MPC](/shared/glossary/#mpc) | ⭐⭐⭐⭐ |
| [AMCL on a known map](projects/49-amcl-on-a-known-map/README.md) | Localize a sim [differential-drive](/shared/glossary/#differential-drive) robot in a known occupancy map | ⭐⭐⭐ |
| [Quadrotor min-snap](projects/50-quadrotor-min-snap/README.md) | Generate and track a [minimum-snap trajectory](/shared/glossary/#minimum-snap-trajectory) through 8 waypoints | ⭐⭐⭐⭐⭐ |
| [Quadruped trotting MPC](projects/51-quadruped-trotting-mpc/README.md) | Reproduce a basic convex [MPC](/shared/glossary/#mpc) trotting controller on a sim quadruped | ⭐⭐⭐⭐⭐ |
| [Learned locomotion](projects/52-learned-locomotion/README.md) | Train an [RL](/shared/glossary/#reinforcement-learning) [policy](/shared/glossary/#policy) in [Isaac Lab](/shared/glossary/#isaac-lab) to walk on flat terrain; deploy to a sim transfer | ⭐⭐⭐⭐⭐ |
| [Social navigation](projects/53-social-navigation/README.md) | Plan around moving "pedestrian" agents with predictive collision avoidance | ⭐⭐⭐⭐ |

### Sample Code: Pure Pursuit for a Differential-Drive Robot

```python
def pure_pursuit(state, path, lookahead, v_des, wheelbase=None):
    """state: (x, y, theta); path: list of (x, y); returns (v, omega)."""
    # Find the look-ahead point on the path
    target = None
    for p in path:
        if np.linalg.norm(p - state[:2]) >= lookahead:
            target = p; break
    if target is None: target = path[-1]

    # Convert to robot frame
    dx = target[0] - state[0]
    dy = target[1] - state[1]
    local_x =  np.cos(state[2]) * dx + np.sin(state[2]) * dy
    local_y = -np.sin(state[2]) * dx + np.cos(state[2]) * dy

    # Curvature γ = 2 y / L²
    curvature = 2 * local_y / (lookahead ** 2)
    omega = v_des * curvature
    return v_des, omega
```

### Key Insight

Mobile robots reveal the most under-appreciated robotics truth: **the world doesn't sit still while you compute.** A perception pipeline running at 5 Hz with 200 ms of latency means by the time the planner sees the dog, the dog is 30 cm closer. The discipline of building reactive stacks — layering a slow-but-smart planner over a fast-and-stupid tracker — is the practical answer. The same principle scales: legged locomotion has a 1 kHz inner loop because the body falls in 100 ms; self-driving has a 30 ms behavior loop because a car at 30 m/s moves 1 m. Always design for the worst-case decision latency, not the average.

### Resources

- *Probabilistic Robotics* (Thrun et al.) — Chapters on mobile-robot localization and mapping
- *Springer Handbook of Robotics* — chapters on legged and wheeled vehicles
- ROS 2 navigation stack (Nav2) documentation
- "MIT Cheetah" papers (Kim, Kim, Wensing) — the modern legged control stack
- "Learning to walk in minutes" (Rudin et al., 2022) — the massively-parallel RL recipe
- Kumar Lab / Sangbae Kim quadrotor and quadruped papers

---

## Phase 8: Learning for Robotics

Through 2020, classical methods dominated practical robotics. Since then, learning has eaten progressively more of the stack — perception first, then policies, then dynamics models, now (sometimes) the whole controller. This phase is the bridge from "I have a robot and a controller" to "I have a robot and a policy" and the engineering it takes for the second to actually work outside a paper.

### Concepts to Learn

- **The taxonomy of robot learning** — where the learning lives in the stack:
  - **Learned perception** — what we did in Phase 3. The least controversial, most universally adopted.
  - **Learned dynamics / world models** — fit `x_{t+1} = f̂(x_t, u_t)` from data; use it for MPC ("model-based RL") or planning. Sample-efficient, model-error-dominated.
  - **Imitation learning (IL)** — clone a demonstrator's behavior. Cheap data, supervised loss, distribution-shift pain.
  - **Reinforcement learning (RL)** — optimize a policy by trial and error against a reward. Expensive data, no demonstrator needed, reward-shaping pain.
  - **Hybrid IL + RL** — pretrain with IL, fine-tune with RL. The frontier recipe for manipulation.
- **Behavior cloning (BC)** — supervised: minimize `‖a − π(s)‖²` (or NLL) over demonstration data. Simple, fast, the right baseline.
- **The [covariate-shift](/shared/glossary/#covariate-shift) problem of BC** — at test time the policy visits states the human never demonstrated, errors compound. The whole field is variations on coping with this.
- **[DAgger](/shared/glossary/#dagger) and online IL** — query the expert at states the policy actually visits; close the distribution gap.
- **[Diffusion policies](/shared/glossary/#diffusion-policy)** — represent the action distribution as a denoising diffusion model conditioned on the current observation. State-of-the-art behavior cloning since 2023; handles multimodal action distributions naturally.
- **Action chunking and ACT-style policies** — predict a chunk of future actions rather than one; smoother behavior, more robust to perception jitter.
- **Reinforcement learning for robotics**:
  - **Policy gradient family** — REINFORCE, A2C, PPO. PPO is the dominant on-policy choice.
  - **Off-policy / actor-critic** — SAC, TD3, DDPG. Better sample efficiency, finickier to tune.
  - **Model-based RL** — Dreamer, TD-MPC, PILCO; learn a world model, plan in it.
  - **Offline RL** — train from a fixed dataset (no rollouts); CQL, IQL, decision transformers. Important when collection is expensive.
  - **Massively-parallel RL** — Isaac Gym / Isaac Lab style: thousands of robots in a single GPU simulation, train a policy in minutes-to-hours. The recipe behind most modern legged-locomotion releases.
- **Reward design** — the central pain of RL for robotics. Dense vs. sparse, hand-engineered vs. learned, terminal vs. shaping. The wrong reward bricks the policy in a way that *looks* like correct learning.
- **Curriculum learning** — start easy, scale difficulty. Often the difference between "policy doesn't learn" and "policy works".
- **Domain randomization** — vary physics parameters (mass, friction, latency), visuals, sensor noise during sim training; the policy learns robustness for free. The single most important sim-to-real technique.
- **Domain adaptation** — pair sim data with a small amount of real data; learn to bridge the gap. Image translation, system identification, online adaptation modules (RMA).
- **Foundation models for robotics** — the 2023+ thread:
  - **VLAs (Vision-Language-Action models)** — large transformers trained on robot trajectories + internet vision-language data; condition on language goals and predict actions. RT-2, OpenVLA, π0 lineage.
  - **Pretrained vision encoders** — DINO, CLIP, MVP, R3M as frozen feature extractors for policies; small downstream policies suffice.
  - **[Cross-embodiment](/shared/glossary/#cross-embodiment)** — training on data from multiple robots so the policy generalizes across grippers, arms, and platforms. [Open X-Embodiment](/shared/glossary/#open-x-embodiment) is the standard cross-robot dataset.
- **[Teleoperation](/shared/glossary/#teleoperation) as a data source** — VR controllers, ALOHA-style leader-follower rigs, smart-glove and exoskeleton interfaces. The dominant way to collect manipulation data at scale.
- **Sim-to-real**:
  - **What transfers easily**: visual policies trained with heavy randomization, locomotion in unstructured terrain
  - **What transfers poorly**: anything stiffness-, friction-, or contact-mode-sensitive; tight tolerances; deformables
  - **Common bridges**: domain randomization, online system ID, residual policy on a classical baseline

### Where Each Learning Recipe Wins

```
                                       What you have:           Best recipe:
                                       ──────────────           ──────────────
  Lots of expert teleop data + sim     +real demonstrations     Diffusion BC / ACT
  Verifiable goal in simulation        +sim only                Massively-parallel RL
  Sim policy that mostly transfers     +small real dataset      Residual / online adapt
  Heterogeneous data across robots     +cross-embodiment        VLA fine-tune
  No demonstrator, fast simulator      +good reward             PPO with curriculum
  Slow simulator or real-only          +offline log             Offline RL (IQL/CQL)
  Need precise dynamics-aware action   +good model              MPC + learned model
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Behavior cloning on a sim arm](projects/54-behavior-cloning-on-a-sim-arm/README.md) | Collect 50 sim teleop demos; train an MLP policy; evaluate success rate | ⭐⭐⭐ |
| [DAgger](projects/55-dagger/README.md) | Add expert queries on rollouts to BC; quantify the improvement | ⭐⭐⭐⭐ |
| [Diffusion policy](projects/56-diffusion-policy/README.md) | Reproduce a small diffusion-policy on a manipulation task; compare to MLP BC | ⭐⭐⭐⭐⭐ |
| [PPO for cart-pole](projects/57-ppo-for-cart-pole/README.md) | The "hello world" of RL; verify your understanding of the loss | ⭐⭐⭐ |
| [SAC for a sim arm reach](projects/58-sac-for-a-sim-arm-reach/README.md) | Train SAC to reach a target; tune `α` (entropy) carefully | ⭐⭐⭐⭐ |
| [Massively-parallel walking](projects/52-learned-locomotion/README.md) | Train a quadruped policy in Isaac Lab with 4 096 envs; deploy in MuJoCo | ⭐⭐⭐⭐⭐ |
| [Domain randomization study](projects/59-domain-randomization-study/README.md) | Same task, with and without randomization; measure transfer gap to a held-out sim env | ⭐⭐⭐⭐ |
| [World-model planning](projects/60-world-model-planning/README.md) | Train a small world model on play data; plan with CEM in latent space | ⭐⭐⭐⭐⭐ |
| [Cross-embodiment fine-tune](projects/61-cross-embodiment-fine-tune/README.md) | Take an Open-X policy; fine-tune on your own robot; report sample efficiency | ⭐⭐⭐⭐⭐ |

### Sample Code: Behavior Cloning, the Minimum Viable Version

```python
import torch, torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, obs): return self.net(obs)

def train_bc(policy, demos, epochs=50, lr=3e-4):
    optim = torch.optim.AdamW(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for obs, act in demos:
            pred = policy(obs)
            loss = loss_fn(pred, act)
            optim.zero_grad(); loss.backward(); optim.step()
```

### Sample Code: PPO Update (Sketch)

```python
def ppo_update(policy, value, batch, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    obs, act, old_logp, adv, ret = batch
    new_logp, entropy = policy.log_prob_and_entropy(obs, act)
    ratio = (new_logp - old_logp).exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)
    pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
    v_loss  = (value(obs) - ret).pow(2).mean()
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy.mean()
    loss.backward()
```

### Key Insight

Robot learning works when (a) data is sufficient and on-distribution, (b) the reward or imitation signal is well-posed, and (c) the sim-to-real gap is small or bridgeable. Each of those three is a research program in itself. The dominant practical pattern in 2026 is *hybrid*: classical perception or estimator for the parts where geometry is clear, a learned policy for the parts where contact and clutter are not, and a hand-engineered safety envelope on top. Anyone telling you "everything will be one big neural network" is right on a long-enough timeline and wrong about every shipping product this year.

### Resources

- *Reinforcement Learning: An Introduction* (Sutton & Barto) — foundations
- The [Reinforcement Learning Guide](../reinforcement-learning/) — algorithmic depth
- *Algorithms for Decision Making* (Kochenderfer et al.) — decision-making spanning POMDPs to RL
- Sergey Levine's deep-RL course (UC Berkeley CS285)
- Karol Hausman et al. RT-2 / RT-X papers
- [Diffusion Policy](/shared/glossary/#diffusion-policy) (Chi et al., 2023)
- Isaac Lab and Isaac Gym documentation
- [Open X-Embodiment](/shared/glossary/#open-x-embodiment) dataset and tutorials

---

## Phase 9: Simulation, Sim-to-Real, and Robot Systems Engineering

A robot is not a single algorithm; it is a system of dozens of interacting processes, sensors, actuators, and people. This phase is the glue: how to simulate believably, how to deploy reliably, and how to run a team or a company shipping robot software without losing the will to live.

### Concepts to Learn

- **Simulator choice and trade-offs** — already enumerated in Phase 0; the deeper considerations:
  - **Physics fidelity** — what contact model, what integrator, what time step
  - **Rendering** — photorealism vs. throughput; raytraced vs. rasterized
  - **Throughput** — how many steps/sec, how parallelizable across GPUs
  - **Determinism** — for reproducibility and debugging
  - **Asset pipeline** — URDF / MJCF / USD; how easily you import meshes
- **Modeling for sim** — collision meshes vs. visual meshes, mass and inertia identification, joint friction, motor models, sensor noise. A simulator's defaults rarely match reality.
- **System identification** — fit dynamics-parameter values from real-robot trajectories. The unglamorous prerequisite of any serious sim-to-real.
- **Domain randomization at scale** — over masses, frictions, latencies, communication delays, control gains, sensor noise, visuals. The list of randomized variables is itself an engineering artifact.
- **Real-time systems** — control loops are hard-real-time-adjacent:
  - **Linux + [PREEMPT_RT](/shared/glossary/#preempt-rt)** — soft to hard real time; sufficient for kHz loops on most hardware
  - **Dedicated MCUs / FPGAs** — for the innermost current loops
  - **Memory allocation** — never `malloc` in the control thread; preallocate everything
  - **Logging discipline** — bounded queues, lock-free where possible, never log from the hot path synchronously
- **Middleware** — the messaging fabric:
  - **ROS 2** — the de facto standard; DDS-based; good for research and many production systems
  - **LCM, ZeroMQ, gRPC** — when ROS 2 is overkill or too slow
  - **DDS QoS** — reliability, durability, history; the place ROS 2 bugs live
- **Drivers and HALs** — every motor, sensor, and bus has its driver, its firmware, its quirks. A well-abstracted hardware-abstraction-layer is a major asset; a leaky one is a major liability.
- **Logging, replay, observability** — `rosbag` / MCAP / custom; the same discipline as production software systems, but with images, point clouds, and 1 kHz time series. Replayability is the single highest-ROI feature you build.
- **Continuous evaluation** — automated test suites for behaviors, integration tests in simulation, hardware-in-the-loop tests. A robotics CI/CD that runs nightly catches [regressions](/shared/glossary/#regression-testing) you would otherwise discover by smashing into a wall.
- **Safety**:
  - **Physical safety** — emergency stops, force/torque limits, watchdogs, light curtains
  - **Software safety** — input validation at every layer, bounded state, fail-safe defaults, redundant estimators
  - **Process safety** — operator training, lockout-tagout, risk assessment, ISO 10218 / ISO/TS 15066 for collaborative robots
- **Calibration as a recurring practice** — sensor extrinsics drift, joint zeros drift, friction parameters change with temperature. Schedule recalibration; instrument for drift detection.
- **Reliability engineering** — MTBF, failure-mode analysis, sentinel tests, canary deployments. Robots run for months; bug latencies measured in operating hours.
- **The cost of going to hardware** — orders of magnitude slower iteration than simulation. A discipline of "simulate to 90% confidence, then validate on hardware" pays for itself many times over.

### A Production-Robotics Diagram

```
                    ┌──────────────────────────────────────────┐
                    │            Cloud / Fleet ops             │
                    │   monitoring, OTA, log warehouse, eval   │
                    └──────────────────────────────────────────┘
                              │                       ▲
                              ▼                       │
                  ┌───────────────────┐     ┌─────────────────────┐
                  │   Robot (one of N)│     │  Eval cluster        │
                  ├───────────────────┤     │  sim regressions /   │
                  │  Behavior         │     │  HW-in-loop nightly  │
                  │  Planner          │     └─────────────────────┘
                  │  Tracker          │
                  │  Estimator        │
                  │  Perception       │
                  │  Drivers          │
                  ├───────────────────┤
                  │  Hardware: arms,  │
                  │  cameras, IMU,    │
                  │  motors, E-stop   │
                  └───────────────────┘

Every production robot company eventually rebuilds something like this.
The companies that do it earlier ship sooner.
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [URDF → MJCF migration](projects/62-urdf-mjcf-migration/README.md) | Take a hobbyist URDF; clean it up; add collision meshes, inertias; load in MuJoCo | ⭐⭐⭐ |
| [System ID on a real arm](projects/63-system-id-on-a-real-arm/README.md) | Excite joints with [chirp signals](/shared/glossary/#chirp); fit a friction + inertia model; compare to URDF defaults | ⭐⭐⭐⭐ |
| [Latency budget instrument](projects/64-latency-budget-instrument/README.md) | Trace photon-to-actuation latency across your stack; report 5/50/95 percentiles | ⭐⭐⭐⭐ |
| [ROS 2 lifecycle node](projects/65-ros-2-lifecycle-node/README.md) | Implement a perception node with managed lifecycle; verify recovery from sensor disconnect | ⭐⭐⭐ |
| [MCAP logging + Foxglove](projects/66-mcap-logging-foxglove/README.md) | Log a complete robot run; replay; visualize trajectories, camera, transforms | ⭐⭐⭐ |
| [Real-time loop drill](projects/67-real-time-loop-drill/README.md) | Run a 1 kHz control loop on [PREEMPT_RT](/shared/glossary/#preempt-rt); measure jitter under load; compare to vanilla Linux | ⭐⭐⭐⭐⭐ |
| [Sim-to-real gap forensics](projects/68-sim-to-real-gap-forensics/README.md) | Pick one transfer failure; isolate whether perception, dynamics, or actuation caused it | ⭐⭐⭐⭐⭐ |
| [Eval harness](projects/69-eval-harness/README.md) | Build a 50-task sim eval that runs nightly with seeded variations; produce a pass-rate dashboard | ⭐⭐⭐⭐ |

### Sample Code: A Hard-Boundary Watchdog

```python
import time, threading

class Watchdog:
    """If kick() isn't called within `timeout` seconds, fire `on_timeout`."""
    def __init__(self, timeout, on_timeout):
        self.timeout = timeout
        self.on_timeout = on_timeout
        self._last = time.monotonic()
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def kick(self): self._last = time.monotonic()
    def stop(self): self._stop = True

    def _run(self):
        while not self._stop:
            if time.monotonic() - self._last > self.timeout:
                self.on_timeout()
                return
            time.sleep(self.timeout / 4)

# Use:
wd = Watchdog(0.1, on_timeout=robot.emergency_stop)
while running:
    cmd = controller.step(state)
    robot.send(cmd)
    wd.kick()
```

### Key Insight

Robotics-as-a-system is the part textbooks never teach and every successful team learns the hard way. The single highest-leverage investment a team can make is **infrastructure for fast, reliable iteration**: deterministic simulators, scriptable hardware, comprehensive logging, replayable bugs, evaluation harnesses that run unattended. The algorithm wins or loses in the lab; the system wins or loses in production. If you find yourself debugging the same class of issue twice, build a tool that prevents the third.

### Resources

- ROS 2 documentation and design articles
- MuJoCo, Isaac Lab, Drake documentation
- "Real-Time Linux" (Linux Foundation OSPM) materials
- Foxglove and Lichtblick (MCAP visualization)
- The Society of Robotics / IEEE-RAS "Best Practices" workshop materials
- ISO 10218, ISO/TS 15066 standards for industrial and collaborative robots

---

## Phase 10: Frontier Topics

Once you have a robot that can pick a block and walk across a room, you discover the problems that no textbook fully covers: general-purpose embodiment, language interfaces, long-horizon autonomy, multi-agent settings, and the safety and societal questions that come with all of it. This phase is the open frontier — what people are actively researching now.

### Concepts to Learn

- **[Vision-Language-Action (VLA) models](/shared/glossary/#vla)** — [transformers](/shared/glossary/#transformer) (often initialized from [VLMs](/shared/glossary/#vlm)) trained on `(image, instruction, action)` triples drawn from large robot datasets. Receive a language command and an image; emit a tokenized or continuous action. The 2023-onward thread that finally lets robots be talked to in natural language.
- **Generalist robot policies and [cross-embodiment](/shared/glossary/#cross-embodiment)** — [Open X-Embodiment](/shared/glossary/#open-x-embodiment)-scale efforts to train one [policy](/shared/glossary/#policy) across many robots, manipulator types, and tasks. Generalization is real, modest, and improving every six months.
- **[Humanoids](/shared/glossary/#humanoid)** — the form factor bet of the late 2020s: anthropomorphic so the world's tools and spaces don't need to change. The combined challenge of bipedal [legged locomotion](/shared/glossary/#legged-locomotion), bimanual [manipulation](/shared/glossary/#manipulation), and whole-body control all at once. Standing-up policies, fall recovery, hand dexterity are all open.
- **[Dexterous manipulation](/shared/glossary/#dexterous-manipulation)** — multi-fingered, in-hand, contact-rich. [RL](/shared/glossary/#reinforcement-learning) in simulation made this tractable for cubes and small objects; transfer to truly varied objects is still hard.
- **[World models](/shared/glossary/#world-model) for embodied AI** — generative video/physics models conditioned on actions ([Genie](/shared/glossary/#genie), [DreamerV3](/shared/glossary/#dreamerv3), NVIDIA Cosmos lineage); the closest thing to "imagined [rollouts](/shared/glossary/#rollout)" for [planning](/shared/glossary/#planning).
- **Foundation-model-driven task and motion planning** — use an [LLM](/shared/glossary/#llm)/[VLM](/shared/glossary/#vlm) to propose subgoals, [primitives](/shared/glossary/#primitives), or code, with classical planners executing each step. [SayCan](/shared/glossary/#saycan), ProgPrompt, Code-as-Policies lineage.
- **[Long-horizon autonomy](/shared/glossary/#long-horizon-autonomy)** — keeping success rates high over thousands of steps. Error compounds; recovery behaviors matter more than headline success rates.
- **Whole-body manipulation** — using the entire robot body, not just the end-effector: leaning on a counter, kicking a box aside, bracing against a wall.
- **Soft robotics** — actuators that deform rather than rotate; pneumatic muscles, dielectric elastomers, hydrogels. New design space, new control challenges, often inherently safe for human interaction.
- **Swarm and multi-robot** — decentralized coordination, communication-aware planning, multi-agent RL. Mostly under-deployed; lots of headroom.
- **Surgical, microscale, and space robotics** — extreme reliability, novel actuation (concentric tubes, magnetic catheters), strict regulatory environments. Domain expertise is decisive.
- **Brain-machine interfaces and assistive robotics** — robot arms controlled from neural decoders, exoskeletons, prosthetics. The intersection where every gram of latency and every Newton of force matters humanely.
- **The safety and societal stack**:
  - **Functional safety** — IEC 61508, ISO 26262 (auto), ISO 13482 (personal-care robots). The vocabulary by which regulators evaluate "safe enough."
  - **Verification and runtime monitoring** — STL / signal temporal logic, [reachability analysis](/shared/glossary/#topp), [control barrier functions (CBFs)](/shared/glossary/#cbf). The formal-methods toolkit.
  - **Labor and economic effects** — automation's distributional impact; deployment ethics; the choices designers and companies make matter.
  - **Misuse and weaponization** — drones and quadrupeds are dual-use; policy and norms are still forming.
  - **Privacy** — robots that look at people are different from robots that move things; the legal regimes haven't caught up.
- **The frontier threads to watch in 2026**:
  - **Robot data scaling laws** — is there a "Chinchilla for robots"? How much data, of what kind, per task?
  - **[Sim-to-real](/shared/glossary/#sim-to-real) that's close enough to real that pretraining transfers** — improving physics, contact, and rendering
  - **Continual on-robot learning** — updating [policies](/shared/glossary/#policy) on the fleet without catastrophic regressions
  - **Human-robot collaboration** — fluent shared-task behavior, intent inference, social norms
  - **Energy and embodiment** — battery density, torque density, drivetrains are still the gating constraints

### A Map of the Open Problems

```
       Hardware ─────  battery, drivetrain, sensors getting cheap; still gating
            │
            ▼
       Locomotion ──── flat ground solved; varied terrain robust; falls still hurt
            │
            ▼
       Manipulation ── tabletop pick-place near-solved; clutter, contact, deformables open
            │
            ▼
       Perception ──── geometry good; semantics good; persistent identity / memory open
            │
            ▼
       Generalization  cross-task, [cross-embodiment](/shared/glossary/#cross-embodiment) improving rapidly
            │
            ▼
       Reliability ──  long-horizon success rates compounded errors; the bottleneck
            │
            ▼
       Safety ──────── physical safety mature; emergent-behavior safety unsolved
            │
            ▼
       Economics ─── cost-per-task vs. labor cost determines what ships
```

### Projects

| Project | Description | Difficulty |
|---------|-------------|------------|
| [Talk-to-robot demo](projects/70-talk-to-robot-demo/README.md) | Use an [LLM](/shared/glossary/#llm) to decompose "make me coffee" into [primitives](/shared/glossary/#primitives); execute in a sim kitchen | ⭐⭐⭐⭐ |
| [VLA fine-tune](projects/71-vla-fine-tune/README.md) | Take an open VLA; fine-tune on a small task; evaluate vs. from-scratch [BC](/shared/glossary/#bc) | ⭐⭐⭐⭐⭐ |
| [World-model rollout](projects/72-world-model-rollout/README.md) | Train a tiny action-conditioned video model; planner picks the action chunk whose imagined rollout matches the goal image | ⭐⭐⭐⭐⭐ |
| [CBF safety filter](projects/73-cbf-safety-filter/README.md) | Wrap a learned [policy](/shared/glossary/#policy) with a control-barrier-function safety filter; show it prevents collisions a naive policy causes | ⭐⭐⭐⭐⭐ |
| [Long-horizon eval](projects/74-long-horizon-eval/README.md) | Build a 50-step task; measure how per-step success rate × N compounds vs. observed task success | ⭐⭐⭐⭐ |
| [Cross-embodiment study](projects/75-cross-embodiment-study/README.md) | Train a [policy](/shared/glossary/#policy) on robot A, deploy on robot B; characterize the transfer gap | ⭐⭐⭐⭐⭐ |

### Key Insight

The capabilities frontier of robotics is dominated by *integration*, not by any single algorithm. We have, in some form, every ingredient: perception, planning, control, learning, language conditioning, simulation, hardware. The hard problems are the *non-capabilities* problems: making the integrated system reliable over thousands of trials, safe under adversarial input and component failure, evaluable in a way that predicts deployment, and economically viable per task. None of these are solved. Most of them are not yet rigorously formalized. If you want to do research that matters in the late 2020s, this is where to look — and if you want to build a robotics company, this is what your engineering hours will be spent on.

### Resources

- [Open X-Embodiment](/shared/glossary/#open-x-embodiment) dataset, paper, and tutorials
- Robotics Transformer (RT-1, RT-2) and successor papers
- "Foundation Models for Decision Making" (Stanford, surveys)
- IEEE-RAS Technical Committees as a map of subfield activity
- Major conferences: ICRA, IROS, RSS, CoRL, Humanoids, ISER
- Workshops at NeurIPS / ICML on "Robot Learning" each year

---

## Suggested Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 0. Prerequisites | 0–2 weeks | Math review done; simulator installed; ROS 2 "hello world" running |
| 1. [Kinematics](/shared/glossary/#kinematics) | 1–2 weeks | FK + Jacobian + IK implemented and tested on a 7-DoF arm |
| 2. Dynamics & control | 2 weeks | Cart-pole LQR, computed-torque tracking, impedance demo all working |
| 3. Perception | 2 weeks | Camera calibrated, AprilTags + ICP + a learned detector in your pipeline |
| 4. Estimation & SLAM | 2 weeks | Working EKF + a small SLAM demo on a public dataset |
| 5. Planning | 2 weeks | [RRT-Connect](/shared/glossary/#rrt-connect) + [trajectory optimization](/shared/glossary/#trajectory-optimization) on a real arm in sim |
| 6. Manipulation | 2–3 weeks | Pick-and-place with vision; one contact-rich task (insertion or wiping) |
| 7. Mobile / legged | 2 weeks | Wheeled nav stack + a small legged-locomotion demo |
| 8. Learning | 3 weeks | BC + a policy trained with PPO; one [diffusion policy](/shared/glossary/#diffusion-policy) run; understand sim-to-real |
| 9. Systems engineering | 1–2 weeks | A reproducible eval harness; a real-time loop with logging and a watchdog |
| 10. Frontier | Ongoing | Picked one thread (humanoids, VLAs, safety, etc.) and going deep |

**Total to "comfortable practitioner":** ~4–6 months of focused study, longer if combined with hardware projects (recommended). To "research-comfortable on one frontier thread": 6–12 months beyond that.

---

## Key Advice

1. **Pick a simulator early and live in it.** Switching simulators mid-project is twice the work of any single project. Pick once, commit, port only if you must.
2. **Frame discipline is non-negotiable.** Name every transform by both endpoints; check the chain rule before every product; visualize frames in 3D before debugging numbers.
3. **Always carry uncertainty.** A pose without a covariance is half a pose. A velocity without bounds is a wish. Filters and planners that ignore uncertainty crash robots.
4. **Start with the simplest controller that meets the spec.** PID with feedforward outperforms a hand-built MPC for 80% of tasks. Reach for complexity only when measurement justifies it.
5. **Calibrate, then calibrate again.** Sensor extrinsics drift, joint zeros drift, the world changes. Recalibration is a feature, not a chore.
6. **Profile the latency budget.** Measure photon-to-actuation. Most "the robot is laggy" stories are 200 ms hiding in a USB poll, not the algorithm.
7. **Build replay before you build features.** A bug that doesn't reproduce wastes more hours than it took to record the data.
8. **Eval before you ship.** A 50-task automated harness running nightly is the single most important infrastructure investment a team can make.
9. **Learn from demonstrations before you learn from rewards.** Behavior cloning sets a baseline; RL polishes; reward-engineering from zero is a research project of its own.
10. **Domain randomize aggressively in sim, then validate ruthlessly on hardware.** The transfer gap is almost always something you didn't randomize.
11. **The fancy algorithm is rarely the bottleneck.** Wiring, drivers, time sync, calibration, data quality, and integration are. Plan engineering hours accordingly.
12. **Treat safety as a system property, not a checkbox.** Watchdogs, force limits, fail-safe defaults, redundancy. The cheap safety layer prevents the expensive incident.

---

## Common Pitfalls

- ❌ Confusing `T_world_camera` and `T_camera_world` → robot reaches into the floor
- ❌ Reading Euler angles into storage → gimbal lock and discontinuities at runtime
- ❌ Quaternion sign flip across frames → SLERP takes the long way around
- ❌ Sampling-based plan without smoothing → robot jerks, tracker overshoots, motors complain
- ❌ Ignoring joint limits in IK → solver returns an "elegant" solution your arm cannot achieve
- ❌ Using visual mesh for collision → planner accepts paths that clip through the table
- ❌ Camera intrinsics not refreshed after lens swap → calibration silently wrong; everything is off
- ❌ Software time stamps on USB cameras → 50 ms of jitter you cannot remove downstream
- ❌ Wheel odometry without slip model → estimate drifts a meter per minute on tile
- ❌ IMU integration without bias estimation → orientation drifts; position is fiction
- ❌ Filter that never reports innovations → silently diverges; no one notices for weeks
- ❌ RL reward shaping that rewards a proxy (joint motion) instead of the task → policy wiggles forever
- ❌ Domain randomization too narrow → policy overfits to sim; transfer fails on first real-world variation
- ❌ Sim-to-real demo on the day a paper is due → the wire harness moves 2 cm and the demo dies
- ❌ Logging in the hot path → 1 kHz loop becomes a 200 Hz loop, controller goes unstable
- ❌ Emergency stop wired but never tested → the day you need it is the day you find it broken
- ❌ Single-seed RL result reported as fact → 30% of those results don't replicate
- ❌ No automated regression suite → every PR is a chance to break the demo
- ❌ Treating contact as if it were rigid → simulator and real robot diverge the moment fingers touch
- ❌ MPC tuned for nominal mass → robot wobbles when you bolt the camera on

---

## Additional Resources

### Books
- *Modern Robotics: Mechanics, Planning, and Control* (Lynch & Park) — the modern textbook; free online with companion videos
- *Probabilistic Robotics* (Thrun, Burgard, Fox) — the estimation and SLAM bible
- *Robotics, Vision and Control* (Corke) — practical and code-first
- *Underactuated Robotics* (Tedrake, MIT 6.832 online notes) — best free modern-controls material
- *A Mathematical Introduction to Robotic Manipulation* (Murray, Li, Sastry) — the screw-theory canon
- *Planning Algorithms* (LaValle) — comprehensive, free online
- *Springer Handbook of Robotics* — encyclopedic reference

### Courses
- MIT 6.832 / 6.4210 — *Underactuated Robotics* and *Robotic Manipulation* (Tedrake)
- Stanford CS223A — *Introduction to Robotics* (Khatib)
- UC Berkeley CS287 — *Advanced Robotics* (Abbeel)
- CMU 16-720 / 16-833 — *Computer Vision* and *Robot Localization and Mapping*
- ETH Zurich — *Robot Dynamics*, *State Estimation*, *Perception*
- Cyrill Stachniss YouTube channel — outstanding SLAM and photogrammetry lectures
- Northwestern *Modern Robotics* Coursera specialization

### Code You Should Read
- `pinocchio` — fast, accurate rigid-body dynamics; the reference implementation
- `drake` — trajectory optimization, planning, and simulation; production-grade
- `mujoco` — physics engine and Python bindings
- `isaac-sim` / `isaac-lab` — GPU-parallel simulation
- `OMPL` — motion-planning library
- `GTSAM` — factor-graph SLAM
- `ORB-SLAM3`, `VINS-Fusion` — open-source SLAM baselines
- `Nav2` — the modern ROS 2 navigation stack
- `MoveIt 2` — manipulation planning
- `legged_gym` / `legged_lab` — massively-parallel locomotion RL recipes
- `lerobot` — open ecosystem for learned-policy manipulation

### Papers Everyone Cites
- Featherstone — *Rigid Body Dynamics Algorithms* (book and survey papers)
- Khatib — operational-space formulation
- Smith, Self, Cheeseman — *Estimating Uncertain Spatial Relationships* (EKF-SLAM origins)
- Thrun, Montemerlo — *FastSLAM*
- Hutter et al. — ANYmal / quadruped MPC papers
- Mnih et al. — *Playing Atari with Deep RL* (the modern RL inflection point)
- Schulman et al. — *PPO*
- Haarnoja et al. — *Soft Actor-Critic*
- Chi et al. — *[Diffusion Policy](/shared/glossary/#diffusion-policy)*
- [Open X-Embodiment](/shared/glossary/#open-x-embodiment) collaboration — *RT-X* / [cross-embodiment](/shared/glossary/#cross-embodiment)
- Tedrake — *Underactuated Robotics* (book/notes are the de facto curriculum)

### Communities and Conferences
- ICRA, IROS, RSS, CoRL, Humanoids — the major conferences
- Robotics Stack Exchange — Q&A
- /r/robotics, /r/ROS, /r/reinforcementlearning — practitioner forums
- ROS Discourse — middleware discussions
- Many active research labs publish weekly tech-talks on YouTube

### Talks Worth Watching
- Marc Raibert — early Boston-Dynamics-style legged-locomotion lectures
- Russ Tedrake — *Underactuated Robotics* lecture series
- Sergey Levine — *Deep Reinforcement Learning* (CS285) full course
- Sangbae Kim — cheetah and mini-cheetah dynamics talks
- Chelsea Finn — meta-learning and robot learning seminars

---

## Quick Start Checklist

- [ ] Have read *Modern Robotics* Chapters 2–5 and can derive FK by hand
- [ ] Have implemented FK and a damped-least-squares IK on a 7-DoF arm
- [ ] Can compose and decompose homogeneous transforms in sleep; know what `T_AB · T_BC` means
- [ ] Have tuned a PID on a real or simulated pendulum and felt the gain interactions
- [ ] Have designed and run an LQR on a linearized cart-pole
- [ ] Have implemented a Kalman filter on a 2D tracker and seen its covariance contract
- [ ] Have run an EKF or particle filter on a known map for mobile-robot localization
- [ ] Have calibrated a camera and recovered an AprilTag's 6-DoF pose
- [ ] Have run [RRT](/shared/glossary/#rrt) or [RRT-Connect](/shared/glossary/#rrt-connect) on a 7-[DoF](/shared/glossary/#degrees-of-freedom) arm in simulation
- [ ] Have done a contact-rich manipulation task (insertion / wiping) with impedance control
- [ ] Have trained a behavior-cloning policy on demonstration data and evaluated it
- [ ] Have trained an RL policy (PPO or SAC) on a sim task
- [ ] Have run a domain-randomized sim policy and characterized its transfer
- [ ] Have built or modified a ROS 2 launch graph with at least three nodes
- [ ] Have built a watchdog and an emergency-stop path you have actually tested
- [ ] Have logged a real or simulated robot run end-to-end and replayed it
- [ ] Can read a contemporary robotics paper and explain which design choices solve which problems

---

## License

MIT License. See the [LICENSE](https://github.com/25621/ai-learning-guides/blob/main/LICENSE) file for details.
