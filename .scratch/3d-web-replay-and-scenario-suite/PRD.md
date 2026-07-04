# PRD: 3D Web Replay and Scenario Suite

Status: implemented

## Progress

- 2026-07-03: Issues 01-14 are implemented and human-reviewed. Status records were synchronized from `ready-for-agent` to `implemented`.

## Problem Statement

The current yaw-only 3D MPPI example proves the Core Python controller can plan in `[x, y, z, yaw]`, but it is hard to evaluate or demonstrate. The matplotlib 3D visualization is dated, awkward to interact with, and not suitable for inspecting rollouts, clearance, T-shaped robot volume orientation, or controller behavior over time.

The user is also dissatisfied with the apparent smoothness of the 3D controller relative to the original 2D examples. That smoothness cannot be judged reliably from animation alone: it needs reproducible 3D smoothness telemetry, scenario-suite metrics, and replay data that can be inspected repeatedly.

The existing 3D obstacle scene is too simple to expose meaningful controller behavior. The next 3D work needs a reproducible 3D scenario suite, an authoritative T-shaped 3D robot volume, and a Passive Web 3D viewer that improves inspection without moving MPPI control or simulation ownership into the browser.

## Solution

Build a 2D-compatible 3D workflow around the existing yaw-only 3D MPPI controller. Python remains responsible for scenario progression, local observation generation, local plan selection, MPPI control, state integration, collision checks, smoothness telemetry, and result summaries.

Add a first Offline Web replay mode: Python completes a deterministic simulation run and exports world-frame replay data. A static Three.js viewer loads that replay data and provides modern 3D inspection controls: orbit/follow cameras, timeline scrubbing, playback speed, layer toggles, T-shaped robot rendering, trajectory overlays, and metric panels.

Replace the single simple 3D obstacle demo as the main evaluation surface with a static 3D scenario suite. Each scenario isolates one capability: baseline tracking, vertical gates, narrow gaps for the T-shaped 3D robot volume, T-shaped traps, and cluttered corridors. Dynamic obstacles are a later phase.

Before tuning MPPI parameters or changing cost functions, establish a 3D replay baseline across the scenario suite. That baseline should report goal success, collisions, minimum clearance, steps to goal, command smoothness, trajectory smoothness, and exported replay artifacts.

## User Stories

1. As a researcher, I want a modern Web-based 3D replay, so that I can inspect yaw-only 3D MPPI behavior more clearly than in matplotlib.
2. As a researcher, I want the Web viewer to be passive, so that Python remains the source of truth for MPPI control and simulation progression.
3. As a researcher, I want Python to keep ownership of the control loop, so that the 3D workflow stays close to the original 2D Core Python examples.
4. As a researcher, I want replay data exported after deterministic simulation runs, so that I can reproduce and inspect the same behavior repeatedly.
5. As a researcher, I want a scenario suite instead of one showcase scene, so that controller behavior is tested across distinct navigation challenges.
6. As a researcher, I want a baseline tracking scenario, so that I can measure control smoothness without obstacle effects.
7. As a researcher, I want a vertical gate scenario, so that I can test altitude changes and 3D clearance constraints.
8. As a researcher, I want a narrow-gap scenario for the T-shaped robot volume, so that non-convex collision geometry is actually exercised.
9. As a researcher, I want a T-shaped trap scenario, so that local-planning failure and oscillation cases become visible.
10. As a researcher, I want a cluttered corridor scenario, so that the planner is tested in a denser static environment.
11. As a planner user, I want dynamic obstacles excluded from the first phase, so that static scene behavior can be understood before adding moving obstacles.
12. As a planner user, I want a T-shaped 3D robot volume to be authoritative collision geometry, so that collision checks and visualization refer to the same robot body.
13. As a planner user, I want the Web viewer to render the robot from the same volume config used by the controller, so that display geometry does not mislead collision analysis.
14. As a planner user, I want the T-shaped robot yaw to be visually obvious, so that orientation errors and turning behavior are easy to inspect.
15. As a planner user, I want Web replay frames in world coordinates, so that the viewer does not need to understand the controller's local-frame internals.
16. As a planner user, I want local plans shown in global coordinates, so that I can compare them against the global reference path and executed trajectory.
17. As a planner user, I want optimal trajectories shown in global coordinates, so that I can inspect what MPPI intended at each step.
18. As a planner user, I want sampled rollouts to be optional and sampled, so that replay files remain usable by default.
19. As a planner user, I want obstacle points shown as static scene data, so that I can inspect clearance relationships during replay.
20. As a planner user, I want the executed path visible across the run, so that I can see drift, oscillation, and final approach behavior.
21. As a planner user, I want the global reference path visible, so that I can compare planned behavior with the route.
22. As a planner user, I want the current local plan visible, so that I can inspect the controller-facing reference window.
23. As a planner user, I want current command values displayed, so that I can correlate visual motion with `[vx, vy, vz, wz]`.
24. As a planner user, I want clearance displayed per frame, so that near-collision behavior is visible without guessing.
25. As a planner user, I want goal distance displayed per frame, so that convergence behavior is measurable during replay.
26. As a planner user, I want smoothness telemetry displayed, so that smoothness is evaluated from control and trajectory data rather than animation frame rate.
27. As a planner user, I want playback interpolation to be display-only, so that it cannot be mistaken for improved controller smoothness.
28. As a planner user, I want playback, pause, single-step, and timeline scrubbing, so that I can inspect critical moments.
29. As a planner user, I want playback speed controls, so that I can quickly scan or slowly examine a run.
30. As a planner user, I want orbit camera controls, so that I can inspect 3D geometry from any angle.
31. As a planner user, I want top, side, front, follow, and free camera modes, so that common analysis views are one action away.
32. As a planner user, I want layer toggles for obstacles, paths, rollouts, optimal trajectory, executed trajectory, and robot volume, so that visual clutter can be managed.
33. As a planner user, I want consistent colors for reference path, local plan, executed path, optimal trajectory, and rollouts, so that replays are easy to read.
34. As a planner user, I want exported summary metrics, so that scenario results can be compared without opening the viewer.
35. As a planner user, I want scenario-suite results to include reached goal, collision status, minimum clearance, steps to goal, and final distance, so that pass/fail behavior is explicit.
36. As a planner user, I want scenario-suite results to include command smoothness metrics, so that controller output quality is quantified.
37. As a planner user, I want scenario-suite results to include trajectory smoothness metrics, so that executed path quality is quantified.
38. As a planner user, I want a baseline report before tuning, so that parameter changes can be compared against current behavior.
39. As a maintainer, I want the 3D workflow to align with 2D configuration vocabulary, so that users can move between 2D and 3D examples without relearning concepts.
40. As a maintainer, I want the 3D controller implementation to remain separate from the 2D optimizer for now, so that this work does not become a dimension-generic refactor.
41. As a maintainer, I want planner/scenario configuration instead of hard-coded demo constants, so that new 3D scenarios can be added without editing controller logic.
42. As a maintainer, I want the exported replay schema to be stable, so that future live streaming can reuse the same conceptual frame shape.
43. As a maintainer, I want the Web viewer to be static Three.js in the first version, so that no Web framework or server lifecycle is required.
44. As a maintainer, I want Python to export data and the viewer to load data, so that the integration boundary stays simple.
45. As a maintainer, I want headless scenario-suite runs, so that tests and remote runs do not require a browser.
46. As a maintainer, I want viewer smoke checks where practical, so that the static viewer does not silently break.
47. As a maintainer, I want tests at the scenario-suite and export seams, so that implementation details can change without invalidating useful behavior.
48. As a future implementer, I want MPPI tuning deferred until after baseline metrics exist, so that tuning work is driven by evidence.
49. As a future implementer, I want realtime Web streaming deferred, so that the first viewer can ship without WebSocket or service complexity.
50. As a future implementer, I want scenario editing deferred, so that the first viewer remains an inspection tool rather than an editor.

## Implementation Decisions

- Preserve Python ownership of yaw-only 3D MPPI control, state integration, local 3D observation points, local plan selection, scenario progression, collision checks, and metrics.
- Implement the browser as a Passive Web 3D viewer. It must not own the simulator, controller, or scenario loop.
- Implement Offline Web replay first. Python should complete a deterministic run, export replay data, and then the browser should load that data for inspection.
- Use a static Three.js viewer for the first version. Do not introduce a frontend framework, live server, WebSocket layer, or parameter-tuning UI in this PRD.
- Keep the 3D optimizer implementation separate from the 2D optimizer implementation. The goal is a 2D-compatible 3D workflow, not a shared dimension-generic optimizer core.
- Align the 3D workflow with the original 2D examples at the external workflow level: scenario config, planner config, local-frame controller inputs, global-to-local transforms, local plan windows, controller command generation, and state application.
- Move important 3D demo constants into scenario/planner configuration where practical, especially scenario geometry, robot volume, MPPI parameters, visualization/export flags, and scenario-suite selection.
- Use a T-shaped 3D robot volume as the authoritative robot body for collision evaluation.
- Render the T-shaped robot in the Web viewer from the same robot volume configuration used by collision checks.
- Export replay scene data separately from per-frame data.
- Scene data should include scenario identity, coordinate conventions, obstacle points, reference path, robot volume config, and relevant static visualization metadata.
- Frame data should include robot state, executed path, local plan in global coordinates, optimal trajectory in global coordinates, command, clearance, goal distance, and smoothness telemetry.
- Sampled rollouts should be disabled or downsampled by default because full MPPI rollout batches can make replay JSON too large.
- When enabled, sampled rollouts should be exported in global coordinates and treated as an optional viewer layer.
- The Web viewer should not need to reconstruct local-frame controller inputs. Local-frame data can remain a Python-side implementation detail.
- Smoothness should be measured from control and trajectory histories, not from animation interpolation.
- Playback interpolation may improve visual continuity, but it must be treated as a viewer concern and not as evidence of controller smoothness.
- Establish baseline metrics across the scenario suite before changing MPPI parameters, critic weights, control smoothing, or cost functions.
- The first scenario suite should focus on static scenes: open tracking, vertical gates, narrow gaps, T-shaped traps, and cluttered corridors.
- Dynamic obstacles are out of scope for the first scenario suite and should be handled after static-scene behavior is understood.
- The viewer should support orbit, follow, top, side, front, and free camera modes.
- The viewer should support play, pause, previous/next frame, timeline scrubbing, and playback speed controls.
- The viewer should support layer toggles for obstacle points, global reference path, local plan, executed path, optimal trajectory, robot volume, and optional rollouts.
- The viewer should include a compact metrics panel with current frame index, command, clearance, goal distance, and smoothness telemetry.
- The viewer should use stable visual conventions for the main layers so that replay interpretation stays consistent across scenarios.
- The scenario-suite runner should emit machine-readable summaries for each scenario and an aggregate baseline report.
- The baseline report should make it clear which scenarios pass, fail, collide, miss the goal, or exhibit poor smoothness.
- The replay data schema should be reusable for a later live streaming mode, but live streaming should not be implemented here.

## Testing Decisions

- The highest-value seam is the scenario-suite runner: it should run selected deterministic scenarios headlessly and emit pass/fail summaries, smoothness telemetry, and optional replay artifacts.
- Scenario-suite tests should verify external behavior: each stable scenario produces finite metrics, valid result summaries, and replay data with the expected shape.
- Collision-oriented tests should verify that the T-shaped 3D robot volume used by scenarios is the same authoritative volume passed into collision evaluation and replay export.
- Smoothness telemetry tests should verify metric calculation from simple known command/state histories, rather than asserting implementation details of the controller.
- Replay export tests should verify that scene data and frame data are world-frame, finite, and sufficient for the viewer to render without local-frame controller reconstruction.
- Rollout export tests should verify that rollouts are optional and bounded/downsampled when enabled.
- Viewer tests should be lightweight smoke checks where practical: the static viewer should load representative replay data and create the expected core scene layers.
- Existing yaw-only 3D controller tests and the current obstacle-avoidance example tests remain prior art for Core Python end-to-end checks.
- Tests should run without ROS, Gazebo, a browser window, or a GPU.
- Tuning-specific assertions should not be added before the baseline exists. The first tests should protect data contracts, deterministic evaluation, and scenario-suite reporting.

## Out of Scope

- Moving the MPPI control loop into the browser.
- Browser-owned simulation, browser-owned state integration, or browser-owned scenario progression.
- Realtime Web streaming, WebSockets, or a Python Web server.
- React, Vite, or another frontend application framework.
- Scenario editing in the browser.
- Online parameter tuning in the browser.
- Switching controllers from the browser.
- Dynamic obstacle scenarios in the first phase.
- MPPI parameter tuning before baseline metrics are established.
- Cost-function redesign before baseline metrics are established.
- Refactoring 2D and 3D optimizers into a shared dimension-generic implementation.
- Full `ir-sim` 3D simulator expansion.
- ROS, Gazebo, or RViz integration for this Web replay feature.
- Full 6DoF drone dynamics, roll, pitch, gravity, thrust, mass, inertia, contact, or aerodynamic modeling.
- Treating Web animation smoothness as proof of controller smoothness.
- Decorative robot models that do not match the collision volume.

## Further Notes

- This PRD follows the current project glossary terms: Yaw-only 3D motion, T-shaped 3D robot volume, Passive Web 3D viewer, Offline Web replay, 3D smoothness telemetry, 3D replay baseline, 2D-compatible 3D workflow, and 3D scenario suite.
- The architectural boundary for the Passive Web 3D viewer is recorded in the project ADRs.
- The current matplotlib visualization can remain available as a lightweight/headless-compatible path, but the new Web replay becomes the primary inspection path for this feature.
- The first implementation should produce evidence before tuning. Once baseline reports exist, follow-up work can tune horizon, batch size, sampling standard deviations, critic weights, and any explicit command smoothing based on scenario-specific failures.
