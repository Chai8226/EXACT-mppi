# MID-360-Like observed cloud tracer bullet

Status: ready-for-agent

## Parent

.scratch/mid360-like-3d-point-cloud-observation/PRD.md

## What to build

Build the first complete MID-360-like 3D point cloud observation path through the scenario runner. A scenario should be able to define a top-level MID-360-like sensor, raycast that sensor against axis-aligned box obstacle geometry at each planning step, produce a world-frame Observed 3D point cloud, transform those observed points into the robot-local yaw frame, and feed them to the 3D controller using the existing controller point-packing behavior.

This slice should be narrow but real: it should prove the new observation source can drive MPPI control and replay export without using Legacy 3D obstacle points as the controller's perception input.

## Acceptance criteria

- [ ] A scenario can configure a MID-360-like sensor with range, FOV, and deterministic angular-grid sample counts.
- [ ] The scenario runner generates a per-step world-frame Observed 3D point cloud by raycasting against axis-aligned box obstacle geometry.
- [ ] Raycast output respects configured minimum range, maximum range, horizontal FOV, and asymmetric vertical FOV.
- [ ] Each ray contributes at most the nearest box hit, so basic occlusion is represented.
- [ ] The controller receives robot-local yaw-frame points derived from the observed cloud, not from Legacy 3D obstacle points.
- [ ] Existing controller point packing remains responsible for enforcing `max_obs_num` and nearest-point selection when over budget.
- [ ] Replay frame data includes the current world-frame observed point cloud for this tracer-bullet scenario.
- [ ] Focused tests verify the runner can execute the tracer-bullet scenario headlessly with finite commands, finite observed points, and JSON-serializable replay data.

## Blocked by

None - can start immediately
