# Use geometry truth for 3D clearance and collision metrics

Status: ready-for-agent

## Parent

.scratch/mid360-like-3d-point-cloud-observation/PRD.md

## What to build

Make 3D scenario clearance and collision reporting use authoritative obstacle geometry truth instead of observed point clouds or Legacy 3D obstacle points. The controller should remain limited to the Observed 3D point cloud, but scenario metrics should answer whether the robot volume intersects or approaches the actual world geometry.

This slice should preserve honest evaluation: a robot can collide with geometry even if that surface was outside the sensor FOV or hidden from the observed cloud.

## Acceptance criteria

- [ ] Minimum clearance for 3D scenario results is computed against obstacle geometry truth.
- [ ] Collision status is derived from geometry truth and the configured collision margin.
- [ ] Metrics do not depend on the current observed point cloud contents.
- [ ] A test demonstrates that collision or low clearance is still reported when the relevant geometry is not present in the observed cloud.
- [ ] Replay summaries and baseline reports continue to expose finite clearance/collision fields where applicable.
- [ ] Existing robot volume semantics remain authoritative for clearance evaluation.

## Blocked by

- .scratch/mid360-like-3d-point-cloud-observation/issues/01-mid360-like-observed-cloud-tracer-bullet.md
