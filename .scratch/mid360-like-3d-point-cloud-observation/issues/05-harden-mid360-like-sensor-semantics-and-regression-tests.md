# Harden MID-360-like sensor semantics and regression tests

Status: ready-for-agent

## Parent

.scratch/mid360-like-3d-point-cloud-observation/PRD.md

## What to build

Harden the MID-360-like 3D point cloud observation feature with focused regression coverage and final semantic cleanup. The resulting test suite should protect the agreed first-version sensor behavior: deterministic regular angular-grid raycasting, MID-360-like asymmetric FOV, configured conservative range, nearest-hit occlusion, disabled noise/dropout defaults, and clear separation between observed controller perception and geometry-based metrics.

This slice is the feature-completion pass. It should catch contract regressions across the sensor generator, runner, replay schema, viewer, and metrics without testing private implementation details.

## Acceptance criteria

- [ ] Pure raycast tests cover vertical FOV behavior for `-7°` to `+52°`.
- [ ] Pure raycast tests cover the configured `0.1m` to `10.0m` range limits.
- [ ] Pure raycast tests cover nearest-hit occlusion for two boxes along the same ray direction.
- [ ] Pure raycast tests cover empty geometry producing an empty observed cloud.
- [ ] Tests confirm noise and dropout defaults are disabled and deterministic.
- [ ] Tests confirm reserved noise/dropout configuration does not change behavior when set to zero.
- [ ] Runner tests confirm controller inputs come from observed raycast points rather than legacy global sampled points.
- [ ] Replay tests confirm scene/frame schema after legacy obstacle point removal.
- [ ] Viewer smoke coverage confirms static geometry and dynamic observed cloud layers work with representative replay data.
- [ ] Metrics tests confirm geometry truth remains the source for clearance and collision.
- [ ] The focused test set runs headlessly on CPU without ROS, Gazebo, RViz, GPU, or a live Web server.
- [ ] Documentation or inline scenario comments no longer describe the new path as Range-based local 3D observation.

## Blocked by

- .scratch/mid360-like-3d-point-cloud-observation/issues/01-mid360-like-observed-cloud-tracer-bullet.md
- .scratch/mid360-like-3d-point-cloud-observation/issues/02-migrate-3d-scenario-configs-off-legacy-observation-fields.md
- .scratch/mid360-like-3d-point-cloud-observation/issues/03-use-geometry-truth-for-3d-clearance-and-collision-metrics.md
- .scratch/mid360-like-3d-point-cloud-observation/issues/04-finalize-replay-schema-and-passive-web-observed-cloud-layer.md
