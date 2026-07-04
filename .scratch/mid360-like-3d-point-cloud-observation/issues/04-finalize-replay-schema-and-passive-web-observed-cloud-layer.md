# Finalize replay schema and Passive Web observed cloud layer

Status: ready-for-agent

## Parent

.scratch/mid360-like-3d-point-cloud-observation/PRD.md

## What to build

Finalize the Offline Web replay contract for MID-360-like perception and update the Passive Web 3D viewer to show sensor output. Replay scene data should expose static obstacle geometry as world truth and should no longer include legacy static obstacle points. Each replay frame should expose the current world-frame observed point cloud. The viewer should render static obstacle geometry separately from a dynamic Observed cloud layer that updates with playback and timeline scrubbing.

This slice should make the new perception model inspectable: users should be able to see both what exists in the world and what the simulated sensor observed at a given frame.

## Acceptance criteria

- [ ] Replay scene data includes obstacle geometry.
- [ ] Replay scene data omits legacy static obstacle points.
- [ ] Every replay frame includes a world-frame observed point cloud.
- [ ] Replay JSON remains finite and serializable without NaN.
- [ ] The viewer renders static obstacle geometry as a separate layer.
- [ ] The viewer renders a dynamic Observed cloud layer from frame data.
- [ ] Timeline scrubbing and playback update the Observed cloud layer to match the current frame.
- [ ] The old static obstacle point layer is removed or renamed so it cannot be confused with controller perception.
- [ ] Viewer smoke tests or equivalent static checks cover loading a representative new-schema replay and creating the observed cloud rendering path.

## Blocked by

- .scratch/mid360-like-3d-point-cloud-observation/issues/01-mid360-like-observed-cloud-tracer-bullet.md
- .scratch/mid360-like-3d-point-cloud-observation/issues/02-migrate-3d-scenario-configs-off-legacy-observation-fields.md
