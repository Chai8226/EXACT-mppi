# Minimal static Three.js replay viewer

Status: implemented

## Parent

.scratch/3d-web-replay-and-scenario-suite/PRD.md

## What to build

Build the first static Three.js Passive Web 3D viewer for Offline Web replay data. The viewer should load a replay artifact, render the core scene layers, and provide basic playback controls without introducing a frontend framework, Python Web server, or realtime streaming.

This slice should be demoable with exported replay data from the scenario runner.

## Acceptance criteria

- [x] A static browser viewer can load a replay artifact produced by the Python exporter.
- [x] The viewer renders obstacle points, global reference path, executed path, current robot volume, and current frame state.
- [x] The viewer provides play, pause, timeline scrubbing, and playback speed controls.
- [x] Orbit camera controls work for inspecting the 3D scene.
- [x] The viewer does not require a Python Web server, WebSocket, React, Vite, or another frontend framework.
- [x] A lightweight smoke check or documented manual check demonstrates loading representative replay data.

## Blocked by

- .scratch/3d-web-replay-and-scenario-suite/issues/05-offline-web-replay-export-schema-writer.md
