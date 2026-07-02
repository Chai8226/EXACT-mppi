# Static 3D Replay Viewer

Generate a replay artifact:

```bash
python -m exact_mppi.scenario_runner_3d --scenario narrow_gap_t_volume_3d --replay-json /tmp/narrow_gap_t_volume_3d.replay.json
```

To include bounded sampled MPPI rollouts:

```bash
python -m exact_mppi.scenario_runner_3d --scenario narrow_gap_t_volume_3d --replay-json /tmp/narrow_gap_t_volume_3d_rollouts.replay.json --replay-rollouts --replay-max-rollouts 8
```

Open `index.html` in a browser, choose the generated replay JSON file, then use playback, timeline scrubbing, speed, layer toggles, and camera modes to inspect the run. The viewer is static Three.js: it does not use a Python Web server, WebSocket, React, Vite, or realtime streaming.
