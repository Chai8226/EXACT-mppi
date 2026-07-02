# Static 3D Replay Viewer

Generate a replay artifact:

```bash
python -m exact_mppi.scenario_runner_3d --scenario narrow_gap_t_volume_3d --replay-json /tmp/narrow_gap_t_volume_3d.replay.json
```

Open `index.html` in a browser, choose the generated replay JSON file, then use the playback controls and orbit camera to inspect the run. The viewer is static Three.js: it does not use a Python Web server, WebSocket, React, Vite, or realtime streaming.
