import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.getElementById("scene");
const fileInput = document.getElementById("replay-file");
const playToggle = document.getElementById("play-toggle");
const timeline = document.getElementById("timeline");
const speedSelect = document.getElementById("speed");
const cameraModeSelect = document.getElementById("camera-mode");
const layerControls = {
  obstacles: document.getElementById("layer-obstacles"),
  reference: document.getElementById("layer-reference"),
  localPlan: document.getElementById("layer-local-plan"),
  executed: document.getElementById("layer-executed"),
  optimal: document.getElementById("layer-optimal"),
  rollouts: document.getElementById("layer-rollouts"),
  robot: document.getElementById("layer-robot"),
};

const metrics = {
  scenario: document.getElementById("metric-scenario"),
  frame: document.getElementById("metric-frame"),
  command: document.getElementById("metric-command"),
  clearance: document.getElementById("metric-clearance"),
  goalDistance: document.getElementById("metric-goal-distance"),
  commandRms: document.getElementById("metric-command-rms"),
  trajectoryRms: document.getElementById("metric-trajectory-rms"),
};

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x111418, 1);

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.7));

const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(2, 5, 3);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
camera.position.set(4.2, 3.2, 5.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(1.6, 0.2, 0);

const layerGroups = {
  obstacles: new THREE.Group(),
  reference: new THREE.Group(),
  localPlan: new THREE.Group(),
  executed: new THREE.Group(),
  optimal: new THREE.Group(),
  rollouts: new THREE.Group(),
  robot: new THREE.Group(),
};
scene.add(...Object.values(layerGroups));

const layerColors = {
  obstacles: 0xe05a47,
  referencePath: 0x4aa3df,
  localPlan: 0xb48cf2,
  executedPath: 0x55c2a3,
  optimalTrajectory: 0xf2c14e,
  rollouts: 0x7d8790,
  robotBody: 0xdce5ea,
  robotHeading: 0xfff1a8,
  robotEdges: 0x2b3036,
};

const materials = {
  obstacle: new THREE.PointsMaterial({ color: layerColors.obstacles, size: 0.055 }),
  reference: new THREE.LineBasicMaterial({ color: layerColors.referencePath }),
  localPlan: new THREE.LineBasicMaterial({ color: layerColors.localPlan }),
  executed: new THREE.LineBasicMaterial({ color: layerColors.executedPath }),
  optimal: new THREE.LineBasicMaterial({ color: layerColors.optimalTrajectory }),
  rollouts: new THREE.LineBasicMaterial({
    color: layerColors.rollouts,
    transparent: true,
    opacity: 0.26,
  }),
  robot: new THREE.MeshStandardMaterial({
    color: layerColors.robotBody,
    roughness: 0.7,
    metalness: 0.05,
    transparent: true,
    opacity: 0.82,
  }),
  robotHeading: new THREE.MeshStandardMaterial({
    color: layerColors.robotHeading,
    roughness: 0.5,
    metalness: 0.05,
  }),
  robotEdges: new THREE.LineBasicMaterial({ color: layerColors.robotEdges }),
};

let replay = null;
let currentFrame = 0;
let playing = false;
let lastTick = 0;
let frameAccumulator = 0;

fileInput.addEventListener("change", handleReplayFile);
playToggle.addEventListener("click", togglePlayback);
timeline.addEventListener("input", () => {
  setFrame(Number(timeline.value));
});
cameraModeSelect.addEventListener("change", () => {
  applyCameraMode(cameraModeSelect.value);
});

for (const control of Object.values(layerControls)) {
  control.addEventListener("change", updateLayerVisibility);
}

window.addEventListener("resize", resize);
resize();
renderFrameState(null);
updateLayerVisibility();
animate(0);

function handleReplayFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.addEventListener("load", () => {
    const parsedReplay = JSON.parse(String(reader.result));
    loadReplay(parsedReplay);
  });
  reader.readAsText(file);
}

function loadReplay(nextReplay) {
  validateReplay(nextReplay);
  replay = nextReplay;
  currentFrame = 0;
  frameAccumulator = 0;
  playing = false;
  playToggle.textContent = "Play";
  timeline.min = "0";
  timeline.max = String(Math.max(replay.frames.length - 1, 0));
  timeline.value = "0";

  clearAllLayers();
  renderObstaclePoints(replay.scene.obstacle_points);
  renderReferencePath(replay.scene.reference_path);
  updateLayerVisibility();
  setFrame(0);
}

function validateReplay(candidate) {
  if (!candidate || candidate.schema_version !== 1) {
    throw new Error("Unsupported replay schema.");
  }
  if (!candidate.scene || !Array.isArray(candidate.frames)) {
    throw new Error("Replay requires scene and frames.");
  }
}

function togglePlayback() {
  if (!replay || replay.frames.length === 0) {
    return;
  }
  playing = !playing;
  playToggle.textContent = playing ? "Pause" : "Play";
}

function setFrame(frameIndex) {
  if (!replay || replay.frames.length === 0) {
    return;
  }
  currentFrame = THREE.MathUtils.clamp(frameIndex, 0, replay.frames.length - 1);
  timeline.value = String(currentFrame);
  renderFrameState(replay.frames[currentFrame]);
  applyCameraMode(cameraModeSelect.value);
}

function renderObstaclePoints(points) {
  if (!Array.isArray(points) || points.length === 0) {
    return;
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(worldToThree));
  layerGroups.obstacles.add(new THREE.Points(geometry, materials.obstacle));
}

function renderReferencePath(path) {
  const line = makeLine(path, materials.reference);
  if (line) {
    layerGroups.reference.add(line);
  }
}

function renderLocalPlan(path) {
  const line = makeLine(path, materials.localPlan);
  if (line) {
    layerGroups.localPlan.add(line);
  }
}

function renderExecutedPath(path) {
  const line = makeLine(path, materials.executed);
  if (line) {
    layerGroups.executed.add(line);
  }
}

function renderOptimalTrajectory(path) {
  const line = makeLine(path, materials.optimal);
  if (line) {
    layerGroups.optimal.add(line);
  }
}

function renderRollouts(rollouts) {
  if (!Array.isArray(rollouts)) {
    return;
  }
  for (const rollout of rollouts) {
    const line = makeLine(rollout, materials.rollouts);
    if (line) {
      layerGroups.rollouts.add(line);
    }
  }
}

function renderRobotVolume(robotVolume, frame) {
  clearGroup(layerGroups.robot);
  if (!robotVolume || !Array.isArray(robotVolume.boxes) || !frame) {
    return;
  }

  const state = frame.state;
  layerGroups.robot.position.copy(worldToThree(state));
  layerGroups.robot.rotation.set(0, -state[3], 0);

  for (const box of robotVolume.boxes) {
    const size = box.size || box.half_extents?.map((v) => v * 2);
    const center = box.center || [0, 0, 0];
    if (!size) {
      continue;
    }
    const geometry = new THREE.BoxGeometry(size[0], size[2], size[1]);
    const mesh = new THREE.Mesh(geometry, materials.robot);
    mesh.position.copy(worldToThree(center));
    layerGroups.robot.add(mesh);

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      materials.robotEdges,
    );
    edges.position.copy(mesh.position);
    layerGroups.robot.add(edges);
  }

  const headingGeometry = new THREE.ConeGeometry(0.09, 0.32, 16);
  const heading = new THREE.Mesh(headingGeometry, materials.robotHeading);
  heading.position.set(0.46, 0.0, 0.0);
  heading.rotation.set(0, 0, -Math.PI / 2);
  layerGroups.robot.add(heading);
}

function renderFrameState(frame) {
  clearDynamicLayers();
  if (!replay || !frame) {
    metrics.scenario.textContent = "No replay";
    metrics.frame.textContent = "0 / 0";
    return;
  }

  renderLocalPlan(frame.local_plan);
  renderExecutedPath(frame.executed_path);
  renderOptimalTrajectory(frame.optimal_trajectory);
  renderRollouts(frame.rollouts);
  renderRobotVolume(replay.scene.robot_volume, frame);
  updateLayerVisibility();

  metrics.scenario.textContent = replay.scene.scenario;
  metrics.frame.textContent = `${frame.frame_index + 1} / ${replay.frames.length}`;
  metrics.command.textContent = formatVector(frame.command);
  metrics.clearance.textContent = formatScalar(frame.clearance);
  metrics.goalDistance.textContent = formatScalar(frame.goal_distance);
  metrics.commandRms.textContent = formatScalar(
    frame.smoothness_telemetry?.command_smoothness?.rms_delta_norm,
  );
  metrics.trajectoryRms.textContent = formatScalar(
    frame.smoothness_telemetry?.trajectory_smoothness
      ?.rms_second_difference_norm,
  );
}

function updateLayerVisibility() {
  for (const [name, group] of Object.entries(layerGroups)) {
    const control = layerControls[name];
    if (control) {
      group.visible = control.checked;
    }
  }
}

function applyCameraMode(mode) {
  if (!replay || replay.frames.length === 0 || mode === "free") {
    controls.enabled = true;
    return;
  }

  const state = replay.frames[currentFrame].state;
  const target = worldToThree(state);
  controls.target.copy(target);
  controls.enabled = mode !== "follow";

  if (mode === "top") {
    camera.up.set(0, 0, -1);
    camera.position.copy(target).add(new THREE.Vector3(0, 5.4, 0));
  } else if (mode === "side") {
    camera.up.set(0, 1, 0);
    camera.position.copy(target).add(new THREE.Vector3(0, 1.2, 5.2));
  } else if (mode === "front") {
    camera.up.set(0, 1, 0);
    camera.position.copy(target).add(new THREE.Vector3(5.2, 1.2, 0));
  } else if (mode === "follow") {
    camera.up.set(0, 1, 0);
    const yaw = state[3];
    const followWorldPosition = [
      state[0] - Math.cos(yaw) * 2.2,
      state[1] - Math.sin(yaw) * 2.2,
      state[2] + 1.1,
    ];
    camera.position.copy(worldToThree(followWorldPosition));
  }

  camera.lookAt(target);
  controls.update();
}

function makeLine(points, material) {
  if (!Array.isArray(points) || points.length < 2) {
    return null;
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(worldToThree));
  return new THREE.Line(geometry, material);
}

function worldToThree(point) {
  return new THREE.Vector3(point[0], point[2], point[1]);
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    child.geometry?.dispose();
  }
}

function clearAllLayers() {
  for (const group of Object.values(layerGroups)) {
    clearGroup(group);
  }
}

function clearDynamicLayers() {
  clearGroup(layerGroups.localPlan);
  clearGroup(layerGroups.executed);
  clearGroup(layerGroups.optimal);
  clearGroup(layerGroups.rollouts);
  clearGroup(layerGroups.robot);
}

function formatScalar(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

function formatVector(value) {
  if (!Array.isArray(value)) {
    return "[0, 0, 0, 0]";
  }
  return `[${value.map((item) => Number(item).toFixed(2)).join(", ")}]`;
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function animate(timestamp) {
  requestAnimationFrame(animate);

  const deltaSeconds = Math.min((timestamp - lastTick) / 1000, 0.25);
  lastTick = timestamp;

  if (playing && replay?.frames.length > 0) {
    frameAccumulator += deltaSeconds * Number(speedSelect.value) * 8;
    if (frameAccumulator >= 1) {
      const step = Math.floor(frameAccumulator);
      frameAccumulator -= step;
      const nextFrame = currentFrame + step;
      if (nextFrame >= replay.frames.length) {
        playing = false;
        playToggle.textContent = "Play";
        setFrame(replay.frames.length - 1);
      } else {
        setFrame(nextFrame);
      }
    }
  }

  controls.update();
  renderer.render(scene, camera);
}
