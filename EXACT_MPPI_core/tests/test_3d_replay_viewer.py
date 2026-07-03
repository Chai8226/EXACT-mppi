import json
import shutil
import subprocess
from importlib import resources

import pytest


def representative_mid360_replay():
    return {
        "schema_version": 1,
        "scene": {
            "scenario": "viewer_smoke",
            "reference_path": [[0.0, 0.0, 0.0, 0.0]],
            "obstacle_geometry": [
                {
                    "type": "box",
                    "center": [1.0, 0.0, 0.0],
                    "size": [0.2, 0.2, 0.2],
                }
            ],
            "robot_volume": {
                "type": "box_union",
                "boxes": [{"center": [0.0, 0.0, 0.0], "size": [0.4, 0.4, 0.4]}],
            },
        },
        "frames": [
            {
                "frame_index": 0,
                "state": [0.0, 0.0, 0.0, 0.0],
                "executed_path": [[0.0, 0.0, 0.0, 0.0]],
                "reference_window": [[0.0, 0.0, 0.0, 0.0]],
                "optimal_trajectory": [[0.0, 0.0, 0.0, 0.0]],
                "observed_point_cloud": [[0.9, 0.0, 0.0]],
                "command": [0.0, 0.0, 0.0, 0.0],
                "clearance": 0.5,
                "goal_distance": 0.0,
                "smoothness_telemetry": {
                    "command_smoothness": {"rms_delta_norm": 0.0},
                    "trajectory_smoothness": {"rms_second_difference_norm": 0.0},
                },
            }
        ],
    }


def test_static_3d_replay_viewer_resources_are_packaged():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")

    assert viewer_files.joinpath("index.html").is_file()
    assert viewer_files.joinpath("app.js").is_file()
    assert viewer_files.joinpath("style.css").is_file()
    assert viewer_files.joinpath("README.md").is_file()


def test_static_3d_replay_viewer_exposes_replay_controls():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    html = viewer_files.joinpath("index.html").read_text(encoding="utf-8")

    for required_id in (
        'id="replay-file"',
        'id="play-toggle"',
        'id="timeline"',
        'id="speed"',
        'id="camera-mode"',
        'id="layer-observed-cloud"',
        'id="layer-obstacle-geometry"',
        'id="layer-reference"',
        'id="layer-local-plan"',
        'id="layer-executed"',
        'id="layer-optimal"',
        'id="layer-rollouts"',
        'id="layer-robot"',
        'id="frame-state"',
        'id="metric-cadence"',
        'id="metric-display"',
    ):
        assert required_id in html
    assert "Reference window" in html
    assert "Local plan" not in html


def test_static_3d_replay_viewer_uses_passive_threejs_runtime():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    html = viewer_files.joinpath("index.html").read_text(encoding="utf-8")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    assert "three.module.js" in html
    assert "OrbitControls.js" in html
    assert "WebSocket" not in app_js
    assert "React" not in html + app_js
    assert "vite" not in html.lower() + app_js.lower()


def test_static_3d_replay_viewer_renders_core_replay_layers():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    for renderer_hook in (
        "renderObservedPointCloud",
        "renderObstacleGeometry",
        "renderReferencePath",
        "renderLocalPlan",
        "renderExecutedPath",
        "renderOptimalTrajectory",
        "renderRollouts",
        "renderRobotVolume",
        "renderFrameState",
    ):
        assert renderer_hook in app_js


def test_static_3d_replay_viewer_supports_camera_modes_and_layer_toggles():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    for camera_mode in ("top", "side", "front", "follow", "free"):
        assert f'"{camera_mode}"' in app_js

    assert "applyCameraMode" in app_js
    assert "updateLayerVisibility" in app_js
    assert "layerControls" in app_js


def test_static_3d_replay_viewer_renders_obstacle_geometry_from_scene_data():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    assert "replay.scene.obstacle_geometry" in app_js
    assert "BoxGeometry" in app_js
    assert "obstacleGeometry" in app_js


def test_static_3d_replay_viewer_renders_dynamic_observed_cloud_from_frame_data():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    assert "frame.observed_point_cloud" in app_js
    assert "renderObservedPointCloud" in app_js
    assert "observedCloud" in app_js
    assert "replay.scene.obstacle_points" not in app_js
    assert "renderObservedPointCloud(fromFrame.observed_point_cloud)" in app_js


def test_static_3d_replay_viewer_load_path_supports_representative_mid360_replay():
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the static replay viewer smoke test.")
    replay = representative_mid360_replay()
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")
    executable_app = "\n".join(
        line
        for line in app_js.splitlines()
        if not line.startswith("import ")
    )
    node_script = f"""
class Vector3 {{
  constructor(x = 0, y = 0, z = 0) {{ this.x = x; this.y = y; this.z = z; }}
  set(x, y, z) {{ this.x = x; this.y = y; this.z = z; return this; }}
  copy(other) {{ this.x = other.x; this.y = other.y; this.z = other.z; return this; }}
  add(other) {{ this.x += other.x; this.y += other.y; this.z += other.z; return this; }}
}}
class Object3D {{
  constructor() {{
    this.children = [];
    this.visible = true;
    this.position = new Vector3();
    this.rotation = {{ set() {{}} }};
  }}
  add(...children) {{ this.children.push(...children); }}
  remove(child) {{ this.children = this.children.filter((item) => item !== child); }}
}}
class Geometry {{
  setFromPoints(points) {{ this.points = points; return this; }}
  dispose() {{}}
}}
class Group extends Object3D {{}}
class Mesh extends Object3D {{
  constructor(geometry, material) {{ super(); this.geometry = geometry; this.material = material; }}
}}
class Points extends Mesh {{}}
class Line extends Mesh {{}}
class LineSegments extends Mesh {{}}
class PerspectiveCamera extends Object3D {{
  constructor() {{ super(); this.up = new Vector3(); }}
  lookAt() {{}}
  updateProjectionMatrix() {{}}
}}
class Scene extends Group {{}}
class DirectionalLight extends Object3D {{}}
const THREE = {{
  AmbientLight: class extends Object3D {{}},
  BoxGeometry: class extends Geometry {{}},
  BufferGeometry: Geometry,
  DirectionalLight,
  EdgesGeometry: class extends Geometry {{
    constructor(source) {{ super(); this.source = source; }}
  }},
  Group,
  Line,
  LineBasicMaterial: class {{ constructor(options) {{ this.options = options; }} }},
  LineSegments,
  MathUtils: {{ clamp(value, min, max) {{ return Math.min(Math.max(value, min), max); }} }},
  Mesh,
  MeshStandardMaterial: class {{ constructor(options) {{ this.options = options; }} }},
  PerspectiveCamera,
  Points,
  PointsMaterial: class {{ constructor(options) {{ this.options = options; }} }},
  Scene,
  Vector3,
  WebGLRenderer: class {{
    constructor(options) {{ this.domElement = options.canvas; }}
    setPixelRatio() {{}}
    setClearColor() {{}}
    setSize() {{}}
    render() {{}}
  }},
}};
class OrbitControls {{
  constructor() {{ this.target = new Vector3(); this.enabled = true; }}
  update() {{}}
}}
const elements = new Map();
function elementFor(id) {{
  if (!elements.has(id)) {{
    elements.set(id, {{
      addEventListener() {{}},
      checked: true,
      textContent: "",
      value: id === "speed" ? "1" : id === "camera-mode" ? "free" : "0",
    }});
  }}
  return elements.get(id);
}}
globalThis.document = {{ getElementById: elementFor }};
globalThis.window = {{
  addEventListener() {{}},
  devicePixelRatio: 1,
  innerHeight: 720,
  innerWidth: 1280,
}};
globalThis.requestAnimationFrame = () => {{}};
{executable_app}
loadReplay({json.dumps(replay)});
if (layerGroups.obstacleGeometry.children.length !== 2) {{
  throw new Error(`Expected obstacle geometry mesh and edges, got ${{layerGroups.obstacleGeometry.children.length}}`);
}}
if (layerGroups.observedCloud.children.length !== 1) {{
  throw new Error(`Expected observed cloud points, got ${{layerGroups.observedCloud.children.length}}`);
}}
if (layerGroups.robot.children.length !== 2) {{
  throw new Error(`Expected robot volume mesh and edges, got ${{layerGroups.robot.children.length}}`);
}}
if (metrics.scenario.textContent !== "viewer_smoke") {{
  throw new Error(`Expected scenario metric to update, got ${{metrics.scenario.textContent}}`);
}}
"""

    assert "obstacle_points" not in replay["scene"]
    assert replay["scene"]["obstacle_geometry"]
    assert replay["frames"][0]["observed_point_cloud"]
    subprocess.run(
        ["node", "--input-type=module"],
        input=node_script,
        text=True,
        check=True,
        capture_output=True,
    )


def test_static_3d_replay_viewer_validates_new_replay_schema():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    assert "candidate.scene.obstacle_points" in app_js
    assert "candidate.scene.obstacle_geometry" in app_js
    assert "observed_point_cloud" in app_js
    assert "Replay scene uses obsolete obstacle_points" in app_js


def test_static_3d_replay_viewer_highlights_local_plan_layer():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    assert "frame.reference_window ?? frame.local_plan" in app_js
    assert "LOCAL_PLAN_RENDER_ORDER" in app_js
    assert "localPlanMarker" in app_js
    assert "renderLocalPlanMarkers" in app_js
    assert "depthTest: false" in app_js


def test_static_3d_replay_viewer_uses_display_only_interpolation():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    assert "DISPLAY_FRAMES_PER_SECOND" in app_js
    assert "interpolateFrameState" in app_js
    assert "interpolateYaw" in app_js
    assert "renderInterpolatedDisplayState" in app_js
    assert "renderExactFrameState" in app_js
    assert "Display interpolation" in app_js


def test_static_3d_replay_viewer_renders_authoritative_robot_volume_without_heading_cone():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    app_js = viewer_files.joinpath("app.js").read_text(encoding="utf-8")

    for color_name in (
        "obstacles",
        "observedCloud",
        "referencePath",
        "localPlan",
        "executedPath",
        "optimalTrajectory",
        "rollouts",
        "robotBody",
        "robotEdges",
    ):
        assert color_name in app_js

    assert "replay.scene.robot_volume" in app_js
    assert "BoxGeometry" in app_js
    assert "EdgesGeometry" in app_js
    assert "ConeGeometry" not in app_js
    assert "robotHeading" not in app_js


def test_static_3d_replay_viewer_documents_manual_smoke_check():
    viewer_files = resources.files("exact_mppi.replay_viewer_3d")
    readme = viewer_files.joinpath("README.md").read_text(encoding="utf-8")

    assert "--replay-json" in readme
    assert "--replay-rollouts" in readme
    assert "index.html" in readme
    assert "narrow_gap_t_volume_3d" in readme
    assert "Observed 3D point cloud" in readme
    assert "obstacle points" not in readme
