from importlib import resources


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
