import numpy as np

from exact_mppi.mppi_3d.geometry import (
    BoxUnionVolume3D,
    point_to_triangle_distance,
    points_inside_halfspaces,
    signed_distance_to_box_union,
    signed_distance_to_convex_polyhedron,
)


def test_box_union_volume_can_be_created_from_readable_config():
    volume = BoxUnionVolume3D.from_config(
        {
            "type": "box_union",
            "boxes": [
                {"center": [1.0, 2.0, 3.0], "size": [2.0, 4.0, 6.0]},
                {"center": [-1.0, 0.0, 0.0], "half_extents": [0.5, 1.0, 1.5]},
            ],
        }
    )

    assert volume.triangles.shape == (2, 12, 3, 3)
    assert volume.halfspace_normals.shape == (2, 6, 3)
    assert volume.halfspace_offsets.shape == (2, 6)
    np.testing.assert_allclose(
        np.asarray(volume.halfspace_normals[0]),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_point_to_triangle_distance_handles_face_edge_and_vertex_cases():
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    points = np.array(
        [
            [0.5, 0.5, 3.0],
            [1.0, -2.0, 0.0],
            [-1.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    distances = point_to_triangle_distance(points, triangle)

    np.testing.assert_allclose(
        np.asarray(distances),
        np.array([3.0, 2.0, np.sqrt(2.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_halfspace_classification_treats_box_surface_as_inside():
    volume = BoxUnionVolume3D.from_config(
        [{"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]}]
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    inside = points_inside_halfspaces(
        points,
        volume.halfspace_normals[0],
        volume.halfspace_offsets[0],
    )

    np.testing.assert_array_equal(np.asarray(inside), np.array([True, True, False]))


def test_single_box_signed_distance_is_negative_inside_zero_on_surface_positive_outside():
    volume = BoxUnionVolume3D.from_config(
        [{"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]}]
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.25, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    distances = signed_distance_to_convex_polyhedron(
        points,
        volume.triangles[0],
        volume.halfspace_normals[0],
        volume.halfspace_offsets[0],
    )

    np.testing.assert_allclose(
        np.asarray(distances),
        np.array([-1.0, 0.0, 0.25, np.sqrt(2.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_box_union_signed_distance_takes_minimum_across_boxes():
    volume = BoxUnionVolume3D.from_config(
        {
            "boxes": [
                {"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]},
                {"center": [3.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]},
            ],
        }
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    distances = signed_distance_to_box_union(points, volume)

    np.testing.assert_allclose(
        np.asarray(distances),
        np.array([-1.0, -1.0, 0.5, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
