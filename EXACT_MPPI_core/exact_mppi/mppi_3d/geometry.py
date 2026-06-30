from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class BoxUnionVolume3D:
    """Robot volume represented as a union of body-frame boxes."""

    triangles: jax.Array
    halfspace_normals: jax.Array
    halfspace_offsets: jax.Array

    def signed_distance(self, points: jax.Array) -> jax.Array:
        return signed_distance_to_box_union(points, self)

    @classmethod
    def from_config(cls, config: Any) -> "BoxUnionVolume3D":
        boxes = _read_boxes_config(config)
        triangles = []
        normals = []
        offsets = []

        for box in boxes:
            center, half_extents = _read_box(box)
            box_triangles, box_normals, box_offsets = _box_to_polyhedron(
                center,
                half_extents,
            )
            triangles.append(box_triangles)
            normals.append(box_normals)
            offsets.append(box_offsets)

        return cls(
            triangles=jnp.asarray(np.stack(triangles), dtype=jnp.float32),
            halfspace_normals=jnp.asarray(np.stack(normals), dtype=jnp.float32),
            halfspace_offsets=jnp.asarray(np.stack(offsets), dtype=jnp.float32),
        )


def _read_boxes_config(config: Any) -> list[Any]:
    if isinstance(config, dict):
        if config.get("type", "box_union") != "box_union":
            raise ValueError("3D robot volume config type must be 'box_union'.")
        boxes = config.get("boxes")
    else:
        boxes = config

    if boxes is None or len(boxes) == 0:
        raise ValueError("box_union volume requires at least one box.")
    return list(boxes)


def point_to_triangle_distance(points: jax.Array, triangle: jax.Array) -> jax.Array:
    """Compute exact unsigned distance from points to a closed triangle."""

    p = jnp.asarray(points, dtype=jnp.float32)
    tri = jnp.asarray(triangle, dtype=jnp.float32)
    a, b, c = tri[0], tri[1], tri[2]

    ab = b - a
    ac = c - a
    normal = jnp.cross(ab, ac)
    normal_sq = jnp.sum(normal * normal)
    normal_len = jnp.sqrt(jnp.maximum(normal_sq, 1e-12))
    unit_normal = normal / normal_len

    plane_distance = jnp.sum((p - a) * unit_normal, axis=-1)
    projection = p - plane_distance[..., None] * unit_normal

    d00 = jnp.sum(ab * ab)
    d01 = jnp.sum(ab * ac)
    d11 = jnp.sum(ac * ac)
    v2 = projection - a
    d20 = jnp.sum(v2 * ab, axis=-1)
    d21 = jnp.sum(v2 * ac, axis=-1)
    denom = d00 * d11 - d01 * d01
    denom_safe = jnp.where(jnp.abs(denom) > 1e-12, denom, 1.0)

    bary_b = (d11 * d20 - d01 * d21) / denom_safe
    bary_c = (d00 * d21 - d01 * d20) / denom_safe
    inside_triangle = (
        (normal_sq > 1e-12)
        & (bary_b >= -1e-6)
        & (bary_c >= -1e-6)
        & ((bary_b + bary_c) <= 1.0 + 1e-6)
    )

    edge_distance = jnp.minimum(
        jnp.minimum(
            point_to_segment_distance(points, a, b),
            point_to_segment_distance(points, b, c),
        ),
        point_to_segment_distance(points, c, a),
    )

    return jnp.where(inside_triangle, jnp.abs(plane_distance), edge_distance)


def point_to_segment_distance(
    points: jax.Array,
    a: jax.Array,
    b: jax.Array,
) -> jax.Array:
    p = jnp.asarray(points, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    ab = b - a
    ab_sq = jnp.sum(ab * ab)
    t = jnp.where(ab_sq > 1e-12, jnp.sum((p - a) * ab, axis=-1) / ab_sq, 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return jnp.sqrt(jnp.sum((p - closest) ** 2, axis=-1))


def points_inside_halfspaces(
    points: jax.Array,
    halfspace_normals: jax.Array,
    halfspace_offsets: jax.Array,
    tolerance: float = 1e-6,
) -> jax.Array:
    p = jnp.asarray(points, dtype=jnp.float32)
    normals = jnp.asarray(halfspace_normals, dtype=jnp.float32)
    offsets = jnp.asarray(halfspace_offsets, dtype=jnp.float32)
    values = jnp.einsum("...d,hd->...h", p, normals) + offsets
    return jnp.all(values <= tolerance, axis=-1)


def signed_distance_to_convex_polyhedron(
    points: jax.Array,
    triangles: jax.Array,
    halfspace_normals: jax.Array,
    halfspace_offsets: jax.Array,
) -> jax.Array:
    p = jnp.asarray(points, dtype=jnp.float32)
    surface_distances = jax.vmap(
        lambda triangle: point_to_triangle_distance(p, triangle)
    )(triangles)
    surface_distance = jnp.min(surface_distances, axis=0)
    inside = points_inside_halfspaces(p, halfspace_normals, halfspace_offsets)
    return jnp.where(inside, -surface_distance, surface_distance)


def signed_distance_to_box_union(
    points: jax.Array,
    volume: BoxUnionVolume3D,
) -> jax.Array:
    part_distances = jax.vmap(
        signed_distance_to_convex_polyhedron,
        in_axes=(None, 0, 0, 0),
    )(
        jnp.asarray(points, dtype=jnp.float32),
        volume.triangles,
        volume.halfspace_normals,
        volume.halfspace_offsets,
    )
    return jnp.min(part_distances, axis=0)


def _read_box(box: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(box, dict):
        raise ValueError("Each box must be a dict with center and size fields.")

    center = np.asarray(box.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
    if center.shape != (3,):
        raise ValueError("Box center must have shape (3,).")

    if "half_extents" in box:
        half_extents = np.asarray(box["half_extents"], dtype=np.float32)
    elif "size" in box:
        half_extents = np.asarray(box["size"], dtype=np.float32) * 0.5
    else:
        raise ValueError("Each box must define either size or half_extents.")

    if half_extents.shape != (3,):
        raise ValueError("Box size or half_extents must have shape (3,).")
    if np.any(half_extents <= 0.0):
        raise ValueError("Box half_extents must be positive.")

    return center, half_extents


def _box_to_polyhedron(
    center: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hx, hy, hz = half_extents
    cx, cy, cz = center
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )

    face_quads = [
        (1, 2, 6, 5),
        (0, 4, 7, 3),
        (2, 3, 7, 6),
        (0, 1, 5, 4),
        (4, 5, 6, 7),
        (0, 3, 2, 1),
    ]
    triangle_indices = []
    for q0, q1, q2, q3 in face_quads:
        triangle_indices.append((q0, q1, q2))
        triangle_indices.append((q0, q2, q3))
    triangles = np.asarray(
        [[vertices[a], vertices[b], vertices[c]] for a, b, c in triangle_indices],
        dtype=np.float32,
    )

    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    max_corner = center + half_extents
    min_corner = center - half_extents
    offsets = np.array(
        [
            -max_corner[0],
            min_corner[0],
            -max_corner[1],
            min_corner[1],
            -max_corner[2],
            min_corner[2],
        ],
        dtype=np.float32,
    )
    return triangles, normals, offsets
