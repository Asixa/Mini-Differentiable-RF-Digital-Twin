"""RayD-backed runtime adapter and DrJit type aliases for RFDT.

This is the only module that imports RayD or backend-specific DrJit types.
All other modules should import runtime aliases and intersection helpers from
here so the ray-tracing backend stays encapsulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
import drjit.cuda.ad as cuda_ad
import rayd


# Backend-resolved DrJit type aliases.
Float = cuda_ad.Float
UInt32 = cuda_ad.UInt32
Int32 = cuda_ad.Int32
Bool = cuda_ad.Bool
Point2f = cuda_ad.Array2f
Point3f = cuda_ad.Array3f
Vector2f = cuda_ad.Array2f
Vector3f = cuda_ad.Array3f
Vector3u = cuda_ad.Array3u
Complex2f = cuda_ad.Complex2f
Matrix4f = cuda_ad.Matrix4f


class Transform4f:
    """Compatibility placeholder for the not-yet-migrated optimize path."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "The legacy Transform4f helper is not available in the RayD runtime. "
            "optimize.py has not been migrated yet."
        )


@dataclass(frozen=True)
class RayIntersection:
    """Minimal surface-interaction wrapper used by the RFDT tracing code."""

    valid: Bool
    t: Float
    p: Point3f
    n: Vector3f
    geo_n: Vector3f
    prim_index: Int32
    prim_id: Int32
    shape_id: Int32

    def is_valid(self):
        return self.valid


def _ray_width(origins, directions) -> int:
    return max(dr.width(origins.x), dr.width(directions.x))


def _zeros_point3(width: int) -> Point3f:
    return Point3f(
        dr.zeros(Float, width),
        dr.zeros(Float, width),
        dr.zeros(Float, width),
    )


def _invalid_intersection(origins, directions) -> RayIntersection:
    width = _ray_width(origins, directions)
    zeros = _zeros_point3(width)
    invalid = dr.full(Bool, False, width)
    inf_t = dr.full(Float, float("inf"), width)
    neg_one = dr.full(Int32, -1, width)
    return RayIntersection(
        valid=invalid,
        t=inf_t,
        p=zeros,
        n=Vector3f(zeros.x, zeros.y, zeros.z),
        geo_n=Vector3f(zeros.x, zeros.y, zeros.z),
        prim_index=neg_one,
        prim_id=neg_one,
        shape_id=neg_one,
    )


def build_scene(vertices, faces):
    """Build a dynamic RayD scene from mesh vertices and face indices.

    Returns
    -------
    scene : rayd.Scene | None
    params : rayd.Scene | None
        Reused as the mutable runtime handle for vertex updates.
    vertex_key : int | None
        Mesh slot index for update_mesh_vertices.
    n_verts : int
    """
    n_verts = dr.width(vertices)
    n_faces = dr.width(faces)

    if n_verts == 0 or n_faces == 0:
        return None, None, None, n_verts

    mesh = rayd.Mesh(vertices, faces)
    scene = rayd.Scene()
    scene.add_mesh(mesh, dynamic=True)
    scene.build()
    return scene, scene, 0, n_verts


def update_vertices(scene_params, vertex_key, vertices):
    """Update vertex positions in an existing RayD scene."""
    if scene_params is None or vertex_key is None:
        return

    scene_params.update_mesh_vertices(int(vertex_key), vertices)
    scene_params.sync()


def ray_intersect(scene, origins, directions, active=None):
    """Cast rays against the RayD scene and return an RFDT-style wrapper."""
    if scene is None:
        return _invalid_intersection(origins, directions)

    ray = rayd.Ray(origins, directions)
    raw = scene.intersect(
        ray,
        active=True if active is None else active,
        flags=rayd.RayFlags.All,
    )

    prim_id = Int32(raw.prim_id)
    return RayIntersection(
        valid=Bool(raw.is_valid()),
        t=Float(raw.t),
        p=Point3f(raw.p.x, raw.p.y, raw.p.z),
        n=Vector3f(raw.n.x, raw.n.y, raw.n.z),
        geo_n=Vector3f(raw.geo_n.x, raw.geo_n.y, raw.geo_n.z),
        prim_index=prim_id,
        prim_id=prim_id,
        shape_id=Int32(raw.shape_id),
    )


def create_adam(lr):
    """Compatibility placeholder for the not-yet-migrated optimize path."""
    del lr
    raise NotImplementedError(
        "The legacy Adam helper is not available in the RayD runtime. "
        "optimize.py has not been migrated yet."
    )


def set_log_level_warn():
    """No-op compatibility helper kept for optimize.py."""


def register_sampler_seed(seed):
    """No-op compatibility helper kept for optimize.py."""
    del seed


__all__ = [
    "Bool",
    "Complex2f",
    "Float",
    "Int32",
    "Matrix4f",
    "Point2f",
    "Point3f",
    "RayIntersection",
    "Transform4f",
    "UInt32",
    "Vector2f",
    "Vector3f",
    "Vector3u",
    "build_scene",
    "create_adam",
    "ray_intersect",
    "register_sampler_seed",
    "set_log_level_warn",
    "update_vertices",
]
