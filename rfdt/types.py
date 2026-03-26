"""Typed structures for RFDT geometry and tracing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

from .rt_backend import Float, Point3f, Vector2f, Vector3f


class Edge2D(NamedTuple):
    p0: Vector2f
    p1: Vector2f
    normal: Vector2f
    name: str


class Corner2D(NamedTuple):
    position: Vector2f
    face0_point: Vector2f
    face_n_point: Vector2f
    name: str
    edge_info: Optional["DiffractionPoint"]


@dataclass
class VerticalEdge:
    vertex_indices: Tuple[int, int]
    p0: Point3f
    p1: Point3f
    adjacent_faces: Sequence[int]
    is_boundary: bool
    edge_vector: Vector3f
    length: Float
    normal_2d: Optional[Vector2f] = None
    wedge_n: Optional[Float] = None
    face_normals_3d: Optional[List[Vector3f]] = None


@dataclass(frozen=True)
class DiffractionPoint:
    position: Point3f
    edge_vector: Vector3f
    length: Float
    wedge_n: Float
    face_normals_3d: Sequence[Vector3f]
