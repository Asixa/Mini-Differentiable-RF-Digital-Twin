"""Typed structures for RFDT geometry and tracing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

import mitsuba as mi


class Edge2D(NamedTuple):
    p0: mi.Vector2f
    p1: mi.Vector2f
    normal: mi.Vector2f
    name: str


class Corner2D(NamedTuple):
    position: mi.Vector2f
    face0_point: mi.Vector2f
    face_n_point: mi.Vector2f
    name: str
    edge_info: Optional["DiffractionPoint"]


@dataclass
class VerticalEdge:
    vertex_indices: Tuple[int, int]
    p0: mi.Point3f
    p1: mi.Point3f
    adjacent_faces: Sequence[int]
    is_boundary: bool
    edge_vector: mi.Vector3f
    length: mi.Float
    normal_2d: Optional[mi.Vector2f] = None
    wedge_n: Optional[mi.Float] = None
    face_normals_3d: Optional[List[mi.Vector3f]] = None


@dataclass(frozen=True)
class DiffractionPoint:
    position: mi.Point3f
    edge_vector: mi.Vector3f
    length: mi.Float
    wedge_n: mi.Float
    face_normals_3d: Sequence[mi.Vector3f]
