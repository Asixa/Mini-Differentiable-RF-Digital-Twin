"""
RFDT - Radio Frequency Diffraction Tracer

GPU-accelerated 2D/3D electromagnetic field simulation using ray tracing
and Uniform Theory of Diffraction (UTD).
"""

# rt_backend is imported first to expose the backend-resolved DrJit aliases.
from .rt_backend import (  # noqa: F401 - re-export type aliases
    Float, UInt32, Int32, Bool,
    Point2f, Point3f, Vector2f, Vector3f, Vector3u, Complex2f,
)

from .tracer import Tracer
from .scene import Scene
from .field import Field
from .scene import create_cube_mesh, create_prism_mesh, create_pentagonal_prism_mesh
from .visualization import (
    draw_edges,
    draw_edges_with_normals,
    draw_corners,
    draw_tx,
    draw_scene,
    plot_field_with_edges,
    plot_gradient_with_edges,
)
from .raygen import generate_circle_directions, generate_sphere_directions
from .types import Corner2D, DiffractionPoint, Edge2D, VerticalEdge
from .utils import edge_xy, corner_xy, to_power_db, to_numpy
from .constants import DEFAULT_VARIANT

__all__ = [
    # Core
    'Tracer',
    'Scene',
    'Field',
    # Mesh constructors
    'create_cube_mesh',
    'create_prism_mesh',
    'create_pentagonal_prism_mesh',
    # Visualization
    'draw_edges',
    'draw_edges_with_normals',
    'draw_corners',
    'draw_tx',
    'draw_scene',
    'plot_field_with_edges',
    'plot_gradient_with_edges',
    # Ray generation
    'generate_circle_directions',
    'generate_sphere_directions',
    # Types
    'Edge2D',
    'Corner2D',
    'VerticalEdge',
    'DiffractionPoint',
    # DrJit type aliases (from rt_backend)
    'Float', 'UInt32', 'Int32', 'Bool',
    'Point2f', 'Point3f', 'Vector2f', 'Vector3f', 'Vector3u', 'Complex2f',
    # Utilities
    'edge_xy',
    'corner_xy',
    'to_power_db',
    'to_numpy',
    'DEFAULT_VARIANT',
]
