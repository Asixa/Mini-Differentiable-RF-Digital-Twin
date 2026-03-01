"""
RFDT - Radio Frequency Diffraction Tracer

GPU-accelerated 2D/3D electromagnetic field simulation using ray tracing
and Uniform Theory of Diffraction (UTD).
"""

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
    'Tracer',
    'Scene',
    'Field',
    'create_cube_mesh',
    'create_prism_mesh',
    'create_pentagonal_prism_mesh',
    'draw_edges',
    'draw_edges_with_normals',
    'draw_corners',
    'draw_tx',
    'draw_scene',
    'plot_field_with_edges',
    'plot_gradient_with_edges',
    'generate_circle_directions',
    'generate_sphere_directions',
    'Edge2D',
    'Corner2D',
    'VerticalEdge',
    'DiffractionPoint',
    'edge_xy',
    'corner_xy',
    'to_power_db',
    'to_numpy',
    'DEFAULT_VARIANT',
]
