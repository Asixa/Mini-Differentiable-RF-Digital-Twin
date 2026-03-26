"""Utility functions for DrJit operations."""
import drjit as dr
import numpy as np

from .constants import EPS, POWER_DB_FLOOR
from .rt_backend import Float, Vector2f, Vector3f, Point3f


# -----------------------------------------------------------------------------
# Power conversion
# -----------------------------------------------------------------------------

def to_power_db(a, imag=None):
    """Convert complex field to power in dB (gradient-preserving).

    Args:
        a: Complex2f, OR Float real part (if imag is provided)
        imag: Float imaginary part (optional, for separate real/imag input)

    Returns:
        Power in dB (Float)
    """
    log10 = dr.log(Float(10))
    if imag is not None:
        # Separate real/imag input
        power = a * a + imag * imag
    else:
        # Complex2f input
        power = dr.squared_norm(a)
    return 10 * dr.log(power + POWER_DB_FLOOR) / log10


# -----------------------------------------------------------------------------
# Conversion utilities
# -----------------------------------------------------------------------------

def to_numpy(drjit_array):
    """Convert DrJit/torch array to numpy array."""
    if hasattr(drjit_array, 'numpy'):
        return np.array(drjit_array)
    if hasattr(drjit_array, 'torch'):
        return drjit_array.torch().cpu().numpy()
    return np.asarray(drjit_array)


def to_numpy_2d(drjit_array, grid_size):
    """Convert DrJit array to numpy (grid_size x grid_size)."""
    return np.array(drjit_array).reshape(grid_size, grid_size)


def to_numpy_complex_2d(drjit_complex, grid_size):
    """Convert DrJit Complex2f to numpy complex array."""
    real = np.array(drjit_complex.real).reshape(grid_size, grid_size)
    imag = np.array(drjit_complex.imag).reshape(grid_size, grid_size)
    return real + 1j * imag


# -----------------------------------------------------------------------------
# DrJit type utilities
# -----------------------------------------------------------------------------

def scalar(v):
    """Extract scalar value from DrJit type.

    Works with both scalar and array types.

    Args:
        v: DrJit value (Float, float, etc.)

    Returns:
        Python float
    """
    if hasattr(v, '__len__') and dr.width(v) > 0:
        return float(np.array(v).flat[0])
    return float(v)


def edge_xy(edge):
    """Extract 2D coordinates from Edge2D.

    Args:
        edge: Edge2D with p0 and p1 points

    Returns:
        tuple: (p0_x, p0_y, p1_x, p1_y)
    """
    return scalar(edge.p0.x), scalar(edge.p0.y), scalar(edge.p1.x), scalar(edge.p1.y)


def corner_xy(corner):
    """Extract 2D coordinates from Corner2D.

    Args:
        corner: Corner2D with position point

    Returns:
        tuple: (x, y)
    """
    return scalar(corner.position.x), scalar(corner.position.y)
