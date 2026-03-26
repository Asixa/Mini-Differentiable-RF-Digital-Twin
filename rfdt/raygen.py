"""Ray direction generation for GPU-accelerated ray tracing."""

import drjit as dr

from .rt_backend import Float, Vector3f

# Golden ratio for Fibonacci lattice
_PHI = (1 + 5 ** 0.5) / 2


def generate_sphere_directions(n_rays: int):
    """
    Generate uniformly distributed directions on a sphere using Fibonacci lattice.
    Pure DrJit implementation (no PyTorch dependency).

    Args:
        n_rays: Number of ray directions to generate

    Returns:
        Vector3f: Unit direction vectors
    """
    indices = dr.arange(Float, n_rays)
    phi = dr.acos(1 - 2 * (indices + 0.5) / n_rays)
    theta = dr.pi * (1 + _PHI) * indices

    sin_phi = dr.sin(phi)
    dx = sin_phi * dr.cos(theta)
    dy = sin_phi * dr.sin(theta)
    dz = dr.cos(phi)

    return Vector3f(dx, dy, dz)


def generate_circle_directions(n_rays: int):
    """
    Generate uniformly distributed directions on a circle (2D, z=0).
    Pure DrJit implementation (no PyTorch dependency).

    Args:
        n_rays: Number of ray directions to generate

    Returns:
        Vector3f: Unit direction vectors (z component is always 0)
    """
    # Create angles from 0 to 2*pi (exclusive of endpoint)
    step = 2 * dr.pi / n_rays
    theta = dr.arange(Float, n_rays) * step

    dx = dr.cos(theta)
    dy = dr.sin(theta)
    dz = dr.zeros(Float, n_rays)

    return Vector3f(dx, dy, dz)
