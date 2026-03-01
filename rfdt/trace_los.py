"""Line-of-Sight (LoS) tracing for radio propagation simulation"""

import drjit as dr
import mitsuba as mi

from .constants import RAY_EPS


def los_blocked(mi_scene, tx_pos, rx_positions):
    """
    Use Mitsuba ray tracing to detect LoS occlusion.

    Args:
        mi_scene: Mitsuba scene object
        tx_pos: Transmitter position - mi.Point3f (gradient-preserving)
        rx_positions: Receiver positions - Mitsuba Point3f array

    Returns:
        blocked: Boolean array (mi.Bool), True means blocked
    """
    # Create rays: from tx to each rx point (tx_pos is mi.Point3f)
    ray_dir = rx_positions - tx_pos

    # Normalize direction and compute distance
    ray_length = dr.norm(ray_dir)
    ray_dir_normalized = ray_dir / ray_length

    # Build rays
    rays = mi.Ray3f(o=tx_pos, d=ray_dir_normalized)

    # Intersection test (no grad needed - just visibility check)
    with dr.suspend_grad():
        si = mi_scene.ray_intersect(rays)
        # Check for occlusion: if intersection distance < rx distance, then blocked
        blocked = si.is_valid() & (si.t < ray_length - RAY_EPS)

    return blocked


def compute_los_field(scene, X, Y, rx_z, tx_pos, wavelength, k):
    """
    Compute LoS field with mesh occlusion using ray tracing.

    Args:
        scene: Scene object (uses scene.mi_scene for ray tracing)
        X, Y: 2D receiver grid coordinates (Mitsuba Float)
        rx_z: Receiver Z coordinate (Mitsuba Float)
        tx_pos: Transmitter position - mi.Point3f (gradient-preserving)
        wavelength: Wavelength in meters
        k: Wave number

    Returns:
        a_los: Complex field amplitude (Mitsuba Complex2f)
    """
    # Build 3D receiver positions
    rx_positions = mi.Point3f(X, Y, rx_z)

    # Use ray tracing to detect occlusion
    los_blocked_mask = los_blocked(scene.mi_scene, tx_pos, rx_positions)

    # Compute LoS field (3D distance) - tx_pos is mi.Point3f
    dx = X - tx_pos.x
    dy = Y - tx_pos.y
    dz = rx_z - tx_pos.z
    d_los = dr.sqrt(dx * dx + dy * dy + dz * dz)

    los_coeff = mi.Float(wavelength) / (4 * dr.pi * d_los)
    los_phase = mi.Complex2f(0, -mi.Float(k) * d_los)
    a_los = los_coeff * dr.exp(los_phase)
    a_los = dr.select(los_blocked_mask, mi.Complex2f(0, 0), a_los)

    return a_los
