"""Line-of-Sight (LoS) tracing for radio propagation simulation"""

import drjit as dr

from .constants import RAY_EPS
from .rt_backend import Float, Point3f, Complex2f, ray_intersect


def los_blocked(rt_scene, tx_pos, rx_positions):
    """
    Use ray tracing to detect LoS occlusion.

    Args:
        rt_scene: Scene object from rt_backend.build_scene
        tx_pos: Transmitter position - Point3f (gradient-preserving)
        rx_positions: Receiver positions - Point3f array

    Returns:
        blocked: Boolean array, True means blocked
    """
    ray_dir = rx_positions - tx_pos
    ray_length = dr.norm(ray_dir)
    ray_dir_normalized = ray_dir / ray_length

    with dr.suspend_grad():
        si = ray_intersect(rt_scene, tx_pos, ray_dir_normalized)
        blocked = si.is_valid() & (si.t < ray_length - RAY_EPS)

    return blocked


def compute_los_field(scene, X, Y, rx_z, tx_pos, wavelength, k):
    """
    Compute LoS field with mesh occlusion using ray tracing.

    Args:
        scene: Scene object (uses scene.rt_scene for ray tracing)
        X, Y: 2D receiver grid coordinates (Float)
        rx_z: Receiver Z coordinate (Float)
        tx_pos: Transmitter position - Point3f (gradient-preserving)
        wavelength: Wavelength in meters
        k: Wave number

    Returns:
        a_los: Complex field amplitude (Complex2f)
    """
    rx_positions = Point3f(X, Y, rx_z)
    los_blocked_mask = los_blocked(scene.rt_scene, tx_pos, rx_positions)

    dx = X - tx_pos.x
    dy = Y - tx_pos.y
    dz = rx_z - tx_pos.z
    d_los = dr.sqrt(dx * dx + dy * dy + dz * dz)

    los_coeff = Float(wavelength) / (4 * dr.pi * d_los)
    los_phase = Complex2f(0, -Float(k) * d_los)
    a_los = los_coeff * dr.exp(los_phase)
    a_los = dr.select(los_blocked_mask, Complex2f(0, 0), a_los)

    return a_los
