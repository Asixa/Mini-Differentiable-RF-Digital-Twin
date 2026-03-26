"""Diffraction tracing using UTD with gradient-preserving GPU computation."""

import math
import drjit as dr

from .constants import DIFFRACTION_MIN_DISTANCE, EPS
from .utd import diffraction_coefficient_2d
from .rt_backend import Float, UInt32, Point3f, Vector3f, Complex2f


def _edge_attr(edge, name, default=None):
    if isinstance(edge, dict):
        return edge.get(name, default)
    return getattr(edge, name, default)


def preload_diffraction_edges(diffraction_points):
    """
    Preload all diffraction edge data to GPU arrays using DrJit Point3f/Vector3f.

    This function preserves gradient connections by concatenating DrJit arrays
    directly instead of extracting scalar values.

    Args:
        diffraction_points: List of DiffractionPoint (DrJit data from scene.py)

    Returns:
        dict with DrJit GPU arrays for all edge properties, or None if no edges
    """
    n_edges = len(diffraction_points)
    if n_edges == 0:
        return None

    # Collect valid edge DrJit arrays (preserves gradient chain)
    pos_list = []      # List of Point3f (each width=1)
    edge_dir_list = [] # List of Vector3f (each width=1)
    n0_list = []       # List of Vector3f (each width=1)
    nn_list = []       # List of Vector3f (each width=1)
    wedge_n_list = []  # List of Float (each width=1)
    valid_edges = []

    for dif_point in diffraction_points:
        face_normals = _edge_attr(dif_point, 'face_normals_3d', [])
        if len(face_normals) < 2:
            continue

        # Keep DrJit types directly (preserves gradient)
        pos = _edge_attr(dif_point, 'position')  # Point3f
        pos_list.append(pos)

        # Edge direction (normalized)
        edge_vec = _edge_attr(dif_point, 'edge_vector')  # Vector3f
        edge_len = dr.norm(edge_vec) + EPS
        edge_dir = edge_vec / edge_len
        edge_dir_list.append(edge_dir)

        # Face normals
        n0 = face_normals[0]  # Vector3f
        nn = face_normals[1]  # Vector3f
        n0_list.append(n0)
        nn_list.append(nn)

        # Wedge parameter
        wn = _edge_attr(dif_point, 'wedge_n')  # Float
        wedge_n_list.append(wn)

        valid_edges.append(dif_point)

    n_valid = len(valid_edges)
    if n_valid == 0:
        return None

    # Concatenate DrJit arrays (preserves gradient chain)
    pos_x = dr.concat([p.x for p in pos_list])
    pos_y = dr.concat([p.y for p in pos_list])
    pos_z = dr.concat([p.z for p in pos_list])
    pos_dr = Point3f(pos_x, pos_y, pos_z)

    edge_x = dr.concat([e.x for e in edge_dir_list])
    edge_y = dr.concat([e.y for e in edge_dir_list])
    edge_z = dr.concat([e.z for e in edge_dir_list])
    edge_dir_dr = Vector3f(edge_x, edge_y, edge_z)

    n0_x = dr.concat([n.x for n in n0_list])
    n0_y = dr.concat([n.y for n in n0_list])
    n0_z = dr.concat([n.z for n in n0_list])
    n0_dr = Vector3f(n0_x, n0_y, n0_z)

    nn_x = dr.concat([n.x for n in nn_list])
    nn_y = dr.concat([n.y for n in nn_list])
    nn_z = dr.concat([n.z for n in nn_list])
    nn_dr = Vector3f(nn_x, nn_y, nn_z)

    wedge_n_dr = dr.concat(wedge_n_list)

    return {
        'n_edges': n_valid,
        'pos': pos_dr,           # Point3f (N,)
        'edge_dir': edge_dir_dr, # Vector3f (N,)
        'n0': n0_dr,             # Vector3f (N,)
        'nn': nn_dr,             # Vector3f (N,)
        'wedge_n': wedge_n_dr,   # Float (N,)
        'valid_edges': valid_edges
    }


def _compute_diffraction_impl(X, Y, rx_z, tx_pos, edge_data, wavelength, k,
                               return_components=False, return_per_edge=True):
    """
    Gradient-preserving diffraction field computation using pure DrJit.

    Computes the UTD diffraction field. Gradients are propagated automatically
    through DrJit's AD system.

    Args:
        X, Y: Receiver grid coordinates (Float)
        rx_z: Receiver Z coordinate (float)
        tx_pos: Transmitter position - Point3f (gradient-preserving)
        edge_data: Preloaded edge data dict with DrJit types
        wavelength: Wavelength in meters
        k: Wave number
        return_components: If True, also return individual components

    Returns:
        total_real: Total field real part (Float)
        total_imag: Total field imaginary part (Float)
        per_edge_list: List of (real, imag) tuples per edge (Float)
        (if return_components) components: dict with utd_coeff, spreading, etc.
    """
    n_rx = dr.width(X)
    n_edges = edge_data['n_edges']
    n_pairs = n_rx * n_edges

    # Create expanded indices
    pair_idx = dr.arange(UInt32, n_pairs)
    rx_idx_exp = pair_idx // n_edges
    edge_idx_exp = pair_idx % n_edges

    # Gather receiver coordinates
    X_exp = dr.gather(Float, X, rx_idx_exp)
    Y_exp = dr.gather(Float, Y, rx_idx_exp)
    rx_pos = Point3f(X_exp, Y_exp, Float(rx_z))

    # Edge data is already DrJit arrays
    pos_dr = edge_data['pos']
    edge_dr = edge_data['edge_dir']
    n0_dr = edge_data['n0']
    wedge_n_dr = edge_data['wedge_n']

    # Gather edge data for each pair
    dif_pos = dr.gather(Point3f, pos_dr, edge_idx_exp)
    e_hat = dr.gather(Vector3f, edge_dr, edge_idx_exp)
    n0 = dr.gather(Vector3f, n0_dr, edge_idx_exp)
    wedge_n = dr.gather(Float, wedge_n_dr, edge_idx_exp)

    # Compute distances (gradient flows through these)
    # tx_pos is Point3f
    tx_to_dif = dif_pos - tx_pos
    s_prime = dr.norm(tx_to_dif) + EPS

    dif_to_rx = rx_pos - dif_pos
    s = dr.norm(dif_to_rx) + EPS

    d_total = s + s_prime

    # Face 0 tangent
    to_hat = dr.normalize(dr.cross(n0, e_hat))

    # Incident direction (TX -> dif point)
    ki = tx_to_dif / s_prime
    ki_proj = ki - dr.dot(ki, e_hat) * e_hat
    ki_proj = dr.normalize(ki_proj)

    # phi_prime (incident angle)
    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, n0)))
    phi_prime = phi_prime + dr.pi

    # Scattering direction (dif point -> RX)
    ko = dif_to_rx / s
    ko_proj = ko - dr.dot(ko, e_hat) * e_hat
    ko_proj = dr.normalize(ko_proj)

    # phi (scattering angle)
    phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat), -1.0, 1.0))
    phi = phi * (-dr.sign(dr.dot(ko_proj, n0)))
    phi = phi + dr.pi

    # Validity check
    n_pi = wedge_n * dr.pi
    valid = (phi_prime >= 0) & (phi_prime <= n_pi) & (phi >= 0) & (phi <= n_pi) & (s > DIFFRACTION_MIN_DISTANCE)

    # UTD diffraction coefficient (complex)
    D = diffraction_coefficient_2d(phi, phi_prime, wedge_n, Float(k), s, s_prime)

    # Spreading factor
    spreading = dr.rsqrt(s * s_prime * (s + s_prime) + EPS)

    # Phase term
    phase = Complex2f(0, -Float(k) * d_total)

    # Full field amplitude per pair
    lambda_factor = Float(wavelength / (4 * math.pi))
    a_pair = D * lambda_factor * spreading * dr.exp(phase)
    a_pair = dr.select(valid, a_pair, Complex2f(0, 0))

    # Sum contributions per receiver using scatter_reduce
    total_real = dr.zeros(Float, n_rx)
    total_imag = dr.zeros(Float, n_rx)
    dr.scatter_reduce(dr.ReduceOp.Add, total_real, a_pair.real, rx_idx_exp)
    dr.scatter_reduce(dr.ReduceOp.Add, total_imag, a_pair.imag, rx_idx_exp)

    # Per-edge contributions
    per_edge_list = []
    if return_per_edge:
        for e_idx in range(n_edges):
            edge_mask = (edge_idx_exp == e_idx)
            edge_real = dr.select(edge_mask, a_pair.real, Float(0))
            edge_imag = dr.select(edge_mask, a_pair.imag, Float(0))
            e_real = dr.zeros(Float, n_rx)
            e_imag = dr.zeros(Float, n_rx)
            dr.scatter_reduce(dr.ReduceOp.Add, e_real, edge_real, rx_idx_exp)
            dr.scatter_reduce(dr.ReduceOp.Add, e_imag, edge_imag, rx_idx_exp)
            per_edge_list.append((e_real, e_imag))

    if return_components:
        D_mag = dr.abs(D)
        D_mag_valid = dr.select(valid, D_mag, Float(0))
        phase_exp = dr.exp(phase)
        valid_float = dr.select(valid, Float(1.0), Float(0.0))

        components = {
            'utd_coeff': D_mag_valid,
            'spreading': spreading,
            'phase_real': phase_exp.real,
            'phase_imag': phase_exp.imag,
            'valid': valid_float,
            'd_total': d_total,
            'n_rx': n_rx,
            'n_edges': n_edges,
        }
        return total_real, total_imag, per_edge_list, components

    return total_real, total_imag, per_edge_list


def compute_diffraction_field(X, Y, rx_z, tx_pos, scene, wavelength, k,
                              return_components=False, return_per_edge=True):
    """
    Compute total diffraction field from all edges using UTD.

    This is the main entry point for diffraction computation. All gradients
    are preserved through DrJit's AD system.

    Args:
        X, Y: 2D receiver grid coordinates (Float)
        rx_z: Receiver Z coordinate (float)
        tx_pos: Transmitter position - Point3f (gradient-preserving)
        scene: Scene object (uses scene.get_edge_data for diffraction points)
        wavelength: Wavelength in meters
        k: Wave number
        return_components: If True, also return individual components
        return_per_edge: If True, also return per-edge field components

    Returns:
        total_real: Total field real part (Float, gradient-enabled)
        total_imag: Total field imaginary part (Float)
        per_edge_list: List of (real, imag) tuples per edge (Float)
        (if return_components) components: dict with utd_coeff, spreading, etc.
    """
    n_rx = dr.width(X)

    # Get edge data from scene
    edge_cache = scene.get_edge_data(rx_z)
    diffraction_points = edge_cache['diffraction_points']

    if len(diffraction_points) == 0:
        zero_real = dr.zeros(Float, n_rx)
        zero_imag = dr.zeros(Float, n_rx)
        if return_components:
            return zero_real, zero_imag, [], None
        return zero_real, zero_imag, []

    # Use preloaded edge data from cache
    data = edge_cache.get('edge_data') or preload_diffraction_edges(diffraction_points)
    if data is None:
        zero_real = dr.zeros(Float, n_rx)
        zero_imag = dr.zeros(Float, n_rx)
        if return_components:
            return zero_real, zero_imag, [], None
        return zero_real, zero_imag, []

    return _compute_diffraction_impl(X, Y, rx_z, tx_pos, data, wavelength, k,
                                     return_components=return_components,
                                     return_per_edge=return_per_edge)
