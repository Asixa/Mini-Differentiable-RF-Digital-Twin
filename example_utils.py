"""Shared utilities for example notebooks."""
import os

import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi
import drjit as dr

from rfdt.constants import DEFAULT_VARIANT

mi.set_variant(DEFAULT_VARIANT)

from rfdt import (
    create_cube_mesh, Tracer, Scene,
    draw_scene, draw_tx, draw_edges_with_normals, draw_corners,
    to_power_db, to_numpy
)
from rfdt.utils import to_numpy_2d


# ---------------------------------------------------------------------------
# Field computation
# ---------------------------------------------------------------------------

def compute_field(center, size, freq, tx_pos, range_x, range_y, grid_size,
                  rotation=None, n_rays=1000000, max_reflections=1, reflection_coef=1.0):
    """Compute total field for a cube mesh.

    Args:
        center: Cube center - tuple or mi.Point3f (pass mi.Point3f for AD)
        size: Cube side length
        freq: Signal frequency in Hz
        tx_pos: Transmitter position - tuple or mi.Point3f (pass mi.Point3f for AD)
        range_x, range_y: Grid bounds
        grid_size: Grid resolution
        rotation: Z-axis rotation in radians (mi.Float for AD, float/None otherwise)
        n_rays: Number of reflection rays
        max_reflections: Maximum reflection bounces
        reflection_coef: Reflection coefficient

    Returns:
        result: Trace result dict
        scene: Scene object
    """
    if isinstance(center, tuple):
        center_pt = mi.Point3f(*center)
    else:
        center_pt = center

    vertices, faces = create_cube_mesh(center=center_pt, size=size, rotation=rotation)
    scene = Scene(vertices, faces)
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_reflections,
        reflection_coef=reflection_coef
    )
    result = tracer.trace(
        tx_pos=tx_pos,
        grid_size=grid_size,
        range_x=range_x,
        range_y=range_y
    )
    return result, scene


# ---------------------------------------------------------------------------
# AD / FD gradient helpers
# ---------------------------------------------------------------------------

def compute_ad_gradient(param_setup_fn, grid_size, target_field='a_tot'):
    """Compute |da/dp| using DrJit forward AD.

    |da/dp| = sqrt((d(Re)/dp)^2 + (d(Im)/dp)^2)

    Args:
        param_setup_fn: Callable returning (result, scene). Must create a
            differentiable parameter, enable grad, set tangent, and call
            compute_field internally. Called twice (once per component).
        grid_size: Grid resolution (for zero fallback)
        target_field: Which field to differentiate ('a_tot', 'a_ref', etc.)

    Returns:
        result, scene, grad_mag (numpy array of |da/dp|)
    """
    def compute_component_grad(target_part):
        result, scene = param_setup_fn()
        a = result[target_field]
        field = a.real if target_part == 'real' else a.imag
        dr.forward_to(field)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(grid_size * grid_size)

    result, scene, grad_re = compute_component_grad('real')
    _, _, grad_im = compute_component_grad('imag')
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result, scene, grad_mag


def compute_fd_gradient(base_fn, perturbed_fn, grid_size, delta=0.01, target_field='a_tot'):
    """Compute |da/dp| using finite differences.

    |da/dp| = sqrt((d(Re)/dp)^2 + (d(Im)/dp)^2)

    Args:
        base_fn: Callable returning (result, scene) for base parameters
        perturbed_fn: Callable returning (result, scene) for perturbed parameters
        grid_size: Grid resolution
        delta: Perturbation size
        target_field: Which field to differentiate ('a_tot', 'a_ref', etc.)

    Returns:
        result_base, scene, grad_mag (numpy array of |da/dp|)
    """
    result_base, scene = base_fn()
    a_base = result_base[target_field]
    re_base = to_numpy(a_base.real)
    im_base = to_numpy(a_base.imag)

    result_perturbed, _ = perturbed_fn()
    a_perturbed = result_perturbed[target_field]
    re_perturbed = to_numpy(a_perturbed.real)
    im_perturbed = to_numpy(a_perturbed.imag)

    grad_re = (re_perturbed - re_base) / delta
    grad_im = (im_perturbed - im_base) / delta
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result_base, scene, grad_mag


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def prepare_gradient_visualization(result, grad_ad_mag, grad_fd_mag, grid_size,
                                   target_field='a_tot'):
    """Convert raw gradient arrays to 2D dB-scale arrays for visualization.

    Returns:
        field_db, grad_ad_db, grad_fd_db  (all 2D numpy arrays)
    """
    field_db = to_numpy(to_power_db(result[target_field])).reshape(grid_size, grid_size)
    grad_ad_db = 20 * np.log10(grad_ad_mag.reshape(grid_size, grid_size) + 1e-20)
    grad_fd_db = 20 * np.log10(grad_fd_mag.reshape(grid_size, grid_size) + 1e-20)
    return field_db, grad_ad_db, grad_fd_db


def get_edges(scene, result):
    """Extract 2D edges from scene at the calculation height."""
    edge_cache = scene.get_edge_data(result['calculation_height'])
    return edge_cache['edges_2d']


def plot_three_panel(field_db, grad_ad_db, grad_fd_db, edges, tx_pos, range_x, range_y,
                     titles, suptitle, save_path=None,
                     field_vmin=-60, field_vmax=-20, grad_vmin=-80, grad_vmax=-20,
                     fontsize=11):
    """Plot 1x3 panel: Total Field | AD Gradient | FD Gradient.

    Args:
        field_db: 2D array of field power in dB
        grad_ad_db: 2D array of AD gradient in dB
        grad_fd_db: 2D array of FD gradient in dB
        edges: Edge list for scene overlay
        tx_pos: Transmitter position tuple
        range_x, range_y: Plot bounds
        titles: List of 3 subplot titles
        suptitle: Figure super-title
        save_path: Path to save figure (None to skip)
        field_vmin/vmax: Colorbar range for field panel
        grad_vmin/vmax: Colorbar range for gradient panels
        fontsize: Font size for subplot titles

    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]

    im1 = axes[0].imshow(field_db, extent=extent, origin='lower', cmap='jet',
                          vmin=field_vmin, vmax=field_vmax)
    draw_scene(axes[0], edges, tx_pos, range_x, range_y)
    axes[0].set_title(titles[0], fontsize=fontsize)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(grad_ad_db, extent=extent, origin='lower', cmap='RdBu_r',
                          vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[1], edges, tx_pos, range_x, range_y)
    axes[1].set_title(titles[1], fontsize=fontsize)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    im3 = axes[2].imshow(grad_fd_db, extent=extent, origin='lower', cmap='RdBu_r',
                          vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[2], edges, tx_pos, range_x, range_y)
    axes[2].set_title(titles[2], fontsize=fontsize)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Figure saved to {save_path}")

    return fig


def print_gradient_summary(name, grad_ad_mag, grad_fd_mag):
    """Print gradient comparison summary."""
    print("\n" + "=" * 60)
    print(f"Summary: {name}")
    print("=" * 60)
    print(f"  AD sum: {np.sum(grad_ad_mag):.2f}")
    print(f"  FD sum: {np.sum(grad_fd_mag):.2f}")
    print(f"  Difference: {abs(np.sum(grad_ad_mag) - np.sum(grad_fd_mag)):.2f}")
    if np.sum(grad_fd_mag) > 0:
        print(f"  Relative diff: "
              f"{abs(np.sum(grad_ad_mag) - np.sum(grad_fd_mag)) / np.sum(grad_fd_mag) * 100:.2f}%")
    if np.sum(grad_ad_mag) > 0:
        print("\n[OK] Gradients are working!")
    else:
        print("\n[WARN] AD gradients are zero - check gradient propagation")


# ---------------------------------------------------------------------------
# Forward simulation visualization (from forward.py)
# ---------------------------------------------------------------------------

def plot_mesh_2d(result, scene, frequency, range_x=(-8, 8), range_y=(-8, 8)):
    """Plot 2D field heatmap with all components (2x3 layout).

    Args:
        result: dict returned from Tracer.trace()
        scene: Scene object (for edges/corners)
        frequency: Signal frequency in Hz
        range_x, range_y: Plot bounds

    Returns:
        fig: matplotlib figure
    """
    grid_size = result['grid_size']
    calculation_height = result['calculation_height']

    edge_cache = scene.get_edge_data(calculation_height)
    edges = edge_cache['edges_2d']
    corners = edge_cache['corners_2d']
    n_edges = len(edges)
    n_corners = len(corners)

    def reshape_2d(arr):
        return to_numpy(arr).reshape(grid_size, grid_size)

    total_db = to_power_db(result['a_tot'])
    los_db = to_power_db(result['a_los'])
    ref_db = to_power_db(result['a_ref'])
    dif_db = to_power_db(result['a_dif'])

    a_ref_dif = mi.Complex2f(
        result['a_ref'].real + result['a_dif'].real,
        result['a_ref'].imag + result['a_dif'].imag
    )
    ref_dif_db = to_power_db(a_ref_dif)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmax = -20
    vmin = vmax - 40
    tx_pos = result['tx_pos_3d']

    def plot_field(ax, data, title):
        im = ax.imshow(reshape_2d(data),
                       extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                       origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        draw_tx(ax, tx_pos)
        draw_edges_with_normals(ax, edges)
        draw_corners(ax, corners)
        ax.set_xlim(range_x)
        ax.set_ylim(range_y)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.7)

    plot_field(axes[0, 0], total_db, 'Total Field (dB)')
    plot_field(axes[0, 1], los_db, 'Direct (LoS) (dB)')
    plot_field(axes[0, 2], ref_db, 'Reflection (dB)')

    plot_field(axes[1, 0], dif_db, 'Diffraction (dB)')
    plot_field(axes[1, 1], ref_dif_db, 'Reflection + Diffraction (dB)')
    axes[1, 2].axis('off')

    fig.suptitle(f'UTD Field Simulation - {frequency/1e9:.1f} GHz\n'
                 f'Height: {calculation_height:.1f}m, '
                 f'Edges: {n_edges}, Corners: {n_corners}', fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reflection gradient helpers (from grad_reflection.py)
# ---------------------------------------------------------------------------

def compute_reflection_ad_gradient(center_coords, size, freq, tx_pos, range_x, range_y,
                                   grid_size, n_rays, max_reflections, reflection_coef,
                                   grad_axis=0, rotation=None):
    """Compute reflection field and its gradient using DrJit forward AD.

    Args:
        center_coords: tuple (cx, cy, cz)
        grad_axis: 0 for x, 1 for y, 2 for z
        rotation: Z-axis rotation in radians or None

    Returns:
        dict with keys: ref_db, ref_grad_db, ref_real_grad, ref_imag_grad, result, scene
    """
    cx, cy, cz = center_coords

    if grad_axis == 0:
        tangent = mi.Vector3f(1.0, 0.0, 0.0)
    elif grad_axis == 1:
        tangent = mi.Vector3f(0.0, 1.0, 0.0)
    else:
        tangent = mi.Vector3f(0.0, 0.0, 1.0)

    def run_trace_and_get_grad(target='ref_db'):
        center = mi.Point3f(cx, cy, cz)
        dr.enable_grad(center)
        dr.set_grad(center, tangent)

        vertices, faces = create_cube_mesh(center=center, size=size, rotation=rotation)
        scene = Scene(vertices, faces)
        tracer = Tracer(
            frequency=freq,
            scene=scene,
            reflection_n_rays=n_rays,
            reflection_max_bounces=max_reflections,
            reflection_coef=reflection_coef
        )
        result = tracer.trace(
            tx_pos=tx_pos,
            grid_size=grid_size,
            range_x=range_x,
            range_y=range_y
        )

        if target == 'ref_db':
            ref_db = to_power_db(result['a_ref'])
            dr.forward_to(ref_db)
            grad = dr.grad(ref_db)
            return result, scene, grad, ref_db
        elif target == 'ref_real':
            dr.forward_to(result['a_ref'].real)
            grad = dr.grad(result['a_ref'].real)
            return result, scene, grad, None
        elif target == 'ref_imag':
            dr.forward_to(result['a_ref'].imag)
            grad = dr.grad(result['a_ref'].imag)
            return result, scene, grad, None

    result, scene, ref_grad, ref_db = run_trace_and_get_grad('ref_db')
    _, _, ref_real_grad, _ = run_trace_and_get_grad('ref_real')
    _, _, ref_imag_grad, _ = run_trace_and_get_grad('ref_imag')

    return {
        'ref_db': to_numpy_2d(ref_db, grid_size),
        'ref_grad_db': to_numpy_2d(ref_grad, grid_size),
        'ref_real_grad': to_numpy_2d(ref_real_grad, grid_size),
        'ref_imag_grad': to_numpy_2d(ref_imag_grad, grid_size),
        'result': result,
        'scene': scene
    }


def compute_reflection_fd_gradient(center_np, size, freq, tx_pos, range_x, range_y,
                                   grid_size, n_rays, max_reflections, reflection_coef,
                                   grad_axis=0, delta=0.01, rotation=None):
    """Compute reflection field gradient using finite difference.

    Args:
        center_np: numpy array [x, y, z]
        grad_axis: 0 for x, 1 for y, 2 for z
        delta: Perturbation size
        rotation: Z-axis rotation in radians or None

    Returns:
        dict with keys: ref_db, ref_grad_db, ref_real_grad, ref_imag_grad, result, scene
    """
    def _compute(c_np):
        center = mi.Point3f(float(c_np[0]), float(c_np[1]), float(c_np[2]))
        vertices, faces = create_cube_mesh(center=center, size=size, rotation=rotation)
        scene = Scene(vertices, faces)
        tracer = Tracer(
            frequency=freq,
            scene=scene,
            reflection_n_rays=n_rays,
            reflection_max_bounces=max_reflections,
            reflection_coef=reflection_coef
        )
        result = tracer.trace(
            tx_pos=tx_pos,
            grid_size=grid_size,
            range_x=range_x,
            range_y=range_y
        )
        return result, scene

    result_base, scene_base = _compute(center_np)
    ref_db_base = to_numpy_2d(to_power_db(result_base['a_ref']), grid_size)
    ref_real_base = to_numpy_2d(result_base['a_ref'].real, grid_size)
    ref_imag_base = to_numpy_2d(result_base['a_ref'].imag, grid_size)

    center_perturbed = center_np.copy()
    center_perturbed[grad_axis] += delta
    result_perturbed, _ = _compute(center_perturbed)
    ref_db_perturbed = to_numpy_2d(to_power_db(result_perturbed['a_ref']), grid_size)
    ref_real_perturbed = to_numpy_2d(result_perturbed['a_ref'].real, grid_size)
    ref_imag_perturbed = to_numpy_2d(result_perturbed['a_ref'].imag, grid_size)

    return {
        'ref_db': ref_db_base,
        'ref_grad_db': (ref_db_perturbed - ref_db_base) / delta,
        'ref_real_grad': (ref_real_perturbed - ref_real_base) / delta,
        'ref_imag_grad': (ref_imag_perturbed - ref_imag_base) / delta,
        'result': result_base,
        'scene': scene_base
    }
