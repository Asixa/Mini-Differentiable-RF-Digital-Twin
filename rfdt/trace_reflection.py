"""Reflection tracing using ray tracing + image method"""

import math
import drjit as dr

from .constants import BARY_EPS, EPS, RAY_ORIGIN_BIAS, SMALL_EPS
from .raygen import generate_sphere_directions, generate_circle_directions
from .rt_backend import (
    Float, UInt32, Bool, Point3f, Vector3f, Complex2f,
    ray_intersect,
)


def _point_in_triangle_3d(p, v0, v1, v2):
    """
    DrJit vectorized point-in-triangle test using barycentric coordinates.
    Uses Point3f for cleaner vector operations.

    Args:
        p: Test point (Point3f)
        v0, v1, v2: Triangle vertices (Point3f)

    Returns:
        Bool: True if point is inside triangle
    """
    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    vp = p - v0

    # Dot products for barycentric coordinates
    dot00 = dr.dot(edge1, edge1)
    dot01 = dr.dot(edge1, edge2)
    dot02 = dr.dot(edge1, vp)
    dot11 = dr.dot(edge2, edge2)
    dot12 = dr.dot(edge2, vp)

    # Barycentric coordinates
    inv_denom = dr.rcp(dot00 * dot11 - dot01 * dot01 + BARY_EPS)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Point is inside if u >= 0, v >= 0, u + v <= 1
    epsilon = Float(SMALL_EPS)
    return (u >= -epsilon) & (v >= -epsilon) & (u + v <= 1 + epsilon)


def _dda_loop_body(state, params):
    """DDA loop body for dr.while_loop - processes one step for all rays."""
    # Unpack state (all per-ray variables that get compressed together)
    (cur_x, cur_y, t, t_max_x, t_max_y, step_count, loop_active,
     blocker_dist, step_x, step_y, dt_x, dt_y,
     prev_refl_p, prev_refl_n, prev_tx, prev_ampl, prev_prim_idx) = state

    # Unpack params (read-only, fixed size or scalars)
    (x_min, x_max, y_min, y_max, cell_size_x, cell_size_y, nx,
     x_coords_dr, y_coords_dr, rx_z, wavelength, k,
     validate_paths, has_mesh_data, n_triangles, tri_v0, tri_v1, tri_v2,
     grid, result_real, result_imag, result_count, bounce_idx) = params

    # Check bounds
    in_bounds = loop_active & (cur_x >= x_min) & (cur_x < x_max) & \
                (cur_y >= y_min) & (cur_y < y_max) & (t < blocker_dist)

    # Get cell index
    cell_idx = grid.pos_to_idx(cur_x, cur_y)

    # Get cell center coordinates
    cell_ix = UInt32(cell_idx % nx)
    cell_iy = UInt32(cell_idx // nx)
    cell_x = dr.gather(Float, x_coords_dr, cell_ix)
    cell_y = dr.gather(Float, y_coords_dr, cell_iy)

    # Compute mirror source
    d_to_plane = dr.dot(prev_tx - prev_refl_p, prev_refl_n)
    mirror = prev_tx - 2 * d_to_plane * prev_refl_n

    # Distance from cell center to mirror source
    cell_pos = Point3f(cell_x, cell_y, Float(rx_z))
    d_mirror = dr.norm(cell_pos - mirror)

    # Path validation
    valid_path = in_bounds
    if validate_paths and has_mesh_data:
        dir_to_rx = cell_pos - mirror
        denom = dr.dot(dir_to_rx, prev_refl_n)
        t_intersect = dr.dot(prev_refl_p - mirror, prev_refl_n) / (denom + EPS)
        int_p = mirror + t_intersect * dir_to_rx

        v0 = dr.gather(Point3f, tri_v0, prev_prim_idx)
        v1 = dr.gather(Point3f, tri_v1, prev_prim_idx)
        v2 = dr.gather(Point3f, tri_v2, prev_prim_idx)
        in_main = _point_in_triangle_3d(int_p, v0, v1, v2)

        # Heuristic: use adjacent triangle when faces are split into pairs.
        sibling_idx = prev_prim_idx ^ UInt32(1)
        sibling_idx = dr.select(sibling_idx < n_triangles, sibling_idx, prev_prim_idx)
        sv0 = dr.gather(Point3f, tri_v0, sibling_idx)
        sv1 = dr.gather(Point3f, tri_v1, sibling_idx)
        sv2 = dr.gather(Point3f, tri_v2, sibling_idx)
        in_sibling = _point_in_triangle_3d(int_p, sv0, sv1, sv2)

        t_valid = (t_intersect > 0) & (t_intersect < 1)
        denom_valid = dr.abs(denom) > EPS
        valid_path = in_bounds & (in_main | in_sibling) & t_valid & denom_valid

    # Compute field contribution
    fspl = Float(wavelength) / (4.0 * math.pi * dr.maximum(d_mirror, Float(0.01)))
    field_amplitude = prev_ampl * fspl
    phase = -Float(k) * d_mirror

    real_contrib = field_amplitude * dr.cos(phase)
    imag_contrib = field_amplitude * dr.sin(phase)

    # Scatter-reduce to result arrays (AD compatible)
    dr.scatter_reduce(dr.ReduceOp.Add, result_real[bounce_idx],
                      dr.select(valid_path, real_contrib, Float(0.0)),
                      cell_idx, valid_path)
    dr.scatter_reduce(dr.ReduceOp.Add, result_imag[bounce_idx],
                      dr.select(valid_path, imag_contrib, Float(0.0)),
                      cell_idx, valid_path)
    dr.scatter_reduce(dr.ReduceOp.Add, result_count[bounce_idx],
                      dr.select(valid_path, Float(1.0), Float(0.0)),
                      cell_idx, valid_path)

    # DDA step
    move_x = t_max_x < t_max_y
    new_t = dr.select(move_x, t_max_x, t_max_y)
    new_cur_x = dr.select(move_x, cur_x + step_x, cur_x)
    new_cur_y = dr.select(~move_x, cur_y + step_y, cur_y)
    new_t_max_x = dr.select(move_x, t_max_x + dt_x, t_max_x)
    new_t_max_y = dr.select(~move_x, t_max_y + dt_y, t_max_y)

    # Update loop active status
    new_loop_active = in_bounds

    return (new_cur_x, new_cur_y, new_t, new_t_max_x, new_t_max_y,
            step_count + 1, new_loop_active,
            blocker_dist, step_x, step_y, dt_x, dt_y,
            prev_refl_p, prev_refl_n, prev_tx, prev_ampl, prev_prim_idx)


def _compute_reflection_field_impl(grid, rx_z, tx_pos, rt_scene, wavelength, k,
                                    n_rays, max_reflections, mode, reflection_coef,
                                    return_per_bounce, validate_paths,
                                    tri_data, grid_data=None):
    """
    Reflection field computation using DrJit with GPU-parallel DDA traversal.
    Uses dr.while_loop for GPU-native loop execution.

    Args:
        grid: Field object with bounds, size, pos_to_idx method
        rx_z: Receiver Z coordinate (scalar float)
        tx_pos: Transmitter position - Point3f (gradient-preserving)
        rt_scene: RayD scene object
        wavelength: Wavelength in meters
        k: Wave number (2*pi/lambda)
        n_rays: Number of rays to emit
        max_reflections: Max reflection bounces
        mode: '2d' (circle) or '3d' (sphere) ray directions
        reflection_coef: Amplitude loss per bounce
        return_per_bounce: Return per-bounce contributions
        validate_paths: Enable path validation (secondary check)
        tri_data: Preloaded triangle data dict with v0x,v0y,v0z,v1x,...
        grid_data: Optional preloaded grid coordinates dict with x_coords, y_coords

    Returns:
        a_ref_total: Total reflection field (Complex2f)
        a_ref_list: List of per-bounce fields
    """
    n_rx = grid.n_cells
    (x_min, x_max), (y_min, y_max) = grid.bounds
    cell_size_x, cell_size_y = grid.cell_size
    nx, ny = grid.size
    max_steps = 2 * (nx + ny)

    # Use preloaded grid coordinates if available, otherwise compute
    if grid_data is not None:
        x_coords_dr = grid_data['x_coords']
        y_coords_dr = grid_data['y_coords']
    else:
        # Pure DrJit linspace
        x_step = (x_max - x_min) / (nx - 1) if nx > 1 else 0
        y_step = (y_max - y_min) / (ny - 1) if ny > 1 else 0
        idx_x = dr.arange(Float, nx)
        idx_y = dr.arange(Float, ny)
        x_coords_dr = Float(x_min) + idx_x * Float(x_step)
        y_coords_dr = Float(y_min) + idx_y * Float(y_step)

    # Use preloaded triangle data (Point3f format)
    has_mesh_data = tri_data is not None
    if has_mesh_data:
        n_triangles = tri_data['n_triangles']
        tri_v0 = tri_data['v0']
        tri_v1 = tri_data['v1']
        tri_v2 = tri_data['v2']
    else:
        n_triangles = 0
        tri_v0 = tri_v1 = tri_v2 = None

    # Initialize result buffers (always use list for dr.while_loop compatibility)
    results_real = [dr.zeros(Float, n_rx) for _ in range(max_reflections)]
    results_imag = [dr.zeros(Float, n_rx) for _ in range(max_reflections)]
    results_count = [dr.zeros(Float, n_rx) for _ in range(max_reflections)]

    # Generate ray directions (pure DrJit, returns Vector3f)
    if mode == '2d':
        ray_dir = generate_circle_directions(n_rays)
    else:
        ray_dir = generate_sphere_directions(n_rays)

    # Initialize ray state (GPU arrays) using Point3f/Vector3f
    # tx_pos is Point3f - broadcast to n_rays using dr.repeat (preserves gradients)
    ray_origin = Point3f(
        dr.repeat(tx_pos.x, n_rays),
        dr.repeat(tx_pos.y, n_rays),
        dr.repeat(tx_pos.z, n_rays)
    )

    active = dr.full(Bool, True, n_rays)
    amplitude = dr.ones(Float, n_rays)

    # Previous reflection data for image method (stored per ray)
    prev_refl_p = Point3f(
        dr.zeros(Float, n_rays),
        dr.zeros(Float, n_rays),
        dr.zeros(Float, n_rays)
    )
    prev_refl_n = Vector3f(
        dr.zeros(Float, n_rays),
        dr.zeros(Float, n_rays),
        dr.zeros(Float, n_rays)
    )
    prev_tx = Point3f(
        dr.repeat(tx_pos.x, n_rays),
        dr.repeat(tx_pos.y, n_rays),
        dr.repeat(tx_pos.z, n_rays)
    )
    prev_ampl = dr.ones(Float, n_rays)
    prev_prim_idx = dr.zeros(UInt32, n_rays)

    for bounce in range(max_reflections + 1):
        if not dr.any(active):
            break

        # Ray-mesh intersection (no grad needed - just finding hit locations)
        # AD path is preserved by recomputing intersection from triangle vertices below
        with dr.suspend_grad():
            si = ray_intersect(rt_scene, ray_origin, ray_dir, active)
            hit = si.is_valid() & active
            blocker_dist = dr.select(hit, si.t, Float(1e10))
        # Derive normals/points from triangle data to keep AD path to vertices (e.g., rotation)
        if has_mesh_data:
            prim_idx = dr.select(hit, UInt32(si.prim_index), UInt32(0))
            v0 = dr.gather(Point3f, tri_v0, prim_idx)
            v1 = dr.gather(Point3f, tri_v1, prim_idx)
            v2 = dr.gather(Point3f, tri_v2, prim_idx)
            geom_n = dr.cross(v1 - v0, v2 - v0)
            geom_n = geom_n / (dr.norm(geom_n) + EPS)
            denom = dr.dot(ray_dir, geom_n)
            t_hit = dr.dot(v0 - ray_origin, geom_n) / (denom + EPS)
            si_p = ray_origin + t_hit * ray_dir
            si_n = geom_n
        else:
            si_p = si.p
            si_n = si.n

        # First bounce: just find the reflection point
        if bounce == 0:
            active = hit
            if not dr.any(active):
                break

        # GPU DDA traversal using dr.while_loop (after first bounce)
        if bounce > 0:
            dr.eval(blocker_dist)

            # DDA initialization - fully vectorized
            t = dr.zeros(Float, n_rays)
            dt_x = dr.abs(Float(cell_size_x) / dr.maximum(dr.abs(ray_dir.x), Float(EPS)))
            dt_y = dr.abs(Float(cell_size_y) / dr.maximum(dr.abs(ray_dir.y), Float(EPS)))

            cur_x = ray_origin.x
            cur_y = ray_origin.y

            step_x = dr.select(ray_dir.x > 0, Float(cell_size_x), Float(-cell_size_x))
            step_y = dr.select(ray_dir.y > 0, Float(cell_size_y), Float(-cell_size_y))

            next_x = dr.select(ray_dir.x > 0,
                               (dr.floor((cur_x - x_min) / cell_size_x) + 1) * cell_size_x + x_min,
                               dr.floor((cur_x - x_min) / cell_size_x) * cell_size_x + x_min)
            next_y = dr.select(ray_dir.y > 0,
                               (dr.floor((cur_y - y_min) / cell_size_y) + 1) * cell_size_y + y_min,
                               dr.floor((cur_y - y_min) / cell_size_y) * cell_size_y + y_min)

            t_max_x = dr.abs((next_x - cur_x) / dr.maximum(dr.abs(ray_dir.x), Float(EPS)))
            t_max_y = dr.abs((next_y - cur_y) / dr.maximum(dr.abs(ray_dir.y), Float(EPS)))

            # Initial loop active state
            loop_active = active & (cur_x >= x_min) & (cur_x < x_max) & \
                          (cur_y >= y_min) & (cur_y < y_max) & (t < blocker_dist)

            # Pack parameters for the loop body (read-only, fixed size or scalars)
            params = (Float(x_min), Float(x_max), Float(y_min), Float(y_max),
                      Float(cell_size_x), Float(cell_size_y), nx,
                      x_coords_dr, y_coords_dr, rx_z, wavelength, k,
                      validate_paths, has_mesh_data, n_triangles, tri_v0, tri_v1, tri_v2,
                      grid, results_real, results_imag, results_count, bounce - 1)

            # Initial state (all per-ray variables that get compressed together)
            state = (cur_x, cur_y, t, t_max_x, t_max_y,
                     UInt32(0), loop_active,
                     blocker_dist, step_x, step_y, dt_x, dt_y,
                     prev_refl_p, prev_refl_n, prev_tx, prev_ampl, prev_prim_idx)

            # Define condition and body functions
            def loop_cond(cur_x, cur_y, t, t_max_x, t_max_y, step_count, loop_active,
                          blocker_dist, step_x, step_y, dt_x, dt_y,
                          prev_refl_p, prev_refl_n, prev_tx, prev_ampl, prev_prim_idx):
                return loop_active & (step_count < max_steps)

            def loop_body(cur_x, cur_y, t, t_max_x, t_max_y, step_count, loop_active,
                          blocker_dist, step_x, step_y, dt_x, dt_y,
                          prev_refl_p, prev_refl_n, prev_tx, prev_ampl, prev_prim_idx):
                return _dda_loop_body(
                    (cur_x, cur_y, t, t_max_x, t_max_y, step_count, loop_active,
                     blocker_dist, step_x, step_y, dt_x, dt_y,
                     prev_refl_p, prev_refl_n, prev_tx, prev_ampl, prev_prim_idx),
                    params
                )

            # Execute DDA loop using dr.while_loop
            # Use evaluated mode without compress for AD compatibility
            # (compress=True breaks gradient chain, symbolic mode can't gather from symbolic vars)
            dr.while_loop(
                state=state,
                cond=loop_cond,
                body=loop_body,
                mode='evaluated',
                compress=False
            )

        # Update ray state for next bounce
        if not dr.any(hit):
            break

        # Update amplitude with reflection coefficient
        amplitude = dr.select(hit, amplitude * (-reflection_coef), amplitude)
        active = hit

        # Store current reflection info as "previous" for next iteration
        prev_refl_p = dr.select(hit, si_p, prev_refl_p)
        prev_refl_n = dr.select(hit, si_n, prev_refl_n)
        prev_tx = dr.select(hit, ray_origin, prev_tx)
        prev_ampl = dr.select(hit, amplitude, prev_ampl)
        prev_prim_idx = dr.select(hit, UInt32(si.prim_index), prev_prim_idx)

        # Reflection direction: r = d - 2*(d.n)*n
        dot_dn = dr.dot(ray_dir, si_n)
        ray_dir = ray_dir - 2.0 * dot_dn * si_n

        # Offset ray origin slightly to avoid self-intersection
        ray_origin = si.p + ray_dir * RAY_ORIGIN_BIAS

    # Average by contribution count
    # Batch evaluate all arrays at once
    all_arrays = []
    for i in range(max_reflections):
        all_arrays.extend([results_real[i], results_imag[i], results_count[i]])
    dr.eval(*all_arrays)

    # Compute averages for each bounce
    a_ref_list = []
    for i in range(max_reflections):
        mask = results_count[i] > 0
        avg_real = dr.select(mask, results_real[i] / results_count[i], Float(0.0))
        avg_imag = dr.select(mask, results_imag[i] / results_count[i], Float(0.0))
        a_ref = Complex2f(avg_real, avg_imag)
        a_ref_list.append(a_ref)

    # Sum for total
    a_ref_total = Complex2f(0, 0)
    for a_ref in a_ref_list:
        a_ref_total = a_ref_total + a_ref

    if return_per_bounce:
        return a_ref_total, a_ref_list
    else:
        return a_ref_total, []


def compute_reflection_field(grid, rx_z, tx_pos, scene, wavelength, k,
                             n_rays=1000, max_reflections=2,
                             mode='2d', reflection_coef=0.7,
                             return_per_bounce=False,
                             validate_paths=True,
                             allow_empty_scene=True,
                             grid_data=None):
    """
    Compute reflection field using Monte Carlo + Image Method.

    Args:
        grid: Field object with bounds, size, pos_to_idx method
        rx_z: Receiver Z coordinate (scalar float)
        tx_pos: Transmitter position - Point3f (gradient-preserving)
        scene: Scene object (uses scene.rt_scene for ray tracing)
        wavelength: Wavelength in meters
        k: Wave number (2*pi/lambda)
        n_rays: Number of rays to emit (default 1000)
        max_reflections: Max reflection bounces (default 2)
        mode: '2d' (circle) or '3d' (sphere) ray directions
        reflection_coef: Amplitude loss per bounce (default 0.7)
        return_per_bounce: Return per-bounce contributions
        validate_paths: Enable path validation (default True)
        allow_empty_scene: Return zeros when scene is None (default True)
        grid_data: Preloaded grid coordinates dict

    Returns:
        a_ref_total: Total reflection field (Complex2f)
        a_ref_list: List of per-bounce fields (if return_per_bounce=True)
    """
    if scene is None:
        if not allow_empty_scene:
            raise ValueError("scene is None; set allow_empty_scene=True to return zeros.")
        n_rx = grid.n_cells
        zero_real = dr.zeros(Float, n_rx)
        zero_imag = dr.zeros(Float, n_rx)
        zero_field = Complex2f(zero_real, zero_imag)
        if return_per_bounce:
            return zero_field, [zero_field] * max_reflections
        else:
            return zero_field, []

    return _compute_reflection_field_impl(
        grid=grid,
        rx_z=rx_z,
        tx_pos=tx_pos,
        rt_scene=scene.rt_scene,
        wavelength=wavelength,
        k=k,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        reflection_coef=reflection_coef,
        return_per_bounce=return_per_bounce,
        validate_paths=validate_paths,
        tri_data=scene.tri_data_gpu,
        grid_data=grid_data
    )
