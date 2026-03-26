"""3D Scene and Mesh Processing for UTD Ray Tracing (Pure DrJit)"""

import drjit as dr
from collections import defaultdict

from .constants import EDGE_2D_EPS, EPS, SMALL_EPS
from .types import Corner2D, DiffractionPoint, Edge2D, VerticalEdge
from .utils import scalar
from .trace_diffraction import preload_diffraction_edges
from .rt_backend import (
    Float, UInt32, Point3f, Point2f, Vector2f, Vector3f, Vector3u,
    build_scene, update_vertices as rt_update_vertices,
)


class Scene:
    """
    Geometry container that owns the RayD scene, edge caches, and mesh data.
    Assumes mesh topology is fixed; only vertex positions move.
    """

    def __init__(self, vertices, faces, vertical_ratio: float = 0.7):
        self.vertices = vertices
        self.faces = faces
        self.vertical_ratio = vertical_ratio
        self._mesh_version = 0

        # Topology caches (static)
        edge_to_faces = extract_edges_with_adjacency(self.vertices, self.faces)
        self._edge_topology = sorted(edge_to_faces.items(), key=lambda x: x[0])

        # RayD scene + params for updates
        self.rt_scene, self._scene_params, self._scene_vertex_key, self._n_verts = self._build_rt_scene(
            self.vertices, self.faces
        )

        # Dynamic caches
        self._edge_cache = {}
        self._build_vertical_edges()
        self._preload_triangle_data()

    # ------------------------------------------------------------------ #
    # Mesh + runtime helpers
    # ------------------------------------------------------------------ #
    def _compute_mesh_centers(self, vertices):
        n_verts = dr.width(vertices)
        mesh_center_3d = Point3f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts,
            dr.sum(vertices.z) / n_verts
        )
        mesh_center_2d = Vector2f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts
        )
        return n_verts, mesh_center_3d, mesh_center_2d

    def _build_rt_scene(self, vertices, faces):
        return build_scene(vertices, faces)

    def _build_vertical_edges(self):
        self._n_verts, self._mesh_center_3d, self._mesh_center_2d = self._compute_mesh_centers(self.vertices)

        face_normals = compute_face_normals(
            self.vertices,
            self.faces,
            mesh_center_3d=self._mesh_center_3d,
            n_verts=self._n_verts
        )
        vertical_edges_raw = filter_vertical_edges(
            self.vertices,
            self._edge_topology,
            self.vertical_ratio
        )

        self.vertical_edges = []
        for edge_info in vertical_edges_raw:
            compute_edge_geometry(
                edge_info,
                self.vertices,
                self.faces,
                mesh_center_3d=self._mesh_center_3d,
                mesh_center_2d=self._mesh_center_2d,
                n_verts=self._n_verts,
                face_normals=face_normals
            )
            self.vertical_edges.append(edge_info)

    def _preload_triangle_data(self):
        n_triangles = dr.width(self.faces)
        if n_triangles == 0:
            self.tri_data_gpu = None
            return

        v0_idx = self.faces.x
        v1_idx = self.faces.y
        v2_idx = self.faces.z

        v0 = dr.gather(Point3f, self.vertices, v0_idx)
        v1 = dr.gather(Point3f, self.vertices, v1_idx)
        v2 = dr.gather(Point3f, self.vertices, v2_idx)

        self.tri_data_gpu = {
            'v0': v0,
            'v1': v1,
            'v2': v2,
            'n_triangles': n_triangles
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update_vertices(self, vertices, recompute_edges: bool = True):
        rt_update_vertices(self._scene_params, self._scene_vertex_key, vertices)

        self.vertices = vertices
        self._mesh_version += 1

        self._edge_cache.clear()
        self._preload_triangle_data()
        if recompute_edges:
            self._build_vertical_edges()

    def get_edge_data(self, calculation_height):
        cache_key = (self._mesh_version, calculation_height)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]

        edges_2d, corners_2d = project_to_2d(self.vertical_edges, calculation_height, self.vertices)

        diffraction_points = []
        for corner_pos, face0_point, face_n_point, name, edge_info in corners_2d:
            if edge_info is not None:
                diffraction_points.append(edge_info)

        edge_data = preload_diffraction_edges(diffraction_points)

        cache_entry = {
            'edge_data': edge_data,
            'edges_2d': edges_2d,
            'corners_2d': corners_2d,
            'diffraction_points': diffraction_points
        }
        self._edge_cache[cache_key] = cache_entry
        return cache_entry

    def get_edges_2d(self, height: float) -> list:
        """Get 2D edge projections at given height."""
        return self.get_edge_data(height)['edges_2d']

    def get_corners_2d(self, height: float) -> list:
        """Get 2D corner projections at given height."""
        return self.get_edge_data(height)['corners_2d']

    @property
    def n_vertical_edges(self) -> int:
        """Number of vertical edges in the mesh."""
        return len(self.vertical_edges)


def create_cube_mesh(center=None, size=None, rotation=None):
    """
    Create cube mesh using DrJit (differentiable).

    Args:
        center: Cube center position (x, y, z) - Point3f (single point) or tuple/list
        size: Cube side length - Float (scalar) or float
        rotation: Rotation angle around Z-axis in radians - Float (scalar) or float
                  If None, no rotation is applied.

    Returns:
        vertices: Point3f with 8 vertices (differentiable, SoA format)
        faces: Vector3u with 12 triangle faces (topology, not differentiable)
    """
    # Default values
    if center is None:
        center = Point3f(0.0, 0.0, 1.0)
    elif isinstance(center, Point3f):
        pass  # Already correct type
    else:
        # tuple/list/array input
        center = Point3f(float(center[0]), float(center[1]), float(center[2]))

    if size is None:
        size = Float(4.0)
    elif not isinstance(size, Float):
        size = Float(float(size))

    # Handle rotation parameter
    if rotation is not None and not isinstance(rotation, Float):
        rotation = Float(float(rotation))

    # Extract components (preserves gradient if center has gradient enabled)
    cx = center.x
    cy = center.y
    cz = center.z
    h = size / 2

    # 8 vertices in local coordinates (relative to center)
    # Bottom face: 0-3, Top face: 4-7
    local_x = dr.concat([-h, h, h, -h, -h, h, h, -h])
    local_y = dr.concat([-h, -h, h, h, -h, -h, h, h])
    local_z = dr.concat([-h, -h, -h, -h, h, h, h, h])

    # Apply rotation around Z-axis if specified
    if rotation is not None:
        cos_r = dr.cos(rotation)
        sin_r = dr.sin(rotation)
        # Rotate x, y coordinates: x' = x*cos - y*sin, y' = x*sin + y*cos
        rotated_x = local_x * cos_r - local_y * sin_r
        rotated_y = local_x * sin_r + local_y * cos_r
        local_x = rotated_x
        local_y = rotated_y

    # Translate to world coordinates
    vertices = Point3f(
        cx + local_x,
        cy + local_y,
        cz + local_z
    )

    # 12 triangle faces (topology, no gradient needed)
    # Winding order: counterclockwise when viewed from outside (outward normal)
    faces = Vector3u(
        # v0 indices
        UInt32(0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1),
        # v1 indices
        UInt32(2, 3, 5, 6, 1, 5, 3, 7, 7, 4, 2, 6),
        # v2 indices
        UInt32(1, 2, 6, 7, 5, 4, 7, 6, 3, 7, 6, 5)
    )

    return vertices, faces


def create_prism_mesh(n_sides=5, center=None, radius=None, height=None, rotation=None):
    """
    Create regular polygonal prism mesh using DrJit (differentiable).

    Args:
        n_sides: Number of sides (3=triangle, 4=square, 5=pentagon, 6=hexagon, etc.)
        center: Prism center position (x, y, z) - Point3f or tuple/list
        radius: Polygon circumradius - Float or float
        height: Prism height - Float or float
        rotation: Rotation angle around Z-axis in radians - Float or float
                  If None, no rotation is applied.

    Returns:
        vertices: Point3f with 2*n_sides vertices (differentiable, SoA format)
        faces: Vector3u with triangle faces (topology, not differentiable)
    """
    import math


    if n_sides < 3:
        raise ValueError("n_sides must be at least 3")

    # Default values
    if center is None:
        center = Point3f(0.0, 0.0, 1.0)
    elif isinstance(center, Point3f):
        pass
    else:
        center = Point3f(float(center[0]), float(center[1]), float(center[2]))

    if radius is None:
        radius = Float(2.0)
    elif not isinstance(radius, Float):
        radius = Float(float(radius))

    if height is None:
        height = Float(4.0)
    elif not isinstance(height, Float):
        height = Float(float(height))

    # Handle rotation parameter
    if rotation is not None and not isinstance(rotation, Float):
        rotation = Float(float(rotation))

    cx = center.x
    cy = center.y
    cz = center.z
    h = height / 2

    # Generate base vertex angles (counterclockwise in XY plane)
    # Rotated -90 degrees so first vertex is at top
    base_angles = [i * 2 * math.pi / n_sides - math.pi / 2 for i in range(n_sides)]

    # Build local vertex coordinate arrays (relative to center, before rotation)
    local_x_list = []
    local_y_list = []
    local_z_list = []

    # Bottom n_sides vertices (indices 0 to n_sides-1)
    for angle in base_angles:
        local_x_list.append(radius * math.cos(angle))
        local_y_list.append(radius * math.sin(angle))
        local_z_list.append(-h)

    # Top n_sides vertices (indices n_sides to 2*n_sides-1)
    for angle in base_angles:
        local_x_list.append(radius * math.cos(angle))
        local_y_list.append(radius * math.sin(angle))
        local_z_list.append(h)

    # Concatenate into DrJit arrays
    local_x = dr.concat(local_x_list)
    local_y = dr.concat(local_y_list)
    local_z = dr.concat(local_z_list)

    # Apply rotation around Z-axis if specified (preserves gradient)
    if rotation is not None:
        cos_r = dr.cos(rotation)
        sin_r = dr.sin(rotation)
        # Rotate x, y coordinates: x' = x*cos - y*sin, y' = x*sin + y*cos
        rotated_x = local_x * cos_r - local_y * sin_r
        rotated_y = local_x * sin_r + local_y * cos_r
        local_x = rotated_x
        local_y = rotated_y

    # Translate to world coordinates
    vertices = Point3f(
        cx + local_x,
        cy + local_y,
        cz + local_z
    )

    # Build faces (topology, no gradient needed)
    v0_list = []
    v1_list = []
    v2_list = []

    # Bottom face (n_sides-2 triangles, fan from vertex 0)
    for i in range(1, n_sides - 1):
        v0_list.append(0)
        v1_list.append(i + 1)
        v2_list.append(i)

    # Top face (n_sides-2 triangles, fan from vertex n_sides)
    for i in range(1, n_sides - 1):
        v0_list.append(n_sides)
        v1_list.append(n_sides + i)
        v2_list.append(n_sides + i + 1)

    # Side faces (n_sides quads = 2*n_sides triangles)
    for i in range(n_sides):
        next_i = (i + 1) % n_sides
        # First triangle
        v0_list.append(i)
        v1_list.append(next_i)
        v2_list.append(next_i + n_sides)
        # Second triangle
        v0_list.append(i)
        v1_list.append(next_i + n_sides)
        v2_list.append(i + n_sides)

    faces = Vector3u(
        UInt32(*v0_list),
        UInt32(*v1_list),
        UInt32(*v2_list)
    )

    return vertices, faces


def create_pentagonal_prism_mesh(center=None, radius=None, height=None, rotation=None):
    """
    Create pentagonal prism mesh (convenience wrapper for create_prism_mesh).

    Args:
        center: Prism center position (x, y, z) - Point3f or tuple/list
        radius: Pentagon circumradius - Float or float
        height: Prism height - Float or float
        rotation: Rotation angle around Z-axis in radians - Float or float

    Returns:
        vertices: Point3f with 10 vertices (differentiable, SoA format)
        faces: Vector3u with 16 triangle faces (topology, not differentiable)
    """
    return create_prism_mesh(n_sides=5, center=center, radius=radius, height=height, rotation=rotation)


def extract_edges_with_adjacency(vertices, faces):
    """
    Extract all edges and their adjacent faces from triangle mesh.

    Args:
        faces: Vector3u with M triangle faces (SoA format)

    Returns:
        edge_to_faces: dict, {(v0_idx, v1_idx): [face_idx_list]}
                       where v0_idx < v1_idx for uniqueness
    """
    edge_to_faces = defaultdict(list)

    f0 = faces.x
    f1 = faces.y
    f2 = faces.z
    n_faces = dr.width(f0)

    for face_idx in range(n_faces):
        v0 = int(f0[face_idx])
        v1 = int(f1[face_idx])
        v2 = int(f2[face_idx])
        # Three edges (ensure v_min < v_max for deduplication)
        edges_in_face = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        for edge in edges_in_face:
            edge_to_faces[edge].append(face_idx)

    return edge_to_faces


def compute_face_normals(vertices, faces, mesh_center_3d=None, n_verts=None):
    """
    Compute oriented face normals (DrJit) once and reuse across edges.
    """
    f0 = faces.x
    f1 = faces.y
    f2 = faces.z
    n_faces = dr.width(f0)

    if n_verts is None:
        n_verts = dr.width(vertices)
    if mesh_center_3d is None:
        mesh_center_3d = Point3f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts,
            dr.sum(vertices.z) / n_verts
        )

    # Vectorized gather per-face
    face_idx = dr.arange(UInt32, n_faces)
    va = dr.gather(Point3f, vertices, dr.gather(UInt32, f0, face_idx))
    vb = dr.gather(Point3f, vertices, dr.gather(UInt32, f1, face_idx))
    vc = dr.gather(Point3f, vertices, dr.gather(UInt32, f2, face_idx))

    normal = dr.cross(vb - va, vc - va)
    norm_len = dr.norm(normal) + EPS
    normal = normal / norm_len

    centroid = (va + vb + vc) / 3
    outward_vec = centroid - mesh_center_3d
    normal = dr.select(dr.dot(normal, outward_vec) < 0, -normal, normal)

    return normal


def filter_vertical_edges(vertices, edge_to_faces, vertical_ratio=0.7):
    """
    Filter edges that are primarily vertical (DrJit, preserves gradient).

    Args:
        vertices: Point3f with N vertices (SoA format)
        edge_to_faces: Edge to faces mapping dict
        vertical_ratio: Vertical component threshold, |z_component| / length > vertical_ratio
                        Default 0.7 corresponds to about 45 degrees

    Returns:
        vertical_edges: list of VerticalEdge
    """
    vertical_edges = []

    for edge_key, face_list in edge_to_faces:
        v0_idx, v1_idx = edge_key

        idx0 = UInt32(v0_idx)
        idx1 = UInt32(v1_idx)
        p0 = dr.gather(Point3f, vertices, idx0)
        p1 = dr.gather(Point3f, vertices, idx1)
        edge_vec = p1 - p0
        edge_len = dr.norm(edge_vec) + EPS

        z_component = dr.abs(edge_vec.z)
        vertical_ratio_val = z_component / edge_len

        is_valid = (edge_len > SMALL_EPS) & (vertical_ratio_val > vertical_ratio)
        if bool(is_valid):
            vertical_edges.append(VerticalEdge(
                vertex_indices=(v0_idx, v1_idx),
                p0=p0,
                p1=p1,
                adjacent_faces=tuple(face_list),
                is_boundary=len(face_list) == 1,
                edge_vector=edge_vec,
                length=edge_len
            ))

    return vertical_edges


def compute_edge_geometry(edge_info, vertices, faces,
                          mesh_center_3d=None, mesh_center_2d=None, n_verts=None,
                          face_normals=None):
    """
    Compute edge normals and wedge angle (DrJit, preserves gradient).

    Args:
        edge_info: VerticalEdge (from filter_vertical_edges)
        vertices: Point3f with N vertices
        faces: Vector3u with M faces
        mesh_center_3d: Optional cached mesh center in 3D
        mesh_center_2d: Optional cached mesh center in 2D
        n_verts: Optional cached vertex count
        face_normals: Optional precomputed face normals (Vector3f array)

    Returns:
        edge_info: VerticalEdge with normal_2d, wedge_n, face_normals_3d set
    """
    import math


    v0_idx, v1_idx = edge_info.vertex_indices
    face_indices = edge_info.adjacent_faces

    # Get vertices using dr.gather (preserves gradient)
    idx0 = UInt32(v0_idx)
    idx1 = UInt32(v1_idx)
    p0 = dr.gather(Point3f, vertices, idx0)
    p1 = dr.gather(Point3f, vertices, idx1)
    edge_vec = p1 - p0

    # Compute mesh centers if not provided
    if n_verts is None:
        n_verts = dr.width(vertices)
    if mesh_center_3d is None:
        mesh_center_3d = Point3f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts,
            dr.sum(vertices.z) / n_verts
        )
    if mesh_center_2d is None:
        mesh_center_2d = Vector2f(
            dr.sum(vertices.x) / n_verts,
            dr.sum(vertices.y) / n_verts
        )

    f0 = faces.x
    f1 = faces.y
    f2 = faces.z

    # Compute face normals
    face_normals_3d = []
    for face_idx in face_indices:
        if face_normals is not None:
            idx = UInt32(int(face_idx))
            normal = Vector3f(
                dr.gather(Float, face_normals.x, idx),
                dr.gather(Float, face_normals.y, idx),
                dr.gather(Float, face_normals.z, idx),
            )
            face_normals_3d.append(normal)
        else:
            idx = UInt32(int(face_idx))
            va_idx = dr.gather(UInt32, f0, idx)
            vb_idx = dr.gather(UInt32, f1, idx)
            vc_idx = dr.gather(UInt32, f2, idx)

            va = dr.gather(Point3f, vertices, va_idx)
            vb = dr.gather(Point3f, vertices, vb_idx)
            vc = dr.gather(Point3f, vertices, vc_idx)

            normal = dr.cross(vb - va, vc - va)
            norm_len = dr.norm(normal) + EPS
            normal = normal / norm_len

            face_centroid = (va + vb + vc) / 3
            outward_vec_3d = face_centroid - mesh_center_3d
            dot_val = dr.dot(normal, outward_vec_3d)
            normal = dr.select(dot_val < 0, -normal, normal)

            face_normals_3d.append(normal)

    # Project to 2D normal
    if len(face_normals_3d) == 2:
        avg_normal_3d = (face_normals_3d[0] + face_normals_3d[1]) / 2
        normal_2d = Vector2f(avg_normal_3d.x, avg_normal_3d.y)
    else:
        normal_2d = Vector2f(face_normals_3d[0].x, face_normals_3d[0].y)

    # Normalize 2D normal
    norm_2d_len = dr.norm(normal_2d) + EPS
    normal_2d = dr.select(norm_2d_len > EPS, normal_2d / norm_2d_len, Vector2f(0, 0))

    # Ensure normal points outward (using mesh center)
    edge_mid_2d = Vector2f((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
    outward_vec = edge_mid_2d - mesh_center_2d

    outward_norm = dr.norm(outward_vec)
    normal_norm = dr.norm(normal_2d)
    use_flip = (outward_norm > EPS) & (normal_norm > EPS)
    dot_2d = dr.dot(normal_2d, outward_vec)
    normal_2d = dr.select(use_flip & (dot_2d < 0), -normal_2d, normal_2d)

    # Compute wedge angle
    if len(face_normals_3d) == 2:
        n0_candidate = face_normals_3d[0]
        n1_candidate = face_normals_3d[1]

        edge_vec_normalized = edge_vec / (dr.norm(edge_vec) + EPS)

        # Config 1: n0 = n0_candidate, nn = n1_candidate
        to_hat_1 = dr.cross(n0_candidate, edge_vec_normalized)
        tn_hat_1 = dr.cross(n1_candidate, edge_vec_normalized)

        # Config 2: n0 = n1_candidate, nn = n0_candidate
        to_hat_2 = dr.cross(n1_candidate, edge_vec_normalized)
        tn_hat_2 = dr.cross(n0_candidate, edge_vec_normalized)

        # Normalize
        to_hat_1 = to_hat_1 / (dr.norm(to_hat_1) + EPS)
        tn_hat_1 = tn_hat_1 / (dr.norm(tn_hat_1) + EPS)
        to_hat_2 = to_hat_2 / (dr.norm(to_hat_2) + EPS)
        tn_hat_2 = tn_hat_2 / (dr.norm(tn_hat_2) + EPS)

        # Compute rotation angles
        cross_1 = dr.cross(to_hat_1, tn_hat_1)
        dot_1 = dr.dot(to_hat_1, tn_hat_1)
        sign_1 = dr.sign(dr.dot(cross_1, edge_vec_normalized))
        angle_1 = dr.atan2(sign_1 * dr.norm(cross_1), dot_1)
        angle_1 = dr.select(angle_1 < 0, angle_1 + 2 * dr.pi, angle_1)

        cross_2 = dr.cross(to_hat_2, tn_hat_2)
        dot_2 = dr.dot(to_hat_2, tn_hat_2)
        sign_2 = dr.sign(dr.dot(cross_2, edge_vec_normalized))
        angle_2 = dr.atan2(sign_2 * dr.norm(cross_2), dot_2)
        angle_2 = dr.select(angle_2 < 0, angle_2 + 2 * dr.pi, angle_2)

        # Choose configuration closer to 270 deg (1.5 pi)
        target_angle = 1.5 * math.pi
        choose_first = dr.abs(angle_1 - target_angle) < dr.abs(angle_2 - target_angle)
        n0 = dr.select(choose_first, n0_candidate, n1_candidate)
        n1 = dr.select(choose_first, n1_candidate, n0_candidate)

        # Interior angle = arccos(-n0 . n1)
        dot_product = dr.clip(-dr.dot(n0, n1), -1.0, 1.0)
        interior_angle = dr.acos(dot_product)

        # Exterior angle = 2 pi - interior angle
        exterior_angle = 2 * dr.pi - interior_angle
        wedge_n = exterior_angle / dr.pi

        face_normals_3d = [n0, n1]
    else:
        # Boundary edge: assume half-plane (exterior angle = 2 pi, n = 2)
        wedge_n = Float(2.0)

    edge_info.normal_2d = normal_2d
    edge_info.wedge_n = wedge_n
    edge_info.face_normals_3d = face_normals_3d
    return edge_info


def _drjit_to_key(v):
    """Convert a DrJit Vector2f to a hashable tuple key."""
    if isinstance(v, Vector2f):
        x = scalar(v.x)
        y = scalar(v.y)
        return (round(x, 6), round(y, 6))
    elif isinstance(v, Point3f):
        x = scalar(v.x)
        y = scalar(v.y)
        return (round(x, 6), round(y, 6))
    elif hasattr(v, '__iter__'):
        return tuple(round(float(x), 6) for x in v)
    else:
        return (round(float(v), 6),)


def _vectors_close(v1, v2, tol=1e-5):
    """Check if two DrJit vectors are close."""
    if isinstance(v1, (Vector2f, Vector3f, Point3f)):
        v1_x = scalar(v1.x)
        v1_y = scalar(v1.y)
        v2_x = scalar(v2.x)
        v2_y = scalar(v2.y)
        return abs(v1_x - v2_x) < tol and abs(v1_y - v2_y) < tol
    else:
        return all(abs(a - b) < tol for a, b in zip(v1, v2))


def project_to_2d(vertical_edges, calculation_height, vertices):
    """
    Project 3D vertical edges to 2D plane (DrJit, preserves gradient).

    Args:
        vertical_edges: List of vertical edges (with geometry attributes)
        calculation_height: Z coordinate of calculation plane
        vertices: Point3f with N vertices

    Returns:
        edges_2d: list of Edge2D (p0_2d, p1_2d, normal_2d, name)
                  Format compatible with field computation functions (DrJit)
        corners_2d: list of Corner2D (vertex_2d, face0_point, face_n_point, name, edge_info)
                    Diffraction corner list (DrJit)
    """
    import math


    edges_2d = []
    corner_vertices = {}

    for i, edge_info in enumerate(vertical_edges):
        p0_3d = edge_info.p0  # Point3f (single point)
        p1_3d = edge_info.p1  # Point3f (single point)

        # Check if edge is within calculation height range
        z0 = scalar(p0_3d.z)
        z1 = scalar(p1_3d.z)
        z_min = min(z0, z1)
        z_max = max(z0, z1)

        if not (z_min <= calculation_height <= z_max):
            continue  # Skip edge not at calculation height

        # Project to 2D
        p0_2d = Vector2f(p0_3d.x, p0_3d.y)
        p1_2d = Vector2f(p1_3d.x, p1_3d.y)

        # Check if degenerate (fully vertical edge)
        edge_len_2d = dr.norm(p1_2d - p0_2d)
        edge_len_2d_val = scalar(edge_len_2d)

        if edge_len_2d_val > EDGE_2D_EPS:
            # Non-degenerate edge: can be used for reflection and diffraction
            normal_2d = edge_info.normal_2d if edge_info.normal_2d is not None else Vector2f(0, 0)
            edges_2d.append(Edge2D(
                p0_2d,
                p1_2d,
                normal_2d,
                f'edge_{i}'
            ))

            # Record endpoints as potential corners
            corner_key_0 = _drjit_to_key(p0_2d)
            corner_key_1 = _drjit_to_key(p1_2d)

            if corner_key_0 not in corner_vertices:
                corner_vertices[corner_key_0] = {
                    'pos': p0_2d,
                    'edges': [],
                    'vertical_edge': None
                }
            corner_vertices[corner_key_0]['edges'].append(edge_info)

            if corner_key_1 not in corner_vertices:
                corner_vertices[corner_key_1] = {
                    'pos': p1_2d,
                    'edges': [],
                    'vertical_edge': None
                }
            corner_vertices[corner_key_1]['edges'].append(edge_info)
        else:
            # Degenerate edge: record vertical edge info (with correct wedge_n)
            corner_key = _drjit_to_key(p0_2d)

            if corner_key not in corner_vertices:
                corner_vertices[corner_key] = {
                    'pos': p0_2d,
                    'edges': [],
                    'vertical_edge': edge_info
                }
            else:
                corner_vertices[corner_key]['vertical_edge'] = edge_info

    # === Build corners ===
    all_vertex_positions = []
    for vertex_key, vertex_data in corner_vertices.items():
        all_vertex_positions.append(vertex_data['pos'])

    # Compute center
    if len(all_vertex_positions) > 0:
        # Compute mean position
        sum_x = sum(scalar(p.x) for p in all_vertex_positions)
        sum_y = sum(scalar(p.y) for p in all_vertex_positions)
        center_x = sum_x / len(all_vertex_positions)
        center_y = sum_y / len(all_vertex_positions)

        # Sort by polar angle (counterclockwise)
        def angle_from_center(pos):
            px = scalar(pos.x)
            py = scalar(pos.y)
            return math.atan2(py - center_y, px - center_x)

        sorted_positions = sorted(all_vertex_positions, key=angle_from_center)

        # Build position to next counterclockwise position mapping
        position_to_next = {}
        for idx, pos in enumerate(sorted_positions):
            prev_pos = sorted_positions[(idx - 1) % len(sorted_positions)]
            key = _drjit_to_key(pos)
            position_to_next[key] = prev_pos
    else:
        position_to_next = {}

    corners_2d = []
    for vertex_key, vertex_data in corner_vertices.items():
        connected_edges = vertex_data['edges']
        vertical_edge = vertex_data['vertical_edge']

        if len(connected_edges) == 0 and vertical_edge is None:
            continue

        vertex_pos = vertex_data['pos']

        # === Determine wedge_n ===
        if vertical_edge is not None:
            wedge_n = vertical_edge.wedge_n if vertical_edge.wedge_n is not None else Float(1.5)
        elif len(connected_edges) == 1:
            wedge_n = Float(2.0)
        else:
            edge0 = connected_edges[0]
            edge1 = connected_edges[1]
            wn0 = edge0.wedge_n if edge0.wedge_n is not None else Float(1.5)
            wn1 = edge1.wedge_n if edge1.wedge_n is not None else Float(1.5)
            wedge_n = (wn0 + wn1) / 2

        # === Build face reference points ===
        if vertical_edge is not None:
            edge_vector = vertical_edge.edge_vector  # Vector3f
            e_hat = edge_vector / (dr.norm(edge_vector) + EPS)
            face_normals = vertical_edge.face_normals_3d or []

            if len(face_normals) >= 2:
                n0 = face_normals[0]  # Vector3f
                nn = face_normals[1]  # Vector3f

                # Compute face 0 tangent direction (3D)
                to_hat = dr.cross(n0, e_hat)
                to_hat = to_hat / (dr.norm(to_hat) + EPS)

                # Compute face n tangent direction (3D)
                tn_hat = dr.cross(nn, e_hat)
                tn_hat = tn_hat / (dr.norm(tn_hat) + EPS)

                # Determine face order
                vertex_key_tuple = _drjit_to_key(vertex_pos)
                if vertex_key_tuple in position_to_next:
                    next_pos = position_to_next[vertex_key_tuple]
                    npx = scalar(next_pos.x)
                    npy = scalar(next_pos.y)
                    vpx = scalar(vertex_pos.x)
                    vpy = scalar(vertex_pos.y)
                    ccw_dx = npx - vpx
                    ccw_dy = npy - vpy
                    ccw_len = math.sqrt(ccw_dx**2 + ccw_dy**2) + EPS
                    ccw_tangent = Vector2f(ccw_dx / ccw_len, ccw_dy / ccw_len)
                else:
                    ccw_tangent = Vector2f(1.0, 0.0)

                # Project to XY plane
                to_hat_2d = Vector2f(to_hat.x, to_hat.y)
                to_hat_2d = to_hat_2d / (dr.norm(to_hat_2d) + EPS)

                tn_hat_2d = Vector2f(tn_hat.x, tn_hat.y)
                tn_hat_2d = tn_hat_2d / (dr.norm(tn_hat_2d) + EPS)

                # Compute dot products with ccw_tangent
                dot_0 = dr.dot(to_hat_2d, ccw_tangent)
                dot_1 = dr.dot(tn_hat_2d, ccw_tangent)
                choose_face0 = dot_0 > dot_1

                face0_dir = dr.select(choose_face0, to_hat_2d, tn_hat_2d)
                face_n_dir = dr.select(choose_face0, tn_hat_2d, to_hat_2d)
                corner_face_normals = [
                    dr.select(choose_face0, n0, nn),
                    dr.select(choose_face0, nn, n0)
                ]

                face0_point = vertex_pos + face0_dir * 0.1
                face_n_point = vertex_pos + face_n_dir * 0.1

                # Build position (3D)
                position_3d = Point3f(vertex_pos.x, vertex_pos.y, Float(calculation_height))

                corner_specific_edge_info = DiffractionPoint(
                    position=position_3d,
                    edge_vector=vertical_edge.edge_vector,
                    length=vertical_edge.length,
                    wedge_n=wedge_n,
                    face_normals_3d=corner_face_normals
                )
            else:
                face0_point = vertex_pos + Vector2f(0.1, 0.0)
                face_n_point = vertex_pos + Vector2f(0.0, 0.1)
                corner_specific_edge_info = None

        elif len(connected_edges) == 0:
            face0_point = vertex_pos + Vector2f(0.1, 0.0)
            face_n_point = vertex_pos + Vector2f(0.0, 0.1)
            corner_specific_edge_info = None

        elif len(connected_edges) == 1:
            edge0 = connected_edges[0]
            p0_edge = Vector2f(edge0.p0.x, edge0.p0.y)
            p1_edge = Vector2f(edge0.p1.x, edge0.p1.y)

            if _vectors_close(p0_edge, vertex_pos):
                edge_dir = p1_edge - p0_edge
            else:
                edge_dir = p0_edge - p1_edge

            edge_dir = edge_dir / (dr.norm(edge_dir) + EPS)

            face0_point = vertex_pos + edge_dir * 0.1
            # Perpendicular direction
            ed_x = scalar(edge_dir.x)
            ed_y = scalar(edge_dir.y)
            perp_dir = Vector2f(-ed_y, ed_x)
            face_n_point = vertex_pos + perp_dir * 0.1
            corner_specific_edge_info = None

        else:
            edge0 = connected_edges[0]
            edge1 = connected_edges[1]

            p0_e0 = Vector2f(edge0.p0.x, edge0.p0.y)
            p1_e0 = Vector2f(edge0.p1.x, edge0.p1.y)
            if _vectors_close(p0_e0, vertex_pos):
                dir0 = p1_e0 - p0_e0
            else:
                dir0 = p0_e0 - p1_e0

            p0_e1 = Vector2f(edge1.p0.x, edge1.p0.y)
            p1_e1 = Vector2f(edge1.p1.x, edge1.p1.y)
            if _vectors_close(p0_e1, vertex_pos):
                dir1 = p1_e1 - p0_e1
            else:
                dir1 = p0_e1 - p1_e1

            # Normalize
            dir0 = dir0 / (dr.norm(dir0) + EPS)
            dir1 = dir1 / (dr.norm(dir1) + EPS)

            # Compute angle
            d0_x = scalar(dir0.x)
            d0_y = scalar(dir0.y)
            d1_x = scalar(dir1.x)
            d1_y = scalar(dir1.y)
            cross_val = d0_x * d1_y - d0_y * d1_x
            dot_val = d0_x * d1_x + d0_y * d1_y
            angle = math.atan2(cross_val, dot_val)

            if angle < 0:
                angle = angle + 2 * math.pi

            if angle < math.pi:
                face0_dir = dir1
                face_n_dir = dir0
            else:
                face0_dir = dir0
                face_n_dir = dir1

            face0_point = vertex_pos + face0_dir * 0.1
            face_n_point = vertex_pos + face_n_dir * 0.1
            corner_specific_edge_info = None

        corners_2d.append(Corner2D(
            vertex_pos,
            face0_point,
            face_n_point,
            f'corner_{len(corners_2d)}',
            corner_specific_edge_info
        ))

    return edges_2d, corners_2d
