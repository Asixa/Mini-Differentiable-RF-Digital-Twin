"""RayD 0.6 runtime and numerical contracts for RFDT."""

from __future__ import annotations

import unittest

import drjit as dr

from rfdt.rt_backend import (
    Float,
    Point3f,
    UInt32,
    Vector3u,
    build_scene,
    ray_intersect,
    update_vertices,
)


def _triangle(z):
    vertices = Point3f(
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 1.0],
        z,
    )
    faces = Vector3u(UInt32([0]), UInt32([1]), UInt32([2]))
    return vertices, faces


def _forward_hit(scene):
    return ray_intersect(
        scene,
        Point3f([0.0], [0.0], [0.0]),
        Point3f([0.0], [0.0], [1.0]),
    )


class RayD060RuntimeTests(unittest.TestCase):
    def test_intersection_and_dynamic_refit_preserve_distance(self):
        vertices, faces = _triangle([2.0, 2.0, 2.0])
        scene, params, vertex_key, _ = build_scene(vertices, faces)

        before = _forward_hit(scene)
        moved_vertices, _ = _triangle([3.0, 3.0, 3.0])
        update_vertices(params, vertex_key, moved_vertices)
        after = _forward_hit(scene)
        dr.eval(before.t, after.t)

        self.assertTrue(bool(before.is_valid()[0]))
        self.assertTrue(bool(after.is_valid()[0]))
        self.assertAlmostEqual(float(before.t[0]), 2.0, places=5)
        self.assertAlmostEqual(float(after.t[0]), 3.0, places=5)

    def test_intersection_keeps_mesh_geometry_gradient(self):
        height = Float(2.0)
        dr.enable_grad(height)
        vertices, faces = _triangle(dr.concat([height, height, height]))
        scene, _, _, _ = build_scene(vertices, faces)

        hit = _forward_hit(scene)
        dr.forward(height)

        self.assertAlmostEqual(float(hit.t[0]), 2.0, places=5)
        self.assertAlmostEqual(float(dr.grad(hit.t)[0]), 1.0, places=5)
        self.assertAlmostEqual(float(dr.grad(hit.p.z)[0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
