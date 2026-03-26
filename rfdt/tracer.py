"""Radio propagation tracer using mesh-based UTD simulation (Pure DrJit)"""

import math
import time
import drjit as dr

from .scene import Scene
from .trace_los import compute_los_field
from .trace_reflection import compute_reflection_field
from .trace_diffraction import compute_diffraction_field
from .utils import scalar
from .field import Field
from .rt_backend import Float, Point3f, Complex2f

C = 299792458.0


class Tracer:
    """
    Radio propagation tracer for computing electromagnetic fields around obstacles.

    Uses 3D mesh representation with UTD (Uniform Theory of Diffraction) for
    accurate modeling of LoS, reflection, and diffraction paths.
    """

    def __init__(self, frequency: float, scene: Scene = None,
                 vertices=None, faces=None,
                 reflection_n_rays: int = 10000,
                 reflection_max_bounces: int = 2,
                 reflection_coef: float = 0.7,
                 resolution_wavelength: float = 0.125):
        """
        Initialize the tracer with an external Scene object.

        Args:
            frequency: Signal frequency in Hz
            scene: Scene object encapsulating mesh and the RayD scene
            vertices, faces: Optional fallback to build a Scene internally
            reflection_n_rays: Number of rays for reflection tracing (default 10000)
            reflection_max_bounces: Maximum reflection bounces (default 2)
            reflection_coef: Reflection coefficient 0-1 (default 0.7)
            resolution_wavelength: Grid cell size in wavelengths (default 0.125 = lambda/8)
        """
        self.frequency = frequency
        if scene is None:
            if vertices is None or faces is None:
                raise ValueError("Provide a Scene or vertices+faces to initialize Tracer.")
            scene = Scene(vertices, faces)
        self.scene = scene
        self.wavelength = C / frequency
        self.k = 2 * math.pi / self.wavelength

        # Reflection parameters
        self.reflection_n_rays = reflection_n_rays
        self.reflection_max_bounces = reflection_max_bounces
        self.reflection_coef = reflection_coef

        # Grid resolution
        self.resolution_wavelength = resolution_wavelength
        self.cell_size = resolution_wavelength * self.wavelength

    def trace(self, tx_pos, grid_size: int = None,
              range_x=(-8, 8), range_y=(-8, 8),
              calculation_height: float = None,
              verbose: bool = False,
              return_timing: bool = False):
        """
        Compute electromagnetic field distribution.

        Args:
            tx_pos: Transmitter position (x, y, z) - tuple, list, or Point3f
            grid_size: Grid resolution (if None, auto-calculate from resolution_wavelength)
            range_x, range_y: Computation area bounds
            calculation_height: Z coordinate for field computation (default: tx height)
            verbose: Print debug information
            return_timing: If True, include timing info in result dict

        Returns:
            Dictionary containing complex field components:
                - a_los: LoS complex field (Complex2f)
                - a_ref: Reflection complex field (Complex2f)
                - a_dif: Diffraction complex field (Complex2f)
                - a_tot: Total complex field (Complex2f)
                - X, Y: Grid coordinates (Float)
                - tx_pos_3d: Transmitter position tuple
                - calculation_height: Z height used
                - grid_size: Grid resolution used
                - timing: (optional) Timing info dict
        """
        timing = {} if return_timing else None

        # Ensure tx_pos is Point3f for gradient-preserving computation
        if not isinstance(tx_pos, Point3f):
            if hasattr(tx_pos, 'item'):
                tx_pos = Point3f(tx_pos[0].item(), tx_pos[1].item(), tx_pos[2].item())
            else:
                tx_pos = Point3f(float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2]))

        if calculation_height is None:
            calculation_height = scalar(tx_pos.z)
        elif hasattr(calculation_height, 'item'):
            calculation_height = calculation_height.item()
        else:
            calculation_height = float(calculation_height)

        # Create Field (auto-size if grid_size not provided)
        bounds = (range_x, range_y)
        if grid_size is None:
            field = Field.from_wavelength(bounds, self.wavelength, self.resolution_wavelength)
            grid_size = field.grid_size
            if verbose:
                print(f"Auto grid_size: {grid_size} (cell_size={self.cell_size:.4f}m = lambda/{self.wavelength/self.cell_size:.1f})")
        else:
            field = Field(bounds=bounds, size=(grid_size, grid_size))

        # Get cached coordinates
        coords = field.get_coordinates()

        if self.scene.n_vertical_edges == 0 and verbose:
            print("Warning: No valid vertical edges found!")

        X, Y = coords['X'], coords['Y']
        rx_z = Float(calculation_height)

        # === Compute LoS field ===
        if return_timing:
            t0 = time.perf_counter()
        a_los = compute_los_field(self.scene, X, Y, rx_z, tx_pos, self.wavelength, self.k)
        if return_timing:
            dr.eval(a_los)
            timing['los'] = time.perf_counter() - t0

        # === Compute reflection field ===
        if return_timing:
            t0 = time.perf_counter()
        a_ref_total, _ = compute_reflection_field(
            grid=field,
            rx_z=calculation_height,
            tx_pos=tx_pos,
            scene=self.scene,
            wavelength=self.wavelength,
            k=self.k,
            n_rays=self.reflection_n_rays,
            max_reflections=self.reflection_max_bounces,
            mode='2d',
            reflection_coef=self.reflection_coef,
            return_per_bounce=False,
            grid_data=coords
        )
        if return_timing:
            dr.eval(a_ref_total)
            timing['reflection'] = time.perf_counter() - t0

        # === Compute diffraction field ===
        if return_timing:
            t0 = time.perf_counter()
        dif_real, dif_imag, _ = compute_diffraction_field(
            X, Y, calculation_height, tx_pos,
            self.scene, self.wavelength, self.k,
            return_per_edge=False
        )
        if return_timing:
            dr.eval(dif_real, dif_imag)
            timing['diffraction'] = time.perf_counter() - t0

        # Compute total field (complex sum)
        tot_real = a_los.real + a_ref_total.real + dif_real
        tot_imag = a_los.imag + a_ref_total.imag + dif_imag

        # Create complex field objects
        a_dif = Complex2f(dif_real, dif_imag)
        a_tot = Complex2f(tot_real, tot_imag)

        dr.eval(a_los, a_ref_total, a_dif, a_tot)

        result = {
            'X': X,
            'Y': Y,
            'a_los': a_los,
            'a_ref': a_ref_total,
            'a_dif': a_dif,
            'a_tot': a_tot,
            'tx_pos_3d': (scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
            'calculation_height': calculation_height,
            'grid_size': grid_size,
        }
        if return_timing:
            result['timing'] = timing
        return result
