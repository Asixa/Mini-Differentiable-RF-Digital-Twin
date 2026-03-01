from __future__ import annotations

"""Field class for receiver field management."""

import math
import drjit as dr
import mitsuba as mi

C = 299792458.0


class Field:
    """2D Field for receiving signal with grid coordinate caching."""

    # Class-level cache for grid coordinates (shared across instances)
    _coord_cache: dict = {}

    def __init__(self, bounds: tuple, size: tuple):
        """
        Initialize a 2D receiver field.

        Args:
            bounds: ((x_min, x_max), (y_min, y_max)) field boundaries
            size: (nx, ny) number of cells in each dimension
        """
        self.bounds = bounds
        self.size = size
        self.n_cells = size[0] * size[1]

        (x_min, x_max), (y_min, y_max) = bounds
        self.cell_size = ((x_max - x_min) / size[0], (y_max - y_min) / size[1])

    @classmethod
    def from_wavelength(cls, bounds: tuple, wavelength: float,
                        resolution: float = 0.125) -> 'Field':
        """
        Create a Field with automatic size based on wavelength.

        Args:
            bounds: ((x_min, x_max), (y_min, y_max)) field boundaries
            wavelength: Signal wavelength in meters
            resolution: Cell size as fraction of wavelength (default 0.125 = lambda/8)

        Returns:
            Field instance with calculated size
        """
        (x_min, x_max), (y_min, y_max) = bounds
        cell_size = resolution * wavelength

        nx = int(math.ceil((x_max - x_min) / cell_size))
        ny = int(math.ceil((y_max - y_min) / cell_size))
        grid_size = max(nx, ny)

        return cls(bounds=bounds, size=(grid_size, grid_size))

    @classmethod
    def from_frequency(cls, bounds: tuple, frequency: float,
                       resolution: float = 0.125) -> 'Field':
        """
        Create a Field with automatic size based on frequency.

        Args:
            bounds: ((x_min, x_max), (y_min, y_max)) field boundaries
            frequency: Signal frequency in Hz
            resolution: Cell size as fraction of wavelength (default 0.125 = lambda/8)

        Returns:
            Field instance with calculated size
        """
        wavelength = C / frequency
        return cls.from_wavelength(bounds, wavelength, resolution)

    def pos_to_idx(self, x: mi.Float, y: mi.Float) -> mi.UInt32:
        """
        Convert (x, y) positions to flattened field cell indices.

        Args:
            x, y: DrJit Float arrays of positions

        Returns:
            DrJit UInt32 array of cell indices
        """
        (x_min, _), (y_min, _) = self.bounds
        nx, ny = self.size
        ix = dr.clamp(mi.Int32((x - x_min) / self.cell_size[0]), 0, nx - 1)
        iy = dr.clamp(mi.Int32((y - y_min) / self.cell_size[1]), 0, ny - 1)
        return mi.UInt32(iy * nx + ix)

    def get_coordinates(self) -> dict:
        """
        Get cached grid coordinates (X, Y arrays).

        Returns:
            dict with 'X', 'Y', 'x_coords', 'y_coords'
        """
        cache_key = (self.size, self.bounds)
        if cache_key in Field._coord_cache:
            return Field._coord_cache[cache_key]

        (x_min, x_max), (y_min, y_max) = self.bounds
        nx, ny = self.size

        x_step = (x_max - x_min) / (nx - 1) if nx > 1 else 0
        y_step = (y_max - y_min) / (ny - 1) if ny > 1 else 0

        idx = dr.arange(mi.Float, nx)
        x_coords = mi.Float(x_min) + idx * mi.Float(x_step)
        y_coords = mi.Float(y_min) + idx * mi.Float(y_step)

        X = dr.tile(x_coords, ny)
        Y = dr.repeat(y_coords, nx)

        dr.eval(x_coords, y_coords, X, Y)

        coord_data = {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'X': X,
            'Y': Y,
        }

        Field._coord_cache[cache_key] = coord_data
        return coord_data

    @property
    def X(self) -> mi.Float:
        """Flattened X coordinates."""
        return self.get_coordinates()['X']

    @property
    def Y(self) -> mi.Float:
        """Flattened Y coordinates."""
        return self.get_coordinates()['Y']

    @property
    def grid_size(self) -> int:
        """Grid size (assumes square grid)."""
        return self.size[0]
