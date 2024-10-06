from collections import namedtuple
import logging
import torch
import numpy as np
import gemmi
import scipy
import torch.nn.functional

# Define namedtuples for different parameter groups
UnitCellParams = namedtuple('UnitCellParams', ['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
GridParams = namedtuple('GridParams', ['nu', 'nv', 'nw', 'mode', 'nxstart', 'nystart', 'nzstart', 'mx', 'my', 'mz'])
CellParams = namedtuple('CellParams', ['cella', 'cellb', 'mapc', 'mapr', 'maps'])
MapParams = namedtuple('MapParams', ['dmin', 'dmax', 'dmean', 'ispg', 'nsymbt'])
StatParams = namedtuple('StatParams', ['min', 'max', 'mean', 'std'])

    
class DensityMapBase(DensityMapPyTorch):
    def __init__(self):
        return super().__init__()
    
    def load_map(self, ccp4_map):
        self.density_map = self._get_gemmi_ccp4_map(ccp4_map)

        if str(self.density_map.grid.axis_order).split('.')[-1] == 'XYZ':
            self.scipy_interpolator = self._compute_interpolator()
        else:
            raise NotImplementedError("Only XYZ axis order is supported.")

    
    def fractionalize(self, xyz):
        return self.density_map.grid.unit_cell.fractionalize(gemmi.Position(*xyz))
    
    def _compute_interpolator(self):
        # Set up scipy interpolator
        grid_data = self.density_map.grid.array
        x = np.linspace(0, self.density_map.grid.unit_cell.a, self.density_map.grid.nu, endpoint=True)
        y = np.linspace(0, self.density_map.grid.unit_cell.b, self.density_map.grid.nv, endpoint=True)
        z = np.linspace(0, self.density_map.grid.unit_cell.c, self.density_map.grid.nw, endpoint=True)
        self.interpolator = scipy.interpolate.RegularGridInterpolator((x, y, z), grid_data, method='linear', bounds_error=True)
    
    def get_interpolated_density_scipy(self, xyz):
        if isinstance(xyz, torch.Tensor):
            return self.interpolator(xyz.detach().numpy())
        else:
            return self.interpolator(xyz)
        
    
    def get_interpolated_density_gemmi(self, xyz):
        """
        Get interpolated density values at given Cartesian coordinates.
        
        Args:
            xyz (torch.Tensor): Cartesian coordinates. Shape has to be (3,).
        
        Returns:
            torch.Tensor: Interpolated density values.
        """
        # Ensure xyz is a tensor with gradients
        # if not xyz.requires_grad:
        #     xyz.requires_grad_(True)
        
        # Convert Cartesian coordinates to fractional coordinates
        frac = self.fractionalize(xyz)
        
        # Get interpolated density and derivatives
        density = self.density_map.grid.interpolate_value(frac)
        
        return density


