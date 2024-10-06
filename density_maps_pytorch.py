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

class DensityMapPyTorch:
    """
    A class to handle density map data as PyTorch tensors.
    
    This class provides functionality to load CCP4 map files, convert them to PyTorch tensors,
    and perform operations such as coordinate fractionalization and density interpolation.
    """

    def __init__(self):
        """Initialize the DensityMapPyTorch object."""
        pass

    def _get_gemmi_ccp4_map(self, path, setup=True):
        # Read the CCP4 map using gemmi
        ccp4_map = gemmi.read_ccp4_map(path)
        if setup:
            ccp4_map.setup(float('nan'))
        return ccp4_map

    def load_map(self, map_path):
        """
        Load a CCP4 map file and convert it to a PyTorch tensor.
        
        Args:
            map_path (str): Path to a cryoEM density map.
        """

        # Read the CCP4 map using gemmi
        ccp4_map = self._get_gemmi_ccp4_map(map_path)

        # Convert the grid data to a PyTorch tensor
        self.grid_data = torch.from_numpy(np.array(ccp4_map.grid, copy=False)).float()
       
        # if ccp4_map.grid.axis_order == gemmi.AxisOrder.XYZ:
        #     # Permute the tensor to match UVW order if necessary
        #     self.grid_data = self.grid_data.permute(2, 1, 0)
        self.axis_order = str(ccp4_map.grid.axis_order).split('.')[-1]

        # Prepare the grid for interpolation
        self._compute_grid(self.grid_data, self.axis_order)
       
        # Convert parameters to tensor namedtuples
        self.unit_cell = UnitCellParams(*[torch.tensor(x) for x in ccp4_map.grid.unit_cell.parameters])
        self.spacegroup = torch.tensor(ccp4_map.grid.spacegroup.number, dtype=torch.int32)
        
        # Store grid parameters
        self.grid_params = GridParams(
            nu=torch.tensor(ccp4_map.grid.nu),
            nv=torch.tensor(ccp4_map.grid.nv),
            nw=torch.tensor(ccp4_map.grid.nw),
            mode=torch.tensor(ccp4_map.header_i32(4)),
            nxstart=torch.tensor(ccp4_map.header_i32(5)),
            nystart=torch.tensor(ccp4_map.header_i32(6)),
            nzstart=torch.tensor(ccp4_map.header_i32(7)),
            mx=torch.tensor(ccp4_map.header_i32(8)),
            my=torch.tensor(ccp4_map.header_i32(9)),
            mz=torch.tensor(ccp4_map.header_i32(10))
        )
        
        # Store map parameters
        self.map_params = MapParams(
            dmin=torch.tensor(ccp4_map.header_float(20)),
            dmax=torch.tensor(ccp4_map.header_float(21)),
            dmean=torch.tensor(ccp4_map.header_float(22)),
            ispg=torch.tensor(ccp4_map.header_i32(23)),
            nsymbt=torch.tensor(ccp4_map.header_i32(24))
        )

        # Compute and store the fractionalization matrix
        self._compute_frac_matrix()

        # Compute stats
        self.stats = StatParams(
            min=torch.min(self.grid_data),
            max=torch.max(self.grid_data),
            mean=torch.mean(self.grid_data),
            std=torch.std(self.grid_data)
        )

    
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


