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

    def __init__(self, mode='trilinear'):
        """Initialize the DensityMapPyTorch object."""
        if mode == 'trilinear':
            # 'bilinear' mode is trilinear when using 5D input
            self.mode = 'bilinear'
        elif mode == 'nearest':
            self.mode = 'nearest'
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}. Supported modes are 'nearest' and 'bilinear'.")

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

    def _compute_frac_matrix(self):
        """Compute the fractionalization matrix using unit cell parameters."""
        # Define pi as a constant
        self.pi = torch.tensor(np.pi, dtype=torch.float32)
        
        # Extract unit cell parameters
        a, b, c = self.unit_cell.a, self.unit_cell.b, self.unit_cell.c
        cos_alpha = torch.cos(self.unit_cell.alpha * self.pi / 180)
        cos_beta = torch.cos(self.unit_cell.beta * self.pi / 180)
        cos_gamma = torch.cos(self.unit_cell.gamma * self.pi / 180)
        sin_gamma = torch.sin(self.unit_cell.gamma * self.pi / 180)
        
        # Compute unit cell volume
        volume = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
        
        # Compute fractionalization matrix
        self.frac_matrix = torch.tensor([
            [1 / a, -cos_gamma / (a * sin_gamma), b * c * (cos_alpha * cos_gamma - cos_beta) / (volume * sin_gamma)],
            [0, 1 / (b * sin_gamma), a * c * (cos_beta * cos_gamma - cos_alpha) / (volume * sin_gamma)],
            [0, 0, a * b * sin_gamma / volume]
        ], dtype=torch.float32)

    def fractionalize(self, xyz):
        """
        Convert Cartesian coordinates to fractional coordinates.
        
        Args:
            xyz (torch.Tensor): Cartesian coordinates. Shape can be (3,) or (N, 3).
        
        Returns:
            torch.Tensor: Fractional coordinates with the same shape as input.
        """
        # Ensure xyz is a tensor with gradients
        # if not isinstance(xyz, torch.Tensor):
        #     xyz = torch.tensor(xyz, dtype=torch.float32)
        # if not xyz.requires_grad:
        #     xyz.requires_grad_(True)
        
        # Handle different input dimensions
        original_shape = xyz.shape
        if len(original_shape) == 1:  # (3,)
            xyz = xyz.view(1, 3)
        elif len(original_shape) == 2:  # (N, 3)
            xyz = xyz.view(-1, 3)
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}. Expected (3,), or (N,3).")
        
        # Convert cartesian coordinates to fractional coordinates
        frac = torch.einsum('ni,ij->nj', xyz, self.frac_matrix)
        
        # Reshape back to original dimensions if necessary
        if len(original_shape) == 3:
            frac = frac.view(original_shape)
        elif len(original_shape) == 1:
            frac = frac.squeeze(0).squeeze(0)
        
        return frac
    
    def _compute_grid(self, voxel_set, axis_order):
        """
        Prepare the voxel grid for interpolation.
        
        Args:
            voxel_set (torch.Tensor): The voxel grid data.
        """
        # Ensure grid is a tensor with gradients
        # if not isinstance(voxel_set, torch.Tensor):
        #     voxel_set = torch.tensor(voxel_set, dtype=torch.float32)
        # if not voxel_set.requires_grad:
        #     voxel_set.requires_grad_(True)

        # # Reshape grid to (1, 1, nx, ny, nz) for interpolation
        self.grid = voxel_set.unsqueeze(0).unsqueeze(0)
        self.grid.requires_grad = False # TODO: Test if it works better like that

        # Permute the grid to match Depth * Height * Width order (cf documentation)
        if axis_order == 'XYZ': # XYZ -> DHW, assuming X is width, Y is height, Z is depth
            self.grid = self.grid.permute(0, 1, 4, 3, 2)
        elif axis_order == 'ZYX': # ZYX -> DHW, assuming X is width, Y is height, Z is depth
            pass

    
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


