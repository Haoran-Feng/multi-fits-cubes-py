from pathlib import Path
from collections import OrderedDict
from astropy.table import Table
from astropy import units as u
from matplotlib import pyplot as plt
from spectral_cube import SpectralCube, Projection
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import log
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
mpl.rcParams['figure.dpi'] = 72
import re

from astropy.visualization import quantity_support

log.setLevel("ERROR")
quantity_support()

from multi_fits_cubes.helpers import MaskCube, ValidationMap


class Cloud:
    def __init__(self, name, mask_cube, big_cubes: OrderedDict):
        self.name = name
        if isinstance(mask_cube, str):
            self.mask_cube_obj = MaskCube.from_file(mask_cube)
        elif isinstance(mask_cube, SpectralCube):
            self.mask_cube_obj = MaskCube(mask_cube)
        else:
            raise TypeError("mask_cube must be a FITS file or an instance of SpectralCube.")

        self.cubes = {}
        self.line_names = []
        self.big_cubes = big_cubes
        for line, big_cube in big_cubes.items():
            self.line_names.append(line)
            datacube = self.mask_cube_obj.cut_and_mask_from_big_cube(big_cube)
            self.cubes[line] = datacube

        self._check_cube_shapes()

    def _check_cube_shapes(self):
        shapes_spatial = [v.shape[1:] for k, v in self.cubes.items()]
        assert all([s == shapes_spatial[0] for s in shapes_spatial]), 'Spatial Shape not matched!'

    def __getitem__(self, item):
        if item not in self.cubes.keys():
            raise KeyError(f"Data for emission line {item} not found in {self.name}")
        return self.cubes[item]

    def __str__(self):
        return f"Cloud: {self.name}, Lines: {[k for k in self.cubes.keys()]}"

    def get_cube_with_extra_v_left_right_range(self, line_name, v_left, v_right):
        cube = self.mask_cube_obj.cut_extra_v_left_v_right_and_2d_mask_from_big_cube(self.big_cubes[line_name], v_left, v_right)
        return cube


class CloudManager:
    def __init__(self,
                 mask_dir: str,
                 big_cubes: dict, catalog=None, catalog_idx_col='_idx', idx_re=r'cloud(?P<idx>\d+)', index_type=int, cls=Cloud):
        self.cls = cls
        self.mask_dir = Path(mask_dir)
        self.mask_fits_list = list(self.mask_dir.glob('*.fits'))
        self.cloud_indices = [index_type(re.search(idx_re, f.stem).group('idx')) for f in self.mask_fits_list]
        self.cloud_indices_set = set(self.cloud_indices)
        self.cloud_idx_to_mask_filepath = {idx: self.mask_fits_list[i] for i, idx in enumerate(self.cloud_indices)}

        self.big_cubes = OrderedDict()


        self.catalog = catalog
        self.catalog_idx_col = catalog_idx_col
        if self.catalog is not None:
            if isinstance(self.catalog, str):
                self.catalog = Table.read(self.catalog)

        for k, v in big_cubes.items():
            if isinstance(v, str):
                big_cube = SpectralCube.read(v)
                big_cube.allow_huge_operations = True
            else:
                big_cube = v

            if not isinstance(big_cube, SpectralCube):
                raise TypeError(f"Big cube for {k} is not a SpectralCube or str")

            self.big_cubes[k] = big_cube

        self.loaded_cloud = {}



    def get_cloud_indices(self):
        return self.cloud_indices

    def load_all(self):
        pass

    def load_cloud(self, idx, name_prefix='Cloud'):
        if idx not in self.cloud_indices:
            raise ValueError(f"Cloud Index {idx} not found!")

        if idx in self.loaded_cloud.keys():
            return self.loaded_cloud[idx]

        cloud = self.cls(name=name_prefix + str(idx),
                      mask_cube=str(self.cloud_idx_to_mask_filepath[idx]),
                      big_cubes=self.big_cubes)

        if self.catalog is not None:
            idx_row = self.catalog[self.catalog_idx_col] == idx
            setattr(cloud, 'catalog', self.catalog[idx_row])

        self.loaded_cloud[idx] = cloud
        return cloud



if __name__ == '__main__':

    class NewCloudClass(Cloud):
        def shape_polygon(self):
            self.polygon = 3
    cm = CloudManager('../test_data/masks', big_cubes={'C18O': '../test_data/bigs/M195_L2.fits'}, cls=NewCloudClass)
    cloud = cm.load_cloud(1246)
    cloud.shape_polygon()
    print(cloud.polygon)



