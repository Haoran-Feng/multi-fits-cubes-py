
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

from astropy.visualization import quantity_support

log.setLevel("ERROR")
quantity_support()

from multi_fits_cubes.helpers import MaskCube

class Cloud:
    mask_dir = '../mask_files_12CO/'
    catalog = Table.read("../catalogs/12CO_less220_catalog.fit")
    big12 = SpectralCube.read("../M195_U.fits")
    big13 = SpectralCube.read("../M195_L.fits")
    big18 = SpectralCube.read("../M195_L2.fits")
    big12.allow_huge_operations = True
    big13.allow_huge_operations = True
    big18.allow_huge_operations = True
    co13_scale = 3
    c18o_scale = 5

    co13_sum_dir = f"../13COsum_files_idx_by12CO/"
    co13_cover_dir = f"../13COcover_files_idx_by12CO/"
    co13_rms_dir = f"../13COrms_files_idx_by12CO/"

    c18o_sum_dir = f"../C18Osum_files_idx_by12CO/"
    c18o_cover_dir = f"../C18Ocover_files_idx_by12CO/"
    c18o_rms_dir = f"../C18Orms_files_idx_by12CO/"

    def __init__(self, idx):
        self.idx = idx
        self.mask_cube = SpectralCube.read(self.mask_dir + f"cloud{idx}mask.fits")

        self.mask_cube_obj = MaskCube(self.mask_cube)

        self.info = self.catalog[self.catalog['_idx'] == self.idx]
        self.v_cen = self.info['v_cen'][0] * self.info['v_cen'].unit

        self.co12_cube = self.mask_cube_obj.cut_and_mask_from_big_cube(self.big12)
        self.co12_cube = self.mask_cube_obj.cut_and_mask_from_big_cube(self.big13)
        self.co12_cube = self.mask_cube_obj.cut_and_mask_from_big_cube(self.big18)

        assert self.co12_cube.shape == self.mask_cube.shape
        assert self.co13_cube.shape[1:] == self.mask_cube.shape[1:]
        assert self.c18o_cube.shape[1:] == self.mask_cube.shape[1:]

        self.map_mask = self.mask_cube.max(axis=0) > 0 * u.dimensionless_unscaled
        cube_mask = np.zeros(self.co12_cube.shape, dtype=bool)
        cube_mask[:, self.map_mask] = True
        self.co12_cube = self.co12_cube.with_mask(self.mask_cube.filled_data[:].value > 0) * u.K

        cube_mask = np.zeros(self.co13_cube.shape, dtype=bool)
        cube_mask[:, self.map_mask] = True
        co13_cube_temp = self.co13_cube.with_mask(cube_mask) * u.K

        self.co13_mask_cube = self.mask_cube.with_fill_value(0).spectral_interpolate(co13_cube_temp.spectral_axis)
        self.co13_cube = self.co13_cube.with_mask(co13_cube_temp > 0 * co13_cube_temp.unit)

        cube_mask = np.zeros(self.c18o_cube.shape, dtype=bool)
        cube_mask[:, self.map_mask] = True
        c18o_cube_temp = self.c18o_cube.with_mask(cube_mask) * u.K

        self.c18o_mask_cube = self.mask_cube.with_fill_value(0).spectral_interpolate(c18o_cube_temp.spectral_axis)
        self.c18o_cube = self.c18o_cube.with_mask(c18o_cube_temp > 0 * c18o_cube_temp.unit)

        sum_map = self.read_projection(self.co13_sum_dir + f"cloud{self.idx}sum.fits")
        cover_map = self.read_projection(self.co13_cover_dir + f"cloud{self.idx}cover.fits")
        rms_map = self.read_projection(self.co13_rms_dir + f"cloud{self.idx}rms.fits")

        sum_np = sum_map.value
        sum_np[np.isnan(sum_np)] = 0

        valid = sum_np > (3 * rms_map * np.sqrt(cover_map)).value
        count = np.sum(valid)
        self.co13_rms_map = rms_map
        self.valid_co13_count = count
        self.valid_co13 = valid

        sum_map = self.read_projection(self.c18o_sum_dir + f"cloud{self.idx}sum.fits")
        cover_map = self.read_projection(self.c18o_cover_dir + f"cloud{self.idx}cover.fits")
        rms_map = self.read_projection(self.c18o_rms_dir + f"cloud{self.idx}rms.fits")

        sum_np = sum_map.value
        sum_np[np.isnan(sum_np)] = 0

        valid = sum_np > (3 * rms_map * np.sqrt(cover_map)).value
        count = np.sum(valid)

        self.c18o_rms_map = rms_map
        self.valid_c18o_count = count
        self.valid_c18o = valid


    def prepare_valid_pix_avg(self, downsample=None, c18o_scale=None, check_13co=True):
        self.co12_mask = self.map_mask
        self.co13_mask = self.valid_co13
        self.c18o_mask = self.valid_c18o

        if check_13co:
            self.c18o_mask = self.c18o_mask & self.co13_mask

        self.co12_spec_axis = self.co12_cube.spectral_axis
        self.co13_spec_axis = self.co13_cube.spectral_axis
        self.c18o_spec_axis = self.c18o_cube.spectral_axis
        self.wcs = self.co12_cube.wcs
        self.co12_avg = self.co12_cube.mean(axis=(1, 2))
        if c18o_scale is None:
            c18o_scale = Cloud.c18o_scale
        else:
            self.c18o_scale = c18o_scale

        if downsample is None:
            self.co13_avg = self.co13_scale * self.co13_cube.with_mask(self.co13_mask).mean(axis=(1, 2))
            self.c18o_avg = c18o_scale * self.c18o_cube.with_mask(self.c18o_mask).mean(axis=(1, 2))
            self.downsample = None
        else:
            downsampled_cube = self.c18o_cube.with_mask(self.c18o_mask).downsample_axis(downsample, axis=0)
            self.c18o_avg = c18o_scale * downsampled_cube.mean(axis=(1, 2))
            self.c18o_spec_axis = downsampled_cube.spectral_axis
            self.downsample = downsample

        self.c18o_n_avg_pix = np.sum(self.c18o_mask)
        self.co13_n_avg_pix = np.sum(self.co13_mask)

    def plot_avg(self, target_dir=None):
        #         plt.style.use("science")
        plt.figure()
        plt.step(self.co12_spec_axis, self.co12_avg, where='mid', label='${}^{12}CO$')
        plt.step(self.co13_spec_axis, self.co13_avg, where='mid',
                 label=f'${self.co13_scale}' + r'\times' + ' {}^{13}CO$')
        plt.step(self.c18o_spec_axis, self.c18o_avg, where='mid',
                 label=f'${self.c18o_scale}' + r'\times' + ' {C}^{18}O$')
        plt.tick_params(direction='in', which='both')
        bottom, top = plt.ylim()
        left, right = plt.xlim()
        plt.hlines(0, left, right, color="black", linestyle=':')
        plt.vlines(self.v_cen, bottom, top, color='black', linestyle=":")
        plt.xlim(left, right)
        plt.ylim(bottom, top)
        plt.legend(loc=1)
        if not hasattr(self, 'downsample') or self.downsample is None:
            plt.title(f"cloud{self.idx}, 13CO_n_pix={self.co13_n_avg_pix}, C18O_n_pix={self.c18o_n_avg_pix}")
        else:
            plt.title(f"cloud{self.idx}, n_pix={self.co13_n_avg_pix}, downsample: {self.downsample} channels")

        ax = plt.gca()
        axins = inset_axes(ax, width="20%", height="20%", loc=3)

        cloud_map = np.zeros_like(self.co12_mask, dtype=np.float)
        cloud_map[self.co12_mask] = 1.0
        cloud_map[self.co13_mask] = 2.0
        #         cloud_map = self.co12_mask.astype(np.float) + self.co13_mask.astype(np.float)
        cloud_map[cloud_map == 0.0] = np.nan
        self.cloud_map = cloud_map

        axins.imshow(cloud_map, origin=0, cmap='Reds')
        axins.tick_params(labelleft=False, labelbottom=False)
        axins.set_title("13CO")
        axins.set_xticks([])
        axins.set_yticks([])

        axins = inset_axes(ax, width="20%", height="20%", loc=4)

        cloud_map = np.zeros_like(self.c18o_mask, dtype=np.float)
        cloud_map[self.co12_mask] = 1.0
        cloud_map[self.c18o_mask] = 2.0
        #         cloud_map = self.co12_mask.astype(np.float) + self.co13_mask.astype(np.float)
        cloud_map[cloud_map == 0.0] = np.nan
        self.cloud_map = cloud_map

        axins.imshow(cloud_map, origin=0, cmap='Reds')
        axins.tick_params(labelleft=False, labelbottom=False)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("C18O")
        #         plt.ylim(-0.1, top)
        #         plt.savefig("test.eps")
        if target_dir is None:
            plt.show()
        else:
            a = plt.savefig(target_dir + "/" + f"cloud{self.idx}.eps")

    @staticmethod
    def read_projection(filename):
        data, header = fits.getdata(filename), fits.getheader(filename)
        return Projection(value=data, wcs=WCS(header))