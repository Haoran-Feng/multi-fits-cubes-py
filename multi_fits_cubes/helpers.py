import numpy as np
from spectral_cube import SpectralCube, Projection
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS


class MaskCube:
    def __init__(self, maskcube: SpectralCube):
        self.mask3d = maskcube
        self.spectral_axis = self.mask3d.spectral_axis
        self.spectral_resolution = self.spectral_axis[-1] - self.spectral_axis[-2]

    def map_spectral_axis_to_new_one(self, new_wcs):
        vel_values = self.spectral_axis.value
        N = len(vel_values)
        dummy_nx3_array = np.vstack([np.ones(N), np.ones(N), vel_values]).T
        pix_coord_on_new = new_wcs.all_world2pix(dummy_nx3_array, 0)
        v_pix = pix_coord_on_new[:, 2]
        v_idx = np.round(v_pix).astype(np.int)

        return v_idx[v_idx >= 0]

    def check_spectral_mapping(self, new_v_idx, new_v_spectral_axis, new_wcs):
        new_v_spectral_axis = new_v_spectral_axis.to(self.spectral_axis.unit)
        N = len( new_v_idx)
        dummy_nx3_array = np.vstack([np.ones(N), np.ones(N),  new_v_idx]).T
        v = new_wcs.all_pix2world(dummy_nx3_array, 0)[:, 2]
        assert np.all(np.isclose(v, new_v_spectral_axis.value[new_v_idx], atol=0.0001)), "Mapping Spectral Axis Failure!"
        return

    def mask_of_a_new_cube(self, new_cube):
        """
        Return 3D mask array for new_cube, this mask corresponds to the mask -- self.mask3d, but with different
        resolution on the third axis (velocity axis)
        :param new_cube:
        :return:
        """
        new_v_idx = self.map_spectral_axis_to_new_one(new_cube.wcs)

        # check if the velocity axis mapping succeed.
        self.check_spectral_mapping(new_v_idx, new_cube.spectral_axis, new_cube.wcs)

        current_mask_np = ~np.isnan(self.mask3d.filled_data[:].value) & (self.mask3d.filled_data[:].value != 0)


        mask_for_new_cube_np = np.zeros(new_cube.shape).astype(bool)
        for i, v_idx in enumerate(new_v_idx):
            mask_for_new_cube_np[v_idx, :, :] = current_mask_np[i, :, :]

        return mask_for_new_cube_np

    def cut_from_a_big_cube_v_range(self, big_cube: SpectralCube, vlo, vhi, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        l, b = self.mask3d.world_extrema
        llo, lhi = l
        blo, bhi = b
        # vlo, vhi = self.mask3d.spectral_extrema

        new_cube = big_cube.subcube(
            xlo=llo, xhi=lhi,
            ylo=blo, yhi=bhi,
            zlo=vlo, zhi=vhi).with_spectral_unit(with_vel_unit)

        if new_cube.unit == u.dimensionless_unscaled:
            new_cube = new_cube * with_value_unit

        return new_cube


    def cut_from_a_big_cube(self, big_cube: SpectralCube, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        """
        Cut the part from a big spectral cube that corresponds to the current mask cube's world extrema
        e.g. Cut the 30° < l < 30.5°, 0° < b < 0.5° from a big mosaic fits.
        :param big_cube:
        :return:
        """
        vlo, vhi = self.mask3d.spectral_extrema
        new_cube = self.cut_from_a_big_cube_v_range(big_cube, vlo, vhi, with_vel_unit, with_value_unit)
        return new_cube

    def cut_from_a_big_cube_with_v_cen_width(self, big_cube: SpectralCube,
                                             v_cen: u.Quantity,
                                             v_width: u.Quantity,
                                             with_vel_unit=u.km / u.s,
                                             with_value_unit=u.K):
        """
        Cut the part from a big spectral cube that corresponds to the current mask cube's world extrema
        e.g. Cut the 30° < l < 30.5°, 0° < b < 0.5° from a big mosaic fits, and on the third dimension,
        from v_cen - 0.5 * v_width to v_cen + 0.5 * v_width
        """
        new_cube = self.cut_from_a_big_cube_v_range(big_cube, v_cen - 0.5 * v_width, v_cen + 0.5 * v_width, with_vel_unit, with_value_unit)
        return new_cube

    def cut_from_a_big_cube_with_extra_v_range(self,
                                               big_cube: SpectralCube,
                                               extra_v_left: u.Quantity,
                                               extra_v_right: u.Quantity,
                                               with_vel_unit=u.km / u.s,
                                               with_value_unit=u.K):
        """
        Cut the part from a big spectral cube that corresponds to the current mask cube's world extrema
        e.g. Cut the 30° < l < 30.5°, 0° < b < 0.5° from a big mosaic fits, and on the third dimension,
        from vlo - extra_v_left to vhi + extra_v_right
        """
        vlo, vhi = self.mask3d.spectral_extrema
        vlo = vlo - extra_v_left
        vhi = vhi + extra_v_right
        new_cube = self.cut_from_a_big_cube_v_range(big_cube, vlo, vhi, with_vel_unit, with_value_unit)
        return new_cube

    def cut_and_mask_from_big_cube(self, big_cube, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        new_cube = self.cut_from_a_big_cube(big_cube, with_vel_unit, with_value_unit)
        mask3d = self.mask_of_a_new_cube(new_cube)
        return new_cube.with_mask(mask3d)

    def get_mask_map2d(self):
        sum_map_np = self.mask3d.sum(axis=0).filled_data[:].value
        mask_map_2d = ~np.isnan(sum_map_np) & (sum_map_np != 0)
        return mask_map_2d

    def cut_extra_v_and_2d_mask_from_big_cube(self, big_cube, v_cen, v_width, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        new_cube = self.cut_from_a_big_cube_with_v_cen_width(big_cube, v_cen, v_width, with_vel_unit, with_value_unit)
        mask2d = self.get_mask_map2d()
        return new_cube.with_mask(mask2d)

    def cut_extra_v_left_v_right_and_2d_mask_from_big_cube(self, big_cube, extra_v_left, extra_v_right, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        new_cube = self.cut_from_a_big_cube_with_extra_v_range(big_cube, extra_v_left, extra_v_right, with_vel_unit, with_value_unit)
        mask2d = self.get_mask_map2d()
        return new_cube.with_mask(mask2d)

    @staticmethod
    def from_file(fits_filename: str):
        cube = SpectralCube.read(fits_filename)
        return MaskCube(cube)


class ValidationMap:
    """
    A Map that can reflects whether each pixel in the data cube contains valid signal
    """
    def __init__(self, n_sigma=3):
        self.sum_map = None
        self.cover_map = None
        self.rms_map = None
        self.valid_map = None

        self.n_sigma = n_sigma

    def calculate_sum_cover_from_datacube(self, data_cube: SpectralCube):
        data_cube.allow_huge_operations = True
        self.sum_map = data_cube.sum(axis=0)
        data_np = data_cube.filled_data[:].value
        cover_3d = ~np.isnan(data_np)
        self.cover_map = cover_3d.sum(axis=0).astype(np.int)

    def set_rms_value(self, rms=0.3):
        self._check_if_loaded()
        self.rms_map = rms * np.ones_like(self.sum_map)

    def cut_rms_map_from_big_rms_map(self, big_rms_map: Projection):
        self._check_if_loaded()
        l, b = self.sum_map.world_extrema
        llo, lhi = l
        blo, bhi = b
        self.rms_map = big_rms_map.subimage(lhi, llo, blo, bhi)
        return self.rms_map

    def get_valid_map(self):
        self._check_if_loaded()
        if not self.valid_map:
            sum_np = self.sum_map.value
            rms_map_np = self.rms_map.value
            self.valid_map = sum_np > (3 * rms_map_np * np.sqrt(self.cover_map))

        return self.valid_map

    def get_n_valid_pix(self):
        if self.valid_map is None:
            self.get_valid_map()

        return np.sum(self.valid_map)

    def _check_if_loaded(self):
        if self.sum_map is None:
            raise ValueError("Haven't load a sum map yet!")

        if self.cover_map is None:
            raise ValueError("Haven't load a cover map yet!")




class Map(Projection):
    @staticmethod
    def from_file(filename):
        data, header = fits.getdata(filename), fits.getheader(filename)
        header_copy = header.copy()
        for kw in header.keys():
            if kw.endswith('4') or kw.endswith('3'):
                del header_copy[kw]
        header['NAXIS'] = 2
        wcs = WCS(header_copy)
        return Map(value=data, wcs=wcs)


if __name__ == '__main__':
    maskcube = MaskCube.from_file("../test_data/cloud415685mask.fits")
    c18o_cube = maskcube.cut_and_mask_from_big_cube(SpectralCube.read('../test_data/M195_L2.fits'))
    big_rms18 = Map.from_file('../test_data/M195_L2_rms.fits')

    valid_map = ValidationMap(n_sigma=3)
    valid_map.calculate_sum_cover_from_datacube(c18o_cube)
    valid_map.cut_rms_map_from_big_rms_map(big_rms18)
    vm = valid_map.get_valid_map()
    n = valid_map.get_n_valid_pix()
    print(n)
    print(c18o_cube.sum())

    print(c18o_cube.with_mask(vm).sum())

    c18o_cube_invalid = c18o_cube.with_mask(~vm)

    valid_map = ValidationMap(n_sigma=3)
    valid_map.calculate_sum_cover_from_datacube(c18o_cube_invalid)
    valid_map.cut_rms_map_from_big_rms_map(big_rms18)
    vm = valid_map.get_valid_map()
    n = valid_map.get_n_valid_pix()

    print(n)
    print(c18o_cube_invalid.sum())



    # from matplotlib import pyplot as plt
    # plt.imshow(vm)
    # plt.show()

    # c18o_cube = SpectralCube.read("../test_data/cloud415685_masked_C18O_data.fits")
    # new_v_idx = maskcube.map_spectral_axis_to_new_one(c18o_cube.wcs)
    # maskcube.check_spectral_mapping(new_v_idx, c18o_cube.spectral_axis, c18o_cube.wcs)

    # new_mask = maskcube.mask_of_a_new_cube(c18o_cube)
    # c18o_cube.with_mask(new_mask).write("test.fits")


