import numpy as np
from spectral_cube import SpectralCube
from astropy import units as u


class MaskCube:
    def __init__(self, cube: SpectralCube):
        self.mask3d = cube
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

    def cut_from_a_big_cube(self, big_cube: SpectralCube, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        """
        Cut the part from a big spectral cube that corresponds to the current mask cube's world extrema
        e.g. Cut the 30° < l < 30.5°, 0° < b < 0.5° from a big mosaic fits.
        :param big_cube:
        :return:
        """
        l, b = self.mask3d.world_extrema
        llo, lhi = l
        blo, bhi = b
        vlo, vhi = self.mask3d.spectral_extrema

        new_cube = big_cube.subcube(
            xlo=llo, xhi=lhi,
            ylo=blo, yhi=bhi,
            zlo=vlo, zhi=vhi).with_spectral_unit(with_vel_unit)

        if new_cube.unit == u.dimensionless_unscaled:
            new_cube = new_cube * with_value_unit

        return new_cube

    def cut_and_mask_from_big_cube(self, big_cube, with_vel_unit=u.km / u.s, with_value_unit=u.K):
        new_cube = self.cut_from_a_big_cube(big_cube, with_vel_unit, with_value_unit)
        mask3d = self.mask_of_a_new_cube(new_cube)
        return new_cube.with_mask(mask3d)

    @staticmethod
    def from_file(fits_filename: str):
        cube = SpectralCube.read(fits_filename)
        return MaskCube(cube)

if __name__ == '__main__':
    maskcube = MaskCube.from_file("../test_data/cloud415685mask.fits")
    c18o_cube = SpectralCube.read("../test_data/cloud415685_masked_C18O_data.fits")
    # new_v_idx = maskcube.map_spectral_axis_to_new_one(c18o_cube.wcs)
    # maskcube.check_spectral_mapping(new_v_idx, c18o_cube.spectral_axis, c18o_cube.wcs)

    new_mask = maskcube.mask_of_a_new_cube(c18o_cube)
    c18o_cube.with_mask(new_mask).write("test.fits")


