from multi_fits_cubes.cloud import Cloud
import numpy as np
from spectral_cube import SpectralCube
from collections import OrderedDict
import astropy.units as u
from astropy.wcs import WCS
kps = u.km / u.s


class FluxRMS(Cloud):
    def __init__(self, signal_v_range, *args, **kwargs):
        super().__init__(*args, **kwargs)

        master_line = list(self.cubes.keys())[0]
        self.masked_cube = self.cubes[master_line]
        self.mask2d = self.mask_cube_obj.get_mask_map2d()

        self.cutted_big_cubes = OrderedDict()
        for line, cube in self.big_cubes.items():
            vlo, vhi = cube.spectral_extrema
            cutted_cube = self.mask_cube_obj.cut_from_a_big_cube_v_range(cube, vlo, vhi, with_vel_unit=kps,
                                                                         with_value_unit=u.K)
            self.cutted_big_cubes[line] = cutted_cube.with_mask(self.mask2d)

        self.signal_vlo, self.signal_vhi = signal_v_range

        n_vox = np.sum(~np.isnan(cloud.mask_cube_obj.mask3d.filled_data[:].value))
        n_pix = np.sum(cloud.mask_cube_obj.get_mask_map2d())
        self.avg_n_channel = int(round(n_vox / n_pix))

        """
        v_resolution = abs(self.big_cube.spectral_axis[0] - self.big_cube.spectral_axis[1])
        self.signal_channel_num = int(((self.signal_vhi - self.signal_vlo) / v_resolution).value)

        self.set_full_v_range(*self.big_cube.spectral_extrema)

        self.unit = self.big_cube.unit
        v_axis = self.big_cube.spectral_axis
        self.channel_width = abs(v_axis[1] - v_axis[0])
        """

    def sliding_rms_values(self, line, n_channel_step_size):
        """
        Calculate the sliding flux rms values
        :param line:
        :param n_channel_step_size:
        :return:
        """
        cube = self.cutted_big_cubes[line]
        n_tot_channels = cube.shape[0]

        v_axis = cube.spectral_axis.to(kps).value

        t = np.where(np.isclose(v_axis, self.signal_vlo.value, atol=0.07))[0]
        #         print(v_axis, self.signal_vlo)
        #         print(t)
        #         assert len(t) == 1
        self.signal_channel_lo = t[0]

        t = np.where(np.isclose(v_axis, self.signal_vhi.value, atol=0.07))[0]
        #         print(v_axis, self.signal_vhi)
        #         print(t)
        #         assert len(t) == 1
        self.signal_channel_hi = t[0]

        self.signal_channel_num = self.signal_channel_hi - self.signal_channel_lo + 1

        part1_upper_bound = self.signal_channel_lo
        part2_lower_bound = self.signal_channel_hi + 1
        part1 = np.arange(0, part1_upper_bound - self.avg_n_channel, n_channel_step_size, dtype=np.int)
        part2 = np.arange(part2_lower_bound, n_tot_channels - self.avg_n_channel, n_channel_step_size, dtype=np.int)
        # print('part1', part1)
        # print('part2', part2)
        start_channel_indices = np.hstack([part1, part2])
        flux_arr = u.Quantity([self._rms_in_one_box(cube, i) for i in start_channel_indices])

        return start_channel_indices, flux_arr

    def _rms_in_one_box(self, cube, start_channel_index):
        i = start_channel_index
        j = i + self.avg_n_channel

        box = cube[i: j, :, :]

        box.allow_huge_operations = True
        m0 = box.moment0()
        flux = np.nansum(m0.value) * m0.unit
        return flux


class FluxRMSold(Cloud):
    def __init__(self, line=None, signal_v_range=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if line is None:
            assert len(self.cubes) == 1, "Arguments `line` can be None iff exactly 1 line in the Cloud object."
            line = list(self.cubes.keys())[0]

        self.masked_cube = self.cubes[line]
        self.mask2d = self.mask_cube_obj.get_mask_map2d()

        self.big_cube = self.big_cubes[line]
        if signal_v_range is None:
            self.signal_vlo, self.signal_vhi = self.masked_cube.spectral_extrema
            self.signal_channel_num = self.masked_cube.shape[0]
        else:
            self.signal_vlo, self.signal_vhi = signal_v_range
            v_resolution = abs(self.big_cube.spectral_axis[0] - self.big_cube.spectral_axis[1])
            self.signal_channel_num = int(((self.signal_vhi - self.signal_vlo) / v_resolution).value)

        self.set_full_v_range(*self.big_cube.spectral_extrema)

        self.unit = self.big_cube.unit
        v_axis = self.big_cube.spectral_axis
        self.channel_width = abs(v_axis[1] - v_axis[0])


    def set_full_v_range(self, vlo, vhi):
        """
        This v range is not for signal, but for the range in which the rms will be calculated.
        If you want to use the full velocity range in the big cube,
        ignore this method, because the constructor will do that for you.
        :param vlo:
        :param vhi:
        :return:
        """
        self.masked_cube_with_larger_v_range = \
            self.mask_cube_obj.cut_from_a_big_cube_v_range(
                big_cube=self.big_cube,
                vlo=vlo,
                vhi=vhi
            ).with_mask(self.mask2d)

        v_axis = self.masked_cube_with_larger_v_range.spectral_axis.value

        t = np.where(np.isclose(v_axis, self.signal_vlo.value))[0]
        if len(t) == 0:
            self.signal_channel_lo = None
        else:
            assert len(t) == 1
            self.signal_channel_lo = t[0]

        t = np.where(np.isclose(v_axis, self.signal_vhi.value))[0]
        if len(t) == 0:
            self.signal_channel_hi = None

        else:
            assert len(t) == 1
            self.signal_channel_hi = t[0] + 1

        self.masked_cube_with_larger_v_range_data = self.masked_cube_with_larger_v_range.filled_data[:].value

        return self.masked_cube_with_larger_v_range

    def sliding_rms_values(self, n_channel_step_size, n_channel_extra_clip, unit=u.K * u.km / u.s):
        """
        Calculate the sliding flux rms values
        :param n_channel_step_size:
        :param n_channel_extra_clip:
        :return:
        """
        if self.signal_channel_lo is None:
            self.signal_channel_lo = 0
        if self.signal_channel_hi is None:
            self.signal_channel_hi = self.big_cube.shape[0]

        part1_upper_bound = self.signal_channel_lo - n_channel_extra_clip
        part2_lower_bound = self.signal_channel_hi + n_channel_extra_clip
        part1 = np.arange(0, part1_upper_bound - self.signal_channel_num, n_channel_step_size)
        part2 = np.arange(part2_lower_bound,
                          self.masked_cube_with_larger_v_range.shape[0] - self.signal_channel_num,
                          n_channel_step_size)
        # print('part1', part1)
        # print('part2', part2)
        start_channel_indices = np.hstack([part1, part2])
        flux_arr = np.array([self._rms_in_one_box(i) for i in start_channel_indices])

        return start_channel_indices, (flux_arr * self.unit * self.channel_width).to(unit)

    def channel_index_to_velocity(self, channel_indices, with_unit=u.km / u.s):
        return self.big_cube.spectral_axis[channel_indices].to(with_unit)

    def _rms_in_one_box(self, start_channel_index):
        i = start_channel_index
        j = i + self.signal_channel_num

        box = self.masked_cube_with_larger_v_range_data[i: j, :, :]
        flux = np.nansum(box)
        return flux

    @staticmethod
    def make_fake_square_cloud(big_cube: SpectralCube, cloud_vlo, cloud_vhi):
        """
        Assume that
        there is a square cloud with a specific vrange,
        we have a datacube with the same spatial coverage (but larger velocity coverage)
        and our goal is to estimate the flux rms level in the datacube.
        :param big_cube:
        :param cloud_vlo:
        :param cloud_vhi:
        :return: FluxRMS object
        """
        bvlo, bvhi = big_cube.spectral_extrema
        v_resolution = abs(big_cube.spectral_axis[0] - big_cube.spectral_axis[1])
        if cloud_vlo >= bvhi or cloud_vhi <= bvlo:
            cloud_number_of_channels = int((abs(cloud_vhi - cloud_vlo) / v_resolution).value)
            template = big_cube.subcube(zlo=0, zhi=cloud_number_of_channels)
            data = np.ones(template.shape)
            header = template.header.copy()
            header['CRPIX3'] = 0
            header['CRVAL3'] = cloud_vlo.to(big_cube.spectral_axis.unit).value
            cloud_cube = SpectralCube(data=data, wcs=WCS(header))
        else:
            cloud_cube = big_cube.subcube(zlo=cloud_vlo, zhi=cloud_vhi)

        print(cloud_cube.spectral_extrema)
        cloud = FluxRMS(name='fakesquarecloud',
                        mask_cube=cloud_cube,
                        big_cubes=OrderedDict({'line': big_cube}),
                        signal_v_range=(cloud_vlo, cloud_vhi))
        return cloud


if __name__ == '__main__':
    # test
    from multi_fits_cubes.cloud import CloudManager
    from matplotlib import pyplot as plt
    cube = SpectralCube.read('../test_data/bigs/M195_L2.fits')
    cube.allow_huge_operations = True
    cube = cube * u.K
    big_cubes = OrderedDict({'C18O':  cube})
    cm = CloudManager('../test_data/masks', big_cubes=big_cubes, cls=FluxRMSold)
    for idx in [1246, 1254, 7474]:
        cloud = cm.load_cloud(idx)
        start_channels, flux_rms_values = cloud.sliding_rms_values(3, 5)
        plt.hist(flux_rms_values, bins=25, alpha=0.5, label='Cloud' + str(idx))
    plt.legend()
    plt.show()
    print(flux_rms_values)

    '''
    big_cube = SpectralCube.read("../test_data/noise/Noise_1_100_195_0_L.fits")
    big_cube.allow_huge_operations = True
    big_cube = big_cube * u.K
    bvlo, bvhi = big_cube.spectral_extrema
    cloud_v_width = 10 * u.km / u.s
    cloud = FluxRMS.make_fake_square_cloud(big_cube=big_cube,
                                   cloud_vlo=bvlo - cloud_v_width,
                                   cloud_vhi=bvlo)
    print(cloud['line'].spectral_extrema)
    start_channels, flux_rms_values = cloud.sliding_rms_values(4, 0)
    plt.hist(flux_rms_values)
    plt.show()
    plt.plot(start_channels, flux_rms_values)
    plt.show()
    v = cloud.channel_index_to_velocity(start_channels, with_unit=u.km / u.s)
    plt.plot(v, flux_rms_values)
    plt.show()
    print()

    cloud_v_width = 20 * u.km / u.s
    cloud = FluxRMS.make_fake_square_cloud(big_cube=big_cube,
                                           cloud_vlo=bvlo - cloud_v_width,
                                           cloud_vhi=bvlo)
    print(cloud['line'].spectral_extrema)
    start_channels, flux_rms_values = cloud.sliding_rms_values(4, 0)
    plt.hist(flux_rms_values)
    plt.show()
    plt.plot(start_channels, flux_rms_values)
    plt.show()
    v = cloud.channel_index_to_velocity(start_channels, with_unit=u.km / u.s)
    plt.plot(v, flux_rms_values)
    plt.show()
    '''






