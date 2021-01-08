from multi_fits_cubes.cloud import Cloud
import numpy as np


class FluxRMS(Cloud):
    def __init__(self, line=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if line is None:
            assert len(self.cubes) == 1, "Arguments `line` can be None iff exactly 1 line in the Cloud object."
            line = list(self.cubes.keys())[0]

        self.masked_cube = self.cubes[line]
        self.mask2d = self.mask_cube_obj.get_mask_map2d()

        self.big_cube = self.big_cubes[line]
        self.signal_vlo, self.signal_vhi = self.masked_cube.spectral_extrema
        self.signal_channel_num = self.masked_cube.shape[0]
        self.set_full_v_range(*self.big_cube.spectral_extrema)

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
        assert len(t) == 1
        self.signal_channel_lo = t[0]
        t = np.where(np.isclose(v_axis, self.signal_vhi.value))[0]
        assert len(t) == 1
        self.signal_channel_hi = t[0]
        if self.signal_channel_hi < self.signal_channel_lo:
            self.signal_channel_hi, self.signal_channel_lo = self.signal_channel_lo, self.signal_channel_hi

        self.signal_channel_hi += 1
        assert (self.signal_channel_hi - self.signal_channel_lo) == self.signal_channel_num

        self.masked_cube_with_larger_v_range_data = self.masked_cube_with_larger_v_range.filled_data[:].value

        return self.masked_cube_with_larger_v_range

    def sliding_rms_values(self, n_channel_step_size, n_channel_extra_clip):
        """
        Calculate the sliding flux rms values
        :param n_channel_step_size:
        :param n_channel_extra_clip:
        :return:
        """
        part1_upper_bound = self.signal_channel_lo - n_channel_extra_clip
        part2_lower_bound = self.signal_channel_hi + n_channel_extra_clip
        part1 = np.arange(0, part1_upper_bound - self.signal_channel_num, n_channel_step_size)
        part2 = np.arange(part2_lower_bound,
                          self.masked_cube_with_larger_v_range.shape[0] - self.signal_channel_num,
                          n_channel_step_size)
        start_channel_indices = np.hstack([part1, part2])
        flux_arr = np.array([self._rms_in_one_box(i) for i in start_channel_indices])
        return flux_arr

    def _rms_in_one_box(self, start_channel_index):
        i = start_channel_index
        j = i + self.signal_channel_num

        box = self.masked_cube_with_larger_v_range_data[i: j, :, :]
        flux = np.nansum(box)
        return flux


if __name__ == '__main__':
    # test
    from multi_fits_cubes.cloud import CloudManager
    from matplotlib import pyplot as plt
    cm = CloudManager('../test_data/masks', big_cubes={'C18O': '../test_data/bigs/M195_L2.fits'}, cls=FluxRMS)
    for idx in [1246, 1254, 7474]:
        cloud = cm.load_cloud(idx)
        flux_rms_values = cloud.sliding_rms_values(3, 5)
        plt.hist(flux_rms_values, bins=25, alpha=0.5, label='Cloud' + str(idx))
    plt.legend()
    plt.show()
    print(flux_rms_values)





