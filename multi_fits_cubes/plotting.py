import numpy as np
from multi_fits_cubes.cloud import Cloud
from multi_fits_cubes.helpers import MaskCube, ValidationMap, Map
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from astropy.visualization import quantity_support


quantity_support()

MWISP_CO_LINE_LATEX = {'12CO': "{}^{12}CO", '13CO': "{}^{13}CO", 'C18O': "C^{18}O"}


class SpecPlotter:
    def __init__(self, cloud: Cloud, figsize=None):
        self.cloud = cloud
        self.figure = plt.figure(figsize=figsize)
        self.fig_num = self.figure.number


class AvgSpec3LineValidPlotter(SpecPlotter):

    def __init__(self, cloud: Cloud, big_rms_files_or_maps: dict, valid_n_sigma=3, figsize=None):
        super().__init__(cloud, figsize)
        self.spectral_axes = {}
        self.extra_v_data_cubes = {}
        self.extra_v_2d_masked_avg_specs = {}

        self.valid_data_cubes = {}
        self.valid_avg_specs = {}

        self.n_sigma = valid_n_sigma
        self.big_rms_maps = {k: Map.from_file(v) if isinstance(v, str) else v for k, v in big_rms_files_or_maps.items()}

        self.valid_map_mask_np = {}
        self.valid_point_counts = {}

        self.cross_checked_valid_map_np = {}

        self.line_name_latex = {}

        self.line_scales = {k: 1 for k in self.cloud.line_names}

    def set_line_scale(self, d: dict):
        """
        e.g.
        {'12CO': 1, '13CO': 3, 'C18O': 5}
        :param d:
        :return:
        """
        self.line_scales = d

    def set_line_name_latex(self, d: dict):
        """
        e.g.
        { '12CO': "{}^{12}CO", '13CO': "{}^{13}CO", 'C18O': "C^{18}O" }
        :param d:
        :return:
        """
        self.line_name_latex = d

    def prepare(self, lines=None, cross_check_valid_rules=('C18O : 13CO & C18O',)):
        if lines is None:
            lines = self.cloud.line_names

        for line in lines:
            self._prepare_extra_v_cube(line)
            self.spectral_axes[line] = self.extra_v_data_cubes[line].spectral_axis

        line = lines[0]
        self.valid_data_cubes[line] = self.cloud[line]
        self.valid_map_mask_np[line] = self.cloud.mask_cube_obj.get_mask_map2d()
        for line in lines[1:]:
            self._prepare_valid_map(line)

        self._prepare_cross_check_valid_map(rules=cross_check_valid_rules)

        for line in lines:
            self._prepare_avg_line(line)

    def _prepare_extra_v_cube(self, line_name):
        vlo, vhi = self.cloud.mask_cube_obj.mask3d.spectral_extrema
        delta_v = (vhi - vlo)
        cube = self.cloud.get_cube_with_extra_v_left_right_range(line_name, delta_v, delta_v)
        self.extra_v_data_cubes[line_name] = cube

    def _prepare_valid_map(self, line_name):
        data_cube = self.cloud[line_name]
        big_rms = self.big_rms_maps[line_name]
        validmap = ValidationMap(n_sigma=self.n_sigma)
        validmap.calculate_sum_cover_from_datacube(data_cube)
        validmap.cut_rms_map_from_big_rms_map(big_rms)

        validmap_np = validmap.get_valid_map()
        n_valid_pix = validmap.get_n_valid_pix()

        self.valid_map_mask_np[line_name] = validmap_np
        self.valid_point_counts[line_name] = n_valid_pix

        valid_cube = data_cube.with_mask(validmap_np)
        valid_cube.allow_huge_operations = True
        self.valid_data_cubes[line_name] = valid_cube

    def _prepare_cross_check_valid_map(self, rules=('C18O : 13CO & C18O',)):

        for rule in rules:
            target, input_lines = rule.split(' : ')
            operand_1, operator, operand_2 = input_lines.split(' ')
            operand_1 = 'self.valid_map_mask_np["' + operand_1 + '"]'
            operand_2 = 'self.valid_map_mask_np["' + operand_2 + '"]'
            expression = operand_1 + ' ' + operator + ' ' + operand_2
            cc_valid_map = eval(expression)
            self.cross_checked_valid_map_np[target] = cc_valid_map
            self.valid_point_counts[target] = np.sum(cc_valid_map)

        for line in self.cloud.line_names:
            if line not in self.cross_checked_valid_map_np:
                self.cross_checked_valid_map_np[line] = self.valid_map_mask_np[line]

    def _prepare_avg_line(self, line_name):
        # valid_cube = self.valid_data_cubes[line_name]
        # valid_avg_spec = valid_cube.mean(axis=(1, 2))
        # self.valid_avg_specs[line_name] = valid_avg_spec
        extra_v_cube = self.extra_v_data_cubes[line_name]
        valid_map = self.cross_checked_valid_map_np[line_name]
        extra_v_avg_spec = extra_v_cube.with_mask(valid_map).mean(axis=(1, 2))
        self.extra_v_2d_masked_avg_specs[line_name] = extra_v_avg_spec

    def plot_avg_spec(self, target_dir=None, overlay_valid_maps=True):
        plt.figure(self.fig_num)

        for line in self.cloud.line_names:
            spec_axis = self.spectral_axes[line]
            avg_spec = self.line_scales[line] * self.extra_v_2d_masked_avg_specs[line]
            plt.step(spec_axis, avg_spec,
                     where='mid',
                     label=f'${self.line_scales[line]}' + r'\times' + ' ' + self.line_name_latex[line] + '$',
                     alpha=0.95,
                     linewidth=1.5)

        ax = plt.gca()
        plt.tick_params(direction='in', which='both')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


        bottom, top = plt.ylim()
        plt.ylim(bottom - 0.25 * (top - bottom), top)
        bottom, top = plt.ylim()
        left, right = plt.xlim()
        plt.hlines(0, left, right, color="black", linestyle=':')
        # plt.vlines(self.cloud.v_cen, bottom, top, color='black', linestyle=":")
        plt.xlim(left, right)
        plt.ylim(bottom, top)

        signal_vlo, signal_vhi = self.cloud.mask_cube_obj.mask3d.spectral_extrema
        vlo, vhi = self.extra_v_data_cubes[self.cloud.line_names[0]].spectral_extrema
        plt.xlim(vlo, vhi)
        plt.axvspan(vlo, signal_vlo, ymin=bottom, ymax=top, facecolor='gray', alpha=0.1)
        plt.axvspan(signal_vhi, vhi, ymin=bottom, ymax=top, facecolor='gray', alpha=0.1)

        if hasattr(self.cloud, 'catalog'):
            v_cen = self.cloud.catalog['v_cen'][0]
            plt.vlines(v_cen, bottom, top, color='black', linestyle=':')


        plt.legend(loc=1)
        line2, line3 = self.cloud.line_names[1:]
        plt.title(
            f"{self.cloud.name}, {line2} N Pixels={self.valid_point_counts[line2]}, {line3} N Pixels={self.valid_point_counts[line3]}")

        if overlay_valid_maps:

            ax = plt.gca()
            loc = 3
            for target_line in self.cloud.line_names[1:]:
                cloud_map = np.zeros_like(self.valid_map_mask_np[target_line], dtype=np.float)
                i = 1.0
                for line in (self.cloud.line_names[0], target_line):
                    cloud_map[self.cross_checked_valid_map_np[line]] = i
                    i += 1.0
                cloud_map[cloud_map == 0.0] = np.nan
                axins = inset_axes(ax, width="20%", height="20%", loc=loc)
                loc += 1
                axins.imshow(cloud_map, origin='lower', cmap='Reds')
                axins.tick_params(labelleft=False, labelbottom=False)
                axins.set_title(target_line)
                axins.set_xticks([])
                axins.set_yticks([])

        if target_dir is None:
            plt.show()
        else:
            a = plt.savefig(target_dir + "/" + f"{self.cloud.name}_avg_spec.pdf")


class HistPlotter:
    def __init__(self, cloud):
        pass

