
from multi_fits_cubes.cloud import Cloud
from multi_fits_cubes.helpers import MaskCube, ValidationMap, Map
from matplotlib import pyplot as plt
from astropy.visualization import quantity_support
quantity_support()


class SpecPlotter:
    def __init__(self, cloud: Cloud, figsize=None):
        self.cloud = cloud
        self.figure = plt.figure(figsize)


class AvgSpec3LineValidPlotter(SpecPlotter):

    def __init__(self, cloud: Cloud, big_rms_files: dict, valid_n_sigma=3, figsize=None, ):
        super().__init__(cloud, figsize)
        self.spectral_axes = {}
        self.valid_data_cubes = {}
        self.valid_avg_specs = {}


        self.n_sigma = valid_n_sigma
        self.big_rms_maps = {k: Map.from_file(v) for k, v in big_rms_files}

        self.valid_map_mask_np = {}
        self.valid_point_counts = {}

        self.cross_checked_valid_map_np = {}






    def prepare(self, lines=('12CO', '13CO', 'C18O')):
        for line in lines:
            self._prepare_avg_line(line)

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

    def _prepare_cross_check_valid_map(self, rules=('C18O : 13CO & C18O')):
        for rule in rules:
            target, input_lines = rule.split(' : ')
            operand_1, operator, operand_2 = input_lines.split(' ')
            operand_1 = 'self.valid_map_mask_np[' + operand_1 + ']'
            operand_2 = 'self.valid_map_mask_np[' + operand_2 + ']'
            expression = operand_1 + ' ' + operator + ' ' + operand_2
            cc_valid_map = eval(expression)
            self.cross_checked_valid_map_np[target] = cc_valid_map

    def _prepare_avg_line(self, line_name):
        valid_cube = self.valid_data_cubes[line_name]
        valid_avg_spec = valid_cube.mean(axis=0)
        self.valid_avg_specs[line_name] = valid_avg_spec

    def plot_avg_spec(self, overlay_valid_maps=True):






class HistPlotter:
    def __init__(self, cloud):
        pass

if __name__ == '__main__':
