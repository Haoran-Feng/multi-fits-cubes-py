from multi_fits_cubes.cloud import CloudManager
from multi_fits_cubes.plotting import AvgSpec3LineValidPlotter, MWISP_CO_LINE_LATEX
from collections import OrderedDict

if __name__ == '__main__':

    cm = CloudManager('../test_data/masks', big_cubes=OrderedDict({'12CO': '../test_data/bigs/M195_U.fits',
                                                                   # '13CO': '../test_data/bigs/M195_L.fits',
                                                                   'C18O': '../test_data/bigs/M195_L2.fits'}))
    cloud = cm.load_cloud(1246)
    plotter = AvgSpec3LineValidPlotter(cloud, {#'12CO': '../test_data/bigs/M195_U_rms.fits',
                                               '13CO': '../test_data/bigs/M195_L_rms.fits',
                                               'C18O': '../test_data/bigs/M195_L2_rms.fits'
                                               },
                                       valid_n_sigma=3)
    plotter.set_line_scale({'12CO': 1, '13CO': 3, 'C18O': 5})
    plotter.set_line_name_latex(MWISP_CO_LINE_LATEX)
    plotter.prepare()
    plotter.plot_avg_spec(target_dir='.')



