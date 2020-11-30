from multi_fits_cubes.cloud import CloudManager
from multi_fits_cubes.plotting import AvgSpec3LineValidPlotter, MWISP_CO_LINE_LATEX
from collections import OrderedDict

if __name__ == '__main__':

    cm = CloudManager('../mask_files_12CO/', big_cubes=OrderedDict({'12CO': '../M195_U.fits',
                                                                   '13CO': '../M195_L.fits',
                                                                   'C18O': '../M195_L2.fits'}))
    cloud = cm.load_cloud(415685)
    plotter = AvgSpec3LineValidPlotter(cloud, {#'12CO': '../test_data/bigs/M195_U_rms.fits',
                                               '13CO': '../M195_L_rms.fits',
                                               'C18O': '../M195_L2_rms.fits'
                                               },
                                       valid_n_sigma=3)
    plotter.set_line_scale({'12CO': 1, '13CO': 3, 'C18O': 5})
    plotter.set_line_name_latex(MWISP_CO_LINE_LATEX)
    plotter.prepare()
    plotter.plot_avg_spec(target_dir='.')
    plotter.cloud['12CO'].write("Cloud415685_12CO_data.fits")
    # plotter.cloud['C18O'].write("Cloud415685_C18O_data.fits")
    # plotter.valid_data_cubes['C18O'].write("Cloud415685_C18O_validdata.fits")
    # plotter.cloud['12CO'].write("Cloud415685_C18O_data.fits")





