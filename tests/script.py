from multi_fits_cubes.cloud import CloudManager
from multi_fits_cubes.plotting import AvgSpec3LineValidPlotter, MWISP_CO_LINE_LATEX
from collections import OrderedDict
from spectral_cube import SpectralCube
from matplotlib import pyplot as plt
from astropy import units as u
kps = u.km / u.s

if __name__ == '__main__':

    big12 = SpectralCube.read("../test_data/bigs/M195_U.fits")
    big13 = SpectralCube.read("../test_data/bigs/M195_L.fits")
    big18 = SpectralCube.read("../test_data/bigs/M195_L2.fits")

    cm = CloudManager('../test_data/masks/', big_cubes=OrderedDict({'12CO': '../test_data/bigs/M195_U.fits',
                                                                   '13CO': '../test_data/bigs/M195_L.fits',
                                                                   # 'C18O': '../test_data/bigs/M195_L2.fits'
                                                                }),
                      catalog='../test_data/catalogs/12CO_less220_catalog.fit')
    cloud = cm.load_cloud(1254)
    plotter = AvgSpec3LineValidPlotter(cloud, {#'12CO': '../test_data/bigs/M195_U_rms.fits',
                                               '13CO': '../test_data/bigs/M195_L_rms.fits',
                                               # 'C18O': '../test_data/bigs/M195_L2_rms.fits'
                                               },
                                       valid_n_sigma=3)
    plotter.set_line_scale({'12CO': 1, '13CO': 3})
    MWISP_CO_LINE_LATEX.pop('C18O')
    plotter.set_line_name_latex(MWISP_CO_LINE_LATEX)
    plotter.prepare(cross_check_valid_rules=[])
    plotter.plot_avg_spec()


    plotter.plot_avg_spec(target_dir='.')
    # plotter.cloud['12CO'].write("Cloud415685_12CO_data.fits")
    # plotter.cloud['C18O'].write("Cloud415685_C18O_data.fits")
    # plotter.valid_data_cubes['C18O'].write("Cloud415685_C18O_validdata.fits")
    # plotter.cloud['12CO'].write("Cloud415685_C18O_data.fits")





