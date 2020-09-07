
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################


def estimate_ribosome_density(spectra_filename):
    spectra = pd.read_csv(spectra_filename)
    # pixel dwell time
    total_photon_per_pixel = np.average(np.sum(spectra.iloc[:,0:95], axis = 1))
    quantum_yield = 0.92
    ribosome_per_voxel = total_photon_per_pixel/quantum_yield
    cell_volume = np.pi*(0.5)**2*1
    voxel_volume = (0.07)*(0.07)*(0.15)
    average_total_ribosome = cell_volume/voxel_volume*ribosome_per_voxel
    average_ribosome_density = average_total_ribosome/cell_volume
    return(average_total_ribosome, average_ribosome_density)

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('-s', '--spectra_filename', dest = 'image_name', nargs = '*', default = [], type = str, help = 'Image filenames')
    args = parser.parse_args()
    sample = re.sub('_avgint.csv', '', args.spectra_filename)
    estimate_ribosome_density(args.spectra_filename)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
