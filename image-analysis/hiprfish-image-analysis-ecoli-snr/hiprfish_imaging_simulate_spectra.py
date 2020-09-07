
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import sys
import glob
import joblib
import skimage
import argparse
import javabridge
import bioformats
import numpy as np
import pandas as pd
import skimage.filters
from skimage import color
from skimage import feature
from skimage import restoration
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################


def simulate_spectra(sample, bkg_spectra_filename, bkg_intensity, dilution):
    spectra_filename = '{}_avgint.csv'.format(sample)
    avgint = pd.read_csv(spectra_filename)
    bkg_spec = pd.read_csv(bkg_spectra_filename, header = None)
    avgint = pd.read_csv(spectra_filename, header = None)
    avgint_filtered = avgint.iloc[:,0:95].copy().values
    avgint_filtered[:,0:32] = avgint_filtered[:,0:32] - (bkg_intensity*bkg_spec.iloc[0:32,0])[None,:]
    avgint_norm = avgint_filtered/np.max(avgint_filtered, axis = 1)[:,None]
    avgint_pc = np.abs(np.average(avgint_norm, axis = 0)*3*500/dilution)
    simulated_spectra = np.zeros((1000, 95))
    for i in range(95):
        simulated_spectra[:,i] = np.random.poisson(avgint_pc[i], 1000)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    simulated_spectra_filename = '{}_avgint_dilution_{}.csv'.format(sample, dilution)
    pd.DataFrame(simulated_spectra_norm).to_csv(simulated_spectra_filename, header = None, index = None)
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('-s', '--spectra_filename', dest = 'spectra_filename', default = '', type = str, help = 'Image filenames')
    parser.add_argument('-b', '--bkg_intensity', dest = 'bkg_intensity', type = float, default = 0.0, help = 'Background intensity')
    parser.add_argument('-bs', '--bkg_spectra_filename', dest = 'bkg_spectra_filename', type = str, default = '', help = 'Background spectra')
    parser.add_argument('-d', '--dilution', dest = 'dilution', type = int, default = 1, help = 'Dilution factor')
    args = parser.parse_args()
    sample = re.sub('_avgint.csv', '', args.spectra_filename)
    simulate_spectra(sample, args.bkg_spectra_filename, args.bkg_intensity, args.dilution)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
