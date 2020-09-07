import os
import re
import sys
import glob
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import color
from ete3 import NCBITaxa
import bioformats
import javabridge
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from neighbor2d import line_profile_2d_v2
from matplotlib import collections


ncbi = NCBITaxa()
javabridge.start_vm(class_path=bioformats.JARS)



#####

def main():
    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-c', '--cell_info', dest = 'cell_info_filename', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, default = '', help = 'Input normalized single cell spectra filename')
    args = parser.parse_args()
    sample = re.sub('_cell_information.csv', '', args.cell_info_filename)
    segmentation_filename = '{}_seg.npy'.format(sample)
    segmentation = np.load(segmentation_filename)
    cell_info = pd.read_csv(args.cell_info_filename, dtype = {'cell_barcode':str})
    taxon_lookup = get_taxon_lookup(args.probe_design_filename)
    taxa_barcode_sciname = get_taxa_barcode_sciname(args.probe_design_filename)
    cell_info_filtered = analyze_cell_info(args.cell_info_filename, taxa_barcode_sciname)
    generate_identification_image(segmentation, cell_info_filtered, sample, taxon_lookup)
    return


if __name__ == '__main__':
    main()
