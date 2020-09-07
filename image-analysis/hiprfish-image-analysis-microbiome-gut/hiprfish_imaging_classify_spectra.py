
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
import numpy as np
import pandas as pd
from skimage import measure

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def classify_spectra(input_spectra, ref_clf):
    sample = re.sub('_avgint.csv', '', input_spectra)
    avgint = pd.read_csv(input_spectra)
    image_seg = np.load('{}_seg.npy'.format(sample))
    image_registered = np.load('{}_registered.npy'.format(sample))
    avgint_norm = avgint.values/np.max(avgint.values, axis = 1)[:,None]
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    avgint_norm[:,63] = (np.max(avgint_norm[:,[3, 7, 11]], axis = 1) > 0.1)*1
    avgint_norm[:,64] = (np.max(avgint_norm[:,[24, 27, 29, 31, 33]], axis = 1) > 0.2)*1
    avgint_norm[:,65] = (np.max(avgint_norm[:,[43, 47]], axis = 1) > 0.25)*1
    avgint_norm[:,66] = (np.max(avgint_norm[:,[57, 60]], axis = 1) > 0.35)*1
    for r in range(20):
        avgint_identification_filename = '{}_cell_information_replicate_{}.csv'.format(sample, r)
        if not os.path.exists(avgint_identification_filename):
            print('Classifying sample {} with batch {}...'.format(os.path.basename(sample), r))
            umap_transform = joblib.load('{}_replicate_{}.pkl'.format(ref_clf, r))
            clf_umap = joblib.load('{}_svc_replicate_{}.pkl'.format(ref_clf, r))
            avgint_umap_transformed, nn_indices, nn_dists = umap_transform.transform(avgint_norm)
            avgint_umap_nn = umap_transform.embedding_[nn_indices[:,0],:]
            cell_ids_norm = clf_umap.predict(avgint_umap_nn)
            cell_info = pd.DataFrame(np.concatenate((avgint, avgint_norm[:,63:67], avgint_umap_nn, cell_ids_norm[:,None], np.min(nn_dists, axis = 1)[:,None]), axis = 1))
            cell_info.columns = ['channel_{}'.format(i) for i in range(63)] + ['intensity_classification_{}'.format(i) for i in range(4)] + ['umap_1', 'umap_2', 'umap_3', 'cell_barcode', 'nn_dists']
            cell_info['sample'] = sample
            cells = skimage.measure.regionprops(image_seg)
            cell_info['label'] = np.asarray([x.label for x in cells])
            cell_info['centroid_x'] = np.asarray([x.centroid[0] for x in cells])
            cell_info['centroid_y'] = np.asarray([x.centroid[1] for x in cells])
            cell_info['major_axis'] = np.asarray([x.major_axis_length for x in cells])
            cell_info['minor_axis'] = np.asarray([x.minor_axis_length for x in cells])
            cell_info['eccentricity'] = np.asarray([x.eccentricity for x in cells])
            cell_info['orientation'] = np.asarray([x.orientation for x in cells])
            cell_info['area'] = np.asarray([x.area for x in cells])
            cell_info['max_intensity'] = cell_info.loc[:, ['channel_{}'.format(i) for i in range(63)]].max(axis = 1).values
            cell_info_filename = '{}_cell_information_replicate_{}.csv'.format(sample,r)
            cell_info.to_csv(cell_info_filename, index = None)
    barcode_consensus = []
    cell_info = pd.read_csv('{}_cell_information_replicate_{}.csv'.format(sample, 0), dtype = {'cell_barcode':str})
    nn_dist = np.zeros((cell_info.shape[0], 20))
    for r in range(20):
        cell_info = pd.read_csv('{}_cell_information_replicate_{}.csv'.format(sample, r), dtype = {'cell_barcode':str})
        barcode_consensus.append(cell_info.cell_barcode.values)
        nn_dist[:,r] = cell_info.nn_dists.values
    barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
    nn_barcode = []
    nn_indices = np.argmin(nn_dist, axis = 1)
    for b in range(barcode_consensus.shape[0]):
        nn_barcode.append(barcode_consensus[b,nn_indices[b]])
    cell_info_consensus = cell_info.copy()
    cell_info_consensus.loc[:,'cell_barcode'] = nn_barcode
    spectral_centroid_distance = []
    for i in range(cell_info.shape[0]):
        barcode = cell_info.loc[i, 'cell_barcode']
        cell_label = cell_info.loc[i, 'label']
        cell_index = np.where(image_seg == cell_label)
        cell_pixel_intensity = image_registered[image_seg == cell_label, :]
        cx = np.average(cell_index[0])
        cy = np.average(cell_index[1])
        cx_spectral = np.average(cell_pixel_intensity*cell_index[0][:,None], axis = 0)/np.average(cell_pixel_intensity, axis = 0)
        cy_spectral = np.average(cell_pixel_intensity*cell_index[1][:,None], axis = 0)/np.average(cell_pixel_intensity, axis = 0)
        scd = np.sqrt((cx - cx_spectral)**2 + (cy - cy_spectral)**2)
        spectral_centroid_distance.append(np.median(scd))
    cell_info_consensus.loc[:,'spectral_centroid_distance'] = spectral_centroid_distance
    cell_info_consensus.to_csv('{}_cell_information_consensus.csv'.format(sample), index = None)
    return

def main():
    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-i', '--input_spectra', dest = 'input_spectra', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()
    classify_spectra(args.input_spectra, args.ref_clf)
    return

if __name__ == '__main__':
    main()






#####
