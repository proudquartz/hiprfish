
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
    segmentation = np.load('{}_seg.npy'.format(sample))
    avgint_norm = avgint.values/np.max(avgint.values, axis = 1)[:,None]
    avgint_norm_cumsum = np.zeros(avgint_norm.shape)
    avgint_norm_cumsum[:,0:23] = np.cumsum(avgint_norm[:,0:23], axis = 1)
    avgint_norm_cumsum[:,23:43] = np.cumsum(avgint_norm[:,23:43], axis = 1)
    avgint_norm_cumsum[:,43:57] = np.cumsum(avgint_norm[:,43:57], axis = 1)
    avgint_norm_cumsum[:,57:63] = np.cumsum(avgint_norm[:,57:63], axis = 1)
    # avgint_norm_cumsum = np.concatenate((avgint_norm, avgint_norm_cumsum, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    avgint_norm_cumsum = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    # avgint_norm_scaled = scaler.transform(avgint_norm[:,0:63], n_jobs = 1)
    # avgint_norm[:,63] = clf[0].predict(avgint_norm[:,[3, 7, 11]])
    # avgint_norm[:,64] = clf[1].predict(avgint_norm[:,[24, 27, 29, 31, 33]])
    # avgint_norm[:,65] = clf[2].predict(avgint_norm[:,[43, 47]])
    # avgint_norm[:,66] = clf[3].predict(avgint_norm[:,[57, 60]])
    # avgint_norm_scaled = scaler.transform(avgint_norm[:,0:63], n_jobs = 1)
    for r in range(20):
        avgint_identification_filename = '{}_cell_information_replicate_{}.csv'.format(sample, r)
        if not os.path.exists(avgint_identification_filename):
            print('Classifying sample {} with batch {}...'.format(os.path.basename(sample), r))
            umap_transform = joblib.load('{}_replicate_{}.pkl'.format(ref_clf, r))
            clf_umap = joblib.load('{}_svc_replicate_{}.pkl'.format(ref_clf, r))
            avgint_norm_cumsum[:,63] = (np.max(avgint_norm[:,[3, 7, 11]], axis = 1) > 0.1)*1
            avgint_norm_cumsum[:,64] = (np.max(avgint_norm[:,[24, 27, 29, 31, 33]], axis = 1) > 0.1)*1
            avgint_norm_cumsum[:,65] = (np.max(avgint_norm[:,[43, 47]], axis = 1) > 0.1)*1
            avgint_norm_cumsum[:,66] = (np.max(avgint_norm[:,[57, 60]], axis = 1) > 0.1)*1
            # avgint_norm_cumsum[:,126] = (np.max(avgint_norm[:,[3, 7, 11]], axis = 1) > 0.1)*1
            # avgint_norm_cumsum[:,127] = (np.max(avgint_norm[:,[24, 27, 29, 31, 33]], axis = 1) > 0.1)*1
            # avgint_norm_cumsum[:,128] = (np.max(avgint_norm[:,[43, 47]], axis = 1) > 0.1)*1
            # avgint_norm_cumsum[:,129] = (np.max(avgint_norm[:,[57, 60]], axis = 1) > 0.1)*1
            avgint_umap_transformed, nn_indices, nn_dists = umap_transform.transform(avgint_norm_cumsum)
            avgint_umap_nn = umap_transform.embedding_[nn_indices[:,0],:]
            cell_ids_norm = clf_umap.predict(avgint_umap_nn)
            avgint_identification_filename = '{}_cell_information_replicate_{}.csv'.format(sample, r)
            avgint_identification = pd.DataFrame(np.concatenate((avgint.values, avgint_norm_cumsum[:,63:67], cell_ids_norm[:,None]), axis = 1))
            # avgint_identification = pd.DataFrame(np.concatenate((avgint.values, avgint_norm_cumsum[:,63:130], cell_ids_norm[:,None]), axis = 1))
            # avgint_identification[131] = nn_dists[:,0]
            # avgint_identification[132] = sample
            avgint_identification[68] = nn_dists[:,0]
            avgint_identification[69] = sample
            cells = skimage.measure.regionprops(segmentation)
            avgint_identification[70] = np.asarray([x.label for x in cells])
            # avgint_identification[133] = np.asarray([x.label for x in cells])
            avgint_identification.to_csv(avgint_identification_filename, header = None, index = None)
        else:
            pass
    # avgint_norm[:,63] = (np.max(avgint_norm[:,[3, 7, 11]], axis = 1) > 0.05)*1
    # avgint_norm[:,64] = (np.max(avgint_norm[:,[24, 27, 29, 31, 33]], axis = 1) > 0.1)*1
    # avgint_norm[:,65] = (np.max(avgint_norm[:,[43, 47]], axis = 1) > 0.25)*1
    # avgint_norm[:,66] = (np.max(avgint_norm[:,[57, 60]], axis = 1) > 0.35)*1
    # avgint_umap_transformed = umap_transform.transform(avgint_norm)
    # cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    # cell_info = pd.DataFrame(np.concatenate((avgint, avgint_norm[:,63:67], avgint_umap_transformed, cell_ids_norm[:,None]), axis = 1))
    # cell_info[70] = sample
    # cells = skimage.measure.regionprops(segmentation)
    # cell_info[71] = np.asarray([x.label for x in cells])
    # cell_info[72] = np.asarray([x.centroid[0] for x in cells])
    # cell_info[73] = np.asarray([x.centroid[1] for x in cells])
    # cell_info[74] = np.asarray([x.major_axis_length for x in cells])
    # cell_info[75] = np.asarray([x.minor_axis_length for x in cells])
    # cell_info[76] = np.asarray([x.eccentricity for x in cells])
    # cell_info[77] = np.asarray([x.orientation for x in cells])
    # cell_info[78] = np.asarray([x.area for x in cells])
    # cellinfofilename = '{}_cell_information.csv'.format(sample)
    # cell_info.to_csv(cellinfofilename, index = None, header = None)
    return

def main():
    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-i', '--input_spectra', dest = 'input_spectra', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()
    # umap_transform_bucket = joblib.load(args.ref_clf)
    # scaler = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_scaler.pkl', args.ref_clf))
    # clf_umap_bucket = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_svc.pkl', args.ref_clf))
    # clf_bucket = joblib.load(re.sub('transform_biofilm_7b.pkl', 'transformed_biofilm_7b_check_svc.pkl', args.ref_clf))
    classify_spectra(args.input_spectra, args.ref_clf)
    return

if __name__ == '__main__':
    main()
