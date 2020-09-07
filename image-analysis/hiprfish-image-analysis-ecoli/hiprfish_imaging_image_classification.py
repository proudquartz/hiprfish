
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import re
import umap
import glob
import numba
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import color
from skimage import measure
import matplotlib.pyplot as plt
from sklearn import preprocessing

numba.config.NUMBA_NUM_THREADS = 1
numba.config.NUMBA_DEFAULT_NUM_THREADS = 1


def save_identification(image_identification, sample):
    seg_color = color.label2rgb(image_identification, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    segfilename = sample + '_identification.png'
    fig.savefig(segfilename, dpi = 300)
    plt.close()
    return

def classify_images(avgint_filename, bkg_intensity, bkg_spectra_filename, ref_clf):
    sample = re.sub('_avgint.csv', '', avgint_filename)
    bkg_spec = pd.read_csv(bkg_spectra_filename, header = None)
    segmentation = np.load('{}_seg.npy'.format(sample))
    avgint = pd.read_csv(avgint_filename, header = None)
    avgint_filtered = np.zeros((avgint.shape[0], 95))
    # avgint_filtered = np.zeros((avgint.shape[0], 100))
    avgint_filtered[:,0:95] = avgint.iloc[:,0:95].copy().values
    avgint_filtered[:,0:32] = avgint_filtered[:,0:32] - (bkg_intensity*bkg_spec.iloc[0:32,0])[None,:]
    avgint_norm = avgint_filtered.copy()
    avgint_norm = avgint_filtered/np.max(avgint_filtered, axis = 1)[:,None]
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 5))), axis = 1)
    # avgint_norm_cumsum = np.zeros(avgint_norm.shape)
    # avgint_norm_cumsum[:,0:32] = np.cumsum(avgint_norm[:,0:32], axis = 1)
    # avgint_norm_cumsum[:,32:55] = np.cumsum(avgint_norm[:,32:55], axis = 1)
    # avgint_norm_cumsum[:,55:75] = np.cumsum(avgint_norm[:,55:75], axis = 1)
    # avgint_norm_cumsum[:,75:89] = np.cumsum(avgint_norm[:,75:89], axis = 1)
    # avgint_norm_cumsum[:,89:95] = np.cumsum(avgint_norm[:,89:95], axis = 1)
    # avgint_norm_full = np.zeros((avgint.shape[0], 195))
    # avgint_norm_full[:,0:190] = np.concatenate([avgint_norm, avgint_norm_cumsum], axis = 1)
    # print('Classifying sample {}...'.format(os.path.basename(sample)))
    # umap_transform = joblib.load('{}.pkl'.format(ref_clf))
    # clf_umap = joblib.load('{}_svc.pkl'.format(ref_clf))
    # avgint_norm[:,95] = (np.max(avgint_norm[:,[2, 5, 11]], axis = 1) > 0.1)*1
    # avgint_norm[:,96] = (np.max(avgint_norm[:,[35, 39, 43]], axis = 1) > 0.1)*1
    # avgint_norm[:,97] = (np.max(avgint_norm[:,[56, 59, 61, 63, 65]], axis = 1) > 0.1)*1
    # avgint_norm[:,98] = (np.max(avgint_norm[:,[75, 79]], axis = 1) > 0.1)*1
    # avgint_norm[:,99] = (np.max(avgint_norm[:,[89, 92]], axis = 1) > 0.1)*1
    # avgint_umap_transformed, nn_indices, nn_dists = umap_transform.transform(avgint_norm)
    # cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    # avgint_identification_filename = '{}_avgint_ids.csv'.for4t(sample)
    # avgint_identification = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None]), axis = 1))
    # avgint_identification[102] = np.min(nn_dists, axis = 1)
    # avgint_identification[103] = sample
    # cells = skimage.measure.regionprops(segmentation)
    # avgint_identification[104] = np.asarray([x.label for x in cells])
    # avgint_identification.to_csv(avgint_identification_filename, header = None, index = None)
    for r in range(40):
        avgint_identification_filename = '{}_avgint_ids_replicate_{}.csv'.format(sample, r)
        umap_transform_filename = '{}_replicate_{}.pkl'.format(ref_clf, r)
        if not os.path.exists(avgint_identification_filename):
            if os.path.exists(umap_transform_filename):
                print('Classifying sample {} with batch {}...'.format(os.path.basename(sample), r))
                umap_transform = joblib.load('{}_replicate_{}.pkl'.format(ref_clf, r))
                clf_umap = joblib.load('{}_svc_replicate_{}.pkl'.format(ref_clf, r))
                avgint_norm[:,95] = (np.max(avgint_norm[:,[2, 5, 11]], axis = 1) > 0.08)*1
                avgint_norm[:,96] = (np.max(avgint_norm[:,[35, 39, 43]], axis = 1) > 0.1)*1
                avgint_norm[:,97] = (np.max(avgint_norm[:,[56, 59, 61, 63, 65]], axis = 1) > 0.1)*1
                avgint_norm[:,98] = (np.max(avgint_norm[:,[75, 79]], axis = 1) > 0.1)*1
                avgint_norm[:,99] = (np.max(avgint_norm[:,[89, 92]], axis = 1) > 0.1)*1
                avgint_umap_transformed, nn_indices, nn_dists = umap_transform.transform(avgint_norm)
                avgint_umap_nn = umap_transform.embedding_[nn_indices[:,0],:]
                cell_ids_norm = clf_umap.predict(avgint_umap_nn)
                avgint_identification_filename = '{}_avgint_ids_replicate_{}.csv'.format(sample, r)
                avgint_identification = pd.DataFrame(np.concatenate((avgint_filtered, avgint_norm[:,95:100], cell_ids_norm[:,None]), axis = 1))
                avgint_identification[101] = np.min(nn_dists, axis = 1)
                avgint_identification[102] = sample
                # avgint_identification[196] = np.min(nn_dists, axis = 1)
                # avgint_identification[197] = sample
                cells = skimage.measure.regionprops(segmentation)
                avgint_identification[103] = np.asarray([x.label for x in cells])
                # avgint_identification[198] = np.asarray([x.label for x in cells])
                avgint_identification.to_csv(avgint_identification_filename, header = None, index = None)
            else:
                pass
        else:
            pass
        # ids = list(set(cell_ids_norm))
        # image_identification = np.zeros(segmentation.shape)
        # for q in range(0, len(ids)):
        #     cell_population = np.where(cell_ids_norm == ids[q])[0]
        #     for r in range(0, len(cell_population)):
        #         image_identification[segmentation == cell_population[r]+1] = int(ids[q], 2)
        # save_identification(image_identification, sample)
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('input_spectra', type = str, default = '', help = 'Average normalized single cell spectra filenname')
    parser.add_argument('-rf', '--reference_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    parser.add_argument('-b', '--bkg_intensity', dest = 'bkg_intensity', type = float, default = 0.0, help = 'Background intensity')
    parser.add_argument('-bs', '--bkg_spectra_filename', dest = 'bkg_spectra_filename', type = str, default = '', help = 'Background spectra')
    args = parser.parse_args()
    classify_images(args.input_spectra, args.bkg_intensity, args.bkg_spectra_filename, args.ref_clf)
    return

if __name__ == '__main__':
    main()
