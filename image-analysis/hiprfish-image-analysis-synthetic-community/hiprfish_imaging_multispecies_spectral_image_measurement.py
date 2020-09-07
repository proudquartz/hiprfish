
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import sys
import glob
import joblib
import argparse
import javabridge
import bioformats
import numpy as np
import pandas as pd
import skimage.filters
from scipy import stats
from ete3 import NCBITaxa
from skimage import color
from skimage import feature
from skimage import restoration
from skimage import segmentation
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from scipy.ndimage import binary_opening
from matplotlib.colors import hsv_to_rgb
from neighbor2d import line_profile_2d_v2
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian
from skimage import registration


###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)

def load_calibration_images(filename):
    calibration_image = np.load(filename)
    calibration_image = calibration_image/np.max(calibration_image)
    return(calibration_image)

def save_segmentation(segmentation, sample):
    seg_color = color.label2rgb(segmentation, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    segfilename = sample + '_seg.pdf'
    fig.savefig(segfilename, dpi = 300, transparent = True)
    plt.close()
    np.save(sample + '_seg', segmentation)
    return

def save_sum_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'jet')
    segfilename = sample + '_sum.pdf'
    fig.savefig(segfilename, dpi = 300, transparent = True)
    plt.close()
    return

def save_enhanced_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'jet')
    segfilename = sample + '_enhanced.pdf'
    fig.savefig(segfilename, dpi = 300, transparent = True)
    plt.close()
    return

def generate_2d_segmentation(sample, calibration = 1):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [bioformats.load_image(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(image_sum[0], image_sum[i])[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape[0]
    for i in range(len(image_stack)):
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        if np.abs(shift_row) > 20:
            shift_row = 0
        if np.abs(shift_col) > 20:
            shift_col = 0
        original_row_min = int(np.maximum(0, shift_row))
        original_row_max = int(image_shape + np.minimum(0, shift_row))
        original_col_min = int(np.maximum(0, shift_col))
        original_col_max = int(image_shape + np.minimum(0, shift_col))
        registered_row_min = int(-np.minimum(0, shift_row))
        registered_row_max = int(image_shape - np.maximum(0, shift_row))
        registered_col_min = int(-np.minimum(0, shift_col))
        registered_col_max = int(image_shape - np.maximum(0, shift_col))
        image_registered[i][original_row_min: original_row_max, original_col_min: original_col_max, :] = image_stack[i][registered_row_min: registered_row_max, registered_col_min: registered_col_max, :]
        shift_filter_mask[i][original_row_min: original_row_max, original_col_min: original_col_max] = True
    image_channel = np.dstack(image_registered)
    calibration_image = load_calibration_images(calibration)
    calibration_image = calibration_image/np.max(calibration_image)
    for c in range(image_channel.shape[2]):
        image_channel[:,:,c] = image_channel[:,:,c]/calibration_image
    image_registered_sum = np.sum(image_channel, axis = 2)
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance, multichannel = False)
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
    image_padded = np.pad(image_registered_sum_nl_log, 5, mode = 'edge')
    image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis = 3)
    image_lp_max = np.max(image_lp, axis = 3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp = image_lp - image_lp_min[:,:,:,None]
    image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
    image_lp_rnc = image_lp_rel_norm[:,:,:,5]
    image_lprns = np.average(image_lp_rnc, axis = 2)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
    image_lprn_qcv = np.zeros(image_lprn_uq.shape)
    image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
    image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
    image_final = image_lprns*(1-image_lprn_qcv)
    image_final_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(image_final_rough_seg == 0)
    image1 = image_final*(image_final_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    image_final_rough_seg_mask = (image_final_rough_seg == np.argmax([i0,i1]))
    image_lprns_rsfbo = skimage.morphology.binary_opening(image_final_rough_seg_mask)
    image_lprns_rsfbo_bfh = binary_fill_holes(image_lprns_rsfbo)
    image_lprns_rsfbosm_bfh = skimage.morphology.remove_small_objects(image_lprns_rsfbo_bfh, 50)
    image_watershed_seeds = skimage.measure.label(image_lprns_rsfbosm_bfh)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum_nl*(image_bkg_filter == 0)
    image1 = image_registered_sum_nl*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    image_bkg_filter_mask = (image_bkg_filter == np.argmax([i0,i1]))
    image_bkg_filter_mask = binary_fill_holes(image_bkg_filter_mask)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_registered_sum_nl_log_bkg_filtered = image_registered_sum_nl_log*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_final_rough_seg_mask*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = skimage.morphology.remove_small_objects(image_watershed_mask_bkg_filtered, 50)
    image_watershed_mask_bkg_filtered = ndi.binary_fill_holes(image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg_bc = skimage.segmentation.clear_border(image_seg)
    image_seg_smbc = skimage.morphology.remove_small_objects(image_seg_bc, 50)
    image_seg_relabeled = skimage.segmentation.relabel_sequential(image_seg_smbc)[0]
    save_segmentation(image_seg_relabeled, sample)
    return(image_registered_sum, image_channel, image_final_bkg_filtered, image_seg_relabeled)

def measure_biofilm_images_no_reference(sample, calibration):
    image_registered_sum, image_registered, image_final_bkg_filtered, segmentation = generate_2d_segmentation(sample, calibration)
    save_sum_images(image_registered_sum, sample)
    save_enhanced_images(image_final_bkg_filtered, sample)
    np.save('{}_seg.npy'.format(sample), segmentation)
    np.save('{}_registered.npy'.format(sample), image_registered)
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[2]))
    for k in range(0, image_registered.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    pd.DataFrame(avgint).to_csv('{}_avgint.csv'.format(sample), index = None)
    return

def main():
    parser = argparse.ArgumentParser('Measure multispecies synthetic spectral images')
    parser.add_argument('-i', '--image_name', dest = 'image_name', nargs = '*', default = [], type = str, help = 'Input image filenames')
    parser.add_argument('-c', '--calibration', dest = 'calibration', type = str, default = '', help = 'calibration image filename')

    args = parser.parse_args()
    s = re.sub('_[0-9][0-9][0-9].czi', '', args.image_name[0])
    measure_biofilm_images_no_reference(s, args.calibration)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
