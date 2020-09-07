
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
from scipy.ndimage import binary_opening
from scipy.ndimage import binary_fill_holes
from neighbor2d import line_profile_2d_v2

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(x):
    return(x/2.54)

def write_xml(image_name):
    xml = bioformats.get_omexml_metadata(confocal_image_name)
    ome = bioformats.OMEXML(xml)
    metadata = ome.to_xml()
    return(creation_date)

def segment_images(image):
    image_cn = np.sum(image, axis = 2)
    image_cn[image_cn == 0] = np.min(image_cn[image_cn > 0])
    image_cn_log = np.log10(image_cn)
    image_padded = skimage.util.pad(image_cn, 5, mode = 'edge')
    image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis = 3)
    image_lp_max = np.max(image_lp, axis = 3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp_max[image_lp_max == 0] = np.min(image_lp_max[image_lp_max > 0])
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
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        intensity_rough_seg_mask = (intensity_rough_seg == 1)
        intensity_rough_seg_bkg = (intensity_rough_seg == 0)
    else:
        intensity_rough_seg_mask = (intensity_rough_seg == 0)
        intensity_rough_seg_bkg = (intensity_rough_seg == 1)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = intensity_rough_seg_mask_bfh
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_cn_log.reshape(np.prod(image_cn_log.shape), 1)).reshape(image_cn_log.shape)
    image0 = image_cn*(image_bkg_filter == 0)
    image1 = image_cn*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_cn*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    return(image_seg)

def save_segmentation(segmentation, sample):
    seg_color = color.label2rgb(segmentation, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    segfilename = '{}_seg.pdf'.format(sample)
    fig.savefig(segfilename, dpi = 300, transparent = True)
    plt.close()
    np.save('{}_seg'.format(sample), segmentation)
    return

def save_intensity_image(image, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(8))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image, cmap = 'inferno')
    intensity_filename = '{}_intensity.pdf'.format(sample)
    fig.savefig(intensity_filename, dpi = 300, transparent = True)
    plt.close()
    return

def measure_image(image_name):
    sample = re.sub('_pc_fov_[0-9]*_[0-9]*.czi', '', image_name)
    image = bioformats.load_image(image_name, rescale = False)
    confocal_image_name = '{}_fov_1_488.czi'.format(sample)
    confocal_image = bioformats.load_image(confocal_image_name, rescale = False)
    segmentation = segment_images(image)
    cells = skimage.measure.regionprops(segmentation)
    average_photon_count = np.empty((len(cells), image.shape[2]))
    total_photon_count = np.empty((len(cells), image.shape[2]))
    for k in range(0, image.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image[:,:,k])
        average_photon_count[:,k] = [x.mean_intensity for x in cells]
        total_photon_count[:,k] = [x.mean_intensity*x.area for x in cells]
    average_confocal_intensity = np.empty((len(cells), image.shape[2]))
    total_confocal_intensity = np.empty((len(cells), image.shape[2]))
    for k in range(0, image.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = confocal_image[:,:,k])
        average_confocal_intensity[:,k] = [x.mean_intensity for x in cells]
        total_confocal_intensity[:,k] = [x.mean_intensity*x.area for x in cells]
    cell_areas = [x.area*0.07*0.07 for x in cells]
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(5))
    plt.hist(cell_areas, bins = 100, color = (0,0.5,1), histtype = 'step')
    plt.xlabel(r'$A_c$ [$\mu m^2$]', fontsize = 8, color = theme_color)
    plt.ylabel('Frequency', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.subplots_adjust(left = 0.22, bottom = 0.25, right = 0.98, top = 0.97)
    plt.savefig('{}_cell_area.pdf'.format(sample), dpi = 300, transparent = True)
    plt.close()
    save_segmentation(segmentation, sample)
    average_photon_counts_filename = '{}_average_photon_counts.csv'.format(sample)
    total_photon_counts_filename = '{}_total_photon_counts.csv'.format(sample)
    pd.DataFrame(average_photon_count).to_csv(average_photon_counts_filename, index = None)
    pd.DataFrame(total_photon_count).to_csv(total_photon_counts_filename, index = None)
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('image_name', type = str, help = 'Image filenames')
    args = parser.parse_args()
    measure_image(args.image_name)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
