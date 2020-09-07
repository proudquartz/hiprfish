
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
import itertools
import javabridge
import bioformats
import numpy as np
import pandas as pd
import skimage.filters
from sklearn import svm
from skimage import color
from ete3 import NCBITaxa
from skimage import feature
from skimage import exposure
from skimage import restoration
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.future import graph
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from neighbor import line_profile_v2
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from scipy.ndimage import binary_opening
from matplotlib.colors import hsv_to_rgb
from neighbor2d import line_profile_2d_v2
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import binary_fill_holes
from matplotlib.ticker import ScalarFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm
from skimage import registration
from neighbor import line_profile_memory_efficient_v2
from neighbor import line_profile_memory_efficient_v3

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS, run_headless = True)

def cm_to_inches(length):
    return(length/2.54)

def load_ztslice(filename, z_index, t_index):
    image = bioformats.load_image(filename, z = z_index, t = t_index)
    return(image)

def load_ztslice_tile(filename, z_index, t_index, tile):
    image = bioformats.load_image(filename, z = z_index, t = t_index, series = tile)
    return(image)

def get_x_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    x_range = ome.image(0).Pixels.get_SizeX()
    return(x_range)

def get_y_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    y_range = ome.image(0).Pixels.get_SizeY()
    return(y_range)

def get_c_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    c_range = ome.image(0).Pixels.get_SizeC()
    return(c_range)

def get_t_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    t_range = ome.image(0).Pixels.get_SizeT()
    return(t_range)

def get_z_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    z_range = ome.image(0).Pixels.get_SizeZ()
    return(z_range)

def get_tile_size(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    return(int(np.sqrt(ome.image_count)))

def get_image_count(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    return(ome.image_count)

def load_image_tile(filename):
    z_range = get_z_range(filename)
    image = np.stack([load_ztslice(filename, k, 0) for k in range(0, z_range)], axis = 2)
    return(image)

def load_image_zstack_fixed_t(filename, t):
    z_range = get_z_range(filename)
    image = np.stack([load_ztslice(filename, k, t) for k in range(0, z_range)], axis = 2)
    return(image)

def load_image_zstack_fixed_t_memory_efficient(filename, t, z_min, z_max):
    image = np.stack([load_ztslice(filename, k, t) for k in range(z_min, z_max)], axis = 2)
    return(image)

def load_image_zstack_fixed_t_tile(filename, t, tile):
    z_range = get_z_range(filename)
    image = np.stack([load_ztslice_tile(filename, k, t, tile) for k in range(0, z_range)], axis = 2)
    return(image)

def get_registered_image_from_tile(filename, tile_size, overlap):
    image_0 = load_ztslice_tile(filename, 0, 0, 0)
    full_size_x = image_0.shape[0]*tile_size[0] - overlap*(tile_size[0] - 1)
    full_size_y = image_0.shape[1]*tile_size[1] - overlap*(tile_size[1] - 1)
    image_full = np.zeros((full_size_x, full_size_y, image_0.shape[2]))
    for i in range(tile_size[0]):
        for j in range(tile_size[1]):
            overlap_compensation_x = 200*i
            overlap_compensation_y = 200*j
            image_full[i*image_0.shape[0] - overlap_compensation_x: (i+1)*image_0.shape[0] - overlap_compensation_x, j*image_0.shape[0] - overlap_compensation_y: (j+1)*image_0.shape[0] - overlap_compensation_y, :] = load_ztslice_tile(filename, 0, 0, i*tile_size[1] + j)
    return(image_full)

def get_registered_average_image_from_tstack(filename):
    image_0 = load_image_zstack_fixed_t(filename, 0)
    image_registered = image_0.copy()
    image_0_sum = np.sum(image_0, axis = 3)
    shift_vector_list = []
    nt = get_t_range(filename)
    for i in range(1, nt):
        image_i = load_image_zstack_fixed_t(filename, i)
        image_i_sum = np.sum(image_i, axis = 3)
        shift_vector = skimage.registration.phase_cross_correlation(image_0_sum, image_i_sum)[0]
        shift_vector = np.insert(shift_vector, 3,0)
        shift_filter_mask = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), False, dtype = bool)
        shift_x = int(shift_vector[0])
        shift_y = int(shift_vector[1])
        shift_z = int(shift_vector[2])
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_0.shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_0.shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_0.shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_0.shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_0.shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_0.shape[2] - np.maximum(0, shift_z))
        image_registered_hold = np.zeros(image_0.shape)
        image_registered_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max,:] = image_i[registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max,:]
        shift_filter_mask[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
        image_registered += image_registered_hold
    return(image_registered/nt)

def get_registered_image_from_tstack(filename):
    print('Loading {} at t = 0...'.format(os.path.basename(filename)))
    image_registered = load_image_zstack_fixed_t(filename, 0)
    image_0_sum = np.sum(image_registered, axis = 3)
    shift_vector_list = []
    nt = get_t_range(filename)
    for i in range(1, nt):
        print('Loading {} at t = {}...'.format(os.path.basename(filename), i))
        image_i = load_image_zstack_fixed_t(filename, i)
        image_i_sum = np.sum(image_i, axis = 3)
        shift_vector = skimage.registration.phase_cross_correlation(image_0_sum, image_i_sum)[0]
        shift_vector = np.insert(shift_vector, 3,0)
        shift_filter_mask = np.full((image_registered.shape[0], image_registered.shape[1], image_registered.shape[2]), False, dtype = bool)
        shift_x = int(shift_vector[0])
        shift_y = int(shift_vector[1])
        shift_z = int(shift_vector[2])
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_registered.shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_registered.shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_registered.shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_registered.shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_registered.shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_registered.shape[2] - np.maximum(0, shift_z))
        image_registered_hold = np.zeros(image_registered.shape)
        image_registered_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max,:] = image_i[registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max,:]
        shift_filter_mask[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
        image_registered += image_registered_hold
    return(image_registered)

def get_registered_image_from_tstack_tile(filename, tile):
    image_0 = load_image_zstack_fixed_t_tile(filename, 0, tile)
    image_registered = image_0.copy()
    image_0_sum = np.sum(image_0, axis = 3)
    shift_vector_list = []
    nt = get_t_range(filename)
    shift_filter_mask = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), True, dtype = bool)
    for i in range(1, nt):
        image_i = load_image_zstack_fixed_t_tile(filename, i, tile)
        image_i_sum = np.sum(image_i, axis = 3)
        shift_vector = skimage.registration.phase_cross_correlation(image_0_sum, image_i_sum)[0]
        shift_vector = np.insert(shift_vector, 3,0)
        shift_filter_mask_hold = np.full((image_0.shape[0], image_0.shape[1], image_0.shape[2]), False, dtype = bool)
        shift_x = int(shift_vector[0])
        shift_y = int(shift_vector[1])
        shift_z = int(shift_vector[2])
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_0.shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_0.shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_0.shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_0.shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_0.shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_0.shape[2] - np.maximum(0, shift_z))
        image_registered_hold = np.zeros(image_0.shape)
        image_registered_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max,:] = image_i[registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max,:]
        shift_filter_mask_hold[original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
        shift_filter_mask = shift_filter_mask*shift_filter_mask_hold
        image_registered += image_registered_hold
    return(image_registered, shift_filter_mask)

def save_segmentation(segmentation, sample):
    seg_color = color.label2rgb(segmentation, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', box_color = 'white')
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    segfilename = sample + '_seg.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    np.save(sample + '_seg', segmentation)
    return

def save_identification(image_identification, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_identification)
    scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    segfilename = sample + '_identification.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_identification_filtered(image_identification, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_identification)
    scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    segfilename = sample + '_identification_filtered.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_identification_bvox(image_identification, sample):
    nx, ny, nz, nframes = image_identification.shape[0], image_identification.shape[1], image_identification.shape[2], 1
    header = np.array([nx,ny,nz,nframes])
    color_filename = ['r', 'g', 'b']
    for i in range(3):
        pointdata = image_identification[:,:,:,i]
        binfile = open('{}_identification_{}.bvox'.format(sample, color_filename[i]),'wb')
        header.astype('<i4').tofile(binfile)
        pointdata.flatten('F').astype('<f4').tofile(binfile)
    return

def save_raw_image_bvox(image_registered_sum, sample):
    nx, ny, nz, nframes = image_registered_sum.shape[0], image_registered_sum.shape[1], image_registered_sum.shape[2], 1
    header = np.array([nx,ny,nz,nframes])
    binfile = open('{}_raw_image.bvox'.format(sample),'wb')
    header.astype('<i4').tofile(binfile)
    image_registered_sum.flatten('F').astype('<f4').tofile(binfile)
    return

def save_sum_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    segfilename = sample + '_sum.png'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_max_images(image_registered_max, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5.5),cm_to_inches(5.5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_registered_max, cmap = 'inferno')
    image_max_filename = sample + '_max.pdf'
    fig.savefig(image_max_filename, dpi = 300)
    plt.close()
    return

def save_enhanced_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    plt.axis('off')
    segfilename = sample + '_enhanced.pdf'
    fig.savefig(segfilename, dpi = 300)
    plt.close()
    return

def denoise_image(image_registered):
    residual = np.zeros(image_registered.shape)
    for i in range(20):
        for j in range(20):
            print('Calculating sub block {}-{}'.format(i,j))
            image_sub_block = image_registered[i*20:(i+1)*20,j*20:(j+1)*20,:]
            k = 0
            term_2 = image_sub_block[:,:,k+1]
            term_3 = np.zeros(image_sub_block[:,:,k].shape)
            term_3[0,:] = image_sub_block[1,:,k]
            term_3[1:,:] = image_sub_block[0:-1,:,k]
            lhs = np.stack([term_2.reshape(np.prod(term_2.shape)),term_3.reshape(np.prod(term_3.shape))], axis = 1)
            rhs = image_sub_block[:,:,k].reshape(np.prod(image_sub_block[:,:,k].shape))
            reg = LinearRegression().fit(lhs, rhs)
            prediction = reg.predict(lhs)
            residual[i*20:(i+1)*20,j*20:(j+1)*20,k] = image_sub_block[:,:,k] - prediction.reshape(image_sub_block[:,:,k].shape)
            k = image_sub_block.shape[2]-1
            term_1 = image_sub_block[:,:,k-1]
            term_3 = np.zeros(image_sub_block[:,:,k].shape)
            term_3[0,:] = image_sub_block[1,:,k]
            term_3[1:,:] = image_sub_block[0:-1,:,k]
            lhs = np.stack([term_1.reshape(np.prod(term_1.shape)),term_3.reshape(np.prod(term_3.shape))], axis = 1)
            rhs = image_sub_block[:,:,k].reshape(np.prod(image_sub_block[:,:,k].shape))
            reg = LinearRegression().fit(lhs, rhs)
            prediction = reg.predict(lhs)
            residual[i*20:(i+1)*20,j*20:(j+1)*20,k] = image_sub_block[:,:,k] - prediction.reshape(image_sub_block[:,:,k].shape)
            for k in range(1, image_sub_block.shape[2] - 1):
                term_1 = image_sub_block[:,:,k-1]
                term_2 = image_sub_block[:,:,k+1]
                term_3 = np.zeros(image_sub_block[:,:,k].shape)
                term_3[0,:] = image_sub_block[1,:,k]
                term_3[1:,:] = image_sub_block[0:-1,:,k]
                lhs = np.stack([term_1.reshape(np.prod(term_1.shape)),term_2.reshape(np.prod(term_2.shape)),term_3.reshape(np.prod(term_3.shape))], axis = 1)
                rhs = image_sub_block[:,:,k].reshape(np.prod(image_sub_block[:,:,k].shape))
                reg = LinearRegression().fit(lhs, rhs)
                prediction = reg.predict(lhs)
                residual[i*20:(i+1)*20,j*20:(j+1)*20,k] = image_sub_block[:,:,k] - prediction.reshape(image_sub_block[:,:,k].shape)
    I = image_registered.reshape((np.prod(image_registered[:,:,0].shape), image_registered.shape[2]))
    r = residual.reshape((np.prod(residual[:,:,0].shape), residual.shape[2]))
    noise_covariance = np.dot(r.transpose(), r)
    signal_cov = np.dot(I.transpose(), I)
    test = np.dot(noise_covariance, np.linalg.inv(signal_cov))
    return

def generate_2d_segmentation(image_name, sample, marker_x = 0, marker_y = 0):
    # excitations = ['488', '514', '561', '633']
    # image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [bioformats.load_image(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1), np.log(image_sum[i]+1))[0] for i in range(1,4)]
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
    image_registered = np.concatenate(image_registered, axis = 2)
    image_registered_max = np.max(image_registered, axis = 2)
    image_registered_sum = np.sum(image_registered, axis = 2)
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance)
    image_padded = np.pad(image_registered_sum_nl, 5, mode = 'edge')
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
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    intensity_rough_seg_mask = intensity_rough_seg == np.argmax([i0,i1])
    image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
    image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
    image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
    image_watershed_mask = image_lprns_rsfsmbo*image_bkg_filter_mask
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    image_density = np.zeros(image_seg.shape)
    # image_registered_max_padded = np.pad(image_registered_max > 0, 20, mode = 'edge')
    image_seg_padded = np.pad(image_seg > 0, 20, mode = 'edge')
    for i in range(image_seg.shape[0]):
        for j in range(image_seg.shape[1]):
            image_density[i,j] = np.average(image_seg_padded[i:i+40,j:j+40])
    image_density_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_density.reshape(np.prod(image_density.shape), 1)).reshape(image_density.shape)
    image0 = image_density*(image_density_seg == 0)
    image1 = image_density*(image_density_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    image_density_seg_mask = image_density_seg == np.argmin([i0, i1])
    image_density_microbiome_seg_mask = image_density_seg == np.argmax([i0, i1])
    image_density_microbiome_objects = skimage.measure.label(image_density_microbiome_seg_mask)
    image_density_background_objects = skimage.measure.label(image_density_seg_mask)
    density_microbiome_objects = skimage.measure.regionprops(image_density_microbiome_objects)
    density_background_objects = skimage.measure.regionprops(image_density_background_objects)
    density_microbiome_objects_size = [x.area for x in density_microbiome_objects]
    density_background_objects_size = [x.area for x in density_background_objects]
    microbiome_label = density_microbiome_objects[np.argmax(density_microbiome_objects_size)].label
    microbiome = image_density_microbiome_objects == microbiome_label
    microbiome = ndi.binary_fill_holes(microbiome)
    microbiome_boundary = skimage.segmentation.find_boundaries(microbiome)
    regions = skimage.measure.label(1 - microbiome_boundary)
    epithelial_region = regions == regions[marker_x, marker_y]
    epithelial_boundary = skimage.segmentation.find_boundaries(epithelial_region)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, adjacency_seg, epithelial_region, epithelial_boundary)

def generate_3d_segmentation(image_name, sample):
    # excitations = ['488', '514', '561', '633']
    # image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_channel = np.concatenate(image_registered, axis = 3)
    image_registered_sum = np.sum(image_channel, axis = (3,4))
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 0.03)
    image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
    image_lp_rnc = line_profile_memory_efficient_v2(image_padded.astype(np.float64), 11, 9, 9)
    image_lprns = np.average(image_lp_rnc, axis = 3)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 3)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 3)
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 1)
    image2 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    i2 = np.average(image2[image2 > 0])
    intensity_rough_seg_mask = np.zeros(image_final.shape).astype(int)
    intensity_rough_seg_mask[image_final > 0] = (intensity_rough_seg[image_final > 0] == np.argmax([i0,i1,i2]))*1
    intensity_rough_seg_mask = skimage.morphology.remove_small_holes(intensity_rough_seg_mask)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    image_watershed_seeds = skimage.measure.label(image_lprns_rsfbosm_bfh*intensity_rough_seg_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl+1e-8)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum_nl*(image_bkg_filter == 0)
    image1 = image_registered_sum_nl*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    # image_final_bkg_filtered = image_registered_sum_nl_log*image_bkg_filter_mask
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    # image_watershed_sauvola = threshold_sauvola(image_registered_sum_nl, k = 0)
    # image_watershed_mask_bkg_filtered = (image_registered_sum_nl > image_watershed_sauvola)*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_lprns_rsfbosm_bfh*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    # image_seg = skimage.morphology.remove_small_objects(image_seg, size_limit)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_bkg_filter_mask)
    # adjacency_seg = skimage.morphology.remove_small_objects(adjacency_seg, size_limit)
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    # save_segmentation(image_seg, sample)
    return(image_registered_sum, image_channel, image_final_bkg_filtered, image_seg)

def get_image(image_name):
    # excitations = ['488', '514', '561', '633']
    # image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack[:,:,:,:,] = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_stack[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
    image_stack = np.concatenate(image_stack, axis = 3)
    return(image_stack)

def get_t_average_image(image_name):
    image_t_average_filename = re.sub('_488.czi', '_image_time_average.npy', image_name[0])
    # excitations = ['488', '514', '561', '633']
    # image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    if not os.path.exists(image_t_average_filename):
        image_stack = [get_registered_average_image_from_tstack(filename) for filename in image_name]
        image_sum = [np.sum(image, axis = 3) for image in image_stack]
        shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
        shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
        image_shape = image_stack[0].shape
        for i in range(len(image_stack)):
            shift_x = int(shift_vectors[i][0])
            shift_y = int(shift_vectors[i][1])
            shift_z = int(shift_vectors[i][2])
            print(shift_x, shift_y, shift_z)
            original_x_min = int(np.maximum(0, shift_x))
            original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
            original_y_min = int(np.maximum(0, shift_y))
            original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
            original_z_min = int(np.maximum(0, shift_z))
            original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
            registered_x_min = int(-np.minimum(0, shift_x))
            registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
            registered_y_min = int(-np.minimum(0, shift_y))
            registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
            registered_z_min = int(-np.minimum(0, shift_z))
            registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
            image_stack[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        image_stack = np.concatenate(image_stack, axis = 3)
        np.save(image_t_average_filename, image_stack)
    else:
        image_stack = np.load(image_t_average_filename)
    return(image_stack)

def generate_2d_segmentation_from_tile(image_name, sample):
    # excitations = ['488', '514', '561', '633']
    # image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    tile_size = (4,2)
    overlap = 200
    image_stack = [get_registered_image_from_tile(filename, tile_size, overlap) for filename in image_name]
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]), np.log(image_sum[i]))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1]), False, dtype = bool) for image in image_stack]
    image_shape_row = image_stack[0].shape[0]
    image_shape_col = image_stack[0].shape[1]
    for i in range(len(image_stack)):
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        print(shift_row, shift_col)
        original_row_min = int(np.maximum(0, shift_row))
        original_row_max = int(image_shape_row + np.minimum(0, shift_row))
        original_col_min = int(np.maximum(0, shift_col))
        original_col_max = int(image_shape_col + np.minimum(0, shift_col))
        registered_row_min = int(-np.minimum(0, shift_row))
        registered_row_max = int(image_shape_row - np.maximum(0, shift_row))
        registered_col_min = int(-np.minimum(0, shift_col))
        registered_col_max = int(image_shape_col - np.maximum(0, shift_col))
        image_registered[i][original_row_min: original_row_max, original_col_min: original_col_max, :] = image_stack[i][registered_row_min: registered_row_max, registered_col_min: registered_col_max, :]
        shift_filter_mask[i][original_row_min: original_row_max, original_col_min: original_col_max] = True
    image_channel = np.concatenate(image_registered, axis = 2)
    image_registered_sum = np.sum(image_channel, axis = 2)
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
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
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum_nl*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    image_bkg = image_bkg_filter_mask == 0
    image_bkg = skimage.morphology.remove_small_objects(image_bkg, 10000)
    image_bkg = binary_fill_holes(image_bkg)
    structuring_element = skimage.morphology.disk(100)
    image_bkg_bc = skimage.morphology.binary_dilation(image_bkg, structuring_element)
    image_bkg_bc_objects = skimage.measure.label(image_bkg_bc)
    image_bkg_bc_objects_props = skimage.measure.regionprops(image_bkg_bc_objects)
    image_bkg_bc_areas = [x.area for x in image_bkg_bc_objects_props]
    image_bkg_final = image_bkg_bc_objects == image_bkg_bc_objects_props[np.argmax(image_bkg_bc_areas)].label
    image_bkg_final_bd = skimage.morphology.binary_dilation(image_bkg_final)
    image_objects_overall = skimage.measure.label(1 - image_bkg_final_bd)
    image_objects_overall_seg = skimage.segmentation.watershed(-image_registered_sum, image_objects_overall)
    image_objects_overall_props = skimage.measure.regionprops(image_objects_overall_seg)
    image_objects_areas = [x.area for x in image_objects_overall_props]
    image_epithelial_area = image_objects_overall_seg != image_objects_overall_props[np.argmax(image_objects_areas)].label
    return(image_registered_sum, image_channel, image_final_bkg_filtered, image_seg, adjacency_seg, epithelial_boundary)

def generate_2d_segmentation_from_zstack(image_stack, sample, z):
    image_registered = image_stack[:,:,z,:,:]
    image_registered_sum = np.sum(image_registered, axis = (2,3))
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
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
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
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
    intensity_rough_seg_mask = skimage.morphology.remove_small_holes(intensity_rough_seg_mask)
    image_lprns_rsfbo = skimage.morphology.binary_opening(intensity_rough_seg_mask)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    image_watershed_seeds = skimage.measure.label(image_lprns_rsfbosm_bfh*intensity_rough_seg_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_bkg_filter_mask)
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, adjacency_seg)

def generate_2d_segmentation_from_zstack_t_sum(image_stack, sample, z, marker_x, marker_y):
    image_registered = image_stack[:,:,z,:]
    image_registered_sum = np.sum(image_registered, axis = 2)
    image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
    image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
    image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance)
    image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
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
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    intensity_rough_seg_mask = intensity_rough_seg == np.argmax([i0,i1])
    image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
    image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
    image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
    image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
    image_watershed_mask = image_lprns_rsfsmbo*image_bkg_filter_mask
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    adjacency_seg = skimage.segmentation.relabel_sequential(adjacency_seg)[0]
    save_segmentation(image_seg, sample)
    image_density = np.zeros(image_seg.shape)
    # image_registered_max_padded = np.pad(image_registered_max > 0, 20, mode = 'edge')
    image_seg_padded = np.pad(image_seg > 0, 20, mode = 'edge')
    for i in range(image_seg.shape[0]):
        for j in range(image_seg.shape[1]):
            image_density[i,j] = np.average(image_seg_padded[i:i+40,j:j+40])
    image_density_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_density.reshape(np.prod(image_density.shape), 1)).reshape(image_density.shape)
    image0 = image_density*(image_density_seg == 0)
    image1 = image_density*(image_density_seg == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    image_density_seg_mask = image_density_seg == np.argmin([i0, i1])
    image_density_microbiome_seg_mask = image_density_seg == np.argmax([i0, i1])
    image_density_microbiome_objects = skimage.measure.label(image_density_microbiome_seg_mask)
    image_density_background_objects = skimage.measure.label(image_density_seg_mask)
    density_microbiome_objects = skimage.measure.regionprops(image_density_microbiome_objects)
    density_background_objects = skimage.measure.regionprops(image_density_background_objects)
    density_microbiome_objects_size = [x.area for x in density_microbiome_objects]
    density_background_objects_size = [x.area for x in density_background_objects]
    microbiome_label = density_microbiome_objects[np.argmax(density_microbiome_objects_size)].label
    microbiome = image_density_microbiome_objects == microbiome_label
    microbiome = ndi.binary_fill_holes(microbiome)
    microbiome_boundary = skimage.segmentation.find_boundaries(microbiome)
    regions = skimage.measure.label(1 - microbiome_boundary)
    epithelial_region = regions == regions[marker_x, marker_y]
    epithelial_boundary = skimage.segmentation.find_boundaries(epithelial_region)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, adjacency_seg, epithelial_region, epithelial_boundary)

def generate_3d_segmentation_memory_efficient(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_registered = np.concatenate(image_registered, axis = 3)
    image_registered_sum = np.sum(image_registered, axis = 3)
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
    image_lp = line_profile_memory_efficient_v2(image_padded.astype(np.float64), 11, 9, 9)
    image_lprns = np.average(image_lp, axis = 3)
    image_lprn_lq = np.percentile(image_lp, 25, axis = 3)
    image_lprn_uq = np.percentile(image_lp, 75, axis = 3)
    image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
    image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
    image_final = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = np.zeros(image_final.shape).astype(int)
    intensity_rough_seg[image_final > 0] = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final[image_final >0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    intensity_rough_seg_mask = np.zeros(image_final.shape).astype(int)
    if i0 < i1:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0]
    else:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0] == 0
    image_final_seg = np.zeros(image_final.shape).astype(int)
    image_final_seg[image_final > 0] = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final[image_final >0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 1)
    image2 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    i2 = np.average(image2[image2 > 0])
    image_lprns_rsf = np.zeros(image_final.shape).astype(int)
    image_lprns_rsf[image_final > 0] = (image_final_seg[image_final > 0] == np.argmax([i0,i1,i2]))*1
    image_lprns_rsfbo = binary_opening(image_lprns_rsf)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_watershed_seeds = skimage.morphology.label(image_watershed_mask)
    image_registered_sum_nl_log = np.log10(image_registered_sum + 1)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_watershed_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg)

def generate_3d_segmentation_tile(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [load_image_zstack_fixed_t(filename, t = 0) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_registered = np.concatenate(image_registered, axis = 3)
    np.save('{}_registered.npy'.format(sample), image_registered)
    image_registered_sum = np.sum(image_registered, axis = 3)
    del image_registered
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_padded = skimage.util.pad(image_registered_sum, 5, mode = 'edge')
    image_final = np.zeros(image_registered_sum.shape)
    for i in range(10):
        for j in range(10):
            print('Calculating tile {}, {}'.format(i,j))
            image_padded_temp = image_padded[i*200:(i+1)*200+10, j*200:(j+1)*200+10,:]
            image_lp = line_profile_v2(image_padded_temp.astype(np.float64), 11, 9, 9)
            image_lp = np.nan_to_num(image_lp)
            image_lp_min = np.min(image_lp, axis = 4)
            image_lp_max = np.max(image_lp, axis = 4)
            image_lp_max = image_lp_max - image_lp_min
            image_lp = image_lp - image_lp_min[:,:,:,:,None]
            image_lp_rel_norm = image_lp/image_lp_max[:,:,:,:,None]
            image_lp_rnc = image_lp_rel_norm[:,:,:,:,5]
            image_lprns = np.average(image_lp_rnc, axis = 3)
            image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 3)
            image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 3)
            image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
            image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
            image_final[i*200:(i+1)*200, j*200:(j+1)*200,:] = image_lprns*(1-image_lprn_qcv)
    intensity_rough_seg = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
    image0 = image_final*(intensity_rough_seg == 0)
    image1 = image_final*(intensity_rough_seg == 1)
    image2 = image_final*(intensity_rough_seg == 2)
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
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg)

def generate_3d_segmentation_slice(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [load_image_zstack_fixed_t(filename, t = 0) for filename in image_name]
    image_sum = [np.sum(image, axis = 3) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_registered = [np.zeros(image.shape) for image in image_stack]
    shift_filter_mask = [np.full((image.shape[0], image.shape[1], image.shape[2]), False, dtype = bool) for image in image_stack]
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_registered[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
        shift_filter_mask[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max] = True
    image_registered = np.concatenate(image_registered, axis = 3)
    np.save('{}_registered.npy'.format(sample), image_registered)
    image_registered_sum = np.sum(image_registered, axis = 3)
    del image_registered
    image_registered_sum = image_registered_sum/np.max(image_registered_sum)
    image_registered_sum = skimage.restoration.denoise_nl_means(image_registered_sum, h = 0.02)
    image_final = np.zeros(image_registered_sum.shape)
    for i in range(image_registered_sum.shape[2]):
        print('Calculating slice {}'.format(i))
        image_padded = skimage.util.pad(image_registered_sum[:,:,i], 5, mode = 'edge')
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
        image_lprn_qcv = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq)
        image_lprn_qcv = np.nan_to_num(image_lprn_qcv)
        image_final[:,:,i] = image_lprns*(1-image_lprn_qcv)
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
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_registered_sum_nl_log = np.log10(image_registered_sum)
    image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
    image0 = image_registered_sum*(image_bkg_filter == 0)
    image1 = image_registered_sum*(image_bkg_filter == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
    image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
    image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
    image_watershed_seeds = skimage.measure.label(image_watershed_mask)
    image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
    adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)
    image_epithelial_area = np.zeros(image_seg.shape)
    for i in range(image_seg.shape[2]):
        print('Calculating slice {}'.format(i))
        image_bkg = image_bkg_filter_mask[:,:,i] == 0
        image_bkg = skimage.morphology.remove_small_objects(image_bkg, 10000)
        image_bkg = binary_fill_holes(image_bkg)
        structuring_element = skimage.morphology.disk(100)
        image_bkg_bc = skimage.morphology.binary_dilation(image_bkg, structuring_element)
        image_bkg_bc_objects = skimage.measure.label(image_bkg_bc)
        image_bkg_bc_objects_props = skimage.measure.regionprops(image_bkg_bc_objects)
        image_bkg_bc_areas = [x.area for x in image_bkg_bc_objects_props]
        image_bkg_final = image_bkg_bc_objects == image_bkg_bc_objects_props[np.argmax(image_bkg_bc_areas)].label
        image_bkg_final_bd = skimage.morphology.binary_dilation(image_bkg_final)
        image_objects_overall = skimage.measure.label(1 - image_bkg_final_bd)
        image_objects_overall_seg = skimage.segmentation.watershed(-image_registered_sum[:,:,i], image_objects_overall)
        image_objects_overall_props = skimage.measure.regionprops(image_objects_overall_seg)
        image_objects_areas = [x.area for x in image_objects_overall_props]
        if image_objects_areas:
            image_epithelial_area[:,:,i] = image_objects_overall_seg != image_objects_overall_props[np.argmax(image_objects_areas)].label

    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg, image_epithelial_area)

def generate_3d_segmentation_tile_memory_efficient(sample):
    image_name = '{}_561.czi'.format(sample)
    image_tile_size = get_tile_size(image_name)
    image_sum_list = []
    shift_filter_mask_list = []
    for i in range(image_tile_size):
        for j in range(image_tile_size):
            print('Analyzing tile [{}, {}]'.format(i,j))
            image_stack, shift_filter_mask = get_registered_image_from_tstack_tile(image_name, i*image_tile_size + j)
            image_sum = np.sum(image_stack, axis = 3)
            image_sum_list.append(image_sum)
            shift_filter_mask_list.append(shift_filter_mask)
    image_sum_filtered_list = [image_sum_list[i]*shift_filter_mask_list[i] for i in range(image_tile_size*image_tile_size)]
    shift_vector_full = np.zeros((image_tile_size, image_tile_size, 3))
    for i in range(image_tile_size):
        for j in range(image_tile_size):
            if (i == 0) & (j == 0):
                shift_vector_full[i,j,:] = np.zeros(3)
            elif (i > 0) & (j == 0):
                shift_vector = skimage.registration.phase_cross_correlation(image_sum_filtered_list[(i-1)*image_tile_size][450:500,:,:], image_sum_filtered_list[i*image_tile_size][0:50,:,:])
                shift_vector_full[i, j, :] = shift_vector[0]
            else:
                shift_vector = skimage.registration.phase_cross_correlation(image_sum_filtered_list[i*image_tile_size + j - 1][:,450:500,:], image_sum_filtered_list[i*image_tile_size + j][:,0:50,:])
                shift_vector_full[i, j, :] = shift_vector[0]
    image_full = np.zeros((2020, 2020, 170))
    image_overlap_full = np.zeros((2020, 2020, 170))
    for i in range(image_tile_size):
        for j in range(image_tile_size):
            x_min = int(i*500 - 50*i + np.sum(shift_vector_full[0:i+1, 0, 0]) + np.sum(shift_vector_full[i, 1:j+1, 0])) + 10
            x_max = int((i+1)*500 - 50*i + np.sum(shift_vector_full[0:i+1, 0, 0]) + np.sum(shift_vector_full[i, 1:j+1, 0])) + 10
            y_min = int(j*500 - 50*j + np.sum(shift_vector_full[i, 0:j+1, 1])) + 10
            y_max = int((j+1)*500 - 50*j + np.sum(shift_vector_full[i, 0:j+1, 1])) + 10
            z_min = int(np.sum(shift_vector_full[i, 0:j+1, 2])) + 10
            z_max = int(150 + np.sum(shift_vector_full[i, 0:j+1, 2])) + 10
            image_full[x_min:x_max,y_min:y_max,z_min:z_max] += image_sum_filtered_list[i*image_tile_size + j]*shift_filter_mask_list[i*image_tile_size + j]
            image_overlap_full[x_min:x_max,y_min:y_max,z_min:z_max][shift_filter_mask_list[i*image_tile_size + j] > 0] += 1
    image_overlap_full[image_overlap_full == 0] = 1
    image_full = image_full/image_overlap_full
    image_norm = image_full/np.max(image_full)
    image_padded = skimage.util.pad(image_norm, 5, mode = 'edge')
    image_final = np.zeros(image_norm.shape)
    for i in range(20):
        for j in range(20):
            x_start = i*100
            x_end = (i+1)*100 + 10
            y_start = j*100
            y_end = (j+1)*100 + 10
            image_chunk = image_padded[x_start: x_end, y_start: y_end, :]
            image_lp_chunk = line_profile_v2(image_chunk.astype(np.float64), 11, 9, 9)
            # image_lp_chunk = np.nan_to_num(image_lp_chunk)
            image_lp_chunk_min = np.min(image_lp_chunk, axis = 4)
            image_lp_chunk_max = np.max(image_lp_chunk, axis = 4)
            image_lp_chunk_max = image_lp_chunk_max - image_lp_chunk_min
            image_lp_chunk = image_lp_chunk - image_lp_chunk_min[:,:,:,:,None]
            image_lp_rel_norm_chunk = image_lp_chunk/(image_lp_chunk_max[:,:,:,:,None] + 1e-8)
            image_lp_rnc_chunk = image_lp_rel_norm_chunk[:,:,:,:,5]
            image_lprns_chunk = np.average(image_lp_rnc_chunk, axis = 3)
            image_lprn_lq_chunk = np.percentile(image_lp_rnc_chunk, 25, axis = 3)
            image_lprn_uq_chunk = np.percentile(image_lp_rnc_chunk, 75, axis = 3)
            image_lprn_qcv_chunk = (image_lprn_uq_chunk - image_lprn_lq_chunk)/(image_lprn_uq_chunk + image_lprn_lq_chunk + 1e-8)
            # image_lprn_qcv_chunk = np.nan_to_num(image_lprn_qcv_chunk)
            image_final_chunk = image_lprns_chunk*(1-image_lprn_qcv_chunk)
            image_final[i*100:(i+1)*100, j*100:(j+1)*100, :] = image_final_chunk
    intensity_rough_seg = np.zeros(image_final.shape).astype(int)
    intensity_rough_seg[image_final > 0] = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final[image_final >0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(intensity_rough_seg[image_final > 0] == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    intensity_rough_seg_mask = np.zeros(image_final.shape).astype(int)
    if i0 < i1:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0]
    else:
        intensity_rough_seg_mask[image_final > 0] = intensity_rough_seg[image_final > 0] == 0
    image_final_seg = np.zeros(image_final.shape).astype(int)
    image_final_seg[image_final > 0] = KMeans(n_clusters = 3, random_state = 0).fit_predict(image_final[image_final > 0].reshape(-1,1))
    image0 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 0)
    image1 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 1)
    image2 = image_final[image_final > 0]*(image_final_seg[image_final > 0] == 2)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    i2 = np.average(image2[image2 > 0])
    image_lprns_rsf = np.zeros(image_final.shape).astype(int)
    image_lprns_rsf[image_final > 0] = (image_final_seg[image_final > 0] == np.argmax([i0,i1,i2]))*1
    image_lprns_rsfbo = binary_opening(image_lprns_rsf)
    image_lprns_rsfbosm = skimage.morphology.remove_small_objects(image_lprns_rsfbo, 10)
    image_lprns_rsfbosm_bfh = binary_fill_holes(image_lprns_rsfbosm)
    intensity_rough_seg_mask_bfh = binary_fill_holes(intensity_rough_seg_mask)
    image_watershed_mask = image_lprns_rsfbosm_bfh*intensity_rough_seg_mask_bfh
    image_watershed_seeds = skimage.morphology.label(image_watershed_mask)
    image_norm_log = np.log10(image_norm + 1e-8)
    image_bkg_filter = np.zeros(image_norm.shape)
    image_bkg_filter[image_norm > 0] =  KMeans(n_clusters = 2, random_state = 0).fit_predict(image_norm_log[image_norm > 0].reshape(-1, 1))
    image0 = image_norm[image_norm > 0]*(image_bkg_filter[image_norm > 0] == 0)
    image1 = image_norm[image_norm > 0]*(image_bkg_filter[image_norm > 0] == 1)
    i0 = np.average(image0[image0 > 0])
    i1 = np.average(image1[image1 > 0])
    if (i0 < i1):
        image_bkg_filter_mask = (image_bkg_filter == 1)
    else:
        image_bkg_filter[~(image_norm > 0)] = 1
        image_bkg_filter_mask = (image_bkg_filter == 0)
    image_final_bkg_filtered = image_final*image_bkg_filter_mask
    image_watershed_seeds_bkg_filtered = image_watershed_seeds*image_bkg_filter_mask
    image_watershed_mask_bkg_filtered = image_watershed_mask*image_bkg_filter_mask
    image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds_bkg_filtered, mask = image_watershed_mask_bkg_filtered)
    image_seg = skimage.segmentation.relabel_sequential(image_seg)[0]
    save_segmentation(image_seg, sample)
    return(image_registered_sum, image_registered, image_final_bkg_filtered, image_seg)

def get_volume(sample):
    excitations = ['488', '514', '561', '633']
    image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
    image_stack = [get_registered_image_from_tstack(filename) for filename in image_name]
    image_sum = [np.sum(image, axis = (3,4)) for image in image_stack]
    shift_vectors = [skimage.registration.phase_cross_correlation(np.log(image_sum[0]+1e-8), np.log(image_sum[i]+1e-8))[0] for i in range(1,4)]
    shift_vectors.insert(0, np.asarray([0.0,0.0,0.0]))
    image_shape = image_stack[0].shape
    for i in range(len(image_stack)):
        shift_x = int(shift_vectors[i][0])
        shift_y = int(shift_vectors[i][1])
        shift_z = int(shift_vectors[i][2])
        print(shift_x, shift_y, shift_z)
        original_x_min = int(np.maximum(0, shift_x))
        original_x_max = int(image_shape[0] + np.minimum(0, shift_x))
        original_y_min = int(np.maximum(0, shift_y))
        original_y_max = int(image_shape[1] + np.minimum(0, shift_y))
        original_z_min = int(np.maximum(0, shift_z))
        original_z_max = int(image_shape[2] + np.minimum(0, shift_z))
        registered_x_min = int(-np.minimum(0, shift_x))
        registered_x_max = int(image_shape[0] - np.maximum(0, shift_x))
        registered_y_min = int(-np.minimum(0, shift_y))
        registered_y_max = int(image_shape[1] - np.maximum(0, shift_y))
        registered_z_min = int(-np.minimum(0, shift_z))
        registered_z_max = int(image_shape[2] - np.maximum(0, shift_z))
        image_stack[i][original_x_min: original_x_max, original_y_min: original_y_max, original_z_min: original_z_max, :] = image_stack[i][registered_x_min: registered_x_max, registered_y_min: registered_y_max, registered_z_min: registered_z_max, :]
    return(image_stack)

def measure_epithelial_distance(cx, cy, ebc):
    distance = np.zeros(ebc.shape[0])
    for i in range(ebc.shape[0]):
        distance[i] = np.sqrt((cx - ebc[i,0])**2 + (cy - ebc[i,1])**2)
    return(np.min(distance))

def save_spectral_image_max(image_registered, sample):
    image_spectral_max = np.max(image_registered, axis = 2)
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(10),cm_to_inches(10))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_spectral_max, cmap = 'inferno')
    scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    spectral_max_filename = '{}_spectral_max.pdf'.format(sample)
    fig.savefig(spectral_max_filename, dpi = 300, transparent = True)
    plt.close()
    return

def save_spectral_image_max_epithelial_boundary(image_registered, epithelial_boundary, sample):
    image_spectral_max = np.max(image_registered, axis = 2)
    image_spectral_max[epithelial_boundary == 1] = np.nan
    cmap = cm.get_cmap('inferno')
    cmap.set_bad((1,1,1))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(10),cm_to_inches(10))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_spectral_max, cmap = 'inferno')
    scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    spectral_max_filename = '{}_spectral_max_epithelial_boundary.pdf'.format(sample)
    fig.savefig(spectral_max_filename, dpi = 300, transparent = True)
    plt.close()
    return

def measure_biofilm_images_2d(image_name, marker_x, marker_y):
    sample = re.sub('_488.czi', '', image_name[0])
    image_registered_sum, image_registered, image_final_bkg_filtered, segmentation, adjacency_seg, epithelial_region, epithelial_boundary = generate_2d_segmentation(image_name, sample, marker_x, marker_y)
    np.save('{}_registered.npy'.format(sample), image_registered)
    np.save('{}_seg.npy'.format(sample), segmentation)
    np.save('{}_adjacency_seg.npy'.format(sample), adjacency_seg)
    np.save('{}_epithelial_region.npy'.format(sample), epithelial_region)
    np.save('{}_epithelial_boundary.npy'.format(sample), epithelial_boundary)
    save_spectral_image_max(image_registered, sample)
    save_spectral_image_max_epithelial_boundary(image_registered, epithelial_boundary, sample)
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[2]))
    for k in range(0, image_registered.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    pd.DataFrame(avgint).to_csv('{}_avgint.csv'.format(sample), index = None)
    return

def measure_biofilm_images_2d_from_zstack(image_name, image_stack, z, marker_x, marker_y):
    sample = re.sub('_488.czi', '', image_name[0])
    image_registered_sum, image_registered, image_final_bkg_filtered, segmentation, adjacency_seg, epithelial_region, epithelial_boundary = generate_2d_segmentation_from_zstack_t_sum(image_stack, sample, z, marker_x, marker_y)
    np.save('{}_z_{}_registered.npy'.format(sample, z), image_registered)
    np.save('{}_z_{}_seg.npy'.format(sample, z), segmentation)
    np.save('{}_z_{}_adjacency_seg.npy'.format(sample, z), adjacency_seg)
    np.save('{}_z_{}_epithelial_region.npy'.format(sample, z), epithelial_region)
    np.save('{}_z_{}_epithelial_boundary.npy'.format(sample, z), epithelial_boundary)
    save_spectral_image_max(image_registered, '{}_z_{}'.format(sample,z))
    save_spectral_image_max_epithelial_boundary(image_registered, epithelial_boundary, '{}_z_{}'.format(sample,z))
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[2]))
    for k in range(0, image_registered.shape[2]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    pd.DataFrame(avgint).to_csv('{}_z_{}_avgint.csv'.format(sample, z), index = None)

    # np.save('{}_z_{}_registered.npy'.format(sample, z), image_registered)
    # np.save('{}_z_{}_seg.npy'.format(sample, z), segmentation)
    # np.save('{}_z_{}_adjacency_seg.npy'.format(sample, z), adjacency_seg)
    # save_segmentation(segmentation, '{}_z_{}'.format(sample, z))
    # cells = skimage.measure.regionprops(segmentation)
    # avgint = np.empty((len(cells), image_registered.shape[2]))
    # for k in range(0, image_registered.shape[2]):
    #     cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,k])
    #     avgint[:,k] = [x.mean_intensity for x in cells]
    # avgint_norm = avgint/np.max(avgint, axis = 1)[:,None]
    # avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    # avgint_norm_scaled = scaler.transform(avgint_norm[:,0:63])
    # avgint_norm[:,63] = clf[0].predict(avgint_norm_scaled[:,0:23])
    # avgint_norm[:,64] = clf[1].predict(avgint_norm_scaled[:,23:43])
    # avgint_norm[:,65] = clf[2].predict(avgint_norm_scaled[:,43:57])
    # avgint_norm[:,66] = clf[3].predict(avgint_norm_scaled[:,57:63])
    # avgint_umap_transformed = umap_transform.transform(avgint_norm)
    # cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    # cell_info = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None]), axis = 1))
    # cell_info[68] = sample
    # cell_info[69] = np.asarray([x.label for x in cells])
    # cell_info[70] = np.asarray([x.centroid[0] for x in cells])
    # cell_info[71] = np.asarray([x.centroid[1] for x in cells])
    # cell_info[72] = np.asarray([x.major_axis_length for x in cells])
    # cell_info[73] = np.asarray([x.minor_axis_length for x in cells])
    # cell_info[74] = np.asarray([x.eccentricity for x in cells])
    # cell_info[75] = np.asarray([x.orientation for x in cells])
    # cell_info[76] = np.asarray([x.area for x in cells])
    # cellinfofilename = '{}_z_{}_cell_information.csv'.format(sample, z)
    # cell_info.to_csv(cellinfofilename, index = None, header = None)
    # ids = list(set(cell_ids_norm))
    # image_identification = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    # image_identification_barcode = np.zeros(segmentation.shape)
    # for q in range(0, len(ids)):
    #     cell_population = np.where(cell_ids_norm == ids[q])[0]
    #     for r in range(0, len(cell_population)):
    #         image_identification_barcode[segmentation == cell_population[r]+1] = int(ids[q], 2)
    #         if ids[q] in taxon_lookup.code.values:
    #             image_identification[segmentation == cell_population[r]+1, :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code.values == ids[q], ['H', 'S', 'V']].values)
    #         else:
    #             image_identification[segmentation == cell_population[r]+1, :] = np.array([1,1,1])
    # np.save('{}_z_{}_identification.npy'.format(sample, z), image_identification)
    # save_identification(image_identification, '{}_z_{}'.format(sample, z))
    # edge_map = skimage.filters.sobel(segmentation > 0)
    # rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    # adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.code.values, columns = taxon_lookup.code.values)
    # for i in range(cell_info.shape[0]):
    #     edges = list(rag.edges(i+1))
    #     for e in edges:
    #         node_1 = e[0]
    #         node_2 = e[1]
    #         if (node_1 != 0) and (node_2 !=0):
    #             barcode_1 = cell_info.iloc[node_1-1,67]
    #             barcode_2 = cell_info.iloc[node_2-1, 67]
    #             adjacency_matrix.loc[barcode_1, barcode_2] += 1
    # adjacencyfilename = '{}_z_{}_adjacency_matrix.csv'.format(sample, z)
    # adjacency_matrix.to_csv(adjacencyfilename)
    return

def measure_biofilm_images_3d(image_name):
    sample = re.sub('_488.czi', '', image_name[0])
    image_registered_sum, image_channel, image_final_bkg_filtered, image_seg = generate_3d_segmentation_slice(image_name, sample)
    np.save('{}_registered.npy'.format(sample), image_registered)
    np.save('{}_seg.npy'.format(sample), segmentation)
    # image_registered = np.average(image_registered, axis = 3)
    cells = skimage.measure.regionprops(segmentation)
    avgint = np.empty((len(cells), image_registered.shape[3]))
    for k in range(0, image_registered.shape[3]):
        cells = skimage.measure.regionprops(segmentation, intensity_image = image_registered[:,:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    avgint_norm = avgint/np.max(avgint, axis = 1)[:,None]
    avgint_norm = np.concatenate((avgint_norm, np.zeros((avgint_norm.shape[0], 4))), axis = 1)
    avgint_norm[:,63] = clf[0].predict(avgint_norm[:,0:23])
    avgint_norm[:,64] = clf[1].predict(avgint_norm[:,23:43])
    avgint_norm[:,65] = clf[2].predict(avgint_norm[:,43:57])
    avgint_norm[:,66] = clf[3].predict(avgint_norm[:,57:63])
    avgint_umap_transformed = umap_transform.transform(avgint_norm)
    cell_ids_norm = clf_umap.predict(avgint_umap_transformed)
    cell_ids_norm_prob = clf_umap.predict_proba(avgint_umap_transformed)
    max_prob = np.max(cell_ids_norm_prob, axis = 1)
    cell_info = pd.DataFrame(np.concatenate((avgint_norm, cell_ids_norm[:,None], max_prob[:,None], cell_ids_norm_prob), axis = 1))
    cell_info.columns = ['channel_{}'.format(i) for i in range(63)] + ['intensity_classification_{}'.format(i) for i in range(4)] + ['cell_barcode', 'max_probability'] + ['{}_prob'.format(x) for x in clf_umap.classes_]
    cell_info['sample'] = sample
    cell_info['label'] = np.asarray([x.label for x in cells])
    cell_info['centroid_x'] = np.asarray([x.centroid[0] for x in cells])
    cell_info['centroid_y'] = np.asarray([x.centroid[1] for x in cells])
    cell_info['centroid_z'] = np.asarray([x.centroid[1] for x in cells])
    cell_info['area'] = np.asarray([x.area for x in cells])
    cell_info['type'] = 'cell'
    cell_info_filename = sample + '_cell_information.csv'
    cell_info.to_csv(cell_info_filename, index = None, header = None)
    ids = list(set(cell_ids_norm))
    image_identification = np.zeros((segmentation.shape[0], segmentation.shape[1], segmentation.shape[2], 3))
    image_identification_barcode = np.zeros(segmentation.shape)
    for q in range(0, len(ids)):
        cell_population = np.where(cell_ids_norm == ids[q])[0]
        for r in range(0, len(cell_population)):
            image_identification_barcode[segmentation == cell_population[r]+1] = int(ids[q], 2)
            if ids[q] in taxon_lookup.code.values:
                image_identification[segmentation == cell_population[r]+1, :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code.values == ids[q], ['H', 'S', 'V']].values)
            else:
                image_identification[segmentation == cell_population[r]+1, :] = np.array([1,1,1])
    np.save('{}_identification.npy'.format(sample), image_identification)
    save_identification_bvox(image_identification, sample)
    debris = segmentation*image_epithelial_area
    image_identification_filtered = image_identification.copy()
    image_identification_filtered[debris > 0] = [0.5,0.5,0.5]
    debris_labels = np.delete(np.unique(debris), 0)
    for i in range(cell_info.shape[0]):
        cell_label = cell_info.loc[i, 'label']
        cell_area = cell_info.loc[i, 'area']
        cell_prob = cell_info.loc[i, 'max_probability']
        if (cell_area > 100000) or (cell_prob <=0.95):
            cell_info.loc[i, 'type'] = 'debris'
            image_identification_filtered[segmentation == cell_label] = [0.5,0.5,0.5]

    np.save('{}_identification_filtered.npy'.format(sample), image_identification_filtered)
    save_identification_bvox(image_identification_filtered, sample)
    return

def main():
    parser = argparse.ArgumentParser('Mesure environmental microbial community spectral images')
    parser.add_argument('-i', '--image_name', dest = 'image_name', nargs = '*', default = [], type = str, help = 'Input image filenames')
    parser.add_argument('-c', '--calibration', dest = 'calibration', type = str, default = '', help = 'calibration image filename')
    parser.add_argument('-d', '--dimension', dest = 'dimension', type = int, help = 'Dimension of images')
    parser.add_argument('-z', '--zslices', dest = 'zslices', type = int, help = 'Indices of z slices to analyze')
    parser.add_argument('-mx', '--marker_x', dest = 'marker_x', type = int, default = 0, help = 'Indices of z slices to analyze')
    parser.add_argument('-my', '--marker_y', dest = 'marker_y', type = int, default = 0, help = 'Indices of z slices to analyze')
    args = parser.parse_args()
    if args.dimension == 2:
        if args.zslices == -1:
            measure_biofilm_images_2d(args.image_name, args.marker_x, args.marker_y)
        else:
            print(args.zslices)
            image_stack = get_t_average_image(args.image_name)
            measure_biofilm_images_2d_from_zstack(args.image_name, image_stack, args.zslices, args.marker_x, args.marker_y)
    else:
        measure_biofilm_images_3d(s)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
