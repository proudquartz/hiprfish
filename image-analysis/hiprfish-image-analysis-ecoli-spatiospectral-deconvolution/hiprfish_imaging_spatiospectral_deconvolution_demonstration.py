
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
from skimage import future
from skimage import feature
from skimage import restoration
from skimage import registration
from skimage import segmentation
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from matplotlib import cm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def generate_nn_pairs(sample):
    sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/09_13_2018_1023_mix/09_13_2018_1023_mix_fov_1'
    image_registered = np.load('{}_registered.npy'.format(sample))
    image_seg = np.load('{}_seg.npy'.format(sample))
    cell_info = pd.read_csv('{}_cell_information_consensus.csv'.format(sample), header = None, dtype = {100:str})
    image_cn = np.sum(image_registered, axis = 2)
    image_cn = np.log(np.sum(image_registered, axis = 2)+1e-2)
    rough = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_cn.reshape(image_cn.shape[0]*image_cn.shape[1],1))
    rough_seg = rough.reshape(image_cn.shape)
    image0 = image_cn*(rough_seg == 0)
    image1 = image_cn*(rough_seg == 1)
    i0 = np.average(image0[rough_seg == 0])
    i1 = np.average(image1[rough_seg == 1])
    rough_seg_mask = rough_seg == np.argmax([i0, i1])
    adjacency_seg = skimage.segmentation.watershed(-np.sum(image_registered, axis = 2), image_seg, mask = rough_seg_mask)
    edge_map = skimage.filters.sobel(image_seg > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    cell_label_list = cell_info.iloc[:,103].values
    edges = list(rag.edges(cell_info.iloc[404,103]))
    e = edges[1]
    node_1 = e[0]
    node_2 = e[1]
    barcode_1 = cell_info.iloc[cell_info.iloc[:,103].values == node_1, 100].values[0]
    barcode_2 = cell_info.iloc[cell_info.iloc[:,103].values == node_2, 100].values[0]
    cell_1_index = np.where(image_seg == node_1)
    cell_1_pixel_intensity = image_registered[image_seg == node_1, :]
    cell_2_index = np.where(image_seg == node_2)
    cell_2_pixel_intensity = image_registered[image_seg == node_2, :]
    cxj = np.average(np.concatenate([cell_1_index[0], cell_2_index[0]]))
    cyj = np.average(np.concatenate([cell_1_index[1], cell_2_index[1]]))
    joint_intensity_x = np.concatenate([cell_1_pixel_intensity*cell_1_index[0][:,None], cell_2_pixel_intensity*cell_2_index[0][:,None]], axis = 0)
    joint_intensity_y = np.concatenate([cell_1_pixel_intensity*cell_1_index[1][:,None], cell_2_pixel_intensity*cell_2_index[1][:,None]], axis = 0)
    joint_intensity = np.concatenate([cell_1_pixel_intensity, cell_2_pixel_intensity], axis = 0)
    cxj_spectral = np.average(joint_intensity_x, axis = 0)/np.average(joint_intensity, axis = 0)
    cyj_spectral = np.average(joint_intensity_y, axis = 0)/np.average(joint_intensity, axis = 0)

    x_low = int(cxj - 50)
    x_high = int(cxj + 50)
    y_low = int(cyj - 50)
    y_high = int(cyj + 50)
    image_seg_local = np.zeros(image_seg.shape)
    image_seg_local[image_seg == node_1] = 1
    image_seg_local[image_seg == node_2] = 1
    image_seg_local = image_seg_local[x_low:x_high,y_low:y_high]
    image_seg_local_boundary = skimage.segmentation.find_boundaries(image_seg_local, mode = 'inner')
    cmap = cm.get_cmap('inferno')
    cmap.set_bad((1,1,1))
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches())
    k = 75
    image_registered_annotated = image_registered[x_low:x_high,y_low:y_high,k].copy()
    image_registered_annotated[image_seg_local_boundary == 1] = np.nan
    plt.imshow(image_registered_annotated,cmap = cmap)
    plt.plot(50, 50, 'o', color = (0,0.5,1))
    plt.plot(cxj_spectral[k] - cxj + 50, cyj_spectral[k] - cyj + 50, 'o', color = (0.5,1,0))



    for i in range(cell_info.shape[0]):
        barcode = cell_info.iloc[i, 100]
        cell_label = cell_info.iloc[i, 103]
        cell_index = np.where(image_seg == cell_label)
        cell_pixel_intensity = image_registered[image_seg == cell_label, :]
        cx = np.average(cell_index[0])
        cy = np.average(cell_index[1])
        cx_spectral = np.average(cell_pixel_intensity*cell_index[0][:,None], axis = 0)/np.average(cell_pixel_intensity, axis = 0)
        cy_spectral = np.average(cell_pixel_intensity*cell_index[1][:,None], axis = 0)/np.average(cell_pixel_intensity, axis = 0)
        centroid_spectral_distance = np.sqrt((cx - cx_spectral)**2 + (cy - cy_spectral)**2)
        centroid_single_object.append([barcode, np.std(centroid_spectral_distance), np.median(centroid_spectral_distance)])
    centroid_spectral_singlet = pd.DataFrame(np.stack(centroid_single_object, axis = 0))
    centroid_spectral_singlet.columns = ['barcode', 'centroid_spectral_std', 'centroid_spectral_median']
    centroid_spectral.to_csv('{}_centroid_spectral.csv'.format(sample), index = None)
    centroid_spectral_singlet.to_csv('{}_centroid_spectral_singlet.csv'.format(sample), index = None)
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('-s', '--sample_filename', dest = 'sample_filename', type = str, help = 'Image filenames')
    args = parser.parse_args()
    sample = re.sub('_cell_information_consensus.csv', '', args.sample_filename)
    generate_nn_pairs(sample)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
