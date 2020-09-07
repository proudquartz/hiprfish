
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib_scalebar.scalebar import ScaleBar

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def cm_to_inches(x):
    return(x/2.54)

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def collect_deconvolution_results(data_folder, image_tab_filename, theme_color):
    image_tab = pd.read_csv(image_tab_filename)
    cell_merger_counts_list = []
    cell_merger_correct_separation_list = []
    cell_merger_hamming_counts_list = []
    cell_merger_hamming_correct_separation_list = []
    for csm in np.arange(0.2,1.6,0.2):
        cell_merger_counts = np.zeros((1023,1023))
        cell_merger_correct_separation = np.zeros((1023,1023))
        cell_merger_hamming_counts = np.zeros(11)
        cell_merger_hamming_correct_separation = np.zeros(11)
        for i in range(image_tab.shape[0]):
            sample = image_tab.loc[i,'SAMPLE']
            image_name = image_tab.loc[i,'IMAGES']
            centroid_spectral = pd.read_csv('{}/{}/{}_centroid_spectral.csv'.format(data_folder, sample, image_name), dtype = {'barcode_1':str, 'barcode_2':str})
            centroid_spectral_singlet = pd.read_csv('{}/{}/{}_centroid_spectral_singlet.csv'.format(data_folder, sample, image_name), dtype = {'barcode':str})
            centroid_spectral['separation'] = centroid_spectral.centroid_spectral_median.values > csm
            centroid_spectral_singlet['separation'] = centroid_spectral_singlet.centroid_spectral_median.values > csm
            for k in range(centroid_spectral.shape[0]):
                bc_1 = int(centroid_spectral.loc[k, 'barcode_1'], 2)
                bc_2 = int(centroid_spectral.loc[k, 'barcode_2'], 2)
                cell_merger_counts[bc_1 - 1, bc_2 - 1] += 1
                cell_merger_correct_separation[bc_1 - 1, bc_2 - 1] += centroid_spectral.loc[k, 'separation']*1
                hd = hamming2(centroid_spectral.loc[k, 'barcode_1'], centroid_spectral.loc[k, 'barcode_2'])
                cell_merger_hamming_counts[int(hd)] += 1
                cell_merger_hamming_correct_separation[int(hd)] += centroid_spectral.loc[k, 'separation']*1
            for k in range(centroid_spectral_singlet.shape[0]):
                bc = int(centroid_spectral_singlet.loc[k, 'barcode'], 2)
                cell_merger_counts[bc - 1, bc - 1] += 1
                cell_merger_correct_separation[bc - 1, bc - 1] += centroid_spectral_singlet.loc[k, 'separation']*1
                cell_merger_hamming_counts[0] += 1
                cell_merger_hamming_correct_separation[0] += centroid_spectral_singlet.loc[k, 'separation']*1
        cell_merger_counts_list.append(cell_merger_counts)
        cell_merger_correct_separation_list.append(cell_merger_correct_separation)
        cell_merger_hamming_counts_list.append(cell_merger_hamming_counts)
        cell_merger_hamming_correct_separation_list.append(cell_merger_hamming_correct_separation)
    cell_merger_hamming_detection_rate_list = []
    for i in range(7):
         cmhdr = cell_merger_hamming_correct_separation_list[i]/cell_merger_hamming_counts_list[i]
         cell_merger_hamming_detection_rate_list.append(cmhdr)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    cmap = cm.get_cmap('tab10')
    for i in range(7):
        plt.plot(cell_merger_hamming_detection_rate_list[i], '-o', markersize = 2, color = cmap(i/10), label = '{}')
    plt.ylim(-0.03,1.1)
    plt.xlabel(r'$d_{Hamming}$', fontsize = 8, color = theme_color)
    plt.ylabel('Merger detection rate', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, color = theme_color, labelsize = 8)
    plt.xticks(np.arange(0, 11, 2))
    plt.subplots_adjust(left = 0.2, bottom = 0.22, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_1023_mix_barcode_neighbor_hamming_parameter_selection.pdf'.format(data_folder), dpi = 300, transparent = True)
    # barcode neighbor hamming
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    cmap = cm.get_cmap('tab10')
    plt.plot(cell_merger_hamming_detection_rate_list[4], '-o', markersize = 2, color = 'navy')
    plt.ylim(-0.03,1.1)
    plt.xlabel('Hamming distance', fontsize = 8, color = theme_color)
    plt.ylabel('Merger detection rate', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, color = theme_color, labelsize = 8)
    plt.xticks(np.arange(0, 11, 2))
    plt.subplots_adjust(left = 0.2, bottom = 0.22, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_1023_mix_barcode_neighbor_hamming.pdf'.format(data_folder), dpi = 300, transparent = True)
    # barcode neighbor hamming presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(12), cm_to_inches(8))
    cmap = cm.get_cmap('tab10')
    plt.plot(cell_merger_hamming_detection_rate_list[4], '-o', markersize = 2, color = (0,0.25,1))
    plt.ylim(-0.03,1.1)
    plt.xlabel(r'Hamming distance', fontsize = font_size, color = theme_color)
    plt.ylabel('Merger detection rate', fontsize = font_size, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.xticks(np.arange(0, 11, 2))
    plt.subplots_adjust(left = 0.2, bottom = 0.22, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_1023_mix_barcode_neighbor_hamming_presentation.svg'.format(data_folder), dpi = 300, transparent = True)
    plt.close()
    cell_merger_counts_zero_removed = cell_merger_counts.copy()
    cell_merger_counts_zero_removed[cell_merger_counts_zero_removed == 0] += 1
    cell_merger_separation_fraction = cell_merger_correct_separation/cell_merger_counts_zero_removed
    cell_merger_separation_fraction[cell_merger_counts == 0] = np.nan
    # merger detection heatmap
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(18), cm_to_inches(18))
    cmap = cm.get_cmap('Blues')
    cmap.set_bad(color='darkorange')
    ax = plt.imshow(cell_merger_separation_fraction, cmap = cmap, interpolation = 'None')
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.xlabel('Barcodes')
    plt.ylabel('Barcodes')
    plt.xticks([127, 255, 383, 511, 639, 767, 895], [128, 256, 384, 512, 640, 768, 896])
    plt.yticks([127, 255, 383, 511, 639, 767, 895], [128, 256, 384, 512, 640, 768, 896])
    cbaxes = fig.add_axes([0.7, 0.05, 0.1, 0.01])
    cb = plt.colorbar(ax, cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize=8, color = theme_color)
    cb.set_ticks([0,0.5,1])
    cb.set_label('Merger detection rate', fontsize = 8, color = theme_color)
    laxes = fig.add_axes([0.2, 0.05, 0.1, 0.01])
    plt.plot(0, 0, 's', markersize = 4, color = 'darkorange')
    plt.text(0, -0.3, 'Undetected barcode-neighbor combinations', fontsize = 8, ha = 'center')
    plt.axis('off')
    plt.subplots_adjust(left = 0.1, bottom = 0.08, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_1023_mix_barcode_neighbor_heatmap.pdf'.format(data_folder), dpi = 300, transparent = True)
    plt.close()
    # merger detection heatmap presentation
    theme_color = 'white'
    font_size= 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(16), cm_to_inches(16))
    cmap = cm.get_cmap('Blues')
    cmap.set_bad(color='darkorange')
    ax = plt.imshow(cell_merger_separation_fraction, cmap = cmap, interpolation = 'None')
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.xlabel('Barcodes', fontsize = font_size, color = theme_color)
    plt.ylabel('Barcodes', fontsize = font_size, color = theme_color)
    plt.xticks([127, 255, 383, 511, 639, 767, 895], [128, 256, 384, 512, 640, 768, 896])
    plt.yticks([127, 255, 383, 511, 639, 767, 895], [128, 256, 384, 512, 640, 768, 896])
    cbaxes = fig.add_axes([0.7, 0.05, 0.1, 0.01])
    cb = plt.colorbar(ax, cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    cb.set_ticks([0,0.5,1])
    cb.set_label('Merger detection rate', fontsize = 8, color = theme_color)
    laxes = fig.add_axes([0.2, 0.05, 0.1, 0.01])
    plt.plot(0, 0, 's', markersize = 4, color = 'darkorange')
    plt.text(0, -0.3, 'Undetected barcode-neighbor combinations', fontsize = 8, color = theme_color, ha = 'center')
    plt.axis('off')
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.subplots_adjust(left = 0.1, bottom = 0.08, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_1023_mix_barcode_neighbor_heatmap_presentation.svg'.format(data_folder), dpi = 300, transparent = True)


    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Collect summary statistics of HiPRFISH probes for a complex microbial community')

    # data directory
    parser.add_argument('data_dir', type = str, help = 'Directory of the data files')

    # input simulation table
    parser.add_argument('simulation_table', type = str, help = 'Input csv table containing simulation information')

    args = parser.parse_args()

    collect_deconvolution_results(args.data_dir, args.simulation_table, 'black')

if __name__ == '__main__':
    main()
