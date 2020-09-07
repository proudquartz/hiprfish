
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
import argparse
import javabridge
import bioformats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import segmentation
from mpl_toolkits.axes_grid1 import make_axes_locatable



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(length):
    return(length/2.54)

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

def plot_count_heatmap(taxa_barcode_sciname_filtered, theme_color, sam_tab_filename):
    detected_count_filename = re.sub('.csv', '_count_summary.pdf', sam_tab_filename)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(27), cm_to_inches(16))
    im = plt.imshow(np.log10(taxa_barcode_sciname_filtered.loc[:,['{}_count'.format(s) for s in sam_tab.SAMPLE.drop_duplicates().values]].values+1), cmap = 'RdBu')
    plt.xlabel('Samples', fontsize = 8, color = theme_color)
    plt.yticks(np.arange(taxa_barcode_sciname_filtered.shape[0]), taxa_barcode_sciname_filtered.sci_name.values, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.subplots_adjust(left = 0.15, bottom = 0.03, top = 0.98, right = 0.98)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    divider = make_axes_locatable(plt.axes())
    cax = divider.append_axes('right', size = '2%', pad = 0.05)
    cbar = plt.colorbar(im, cax = cax, orientation = 'vertical')
    cbar.ax.tick_params(direction = 'in', length = 1, labelsize = 8, colors = theme_color)
    cbar.outline.set_edgecolor(theme_color)
    cbar.set_label(r'$\log_{10}[Cell Count]$', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.1, bottom = 0.07, right = 0.97, top = 0.98)
    plt.savefig(detected_count_filename, dpi = 300, transparent = True)
    return

def plot_shannon_diversity(shannon_diversity, theme_color, sam_tab, summary_dir):
    shannon_diversity_filename = re.sub('.csv', '_shannon_diversity.pdf', sam_tab_filename)
    shannon_list = []
    sampling_time = []
    sampling_time_labels = []
    for s in sam_tab.SAMPLING_TIME.drop_duplicates():
        image_names = sam_tab.loc[sam_tab.SAMPLING_TIME.values == s, 'IMAGES']
        shannon_list.append(shannon_diversity.loc[image_names].values)
    flierprops = dict(marker='o', markerfacecolor= (0,0.5,1), markeredgewidth = 0, markersize=4, alpha = 0.7, linestyle='none')
    capprops = dict(color = theme_color)
    whiskerprops = dict(color = theme_color)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    bp = plt.boxplot(shannon_list, positions = 0.4*sam_tab.SAMPLING_DAY_REFERENCE.drop_duplicates()/30, patch_artist = True, flierprops = flierprops, capprops = capprops, whiskerprops = whiskerprops)
    for b in bp['boxes']:
        b.set_facecolor((0,0.5,1))

    plt.xticks(fontsize = 8)
    plt.xticks(np.arange(0, 0.4*sam_tab.SAMPLING_DAY_REFERENCE.max()/30, 0.4*6), [0, 6, 12, 18, 24, 30])
    plt.xlabel('Sampling Time [month]', color = theme_color, fontsize = 8)
    plt.ylabel('Shannon Diversity', color = theme_color, fontsize = 8)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.15, bottom = 0.25, right = 0.98, top = 0.98)
    plt.savefig('{}/shannon_diversity_time.pdf'.format(summary_dir), dpi = 300, transparent = True)
    plt.close()
    return

def get_cell_image(cell_stat, data_folder):
    image_name = cell_stat['sample']
    centroid_x = cell_stat['centroid_x']
    centroid_y = cell_stat['centroid_y']
    major_axis_length = cell_stat['major_axis']
    cell_label = cell_stat['label']
    image_registered_filename = '{}_registered.npy'.format(image_name)
    image_seg_filename = '{}_seg.npy'.format(image_name)
    image_registered = np.load(image_registered_filename)
    image_seg = np.load(image_seg_filename)
    x_start = max(0, int(centroid_x - major_axis_length - 50))
    x_end = min(image_registered.shape[0], int(centroid_x + major_axis_length + 50))
    y_start = max(0, int(centroid_y - major_axis_length - 50))
    y_end = min(image_registered.shape[1], int(centroid_y + major_axis_length + 50))
    image_registered_sub = image_registered[x_start:x_end,y_start:y_end,:]
    fig = plt.figure(1)
    fig.set_size_inches(cm_to_inches(16), cm_to_inches(6))
    gs = GridSpec(4, 23)
    rem_indices = [0, 23, 43, 57]
    channel_indices = [0, 3, 9, 17]
    for i in range(4):
        for j in range(channel_indices[i], 23):
            ax = plt.subplot(gs[i,j])
            ax.imshow(image_registered_sub[:,:,rem_indices[i] + j - channel_indices[i]], cmap = 'inferno')
            plt.tick_params(length = 0)
            plt.text(0.8,0.8,rem_indices[i] + j - channel_indices[i])
    plt.figure(2)
    plt.imshow(np.sum(image_registered[x_start:x_end,y_start:y_end,:], axis = 2), cmap = 'inferno')
    plt.figure(3)
    plt.imshow(image_seg[x_start:x_end,y_start:y_end] == cell_label)
    plt.show()
    return

def summarize_cell_snr(sam_tab_filename, data_folder, taxa_barcode_sciname, theme_color):
    sam_tab = pd.read_csv(sam_tab_filename)
    cell_info_filtered_list = []
    sam_tab_basename = os.path.basename(sam_tab_filename)
    summary_dir = '{}/{}'.format(data_folder, re.sub('.csv', '_summary', sam_tab_basename))
    os.makedirs(summary_dir)
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        # save_spectral_image_max(image_registered, sample)
        cell_snr_filename = '{}/{}/{}_cell_snr_summary.csv'.format(data_folder, sample, image_name)
        cell_snr = pd.read_csv(cell_snr_filename, dtype = {'cell_barcode':str})
        cell_info_filename = '{}/{}/{}_cell_information.csv'.format(data_folder, sample, image_name)
        cell_info = pd.read_csv(cell_info_filename, dtype = {'cell_barcode':str})
        cell_info_filtered_filename = '{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name)
        cell_info_filtered = pd.read_csv(cell_info_filtered_filename, dtype = {'cell_barcode':str})
        cell_info_filtered_list.append(cell_info_filtered)
        taxa_barcode_sciname = taxa_barcode_sciname.merge(cell_snr.loc[:,['cell_barcode','count', 'mean']], on = 'cell_barcode', how = 'left')
        taxa_barcode_sciname = taxa_barcode_sciname.rename(columns = {'count': '{}_count'.format(image_name), 'mean': '{}_mean'.format(image_name)})
    cell_info_full = pd.concat(cell_info_filtered_list)
    cell_info_full = cell_info_full.reset_index().drop(columns = 'index')
    taxa_barcode_sciname = taxa_barcode_sciname.fillna(0)
    taxa_barcode_sciname['average_count'] = taxa_barcode_sciname.loc[:,['{}_count'.format(s) for s in sam_tab.IMAGES.values]].mean(axis = 1)
    taxa_barcode_sciname['average_mean'] = taxa_barcode_sciname.loc[:,['{}_mean'.format(s) for s in sam_tab.IMAGES.values]].mean(axis = 1)
    taxa_barcode_sciname_filtered = taxa_barcode_sciname.loc[taxa_barcode_sciname.average_count.values > 0,:]
    taxa_barcode_sciname_filtered = taxa_barcode_sciname_filtered.sort_values(by = ['average_mean', 'average_count'], ascending = [False, False]).reset_index().drop(columns = 'index')
    for t in range(0,taxa_barcode_sciname_filtered.shape[0]):
        sci_name = taxa_barcode_sciname_filtered.loc[t, 'sci_name']
        cell_barcode = taxa_barcode_sciname_filtered.loc[t, 'cell_barcode']
        tax_id = taxa_barcode_sciname_filtered.loc[t, 'target_taxon']
        cell_taxa = cell_info_full.loc[cell_info_full.cell_barcode.values == cell_barcode,:].sort_values(by = 'max_intensity', ascending = False).reset_index().drop(columns = ['index'])
        cell_spec = cell_info_full.loc[cell_info_full.cell_barcode.values == cell_barcode, ['channel_{}'.format(cn) for cn in range(63)]].values
        cell_spec_norm = cell_spec/np.max(cell_spec, axis = 1)[:,None]
        spec_heatmap = np.zeros((100,63))
        for cn in range(63):
            hist_data = plt.hist(cell_spec_norm[:,cn], bins = np.arange(0,1.01,0.01))
            spec_heatmap[:,cn] = np.log10(hist_data[0]+1)
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(16), cm_to_inches(14))
        gs = GridSpec(4, 4)
        ax = plt.subplot(gs[0,0])
        ax.text(0,0.8, 'Name: {}'.format(sci_name), fontsize = 8, color = theme_color)
        ax.text(0,0.6, 'TaxID: {}'.format(tax_id), fontsize = 8, color = theme_color)
        ax.text(0,0.4, 'Barcode: {}'.format(cell_barcode), fontsize = 8, color = theme_color)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax = plt.subplot(gs[0,1:3])
        ax.imshow(spec_heatmap, cmap = 'inferno', origin = 'lower')
        ax.set_aspect(0.5)
        plt.xlabel('Channel', fontsize = 8, color = theme_color)
        plt.ylabel('Intensity', fontsize = 8, color = theme_color)
        plt.tick_params(direction = 'in', labelsize = 8, colors = theme_color)
        ax.spines['left'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax = plt.subplot(gs[1,0])
        plt.hist(cell_taxa.loc[:,'major_axis'].values*0.07, bins = 100, color = (0,0.5,1), histtype = 'step')
        plt.xlabel(r'Major Axis [$\mu$m]', fontsize = 8, color = theme_color)
        plt.ylabel('Frequency', fontsize = 8, color = theme_color)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.ticklabel_format(axis = 'y', style = 'sci', fontsize = 8, scilimits = (0,0), useMathText = True)
        ax.spines['left'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax = plt.subplot(gs[1,1])
        plt.hist(cell_taxa.loc[:,'minor_axis'].values*0.07, bins = 100, color = (0,0.5,1), histtype = 'step')
        plt.xlabel(r'Minor Axis [$\mu$m]', fontsize = 8, color = theme_color)
        plt.ylabel('Frequency', fontsize = 8, color = theme_color)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.ticklabel_format(axis = 'y', style = 'sci', fontsize = 8, scilimits = (0,0), useMathText = True)
        ax.spines['left'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax = plt.subplot(gs[1,2])
        plt.hist(cell_taxa.loc[:,'major_axis'].values/cell_taxa.loc[:,'minor_axis'].values, bins = 100, color = (0,0.5,1), histtype = 'step')
        plt.xlabel('Aspect Ratio', fontsize = 8, color = theme_color)
        plt.ylabel('Frequency', fontsize = 8, color = theme_color)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.ticklabel_format(axis = 'y', style = 'sci', fontsize = 8, scilimits = (0,0), useMathText = True)
        ax.spines['left'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax = plt.subplot(gs[1,3])
        plt.hist(cell_taxa.loc[:,'max_intensity'].values, bins = 100, color = (0,0.5,1), histtype = 'step')
        plt.xlabel('SNR', fontsize = 8, color = theme_color)
        plt.ylabel('Frequency', fontsize = 8, color = theme_color)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.ticklabel_format(axis = 'y', style = 'sci', fontsize = 8, scilimits = (0,0), useMathText = True)
        ax.spines['left'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        for c in range(min(cell_taxa.shape[0], 8)):
            row, col = np.divmod(c,4)
            ax = plt.subplot(gs[2 + row, col])
            image_name = cell_taxa.loc[c, 'sample']
            image_registered_filename = '{}_registered.npy'.format(image_name)
            image_registered = np.load(image_registered_filename)
            image_seg_filename = '{}_seg.npy'.format(image_name)
            image_seg = np.load(image_seg_filename)
            centroid_x = cell_taxa.loc[c,'centroid_x']
            centroid_y = cell_taxa.loc[c,'centroid_y']
            major_axis_length = cell_taxa.loc[c,'major_axis']
            cell_label = cell_taxa.loc[c,'label']
            x_start = max(0, int(centroid_x - major_axis_length - 50))
            x_end = min(image_registered.shape[0], int(centroid_x + major_axis_length + 50))
            y_start = max(0, int(centroid_y - major_axis_length - 50))
            y_end = min(image_registered.shape[1], int(centroid_y + major_axis_length + 50))
            image_registered_sub = image_registered[x_start:x_end,y_start:y_end,:]
            image_seg_sub = image_seg[x_start:x_end,y_start:y_end]
            image_seg_cell = ndi.binary_fill_holes(image_seg_sub == cell_label)
            cell_boundary = skimage.segmentation.find_boundaries(image_seg_cell, mode = 'outer')
            image_registered_sub_sum = np.sum(image_registered_sub, axis = 2)
            image_registered_sub_sum[cell_boundary] = np.nan
            cmap = plt.get_cmap('inferno')
            cmap.set_bad(color = 'white')
            ax.imshow(image_registered_sub_sum, cmap = cmap)
            plt.tick_params(length = 0)
            ax.axis('off')
        plt.subplots_adjust(left = 0.06, bottom = 0.02, right = 0.98, top = 0.98, hspace = 0.5, wspace = 0.5)
        plt.savefig('{}/{}_dashboard.pdf'.format(summary_dir, sci_name), dpi = 300, transparent = True)
    plot_count_heatmap(taxa_barcode_sciname_filtered, theme_color, sam_tab_filename)
    relative_abundance = taxa_barcode_sciname_filtered.loc[:,['{}_count'.format(s) for s in sam_tab.IMAGES.values]].values/taxa_barcode_sciname_filtered.loc[:,['{}_count'.format(s) for s in sam_tab.IMAGES.values]].max(axis = 0)[None,:]
    shannon_diversity = pd.DataFrame(np.zeros(relative_abundance.shape[1]), index = sam_tab.IMAGES.values)
    for i in range(shannon_diversity.shape[0]):
        image_name = sam_tab.IMAGES.values[i]
        ra_sample = relative_abundance[:,i]
        ra_filtered = ra_sample[ra_sample > 0]
        shannon_diversity.loc[image_name] = -np.sum(ra_filtered*np.log(ra_filtered))
    taxa_local_order = pd.DataFrame(index = taxa_barcode_sciname_filtered.sci_name.values, columns = sam_tab.IMAGES.values)
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        cell_info_filtered_filename = '{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name)
        cell_info_filtered = pd.read_csv(cell_info_filtered_filename, dtype = {'cell_barcode':str})
        for t in range(0,taxa_barcode_sciname_filtered.shape[0]):
            sci_name = taxa_barcode_sciname_filtered.loc[t, 'sci_name']
            cell_barcode = taxa_barcode_sciname_filtered.loc[t, 'cell_barcode']
            tax_id = taxa_barcode_sciname_filtered.loc[t, 'target_taxon']
            cell_info_filtered_taxa_alignment_eigenvalue = cell_info_filtered.loc[cell_info_filtered.cell_barcode.values == cell_barcode, 'q_eigenvalue'].values
            taxa_local_order.loc[sci_name,image_name] = np.average(cell_info_filtered_taxa_alignment_eigenvalue)
    taxa_local_order = taxa_local_order.fillna(0)
    taxa_local_order['average_eigenvalue'] = np.average(taxa_local_order.values, axis = 1)
    taxa_local_order_filtered = taxa_local_order.loc[taxa_local_order.average_eigenvalue > 0, :]
    taxa_local_order_filtered = taxa_local_order_filtered.sort_values(by = 'average_eigenvalue', ascending = False).reset_index().drop(columns = ['index'])
    taxa_local_order_filtered_tp = pd.DataFrame(index = taxa_local_order_filtered.index, columns = sam_tab.SAMPLING_TIME.drop_duplicates())
    for st in sam_tab.SAMPLING_TIME.values:
        image_list = sam_tab.loc[sam_tab.SAMPLING_TIME.values == st, 'IMAGES'].values
        taxa_local_order_filtered_tp.loc[:,st] = np.average(taxa_local_order_filtered.loc[:,image_list].values, axis = 1)

    return

def summarize_cell_spec(sam_tab_filename, data_folder, taxa_barcode_sciname, theme_color):
    sam_tab = pd.read_csv(sam_tab_filename)
    cell_info_filtered_list = []
    sam_tab_basename = os.path.basename(sam_tab_filename)
    summary_dir = '{}/{}'.format(data_folder, re.sub('.csv', '_summary', sam_tab_basename))
    os.makedirs(summary_dir)
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        # save_spectral_image_max(image_registered, sample)
        cell_snr_filename = '{}/{}/{}_cell_snr_summary.csv'.format(data_folder, sample, image_name)
        cell_snr = pd.read_csv(cell_snr_filename, dtype = {'cell_barcode':str})
        cell_info_filename = '{}/{}/{}_cell_information.csv'.format(data_folder, sample, image_name)
        cell_info = pd.read_csv(cell_info_filename, dtype = {'cell_barcode':str})
        cell_info_filtered_filename = '{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name)
        cell_info_filtered = pd.read_csv(cell_info_filtered_filename, dtype = {'cell_barcode':str})
        cell_info_filtered_list.append(cell_info_filtered)
        taxa_barcode_sciname = taxa_barcode_sciname.merge(cell_snr.loc[:,['cell_barcode','count', 'mean']], on = 'cell_barcode', how = 'left')
        taxa_barcode_sciname = taxa_barcode_sciname.rename(columns = {'count': '{}_count'.format(image_name), 'mean': '{}_mean'.format(image_name)})
    cell_info_full = pd.concat(cell_info_filtered_list)
    cell_info_full = cell_info_full.reset_index().drop(columns = 'index')
    taxa_barcode_sciname = taxa_barcode_sciname.fillna(0)
    taxa_barcode_sciname['average_count'] = taxa_barcode_sciname.loc[:,['{}_count'.format(s) for s in sam_tab.IMAGES.values]].mean(axis = 1)
    taxa_barcode_sciname['average_mean'] = taxa_barcode_sciname.loc[:,['{}_mean'.format(s) for s in sam_tab.IMAGES.values]].mean(axis = 1)
    taxa_barcode_sciname_filtered = taxa_barcode_sciname.loc[taxa_barcode_sciname.average_count.values > 0,:]
    taxa_barcode_sciname_filtered = taxa_barcode_sciname_filtered.sort_values(by = ['average_mean', 'average_count'], ascending = [False, False]).reset_index().drop(columns = 'index')
    spec_heatmap = pd.DataFrame(index = taxa_barcode_sciname_filtered.sci_name.values, columns = ['channel_{}'.format(i) for i in range(63)])
    for t in range(0,taxa_barcode_sciname_filtered.shape[0]):
        sci_name = taxa_barcode_sciname_filtered.loc[t, 'sci_name']
        cell_barcode = taxa_barcode_sciname_filtered.loc[t, 'cell_barcode']
        tax_id = taxa_barcode_sciname_filtered.loc[t, 'target_taxon']
        cell_taxa = cell_info_full.loc[cell_info_full.cell_barcode.values == cell_barcode,:].sort_values(by = 'max_intensity', ascending = False).reset_index().drop(columns = ['index'])
        cell_spec = cell_info_full.loc[cell_info_full.cell_barcode.values == cell_barcode, ['channel_{}'.format(cn) for cn in range(63)]].values
        cell_spec_norm = cell_spec/np.max(cell_spec, axis = 1)[:,None]
        cell_spec_norm_avg = np.average(cell_spec_norm, axis = 0)
        spec_heatmap.loc[sci_name, ['channel_{}'.format(i) for i in range(63)]] = cell_spec_norm_avg
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(15), cm_to_inches(14))
    plt.imshow(spec_heatmap.values.astype(float), cmap = 'inferno')
    plt.xlabel('Channels', fontsize = 8, color = theme_color)
    plt.yticks(np.arange(taxa_barcode_sciname_filtered.shape[0]), taxa_barcode_sciname_filtered.sci_name.values, fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.subplots_adjust(left = 0.28, bottom = 0.1, right = 0.98, top = 0.98)
    return

def summarize_abundance_comparison(sam_tab_filename, data_folder, taxa_barcode_sciname, blast_lineage_filename, theme_color):
    sam_tab = pd.read_csv(sam_tab_filename)
    cell_info_filtered_list = []
    sam_tab_basename = os.path.basename(sam_tab_filename)
    summary_dir = '{}/{}'.format(data_folder, re.sub('.csv', '_summary', sam_tab_basename))
    os.makedirs(summary_dir)
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        # save_spectral_image_max(image_registered, sample)
        cell_snr_filename = '{}/{}/{}_cell_snr_summary.csv'.format(data_folder, sample, image_name)
        cell_snr = pd.read_csv(cell_snr_filename, dtype = {'cell_barcode':str})
        cell_info_filename = '{}/{}/{}_cell_information.csv'.format(data_folder, sample, image_name)
        cell_info = pd.read_csv(cell_info_filename, dtype = {'cell_barcode':str})
        cell_info_filtered_filename = '{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name)
        cell_info_filtered = pd.read_csv(cell_info_filtered_filename, dtype = {'cell_barcode':str})
        cell_info_filtered_list.append(cell_info_filtered)
        taxa_barcode_sciname = taxa_barcode_sciname.merge(cell_snr.loc[:,['cell_barcode','count', 'mean']], on = 'cell_barcode', how = 'left')
        taxa_barcode_sciname = taxa_barcode_sciname.rename(columns = {'count': '{}_count'.format(image_name), 'mean': '{}_mean'.format(image_name)})
    cell_info_full = pd.concat(cell_info_filtered_list)
    cell_info_full = cell_info_full.reset_index().drop(columns = 'index')
    taxa_barcode_sciname = taxa_barcode_sciname.fillna(0)
    taxa_barcode_sciname['average_count'] = taxa_barcode_sciname.loc[:,['{}_count'.format(s) for s in sam_tab.IMAGES.values]].mean(axis = 1)
    taxa_barcode_sciname['average_mean'] = taxa_barcode_sciname.loc[:,['{}_mean'.format(s) for s in sam_tab.IMAGES.values]].mean(axis = 1)
    taxa_barcode_sciname_filtered = taxa_barcode_sciname.loc[taxa_barcode_sciname.average_count.values > 0,:]
    taxa_barcode_sciname_filtered = taxa_barcode_sciname_filtered.sort_values(by = ['average_mean', 'average_count'], ascending = [False, False]).reset_index().drop(columns = 'index')
    barcode_abundance = cell_info_full.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['cell_barcode', 'imaging_abundance']
    barcode_abundance = barcode_abundance.merge(taxa_barcode_sciname.loc[:,['target_taxon', 'cell_barcode', 'mode_count']], on = 'cell_barcode', how = 'left')
    blast_lineage = pd.read_table(blast_lineage_filename)
    pacbio_abundance = blast_lineage.genus.value_counts().reset_index()
    pacbio_abundance.columns = ['target_taxon', 'pacbio_abundance']
    abundance_comparison = barcode_abundance.merge(pacbio_abundance, on = 'target_taxon', how = 'left')
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    plt.plot(abundance_comparison.pacbio_abundance.values, abundance_comparison.imaging_abundance.values, 'o', color = (0,0.5,1), markersize = 4, alpha = 0.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('pacbio abundance')
    plt.ylabel('imaging abundance')
    plt.minorticks_off()
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.2, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_microbiome_abundance_comparison.pdf'.format(summary_dir), dpi = 300, transparent = True)
    plt.close()

    return

@dask.delayed
def generate_random_spatial_adjacency_network(image_seg, adjacency_seg, cell_info_filtered, taxa_barcode_sciname):
    edge_map = skimage.filters.sobel(image_seg > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    adjacency_matrix = pd.DataFrame(np.zeros((taxa_barcode_sciname.shape[0], taxa_barcode_sciname.shape[0])), index = taxa_barcode_sciname.cell_barcode.values, columns = taxa_barcode_sciname.cell_barcode.values)
    cell_label_list = cell_info_filtered.label.values
    cell_info_scrambled = cell_info_filtered.copy()
    cell_info_scrambled['cell_barcode'] = cell_info_filtered.cell_barcode.sample(frac = 1).values
    for i in range(cell_info_filtered.shape[0]):
        edges = list(rag.edges(i+1))
        for e in edges:
            node_1 = e[0]
            node_2 = e[1]
            if (node_1 != 0) and (node_2 !=0) and node_1 in cell_label_list and node_2 in cell_label_list:
                barcode_1 = cell_info_scrambled.loc[cell_info_scrambled.label.values == node_1, 'cell_barcode'].values[0]
                barcode_2 = cell_info_scrambled.loc[cell_info_scrambled.label.values == node_2, 'cell_barcode'].values[0]
                adjacency_matrix.loc[barcode_1, barcode_2] += 1
    return(adjacency_matrix)

def summarize_spatial_adjacency(sam_tab_filename, data_folder, taxa_barcode_sciname, theme_color):
    sam_tab = pd.read_csv(sam_tab_filename)
    adjacency_matrix_list = []
    sampling_times = sam_tab.SAMPLING_TIME.drop_duplicates()
    for st in sampling_times:
        sam_tab_sub = sam_tab.loc[sam_tab.SAMPLING_TIME.values == st, :].reset_index().drop(columns = 'index')
        adjacency_matrix_tp = pd.DataFrame(np.zeros((taxa_barcode_sciname.shape[0], taxa_barcode_sciname.shape[0])), index = taxa_barcode_sciname.cell_barcode.values, columns = taxa_barcode_sciname.cell_barcode.values)
        for i in range(sam_tab_sub.shape[0]):
            sample = sam_tab_sub.loc[i,'SAMPLE']
            image_name = sam_tab_sub.loc[i,'IMAGES']
            image_seg = np.load('{}/{}/{}_seg.npy'.format(data_folder, sample, image_name))
            adjacency_seg = np.load('{}/{}/{}_adjacency_seg.npy'.format(data_folder, sample, image_name))
            image_identification = np.load('{}/{}/{}_identification.npy'.format(data_folder, sample, image_name))
            cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
            adjacency_matrix = pd.read_csv('{}/{}/{}_adjacency_matrix.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
            adjacency_matrix_tp.loc[:,:] = adjacency_matrix_tp.values + adjacency_matrix.values
        adjacency_matrix_list.append(adjacency_matrix_tp)
    adjacency_matrix_stack = np.stack(adjacency_matrix_list, axis = 2)
    adjacency_matrix_mask = np.prod(adjacency_matrix_stack > 0, axis = 2)
    adj_filtered = adjacency_matrix_mask[:,np.sum(adjacency_matrix_mask, axis = 0) > 0]
    adj_filtered = adj_filtered[np.sum(adj_filtered, axis = 1) > 0,:]
    network_distances = np.zeros((sampling_times.shape[0], sampling_times.shape[0]))
    random_network = np.random.random(adjacency_matrix_list[0].shape)
    random_network_norm = random_network/np.max(random_network)
    for i in range(network_distances.shape[0]):
        for j in range(network_distances.shape[0]):
            adj_i = adjacency_matrix_list[i].values
            adj_j = adjacency_matrix_list[j].values
            np.fill_diagonal(adj_i, 0)
            np.fill_diagonal(adj_j, 0)
            adj_i_norm = adj_i/np.max(adj_i)
            adj_j_norm = adj_j/np.max(adj_j)
            if i != j:
                network_distances[i,j] = np.sum((adj_i_norm - adj_j_norm)**2)
            else:
                network_distances[i,j] = np.sum((adj_i_norm - random_network_norm)**2)

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4), cm_to_inches(4))
    im = plt.imshow(np.log(network_distances), cmap = 'RdBu')
    plt.xticks(np.arange(7), ['0', '3', '9', '12', '15', '21', '27'], rotation = 90)
    plt.yticks(np.arange(7), ['0', '3', '9', '12', '15', '21', '27'])
    plt.xlabel('Sampling Time [month]', fontsize = 8, color = theme_color)
    plt.ylabel('Sampling Time [month]', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    divider = make_axes_locatable(plt.axes())
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    cbar = plt.colorbar(im, cax = cax)
    cbar.ax.tick_params(direction = 'in', length = 1, labelsize = 8, colors = theme_color)
    cbar.outline.set_edgecolor(theme_color)
    cbar.set_label('Network Difference', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.15, right = 0.85, top = 0.98)
    plt.savefig('{}/images_table_microbiome_2_summary/adjacency_matrix_comparison.pdf'.format(data_folder), dpi = 300, transparent = True)
    return

def main():
    parser = argparse.ArgumentParser('Measure environmental microbial community spectral images')
    parser.add_argument('sam_tab_filename', type = str, help = 'Input folder containing spectral images')
    parser.add_argument('probe_design_filename', type = str, help = 'Input folder containing spectral images')
    parser.add_argument('data_folder', type = str, help = 'Input folder containing spectral images')
    parser.add_argument('-t', '--theme_color', dest = 'theme_color', type = str, help = 'Input folder containing spectral images')
    args = parser.parse_args()
    probes = pd.read_csv(args.probe_design_filename, dtype = {'code':str})
    summarize_cell_snr(args.sam_tab_filename, args.data_folder, taxa_barcode_sciname, theme_color)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
