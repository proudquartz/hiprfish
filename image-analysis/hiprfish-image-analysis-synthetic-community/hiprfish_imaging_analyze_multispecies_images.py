
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from ete3 import NCBITaxa
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 0.5

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

ncbi = NCBITaxa()

def cm_to_inches(length):
    return(length*0.393701)

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def save_classification_image(image, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image)
    plt.axis('off')
    classification_filename = '{}_classification.pdf'.format(sample)
    fig.savefig(classification_filename, dpi = 300, transparent = True)
    plt.close()
    return

def generate_classification_image(cell_info, barcode_assignment):
    sample = cell_info.iloc[0,68]
    image_seg_filename = '{}_seg.npy'.format(sample)
    image_seg = np.load(image_seg_filename)
    image_classification = np.zeros((image_seg.shape[0], image_seg.shape[1], 3))
    cell_info_correct_assignment = cell_info.loc[cell_info.correct_assignment.values == True, :]
    cell_info_incorrect_assignment = cell_info.loc[cell_info.correct_assignment.values == False, :]
    for i in cell_info_correct_assignment[69].values:
        image_classification[image_seg == i] = [0,0.5,1]
    for i in cell_info_incorrect_assignment[69].values:
        image_classification[image_seg == i] = [1,0.5,0]
    save_classification_image(image_classification, sample)
    return

def summarize_error_rate(image_tab, input_folder, probe_design_filename, theme_color = 'black', nbit = 7):
    plotting_info = image_tab.loc[:,['PRIMERSET', 'ENCODING', 'COLOR']].drop_duplicates().reset_index().drop(columns = ['index'])
    for i in range(nbit):
        image_tab['HD{}'.format(i)] = 0
    image_tab['ErrorRate'] = np.nan
    image_tab['UpperLimit'] = 0
    taxid_list = [104102, 108981, 1353, 140100, 1580, 1590, 1718, 285, 438, 56459, 564]
    taxid_sciname = pd.DataFrame.from_dict({564: 'E. coli',
                1718: 'C. glutamicum',
                1590: 'L. plantarum',
                140100: 'V. albensis',
                1580: 'L. brevis',
                438: 'A. plantarum',
                104102: 'A. tropicalis',
                108981: 'A. schindleri',
                285: 'C. testosteroni',
                1353: 'E. gallinarum',
                56459: 'X. vasicola'}, orient = 'index').reset_index()
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8.75), cm_to_inches(7.25))
    gs = GridSpec(2, 1)
    for k in range(plotting_info.shape[0]):
        encoding = plotting_info.loc[k, 'ENCODING']
        image_tab_encoding = image_tab.loc[image_tab.ENCODING.values == encoding, :].reset_index().drop(columns = ['index'])
        enc_set = plotting_info.loc[k, 'PRIMERSET']
        # filenames = ['{}/{}/{}_cell_information.csv'.format(input_folder, image_tab_encoding.loc[i,'SAMPLE'], image_tab_encoding.loc[i,'IMAGES']) for i in range(image_tab_encoding.shape[0])]
        probes = pd.read_csv(probe_design_filename[k], dtype = {'code': str})
        multispecies_summary = probes.loc[:,['target_taxon', 'code']].drop_duplicates().reset_index().drop(columns = ['index'])
        multispecies_summary['ErrorRate'] = 0
        multispecies_summary['UpperLimit'] = 0
        taxid_sciname.columns = ['target_taxon', 'sci_name']
        multispecies_summary = multispecies_summary.merge(taxid_sciname, on = 'target_taxon', how = 'left')
        # taxid_list = taxid_sciname.target_taxon.astype(str).sort_values().reset_index().drop(columns = ['index'])
        multispecies_summary.loc[:,'target_taxon'] = multispecies_summary.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.sort_values(by = ['target_taxon'], ascending = True)
        hamming_distance_list = []
        for t in multispecies_summary.target_taxon.values:
            hamming_distance_list_sub = []
            image_tab_sub = image_tab_encoding.loc[image_tab_encoding.TARGET_TAXON.values == str(t), :].reset_index().drop(columns = ['index'])
            for i in range(image_tab_sub.shape[0]):
                target_taxon = image_tab_sub.loc[i, 'TARGET_TAXON']
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, sample, image_name), header = None, dtype = {68: str})
                # cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, sample, image_name), header = None, dtype = {130: str})
                cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
                cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
                cell_info['Sample'] = sample
                barcode_consensus = []
                nn_dist = np.zeros((cell_info.shape[0], 10))
                for r in range(10):
                    cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_{}.csv'.format(input_folder, sample, image_name, r), header = None, dtype = {67:str})
                    barcode_consensus.append(cell_info.iloc[:,67].values)
                    # cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_{}.csv'.format(input_folder, sample, image_name, r), header = None, dtype = {130:str})
                    # barcode_consensus.append(cell_info.iloc[:,130].values)
                    nn_dist[:,r] = cell_info.iloc[:,68].values
                    # nn_dist[:,r] = cell_info.iloc[:,131].values
                    cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
                    cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
                    cell_info['Sample'] = sample
                barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
                nn_barcode = []
                nn_indices = np.argmin(nn_dist, axis = 1)
                for b in range(barcode_consensus.shape[0]):
                    nn_barcode.append(barcode_consensus[b,nn_indices[b]])
                barcode_assignment = multispecies_summary.loc[multispecies_summary.target_taxon.values == str(target_taxon), 'code'].values[0]
                max_intensity_mean = cell_info.max_intensity.mean()
                max_intensity_std = cell_info.max_intensity.std()
                cell_info_filtered = cell_info.loc[cell_info.max_intensity.values > max_intensity_mean - max_intensity_std, :].reset_index().drop(columns = 'index')
                nn_barcode_filtered = np.array(nn_barcode)[cell_info.max_intensity.values > max_intensity_mean - max_intensity_std]
                cell_info_filtered['correct_assignment'] = nn_barcode_filtered == barcode_assignment
                correct_assignment = np.sum(cell_info_filtered.correct_assignment.values)
                incorrect_assignment = cell_info_filtered.shape[0] - correct_assignment
                cell_info_filtered['barcode'] = np.array(nn_barcode_filtered)
                cell_info_filtered['hamming_distance'] = pd.DataFrame(nn_barcode_filtered)[0].apply(hamming2, args = (barcode_assignment,))
                cell_info_filtered.to_csv('{}/{}/{}_cell_information.csv'.format(input_folder, sample, image_name), index = None)
                hamming_distance_list_sub.append(cell_info_filtered.hamming_distance.values)
                image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'CorrectAssignment'] = correct_assignment
                image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'IncorrectAssignment'] = incorrect_assignment
                for h in range(nbit):
                    image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'HD{}'.format(h)] = np.sum(cell_info_filtered.hamming_distance.values == h)
            hamming_distance_sub = np.concatenate(hamming_distance_list_sub, axis = 0)
            hamming_distance_list.append(hamming_distance_sub)
        assignment_result = image_tab_encoding.groupby('TARGET_TAXON').agg({'CorrectAssignment': 'sum', 'IncorrectAssignment': 'sum'}).reset_index()
        assignment_result = assignment_result.rename(columns = {'TARGET_TAXON': 'target_taxon'})
        assignment_result['target_taxon'] = assignment_result.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.merge(assignment_result, on = 'target_taxon', how = 'left')
        multispecies_summary['target_taxon'] = multispecies_summary.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.sort_values(['target_taxon'], ascending = [True])
        multispecies_summary.loc[:,'ErrorRate'] = multispecies_summary.IncorrectAssignment.values/(multispecies_summary.CorrectAssignment.values + multispecies_summary.IncorrectAssignment.values)
        multispecies_summary.loc[:,'UpperLimit'] = (multispecies_summary.IncorrectAssignment.values == 0)*1
        for i in range(multispecies_summary.shape[0]):
            if multispecies_summary.loc[i, 'UpperLimit'] == 1:
                multispecies_summary.loc[i, 'ErrorRate'] = 1/multispecies_summary.loc[i,'CorrectAssignment']
        ax = plt.subplot(gs[0,0])
        ax.plot(np.arange(11)[multispecies_summary.UpperLimit.values == 0], multispecies_summary.loc[multispecies_summary.UpperLimit.values == 0, 'ErrorRate'], 'o', alpha = 0.8, markersize = 4, color = plotting_info.loc[k, 'COLOR'], marker = 'o', markeredgewidth = 0)
        ax.plot(np.arange(11)[multispecies_summary.UpperLimit.values == 1], multispecies_summary.loc[multispecies_summary.UpperLimit.values == 1, 'ErrorRate'], 'o', alpha = 0.8, markersize = 4, color = plotting_info.loc[k, 'COLOR'], marker = 'v', markeredgewidth = 0)
        ax = plt.subplot(gs[1,0])
        parts = ax.violinplot(hamming_distance_list, np.arange(1 + (k-1)*0.1, 12 + (k-1)*0.1, 1), showmedians = False, showmeans = True, showextrema = False, bw_method = 0.2, widths = 0.5, points = 100)
        for pc in parts['bodies']:
            pc.set_facecolor(plotting_info.loc[k, 'COLOR'])
            pc.set_edgecolor(plotting_info.loc[k, 'COLOR'])
            pc.set_linewidth(0.5)
            pc.set_alpha(0.8)
        parts['cmeans'].set_color(plotting_info.loc[k, 'COLOR'])
        parts['cmeans'].set_linewidth(0.5)
    ax = plt.subplot(gs[0,0])
    plt.yscale('log')
    plt.ylabel('Error Rate', fontsize = 8, color = theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', colors = theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    ax.spines['left'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    plt.xticks([])
    plt.ylim(0.5e-6,1)
    ax = plt.subplot(gs[1,0])
    patches = [mpatches.Patch(alpha = 0), mpatches.Patch(alpha = 0), mpatches.Patch(alpha = 0)]
    l = ax.legend(patches, plotting_info.ENCODING.values, loc = 2, fontsize = 8, framealpha = 0, bbox_to_anchor = (-0.065,-0.065,1.1,1.1), bbox_transform = ax.transAxes)
    for k in range(plotting_info.shape[0]):
        l.get_texts()[k].set_color(plotting_info.loc[k, 'COLOR'])
    xlabel_list = []
    for t in taxid_list:
        xlabel_list.append(multispecies_summary.loc[multispecies_summary.target_taxon.values == str(t), 'sci_name'].values[0])
    plt.xticks(np.arange(1,12), xlabel_list, rotation = 30, horizontalalignment = 'right', rotation_mode = 'anchor', fontsize = 8, color = theme_color, style = 'italic')
    plt.tick_params(labelsize = 8, direction = 'in', colors = theme_color)
    plt.ylabel('Hamming distance', fontsize = 8, labelpad = 15, color = theme_color)
    plt.ylim(-0.4,6.4)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    ax.spines['left'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.2)
    plt.savefig('{}/multispecies_error_rate.pdf'.format(input_folder), dpi = 300, transparent = True)
    plt.close()
    return

def summarize_error_rate_panel(image_tab, input_folder, probe_design_filename, theme_color = 'black', nbit = 7):
    plotting_info = image_tab.loc[:,['PRIMERSET', 'ENCODING', 'COLOR']].drop_duplicates().reset_index().drop(columns = ['index'])
    for i in range(nbit):
        image_tab['HD{}'.format(i)] = 0
    image_tab['ErrorRate'] = np.nan
    image_tab['UpperLimit'] = 0
    taxid_list = [104102, 108981, 1353, 140100, 1580, 1590, 1718, 285, 438, 56459, 564]
    taxid_sciname = pd.DataFrame.from_dict({564: 'E. coli',
                1718: 'C. glutamicum',
                1590: 'L. plantarum',
                140100: 'V. albensis',
                1580: 'L. brevis',
                438: 'A. plantarum',
                104102: 'A. tropicalis',
                108981: 'A. schindleri',
                285: 'C. testosteroni',
                1353: 'E. gallinarum',
                56459: 'X. vasicola'}, orient = 'index').reset_index()
    multispecies_summary_full_list = []
    hamming_distance_full_list = []
    for k in range(plotting_info.shape[0]):
        encoding = plotting_info.loc[k, 'ENCODING']
        image_tab_encoding = image_tab.loc[image_tab.ENCODING.values == encoding, :].reset_index().drop(columns = ['index'])
        enc_set = plotting_info.loc[k, 'PRIMERSET']
        # filenames = ['{}/{}/{}_cell_information.csv'.format(input_folder, image_tab_encoding.loc[i,'SAMPLE'], image_tab_encoding.loc[i,'IMAGES']) for i in range(image_tab_encoding.shape[0])]
        probes = pd.read_csv(probe_design_filename[k], dtype = {'code': str})
        multispecies_summary = probes.loc[:,['target_taxon', 'code']].drop_duplicates().reset_index().drop(columns = ['index'])
        multispecies_summary['ErrorRate'] = 0
        multispecies_summary['UpperLimit'] = 0
        taxid_sciname.columns = ['target_taxon', 'sci_name']
        multispecies_summary = multispecies_summary.merge(taxid_sciname, on = 'target_taxon', how = 'left')
        # taxid_list = taxid_sciname.target_taxon.astype(str).sort_values().reset_index().drop(columns = ['index'])
        multispecies_summary.loc[:,'target_taxon'] = multispecies_summary.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.sort_values(by = ['target_taxon'], ascending = True)
        hamming_distance_list = []
        for t in multispecies_summary.target_taxon.values:
            hamming_distance_list_sub = []
            image_tab_sub = image_tab_encoding.loc[image_tab_encoding.TARGET_TAXON.values == str(t), :].reset_index().drop(columns = ['index'])
            for i in range(image_tab_sub.shape[0]):
                target_taxon = image_tab_sub.loc[i, 'TARGET_TAXON']
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, sample, image_name), header = None, dtype = {68: str})
                # cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, sample, image_name), header = None, dtype = {130: str})
                cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
                cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
                cell_info['Sample'] = sample
                barcode_consensus = []
                nn_dist = np.zeros((cell_info.shape[0], 20))
                for r in range(20):
                    cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_{}.csv'.format(input_folder, sample, image_name, r), header = None, dtype = {67:str})
                    barcode_consensus.append(cell_info.iloc[:,67].values)
                    # cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_{}.csv'.format(input_folder, sample, image_name, r), header = None, dtype = {130:str})
                    # barcode_consensus.append(cell_info.iloc[:,130].values)
                    nn_dist[:,r] = cell_info.iloc[:,68].values
                    # nn_dist[:,r] = cell_info.iloc[:,131].values
                    cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
                    cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
                    cell_info['Sample'] = sample
                barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
                nn_barcode = []
                nn_indices = np.argmin(nn_dist, axis = 1)
                for b in range(barcode_consensus.shape[0]):
                    nn_barcode.append(barcode_consensus[b,nn_indices[b]])
                barcode_assignment = multispecies_summary.loc[multispecies_summary.target_taxon.values == str(target_taxon), 'code'].values[0]
                max_intensity_mean = cell_info.max_intensity.mean()
                max_intensity_std = cell_info.max_intensity.std()
                cell_info_filtered = cell_info.loc[cell_info.max_intensity.values > max_intensity_mean - max_intensity_std, :].reset_index().drop(columns = 'index')
                nn_barcode_filtered = np.array(nn_barcode)[cell_info.max_intensity.values > max_intensity_mean - max_intensity_std]
                cell_info_filtered['correct_assignment'] = nn_barcode_filtered == barcode_assignment
                correct_assignment = np.sum(cell_info_filtered.correct_assignment.values)
                incorrect_assignment = cell_info_filtered.shape[0] - correct_assignment
                cell_info_filtered['barcode'] = np.array(nn_barcode_filtered)
                cell_info_filtered['hamming_distance'] = pd.DataFrame(nn_barcode_filtered)[0].apply(hamming2, args = (barcode_assignment,))
                cell_info_filtered.to_csv('{}/{}/{}_cell_information.csv'.format(input_folder, sample, image_name), index = None)
                hamming_distance_list_sub.append(cell_info_filtered.hamming_distance.values)
                image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'CorrectAssignment'] = correct_assignment
                image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'IncorrectAssignment'] = incorrect_assignment
                for h in range(nbit):
                    image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'HD{}'.format(h)] = np.sum(cell_info_filtered.hamming_distance.values == h)
            hamming_distance_sub = np.concatenate(hamming_distance_list_sub, axis = 0)
            hamming_distance_list.append(hamming_distance_sub)
        assignment_result = image_tab_encoding.groupby('TARGET_TAXON').agg({'CorrectAssignment': 'sum', 'IncorrectAssignment': 'sum'}).reset_index()
        assignment_result = assignment_result.rename(columns = {'TARGET_TAXON': 'target_taxon'})
        assignment_result['target_taxon'] = assignment_result.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.merge(assignment_result, on = 'target_taxon', how = 'left')
        multispecies_summary['target_taxon'] = multispecies_summary.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.sort_values(['target_taxon'], ascending = [True])
        multispecies_summary.loc[:,'ErrorRate'] = multispecies_summary.IncorrectAssignment.values/(multispecies_summary.CorrectAssignment.values + multispecies_summary.IncorrectAssignment.values)
        multispecies_summary.loc[:,'TotalCells'] = multispecies_summary.CorrectAssignment.values + multispecies_summary.IncorrectAssignment.values
        multispecies_summary.loc[:,'UpperLimit'] = (multispecies_summary.IncorrectAssignment.values == 0)*1
        for i in range(multispecies_summary.shape[0]):
            if multispecies_summary.loc[i, 'UpperLimit'] == 1:
                multispecies_summary.loc[i, 'ErrorRate'] = 1/multispecies_summary.loc[i,'CorrectAssignment']
        multispecies_summary_full_list.append(multispecies_summary)
        hamming_distance_full_list.append(hamming_distance_list)
    median_error_rate = np.median(np.concatenate([multispecies_summary_full_list[k].ErrorRate.values for k in range(3)]))
    print(median_error_rate)
    median_total_cells = np.median(np.concatenate([multispecies_summary_full_list[k].TotalCells.values for k in range(3)]))
    print(median_total_cells)
    upper_limit_list = np.concatenate([multispecies_summary_full_list[k].UpperLimit.values for k in range(3)])
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5.75), cm_to_inches(4.5))
    gs = GridSpec(2, 1)
    color_list = ['dodgerblue', 'olivedrab', 'maroon']
    ax = plt.subplot(gs[0,0])
    for i in range(3):
        for j in range(11):
            barcode = list(multispecies_summary_full_list[i].loc[j,'code'])
            for h in range(7):
                if barcode[h] == '0':
                    plt.plot(j*3 + i - 0.2*(i-1), h, 's', markersize = 2, markeredgewidth = 0, color = color_list[i], fillstyle = 'full', alpha = 0.3)
                else:
                    plt.plot(j*3 + i - 0.2*(i-1), h, 's', markersize = 2, markeredgewidth = 0, color = color_list[i], fillstyle = 'full', alpha = 1)

    plt.xlim(-0.5, 32.5)
    plt.tick_params(length = 0)
    plt.axis('off')
    plt.text(-3, 2.5, 'Bits', rotation = 90, fontsize = 5)
    plt.plot(1, 7.375, 's', markersize = 3, color = color_list[0], fillstyle = 'full')
    plt.text(2, 7, 'Least Complex', color = color_list[0], fontsize = 5)
    plt.plot(13.5, 7.375, 's', markersize = 3, color = color_list[1], fillstyle = 'full')
    plt.text(14.5, 7, 'Random', color = color_list[1], fontsize = 5)
    plt.plot(22, 7.375, 's', markersize = 3, color = color_list[2], fillstyle = 'full')
    plt.text(23, 7, 'Most Complex', color = color_list[2], fontsize = 5)
    ax = plt.subplot(gs[1,0])
    hamming_distance_heatmap = np.zeros((33,7))
    for i in range(3):
        for j in range(11):
            hd = pd.DataFrame(hamming_distance_full_list[i][j]).iloc[:,0].value_counts()
            hd_relative = hd/np.sum(hd)
            for h in range(hd_relative.shape[0]):
                hamming_distance_heatmap[3*j + i, h] = hd_relative.iloc[h]

    hm = plt.imshow(hamming_distance_heatmap.transpose(), cmap = 'coolwarm', origin = 'lower')
    plt.ylim(-0.5,6.5)
    for i in range(10):
        plt.vlines(0.5 + i*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 0.5)
        plt.vlines(1.5 + i*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 0.5)
        plt.vlines(2.5 + i*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 1)

    plt.vlines(0.5 + 10*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 0.5)
    plt.vlines(1.5 + 10*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 0.5)

    xlabel_list = []
    for t in taxid_list:
        xlabel_list.append(multispecies_summary.loc[multispecies_summary.target_taxon.values == str(t), 'sci_name'].values[0])

    plt.xticks(np.arange(1,32,3), xlabel_list, rotation = 30, horizontalalignment = 'right', rotation_mode = 'anchor', fontsize = 6, color = theme_color, style = 'italic')
    plt.tick_params(direction = 'in', length = 2, labelsize = 6, pad = 2)
    plt.ylabel('Hamming\ndistance', fontsize = 6, color = theme_color, labelpad = 0)
    cbaxes = fig.add_axes([0.81, 0.08, 0.15, 0.03])
    cb = plt.colorbar(hm, cax = cbaxes, orientation = 'horizontal')
    cb.set_ticks([])
    plt.text(-0.2,-0.3,'0', fontsize = 5)
    plt.text(1.03,-0.3,'1', fontsize = 5)
    cb.set_label('Frequency', fontsize = 5)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.25, hspace = 0)
    plt.savefig('{}/multispecies_hamming_heatmap.pdf'.format(input_folder), dpi = 300, transparent = True)
    plt.close()

    # Hamming distance decay linear scale
    color_list = ['darkviolet', 'dodgerblue', 'darkorange']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(15), cm_to_inches(18))
    gs = GridSpec(11, 3)
    for i in range(3):
        for j in range(11):
            ax = plt.subplot(gs[j,i])
            ax.plot(hamming_distance_heatmap[j*3 + i, :], '-o', markersize = 4, color = color_list[i])
            ax.tick_params(direction = 'in', length = 2, labelsize = 8)
            plt.ylim(-0.2, 1.2)
            if (i == 0) & (j == 10):
                plt.xlabel(r'$d_{Hamming}$', fontsize = 8, color = theme_color)
                plt.ylabel('Abundance', fontsize = 8, color = theme_color)
            else:
                plt.xticks([])
                plt.yticks([])

    ax = plt.subplot(gs[0,0])
    plt.title('Simplest' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,1])
    plt.title('Random' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,2])
    plt.title('Most Complex' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    for j in range(11):
        ax = plt.subplot(gs[j,2])
        plt.ylabel(r'${}$'.format(xlabel_list[j]), fontsize = 8, color = theme_color, rotation = 0, labelpad = 32)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(length = 0)

    plt.subplots_adjust(left = 0.06, bottom = 0.06, right = 0.85, top = 0.95, wspace = 0.1)
    plt.savefig('{}/multispecies_hamming_distance_breakout_plot.pdf'.format(input_folder), dpi = 300, transparent = True)

    # Hamming distance decay linear scale presentation
    theme_color = 'white'
    font_size = 10
    color_list = ['darkviolet', 'dodgerblue', 'darkorange']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(16), cm_to_inches(18))
    gs = GridSpec(11, 3)
    for i in range(3):
        for j in range(11):
            ax = plt.subplot(gs[j,i])
            ax.plot(hamming_distance_heatmap[j*3 + i, :], '-o', markersize = 4, color = color_list[i])
            ax.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
            plt.xticks([0,2,4,6])
            plt.ylim(-0.2, 1.2)
            ax.spines['left'].set_color(theme_color)
            ax.spines['bottom'].set_color(theme_color)
            ax.spines['right'].set_color(theme_color)
            ax.spines['top'].set_color(theme_color)
            if (i == 0) & (j == 10):
                plt.xlabel(r'$Hamming distance$', fontsize = font_size, color = theme_color)
                plt.ylabel('Abundance', fontsize = font_size, color = theme_color)
            else:
                plt.xticks([])
                plt.yticks([])

    ax = plt.subplot(gs[0,0])
    plt.title('Simplest' "\n" 'Barcodes', fontsize = font_size, color = theme_color)
    ax = plt.subplot(gs[0,1])
    plt.title('Random' "\n" 'Barcodes', fontsize = font_size, color = theme_color)
    ax = plt.subplot(gs[0,2])
    plt.title('Most Complex' "\n" 'Barcodes', fontsize = font_size, color = theme_color)
    for j in range(11):
        ax = plt.subplot(gs[j,2])
        plt.ylabel(r'${}$'.format(xlabel_list[j]), fontsize = font_size, color = theme_color, rotation = 0, labelpad = 48)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(length = 0)

    plt.subplots_adjust(left = 0.06, bottom = 0.06, right = 0.8, top = 0.95, wspace = 0.1)
    plt.savefig('{}/multispecies_hamming_distance_breakout_plot.svg'.format(input_folder), dpi = 300, transparent = True)
    plt.close()


    # Hamming distance decay log scale
    color_list = ['darkviolet', 'dodgerblue', 'darkorange']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(15), cm_to_inches(18))
    gs = GridSpec(11, 3)
    for i in range(3):
        for j in range(11):
            ax = plt.subplot(gs[j,i])
            hd_sub = hamming_distance_heatmap[j*3 + i, :]
            hd_sub[hd_sub == 0] += 1e-7
            ax.plot(hd_sub, '-o', markersize = 4, color = color_list[i])
            ax.tick_params(direction = 'in', length = 2, labelsize = 8)
            plt.ylim(1e-9, 12)
            plt.yscale('log')
            if (i == 0) & (j == 10):
                plt.xlabel('Hamming Distance', fontsize = 8, color = theme_color)
                plt.ylabel('Abundance', fontsize = 8, color = theme_color)
                plt.yticks([1e-8, 1e-4, 1e0])
            else:
                plt.xticks([])
                plt.yticks([])

    ax = plt.subplot(gs[0,0])
    plt.title('Simplest' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,1])
    plt.title('Random' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,2])
    plt.title('Most Complex' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    for j in range(11):
        ax = plt.subplot(gs[j,2])
        plt.ylabel(r'${}$'.format(xlabel_list[j]), fontsize = 8, color = theme_color, rotation = 0, labelpad = 32)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(length = 0)

    plt.subplots_adjust(left = 0.1, bottom = 0.06, right = 0.85, top = 0.95, wspace = 0.1)
    plt.savefig('{}/multispecies_hamming_distance_breakout_plot_log.pdf'.format(input_folder), dpi = 300, transparent = True)



    return

def summarize_error_rate_panel_presentation(image_tab, input_folder, probe_design_filename, theme_color = 'white', nbit = 7):
    plotting_info = image_tab.loc[:,['PRIMERSET', 'ENCODING', 'COLOR']].drop_duplicates().reset_index().drop(columns = ['index'])
    for i in range(nbit):
        image_tab['HD{}'.format(i)] = 0
    image_tab['ErrorRate'] = np.nan
    image_tab['UpperLimit'] = 0
    taxid_list = [104102, 108981, 1353, 140100, 1580, 1590, 1718, 285, 438, 56459, 564]
    taxid_sciname = pd.DataFrame.from_dict({564: 'E. coli',
                1718: 'C. glutamicum',
                1590: 'L. plantarum',
                140100: 'V. albensis',
                1580: 'L. brevis',
                438: 'A. plantarum',
                104102: 'A. tropicalis',
                108981: 'A. schindleri',
                285: 'C. testosteroni',
                1353: 'E. gallinarum',
                56459: 'X. vasicola'}, orient = 'index').reset_index()
    multispecies_summary_full_list = []
    hamming_distance_full_list = []
    for k in range(plotting_info.shape[0]):
        encoding = plotting_info.loc[k, 'ENCODING']
        image_tab_encoding = image_tab.loc[image_tab.ENCODING.values == encoding, :].reset_index().drop(columns = ['index'])
        enc_set = plotting_info.loc[k, 'PRIMERSET']
        # filenames = ['{}/{}/{}_cell_information.csv'.format(input_folder, image_tab_encoding.loc[i,'SAMPLE'], image_tab_encoding.loc[i,'IMAGES']) for i in range(image_tab_encoding.shape[0])]
        probes = pd.read_csv(probe_design_filename[k], dtype = {'code': str})
        multispecies_summary = probes.loc[:,['target_taxon', 'code']].drop_duplicates().reset_index().drop(columns = ['index'])
        multispecies_summary['ErrorRate'] = 0
        multispecies_summary['UpperLimit'] = 0
        taxid_sciname.columns = ['target_taxon', 'sci_name']
        multispecies_summary = multispecies_summary.merge(taxid_sciname, on = 'target_taxon', how = 'left')
        # taxid_list = taxid_sciname.target_taxon.astype(str).sort_values().reset_index().drop(columns = ['index'])
        multispecies_summary.loc[:,'target_taxon'] = multispecies_summary.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.sort_values(by = ['target_taxon'], ascending = True)
        hamming_distance_list = []
        for t in multispecies_summary.target_taxon.values:
            hamming_distance_list_sub = []
            image_tab_sub = image_tab_encoding.loc[image_tab_encoding.TARGET_TAXON.values == str(t), :].reset_index().drop(columns = ['index'])
            for i in range(image_tab_sub.shape[0]):
                target_taxon = image_tab_sub.loc[i, 'TARGET_TAXON']
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, sample, image_name), header = None, dtype = {68: str})
                # cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, sample, image_name), header = None, dtype = {130: str})
                cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
                cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
                cell_info['Sample'] = sample
                barcode_consensus = []
                nn_dist = np.zeros((cell_info.shape[0], 20))
                for r in range(20):
                    cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_{}.csv'.format(input_folder, sample, image_name, r), header = None, dtype = {67:str})
                    barcode_consensus.append(cell_info.iloc[:,67].values)
                    # cell_info = pd.read_csv('{}/{}/{}_cell_information_replicate_{}.csv'.format(input_folder, sample, image_name, r), header = None, dtype = {130:str})
                    # barcode_consensus.append(cell_info.iloc[:,130].values)
                    nn_dist[:,r] = cell_info.iloc[:,68].values
                    # nn_dist[:,r] = cell_info.iloc[:,131].values
                    cell_info['intensity'] = cell_info.iloc[:,0:63].values.sum(axis = 1)
                    cell_info['max_intensity'] = cell_info.iloc[:,0:63].values.max(axis = 1)
                    cell_info['Sample'] = sample
                barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
                nn_barcode = []
                nn_indices = np.argmin(nn_dist, axis = 1)
                for b in range(barcode_consensus.shape[0]):
                    nn_barcode.append(barcode_consensus[b,nn_indices[b]])
                barcode_assignment = multispecies_summary.loc[multispecies_summary.target_taxon.values == str(target_taxon), 'code'].values[0]
                max_intensity_mean = cell_info.max_intensity.mean()
                max_intensity_std = cell_info.max_intensity.std()
                cell_info_filtered = cell_info.loc[cell_info.max_intensity.values > max_intensity_mean - max_intensity_std, :].reset_index().drop(columns = 'index')
                nn_barcode_filtered = np.array(nn_barcode)[cell_info.max_intensity.values > max_intensity_mean - max_intensity_std]
                cell_info_filtered['correct_assignment'] = nn_barcode_filtered == barcode_assignment
                correct_assignment = np.sum(cell_info_filtered.correct_assignment.values)
                incorrect_assignment = cell_info_filtered.shape[0] - correct_assignment
                cell_info_filtered['barcode'] = np.array(nn_barcode_filtered)
                cell_info_filtered['hamming_distance'] = pd.DataFrame(nn_barcode_filtered)[0].apply(hamming2, args = (barcode_assignment,))
                cell_info_filtered.to_csv('{}/{}/{}_cell_information.csv'.format(input_folder, sample, image_name), index = None)
                hamming_distance_list_sub.append(cell_info_filtered.hamming_distance.values)
                image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'CorrectAssignment'] = correct_assignment
                image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'IncorrectAssignment'] = incorrect_assignment
                for h in range(nbit):
                    image_tab_encoding.loc[image_tab_encoding.IMAGES.values == image_name, 'HD{}'.format(h)] = np.sum(cell_info_filtered.hamming_distance.values == h)
            hamming_distance_sub = np.concatenate(hamming_distance_list_sub, axis = 0)
            hamming_distance_list.append(hamming_distance_sub)
        assignment_result = image_tab_encoding.groupby('TARGET_TAXON').agg({'CorrectAssignment': 'sum', 'IncorrectAssignment': 'sum'}).reset_index()
        assignment_result = assignment_result.rename(columns = {'TARGET_TAXON': 'target_taxon'})
        assignment_result['target_taxon'] = assignment_result.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.merge(assignment_result, on = 'target_taxon', how = 'left')
        multispecies_summary['target_taxon'] = multispecies_summary.target_taxon.astype(str)
        multispecies_summary = multispecies_summary.sort_values(['target_taxon'], ascending = [True])
        multispecies_summary.loc[:,'ErrorRate'] = multispecies_summary.IncorrectAssignment.values/(multispecies_summary.CorrectAssignment.values + multispecies_summary.IncorrectAssignment.values)
        multispecies_summary.loc[:,'TotalCells'] = multispecies_summary.CorrectAssignment.values + multispecies_summary.IncorrectAssignment.values
        multispecies_summary.loc[:,'UpperLimit'] = (multispecies_summary.IncorrectAssignment.values == 0)*1
        for i in range(multispecies_summary.shape[0]):
            if multispecies_summary.loc[i, 'UpperLimit'] == 1:
                multispecies_summary.loc[i, 'ErrorRate'] = 1/multispecies_summary.loc[i,'CorrectAssignment']
        multispecies_summary_full_list.append(multispecies_summary)
        hamming_distance_full_list.append(hamming_distance_list)
    median_error_rate = np.median(np.concatenate([multispecies_summary_full_list[k].ErrorRate.values for k in range(3)]))
    print(median_error_rate)
    median_total_cells = np.median(np.concatenate([multispecies_summary_full_list[k].TotalCells.values for k in range(3)]))
    print(median_total_cells)
    upper_limit_list = np.concatenate([multispecies_summary_full_list[k].UpperLimit.values for k in range(3)])
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(17.5), cm_to_inches(12))
    gs = GridSpec(2, 1)
    color_list = ['dodgerblue', 'olivedrab', 'maroon']
    ax = plt.subplot(gs[0,0])
    for i in range(3):
        for j in range(11):
            barcode = list(multispecies_summary_full_list[i].loc[j,'code'])
            for h in range(7):
                if barcode[h] == '0':
                    plt.plot(j*3 + i - 0.2*(i-1), h, 's', markersize = 6, markeredgewidth = 0, color = color_list[i], fillstyle = 'full', alpha = 0.3)
                else:
                    plt.plot(j*3 + i - 0.2*(i-1), h, 's', markersize = 6, markeredgewidth = 0, color = color_list[i], fillstyle = 'full', alpha = 1)

    plt.xlim(-0.5, 32.5)
    plt.tick_params(length = 0)
    plt.axis('off')
    plt.text(-3, 2.5, 'Bits', rotation = 90, fontsize = font_size, color = theme_color)
    plt.plot(1, 7.375, 's', markersize = 4, color = color_list[0], fillstyle = 'full')
    plt.text(2, 7, 'Least Complex', color = color_list[0], fontsize = font_size)
    plt.plot(13.5, 7.375, 's', markersize = 4, color = color_list[1], fillstyle = 'full')
    plt.text(14.5, 7, 'Random', color = color_list[1], fontsize = font_size)
    plt.plot(22, 7.375, 's', markersize = 4, color = color_list[2], fillstyle = 'full')
    plt.text(23, 7, 'Most Complex', color = color_list[2], fontsize = font_size)
    ax = plt.subplot(gs[1,0])
    hamming_distance_heatmap = np.zeros((33,7))
    for i in range(3):
        for j in range(11):
            hd = pd.DataFrame(hamming_distance_full_list[i][j]).iloc[:,0].value_counts()
            hd_relative = hd/np.sum(hd)
            for h in range(hd_relative.shape[0]):
                hamming_distance_heatmap[3*j + i, h] = hd_relative.iloc[h]

    hm = plt.imshow(hamming_distance_heatmap.transpose(), cmap = 'coolwarm', origin = 'lower')
    plt.ylim(-0.5,6.5)
    for i in range(11):
        plt.vlines(0.5 + i*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 0.5)
        plt.vlines(1.5 + i*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 0.5)
        plt.vlines(2.5 + i*3, -0.5, 6.5, colors = (1,1,1), linestyle = '--', linewidth = 1)

    xlabel_list = []
    for t in taxid_list:
        xlabel_list.append(multispecies_summary.loc[multispecies_summary.target_taxon.values == str(t), 'sci_name'].values[0])

    plt.xticks(np.arange(1,32,3), xlabel_list, rotation = 30, horizontalalignment = 'right', rotation_mode = 'anchor', fontsize = font_size, color = theme_color, style = 'italic')
    plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
    plt.ylabel('Hamming\ndistance', fontsize = font_size, color = theme_color)
    cbaxes = fig.add_axes([0.81, 0.08, 0.15, 0.03])
    cb = plt.colorbar(hm, cax = cbaxes, orientation = 'horizontal')
    cb.set_ticks([])
    plt.text(-0.2,-0.2,'0', color = theme_color, fontsize = font_size)
    plt.text(1.05,-0.2,'1', color = theme_color, fontsize = font_size)
    cb.set_label('Frequency', fontsize = font_size, color = theme_color)
    cbaxes.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.25, hspace = 0)
    plt.savefig('{}/multispecies_hamming_heatmap.svg'.format(input_folder), dpi = 300, transparent = True)
    plt.close()

    # Hamming distance decay linear scale
    color_list = ['darkviolet', 'dodgerblue', 'darkorange']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(15), cm_to_inches(18))
    gs = GridSpec(11, 3)
    for i in range(3):
        for j in range(11):
            ax = plt.subplot(gs[j,i])
            ax.plot(hamming_distance_heatmap[j*3 + i, :], '-o', markersize = 4, color = color_list[i])
            ax.tick_params(direction = 'in', length = 2, labelsize = 8)
            plt.ylim(-0.2, 1.2)
            if (i == 0) & (j == 10):
                plt.xlabel(r'$d_{Hamming}$', fontsize = 8, color = theme_color)
                plt.ylabel('Abundance', fontsize = 8, color = theme_color)
            else:
                plt.xticks([])
                plt.yticks([])

    ax = plt.subplot(gs[0,0])
    plt.title('Simplest' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,1])
    plt.title('Random' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,2])
    plt.title('Most Complex' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    for j in range(11):
        ax = plt.subplot(gs[j,2])
        plt.ylabel(r'${}$'.format(xlabel_list[j]), fontsize = 8, color = theme_color, rotation = 0, labelpad = 32)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(length = 0)

    plt.subplots_adjust(left = 0.06, bottom = 0.06, right = 0.85, top = 0.95, wspace = 0.1)
    plt.savefig('{}/multispecies_hamming_distance_breakout_plot.pdf'.format(input_folder), dpi = 300, transparent = True)

    # Hamming distance decay log scale
    color_list = ['darkviolet', 'dodgerblue', 'darkorange']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(15), cm_to_inches(18))
    gs = GridSpec(11, 3)
    for i in range(3):
        for j in range(11):
            ax = plt.subplot(gs[j,i])
            hd_sub = hamming_distance_heatmap[j*3 + i, :]
            hd_sub[hd_sub == 0] += 1e-7
            ax.plot(hd_sub, '-o', markersize = 4, color = color_list[i])
            ax.tick_params(direction = 'in', length = 2, labelsize = 8)
            plt.ylim(1e-9, 12)
            plt.yscale('log')
            if (i == 0) & (j == 10):
                plt.xlabel('Hamming Distance', fontsize = 8, color = theme_color)
                plt.ylabel('Abundance', fontsize = 8, color = theme_color)
                plt.yticks([1e-8, 1e-4, 1e0])
            else:
                plt.xticks([])
                plt.yticks([])

    ax = plt.subplot(gs[0,0])
    plt.title('Simplest' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,1])
    plt.title('Random' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0,2])
    plt.title('Most Complex' "\n" 'Barcodes', fontsize = 8, color = theme_color)
    for j in range(11):
        ax = plt.subplot(gs[j,2])
        plt.ylabel(r'${}$'.format(xlabel_list[j]), fontsize = 8, color = theme_color, rotation = 0, labelpad = 32)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(length = 0)

    plt.subplots_adjust(left = 0.1, bottom = 0.06, right = 0.85, top = 0.95, wspace = 0.1)
    plt.savefig('{}/multispecies_hamming_distance_breakout_plot_log.pdf'.format(input_folder), dpi = 300, transparent = True)



    return

def plot_representative_cell_image(image_tab, input_folder, probe_design_filename, theme_color):
    plotting_info = image_tab.loc[:,['PRIMERSET', 'ENCODING', 'COLOR']].drop_duplicates().reset_index().drop(columns = ['index'])
    taxid_list = [104102, 108981, 1353, 140100, 1580, 1590, 1718, 285, 438, 564, 56459]
    taxid_sciname = pd.DataFrame.from_dict({564: 'E. coli',
                1718: 'C. glutamicum',
                1590: 'L. plantarum',
                140100: 'V. albensis',
                1580: 'L. brevis',
                438: 'A. plantarum',
                104102: 'A. tropicalis',
                108981: 'A. schindleri',
                285: 'C. testosteroni',
                1353: 'E. gallinarum',
                56459: 'X. vasicola'}, orient = 'index').reset_index()
    taxid_sciname.columns = ['target_taxon', 'sci_name']
    taxid_sciname.loc[:, 'target_taxon'] = taxid_sciname.target_taxon.astype(str)
    taxid_sciname = taxid_sciname.sort_values(by = 'target_taxon').reset_index().drop(columns = ['index'])
    # taxid_list = taxid_sciname.target_taxon.astype(str).sort_values().reset_index().drop(columns = ['index'])
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4 + 2.5*(plotting_info.shape[0] - 1)), cm_to_inches(7))
    gs = GridSpec(11, 2*plotting_info.shape[0])
    for k in range(plotting_info.shape[0]):
        encoding = plotting_info.loc[k, 'ENCODING']
        image_tab_encoding = image_tab.loc[image_tab.ENCODING.values == encoding, :].reset_index().drop(columns = ['index'])
        enc_set = plotting_info.loc[k, 'PRIMERSET']
        filenames = ['{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, image_tab_encoding.loc[i,'SAMPLE'], image_tab_encoding.loc[i,'IMAGES']) for i in range(image_tab_encoding.shape[0])]
        # taxon_list = image_tab.TARGET_TAXON.drop_duplicates().sort_values().values
        # ordered_taxid_list = [108981, 140100, 56459, 104102, 1580, 1590, 1353, 438, 1718, 285, 564]
        for i in range(len(taxid_list)):
            t = taxid_list[i]
            image_tab_sub = image_tab_encoding.loc[image_tab_encoding.TARGET_TAXON.values == str(t), :].reset_index().drop(columns = ['index'])
            image_name_list = image_tab_sub.IMAGES.values
            cell_information = [pd.read_csv('{}/{}/{}_cell_information_replicate_0.csv'.format(input_folder, image_tab_sub.loc[s, 'SAMPLE'], image_tab_sub.loc[s, 'IMAGES']), header = None, dtype = {67: str}) for s in range(image_tab_sub.shape[0])]
            cell_information = pd.concat(cell_information)
            avgint_norm = cell_information.loc[:,0:62].values/np.max(cell_information.loc[:,0:62].values, axis = 1)[:,None]
            spec_average = np.average(avgint_norm, axis = 0)
            spec_std = np.std(avgint_norm, axis = 0)
            ax = plt.subplot(gs[i, 2*k:2*k+2])
            ax.errorbar(np.arange(0,23), spec_average[0:23], yerr = spec_std[0:23], color = 'limegreen', fmt = '-o', markersize = 0.1, ecolor = theme_color, capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.errorbar(np.arange(23,43), spec_average[23:43], yerr = spec_std[23:43], color = 'yellowgreen', fmt = '-o', markersize = 0.1, ecolor = theme_color, capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.errorbar(np.arange(43,57), spec_average[43:57], yerr = spec_std[43:57], color = 'darkorange', fmt = '-o', markersize = 0.1, ecolor = theme_color, capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.errorbar(np.arange(57,63), spec_average[57:63], yerr = spec_std[57:63], color = 'red', fmt = '-o', markersize = 0.1, ecolor = theme_color, capsize = 0.4, linewidth = 2, elinewidth = 0.2, capthick = 0.2, markeredgewidth = 0)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['left'].set_color(theme_color)
            ax.spines['bottom'].set_color(theme_color)
            ax.tick_params(colors = theme_color, direction = 'in', labelsize = 6)
            if k == 0:
                ax.axes.get_yaxis().set_visible(True)
                plt.yticks([])
                plt.ylabel(taxid_sciname.loc[taxid_sciname.target_taxon.values == str(t), 'sci_name'].values[0], rotation = 0, horizontalalignment = 'right', rotation_mode = 'anchor', fontsize = 6, fontstyle = 'italic', color = theme_color)
    ax = plt.subplot(gs[10,0:2])
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    plt.yticks([0,1])
    plt.xticks([0,20,40,60])
    plt.xlabel('Channel', fontsize = 6, color = theme_color)
    for k in range(plotting_info.shape[0]):
        ax = plt.subplot(gs[0, k*2:(k+1)*2])
        ax.axes.get_xaxis().set_visible(True)
        plt.tick_params(axis = 'x', bottom = False, colors = theme_color)
        plt.xticks([])
        ax.xaxis.set_label_position('top')
        plt.xlabel(plotting_info.loc[k, 'ENCODING'], fontsize = 6, color = theme_color)
    plt.subplots_adjust(left=0.42 - 0.11*(plotting_info.shape[0] - 1), right=0.98, top=0.95, bottom=0.1)
    plt.savefig('{}/multispecies_representative_cell_spectra.pdf'.format(input_folder), dpi = 300, transparent = True)
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser('Summarize multispecies synthetic community measurement results')
    parser.add_argument('input_image_list', type = str, help = 'Input folder containing image analysis results')
    parser.add_argument('input_folder', type = str, help = 'Input folder containing image analysis results')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, nargs = '*', help = 'Probe design filenames')
    parser.add_argument('-t', '--theme_color', dest = 'theme_color', type = str, help = 'Probe design filenames')
    args = parser.parse_args()
    image_tab = pd.read_csv(args.input_image_list, dtype = {'TARGET_TAXON': str})
    # summarize_error_rate(image_tab, args.input_folder, args.probe_design_filename, args.theme_color)
    summarize_error_rate_panel(image_tab, args.input_folder, args.probe_design_filename, args.theme_color)
    plot_representative_cell_image(image_tab, args.input_folder, args.probe_design_filename, args.theme_color)
    return

if __name__ == '__main__':
    main()
