
"""
Hao Shi 2020
De Vlaminck Lab
Cornell University
"""

import os
import re
import glob
import dask
import time
import argparse
import xmltodict
import bioformats
import javabridge
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib.gridspec import GridSpec
from dask.distributed import LocalCluster, Client

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(length):
    return(length/2.54)

@dask.delayed
def get_error_rate(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder):
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    dilution = sam_tab_bkg_intensity.loc[i,'DILUTION']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_dilution_{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, dilution, 0), header = None, dtype = {100:str})
    # error_rate = 1 - np.sum(cell_info.iloc[:,100].values == barcode)/cell_info.shape[0]
    barcode_consensus = []
    nn_dist = np.zeros((cell_info.shape[0], batch_size))
    for r in range(batch_size):
        cell_info_filename = '{}/{}/{}_avgint_dilution_{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, dilution, r)
        if os.path.exists(cell_info_filename):
            cell_info = pd.read_csv('{}/{}/{}_avgint_dilution_{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, dilution, r), header = None, dtype = {100:str})
            barcode_consensus.append(cell_info.iloc[:,100].values)
            nn_dist[:,r] = cell_info.iloc[:,101].values
        else:
            barcode_consensus.append(np.repeat('0000000000', cell_info.shape[0]))
            nn_dist[:,r] = 1
    barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
    nn_barcode = []
    nn_indices = np.argmin(nn_dist, axis = 1)
    for b in range(barcode_consensus.shape[0]):
        nn_barcode.append(barcode_consensus[b,nn_indices[b]])
    error_rate = 1 - np.sum(np.array(nn_barcode) == barcode)/cell_info.shape[0]
    cell_info_consensus = cell_info.copy()
    cell_info_consensus.iloc[:,100] = nn_barcode
    cell_info_consensus.to_csv('{}/{}/{}_dilution_{}_cell_information_consensus.csv'.format(data_folder, sample, image_name, dilution), index = None)
    if error_rate == 0:
        error_rate = 1/cell_info.shape[0]
        detection_limit = 1
    else:
        detection_limit = 0
    # max_consensus_barcode_list = []
    # for b in range(barcode_consensus.shape[0]):
    #     barcode_counts_barcode ,barcode_counts_counts = np.unique(barcode_consensus[b,:], return_counts = True)
    #     max_consensus_barcode = barcode_counts_barcode[np.argmax(barcode_counts_counts)]
    #     max_consensus_barcode_list.append(max_consensus_barcode)
    # max_consensus_barcode_list = []
    # for b in range(barcode_consensus.shape[0]):
    #     barcode_counts_barcode ,barcode_counts_counts = np.unique(barcode_consensus[b,:], return_counts = True)
    #     max_consensus_barcode = barcode_counts_barcode[np.argmax(barcode_counts_counts)]
    #     max_consensus_barcode_list.append(max_consensus_barcode)
    # error_rate_consensus = 1 - np.sum(np.array(max_consensus_barcode_list) == barcode)/cell_info.shape[0]
    # cell_info_correct = cell_info.loc[cell_info[101].values == barcode,:]
    # cell_info_incorrect = cell_info.loc[cell_info[101].values != barcode,:]
    # nn_barcode_list.append(nn_barcode)
    # barcode_hd = pd.DataFrame(nn_barcode).astype(str).iloc[:,0].apply(hamming2, args = (barcode,))
    # spec_heatmap[i,:] = np.average(cell_info.iloc[:,0:95].values, axis = 0)
    # for k in range(10):
    #     hamming_distance[i,k] = np.sum(barcode_hd == k)/barcode_hd.shape[0]
    return(np.array([error_rate, detection_limit]))

@dask.delayed
def get_average_nn_dists(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder):
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    dilution = sam_tab_bkg_intensity.loc[i,'DILUTION']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_dilution_{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, dilution, 0), header = None, dtype = {100:str})
    # error_rate = 1 - np.sum(cell_info.iloc[:,100].values == barcode)/cell_info.shape[0]
    barcode_consensus = []
    nn_dist = np.zeros((cell_info.shape[0], batch_size))
    for r in range(batch_size):
        cell_info_filename = '{}/{}/{}_avgint_dilution_{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, dilution, r)
        if os.path.exists(cell_info_filename):
            cell_info = pd.read_csv('{}/{}/{}_avgint_dilution_{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, dilution, r), header = None, dtype = {100:str})
            barcode_consensus.append(cell_info.iloc[:,100].values)
            nn_dist[:,r] = cell_info.iloc[:,101].values
        else:
            barcode_consensus.append(np.repeat('0000000000', cell_info.shape[0]))
            nn_dist[:,r] = 1
    barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
    nn_barcode = []
    nn_indices = np.argmin(nn_dist, axis = 1)
    for b in range(barcode_consensus.shape[0]):
        nn_barcode.append(barcode_consensus[b,nn_indices[b]])
    error_rate = 1 - np.sum(np.array(nn_barcode) == barcode)/cell_info.shape[0]
    cell_info_consensus = cell_info.copy()
    cell_info_consensus.iloc[:,100] = nn_barcode
    cell_info_consensus.to_csv('{}/{}/{}_dilution_{}_cell_information_consensus.csv'.format(data_folder, sample, image_name, dilution), index = None)
    average_nn_dists = cell_info_consensus.iloc[:,101].mean()
    # max_consensus_barcode_list = []
    # for b in range(barcode_consensus.shape[0]):
    #     barcode_counts_barcode ,barcode_counts_counts = np.unique(barcode_consensus[b,:], return_counts = True)
    #     max_consensus_barcode = barcode_counts_barcode[np.argmax(barcode_counts_counts)]
    #     max_consensus_barcode_list.append(max_consensus_barcode)
    # max_consensus_barcode_list = []
    # for b in range(barcode_consensus.shape[0]):
    #     barcode_counts_barcode ,barcode_counts_counts = np.unique(barcode_consensus[b,:], return_counts = True)
    #     max_consensus_barcode = barcode_counts_barcode[np.argmax(barcode_counts_counts)]
    #     max_consensus_barcode_list.append(max_consensus_barcode)
    # error_rate_consensus = 1 - np.sum(np.array(max_consensus_barcode_list) == barcode)/cell_info.shape[0]
    # cell_info_correct = cell_info.loc[cell_info[101].values == barcode,:]
    # cell_info_incorrect = cell_info.loc[cell_info[101].values != barcode,:]
    # nn_barcode_list.append(nn_barcode)
    # barcode_hd = pd.DataFrame(nn_barcode).astype(str).iloc[:,0].apply(hamming2, args = (barcode,))
    # spec_heatmap[i,:] = np.average(cell_info.iloc[:,0:95].values, axis = 0)
    # for k in range(10):
    #     hamming_distance[i,k] = np.sum(barcode_hd == k)/barcode_hd.shape[0]
    return(np.array([average_nn_dists]))

@dask.delayed
def get_hamming_distance(sam_tab_bkg_intensity, i, batch_size, nbit):
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    # error_rate = 1 - np.sum(cell_info.iloc[:,100].values == barcode)/cell_info.shape[0]
    barcode_consensus = []
    nn_dist = np.zeros((cell_info.shape[0], batch_size))
    for r in range(batch_size):
        cell_info_filename = '{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, r)
        if os.path.exists(cell_info_filename):
            cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, r), header = None, dtype = {100:str})
            barcode_consensus.append(cell_info.iloc[:,100].values)
            nn_dist[:,r] = cell_info.iloc[:,101].values
        else:
            barcode_consensus.append(np.repeat('0000000000', cell_info.shape[0]))
            nn_dist[:,r] = 1
    barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
    nn_barcode = []
    nn_indices = np.argmin(nn_dist, axis = 1)
    for b in range(barcode_consensus.shape[0]):
        nn_barcode.append(barcode_consensus[b,nn_indices[b]])
    error_rate = 1 - np.sum(np.array(nn_barcode) == barcode)/cell_info.shape[0]
    cell_info_consensus = cell_info.copy()
    cell_info_consensus.iloc[:,100] = nn_barcode
    cell_info_consensus.to_csv('{}/{}/{}_cell_information_consensus.csv'.format(data_folder, sample, image_name), index = None)
    if error_rate == 0:
        error_rate = 1/cell_info.shape[0]
        detection_limit = 1
    else:
        detection_limit = 0
    # max_consensus_barcode_list = []
    # for b in range(barcode_consensus.shape[0]):
    #     barcode_counts_barcode ,barcode_counts_counts = np.unique(barcode_consensus[b,:], return_counts = True)
    #     max_consensus_barcode = barcode_counts_barcode[np.argmax(barcode_counts_counts)]
    #     max_consensus_barcode_list.append(max_consensus_barcode)
    # max_consensus_barcode_list = []
    # for b in range(barcode_consensus.shape[0]):
    #     barcode_counts_barcode ,barcode_counts_counts = np.unique(barcode_consensus[b,:], return_counts = True)
    #     max_consensus_barcode = barcode_counts_barcode[np.argmax(barcode_counts_counts)]
    #     max_consensus_barcode_list.append(max_consensus_barcode)
    # error_rate_consensus = 1 - np.sum(np.array(max_consensus_barcode_list) == barcode)/cell_info.shape[0]
    # cell_info_correct = cell_info.loc[cell_info[101].values == barcode,:]
    # cell_info_incorrect = cell_info.loc[cell_info[101].values != barcode,:]
    # nn_barcode_list.append(nn_barcode)
    hamming_distance = np.zeros(10)
    barcode_hd = pd.DataFrame(nn_barcode).astype(str).iloc[:,0].apply(hamming2, args = (barcode,))
    # spec_heatmap[i,:] = np.average(cell_info.iloc[:,0:95].values, axis = 0)
    for k in range(10):
        hamming_distance[k] = np.sum(barcode_hd == k)/barcode_hd.shape[0]
    return(hamming_distance)

@dask.delayed
def get_spec_heatmap(sam_tab_bkg_intensity, i, batch_size, nbit):
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    spec_heatmap = np.average(cell_info.iloc[:,0:95].values, axis = 0)
    return(spec_heatmap)

def summarize_reference_experiment(sam_tab_bkg_filename, data_folder):
    sam_tab_bkg_intensity = pd.read_csv(sam_tab_bkg_filename)
    sam_tab_bkg_intensity['error_rate'] = 0
    sam_tab_bkg_intensity['detection_limit'] = 0
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    barcode_error_rate_list = []
    hamming_distance_list = []
    spec_heatmap_list = []
    average_nn_dists_list = []
    batch_size = 20
    for i in range(sam_tab_bkg_intensity.shape[0]):
        results = get_error_rate(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder)
        nn_dists_results = get_average_nn_dists(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder)
        hamming_results = get_hamming_distance(sam_tab_bkg_intensity, i, batch_size, nbit)
        spec_heatmap = get_spec_heatmap(sam_tab_bkg_intensity, i, batch_size, nbit)
        barcode_error_rate_list.append(results)
        hamming_distance_list.append(hamming_results)
        spec_heatmap_list.append(spec_heatmap)
        average_nn_dists_list.append(nn_dists_results)

    barcode_error_rate = dask.delayed(np.stack)(barcode_error_rate_list, axis = 0).compute()
    barcode_error_rate = pd.DataFrame(barcode_error_rate, columns = ['error_rate','detection_limit'])
    average_nn_dists = dask.delayed(np.stack)(average_nn_dists_list, axis = 0).compute()
    sam_tab_bkg_intensity.loc[:,'error_rate'] = barcode_error_rate.error_rate.values
    sam_tab_bkg_intensity.loc[:,'detection_limit'] = barcode_error_rate.detection_limit.values

    theme_color = 'black'
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
    dilution_list = sam_tab_bkg_intensity.DILUTION.drop_duplicates()
    error_rate_dilution = [sam_tab_bkg_intensity.loc[sam_tab_bkg_intensity.DILUTION.values == dilution,'error_rate'].values for dilution in dilution_list]
    detection_limit_dilution = [sam_tab_bkg_intensity.loc[sam_tab_bkg_intensity.DILUTION.values == dilution,'detection_limit'].values for dilution in dilution_list]
    perfect_classification = [np.sum(sam_tab_bkg_intensity.loc[sam_tab_bkg_intensity.DILUTION.values == dilution,'error_rate'].values < 0.01) for dilution in dilution_list]
    ribosome_per_cell_list = [np.log10(7903/dilution) for dilution in dilution_list]
    error_rate_fraction = [np.sum(x < 0.05) for x in error_rate_dilution]
    parts = plt.violinplot(error_rate_dilution, positions = ribosome_per_cell_list, widths = 0.2, showextrema = False)
    for pc in parts['bodies']:
        pc.set_facecolor((0,0.5,1))
        pc.set_edgecolor((0,0.5,1))
        pc.set_linewidth(0.5)
        pc.set_alpha(0.8)

    plt.xscale('log')
    plt.xlim(np.log10(50), np.log10(12000))
    plt.xticks([2,3,4], [r'$10^2$',r'$10^3$',r'$10^4$'])
    plt.xlabel('Estimated Ribosome per Cell', color = theme_color, fontsize = 8)
    plt.ylabel('Error Rate', color = theme_color, fontsize = 8)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.minorticks_off()
    plt.subplots_adjust(left = 0.2,bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_error_rate_ribosome_density_dilution.pdf'.format(data_folder), dpi = 300, transparent = True)
    plt.close()

    # ribosome dilution presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(4))
    dilution_list = sam_tab_bkg_intensity.DILUTION.drop_duplicates()
    error_rate_dilution = [sam_tab_bkg_intensity.loc[sam_tab_bkg_intensity.DILUTION.values == dilution,'error_rate'].values for dilution in dilution_list]
    detection_limit_dilution = [sam_tab_bkg_intensity.loc[sam_tab_bkg_intensity.DILUTION.values == dilution,'detection_limit'].values for dilution in dilution_list]
    perfect_classification = [np.sum(sam_tab_bkg_intensity.loc[sam_tab_bkg_intensity.DILUTION.values == dilution,'error_rate'].values < 0.01) for dilution in dilution_list]
    ribosome_per_cell_list = [np.log10(7903/dilution) for dilution in dilution_list]
    error_rate_fraction = [np.sum(x < 0.05) for x in error_rate_dilution]
    parts = plt.violinplot(error_rate_dilution, positions = ribosome_per_cell_list, widths = 0.2, showextrema = False)
    for pc in parts['bodies']:
        pc.set_facecolor((0,0.5,1))
        pc.set_edgecolor((0,0.5,1))
        pc.set_linewidth(0.5)
        pc.set_alpha(0.8)

    plt.xscale('log')
    plt.xlim(np.log10(50), np.log10(12000))
    plt.xticks([2,3,4], [r'$10^2$',r'$10^3$',r'$10^4$'])
    plt.xlabel('Estimated Ribosome per Cell', color = theme_color, fontsize = font_size)
    plt.ylabel('Error Rate', color = theme_color, fontsize = font_size)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.minorticks_off()
    plt.subplots_adjust(left = 0.2,bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/ecoli_error_rate_ribosome_density_dilution_presentation.svg'.format(data_folder), dpi = 300, transparent = True)
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('summary_table_filename', type = str, help = 'Summary file path')
    args = parser.parse_args()
    data_dir = os.path.split(args.summary_table_filename)[0]
    sum_tab = pd.read_csv(args.summary_table_filename)
    plot_mean_abundance_barcodes(args.summary_table_filename, sum_tab)
    plot_mean_abundance_distribution(args.summary_table_filename, sum_tab)
    plot_mean_abundance_distribution_presentation(args.summary_table_filename, sum_tab)

if __name__ == '__main__':
    main()
