
"""
Hao Shi 2019
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

def get_date(image_name):
    xml = bioformats.get_omexml_metadata(image_name)
    ome = bioformats.OMEXML(xml)
    metadata = xmltodict.parse(ome.to_xml())
    creation_date = metadata['ome:OME']['ome:StructuredAnnotations']['ome:XMLAnnotation'][72]['ome:Value']['ome:OriginalMetadata']['ome:Value']
    return(creation_date)

def get_pixel_dwelltime(image_name):
    xml = bioformats.get_omexml_metadata(image_name)
    ome = bioformats.OMEXML(xml)
    metadata = xmltodict.parse(ome.to_xml())
    dwell_time = metadata['ome:OME']['ome:StructuredAnnotations']['ome:XMLAnnotation'][190]['ome:Value']['ome:OriginalMetadata']['ome:Value']
    dt = re.sub("''",'', dwell_time)
    return(creation_date)

def convert_to_timestamp(t):
    timestamp = time.mktime(time.strptime(t, '[%Y-%m-%dT%H:%M:%S]'))
    return(timestamp)

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return(z)

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def get_image_collection_date(sam_tab_filename, data_folder):
    sam_tab = pd.read_csv(sam_tab_filename)
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        sam_tab.loc[i, 'CreationDate'] = get_date('{}/{}/{}_488.czi'.format(data_folder, sample, image_name))

    sam_tab.loc[:,'TimeStamp'] = sam_tab.CreationDate.apply(convert_to_timestamp)
    for i in range(32):
        sam_tab.loc[:, 'channel_{}'.format(i)] = 0

    sam_tab.loc[:,'UVDye'] = 0
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
        barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
        cell_info = pd.read_csv('{}/{}/{}_avgint.csv'.format(data_folder, sample, image_name), header = None, dtype = {132:str})
        sam_tab.loc[i, ['channel_{}'.format(i) for i in range(32)]] = np.average(cell_info.iloc[:,0:32], axis = 0)
        sam_tab.loc[i, 'UVDye'] = ((barcode[1] == '0') & (barcode[5] == '0') & (barcode[6] == '0'))*1

    sam_tab.loc[:,'BkgFactor'] = 1
    sam_tab.loc[:,'SampleAdjacent'] = ''
    sam_tab.loc[:,'TimeAdjacent'] = 0
    non_uv_spec = sam_tab.loc[sam_tab.UVDye.values == 1, :].reset_index().drop(columns = 'index')

    for i in range(sam_tab.shape[0]):
        print(i)
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
        barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
        cell_info = pd.read_csv('{}/{}/{}_avgint.csv'.format(data_folder, sample, image_name), header = None, dtype = {132:str})
        collection_time = sam_tab.loc[i,'TimeStamp']
        bkg_nc = np.load('{}/{}/{}_bkg.npy'.format(data_folder, sample, image_name))
        image = np.load('{}/{}/{}_registered.npy'.format(data_folder, sample, image_name))
        bkg_nc_max = np.max(image[:,:,0:32]*bkg_nc[:,:,None], axis = 2)
        bkg_nc_avg = np.average(bkg_nc_max[bkg_nc_max > 0])
        if sam_tab.loc[i,'UVDye'] == 1:
            sam_tab.loc[i,'BkgFactor'] = 1
            sam_tab.loc[i,'SampleAdjacent'] = image_name
            sam_tab.loc[i,'TimeAdjacent'] = 0
        else:
            delta_t = np.abs(non_uv_spec.TimeStamp.values - collection_time)
            min_delta_t = np.min(delta_t)
            t_index = np.argmin(delta_t)
            sample_adjacent = non_uv_spec.loc[t_index, 'SAMPLE']
            image_name_adjacent = non_uv_spec.loc[t_index, 'IMAGES']
            cell_info_adjacent = pd.read_csv('{}/{}/{}_avgint.csv'.format(data_folder, sample_adjacent, image_name_adjacent), header = None, dtype = {132:str})
            bkg_nc_adjacent = np.load('{}/{}/{}_bkg.npy'.format(data_folder, sample_adjacent, image_name_adjacent))
            image_adjacent = np.load('{}/{}/{}_registered.npy'.format(data_folder, sample_adjacent, image_name_adjacent))
            bkg_nc_adjacent_max = np.max(image_adjacent[:,:,0:32]*bkg_nc_adjacent[:,:,None], axis = 2)
            bkg_nc_adjacent_avg = np.average(bkg_nc_adjacent_max[bkg_nc_adjacent_max>0])
            bkg_nc_factor = bkg_nc_avg/bkg_nc_adjacent_avg
            sam_tab.loc[i,'BkgFactor'] = bkg_nc_factor
            sam_tab.loc[i,'SampleAdjacent'] = image_name_adjacent
            sam_tab.loc[i,'TimeAdjacent'] = collection_time - non_uv_spec.loc[t_index, 'TimeStamp']

    sam_tab.loc[:,'BkgFactorCorrected'] = sam_tab.BkgFactor.values
    for i in range(sam_tab.shape[0]):
        print(i)
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
        barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
        cell_info = pd.read_csv('{}/{}/{}_avgint.csv'.format(data_folder, sample, image_name), header = None, dtype = {132:str})
        cell_info_bkg = cell_info.iloc[:,0:95].copy()
        if sam_tab.loc[i,'UVDye'] == 0:
            image_name_adjacent = sam_tab.loc[i,'SampleAdjacent']
            bkg_nc_factor = sam_tab.loc[i,'BkgFactor']
            bkg_spec = pd.read_csv('{}/{}/{}_bkg.csv'.format(data_folder, sample, image_name_adjacent), header = None).iloc[0:32,0].values
            min_intensity = np.min(np.average(cell_info_bkg.values[:,0:32], axis = 0) - bkg_nc_factor*bkg_spec)
            if min_intensity > -0.001:
                bkg_nc_factor = sam_tab.loc[i,'BkgFactor']
            elif min_intensity < -0.005:
                delta = 0.01
                min_intensity_temp = min_intensity
                while min_intensity_temp < -0.005:
                    bkg_nc_factor -= delta
                    min_intensity_temp = np.min(np.average(cell_info_bkg.values[:,0:32], axis = 0) - bkg_nc_factor*bkg_spec)
            sam_tab.loc[i, 'BkgFactorCorrected'] = bkg_nc_factor
    sam_tab_bkg_intensity = sam_tab.loc[:,['SAMPLE', 'IMAGES', 'CALIBRATION', 'REFERENCE', 'CALIBRATION_FILENAME',
                                           'REFERENCE_FOLDER', 'REFERENCE_TYPE', 'REFERENCE_NORMALIZATION',
                                           'REFERENCE_SCOPE', 'SPC', 'CreationDate', 'TimeStamp', 'BkgFactor', 'BkgFactorCorrected',
                                           'SampleAdjacent', 'TimeAdjacent', 'UVDye']]
    sam_tab_bkg_intensity.to_csv('{}/images_table_1023_reference_bkg.csv'.format(data_folder), index = None)
    for i in range(sam_tab_bkg_intensity.shape[0]):
        print(i)
        sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
        image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
        enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
        barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
        cell_info = pd.read_csv('{}/{}/{}_avgint.csv'.format(data_folder, sample, image_name), header = None, dtype = {132:str})
        bkg_intensity = sam_tab_bkg_intensity.loc[i,'BkgIntensity']
        bkg_intensity_corrected = sam_tab_bkg_intensity.loc[i,'BkgIntensityCorrected']
        delta_spec = cell_info.iloc[:,0:32].mean(axis = 0) - bkg_intensity*bkg_spec.iloc[0:32,0].values
        delta_spec_corrected = cell_info.iloc[:,0:32].mean(axis = 0) - bkg_intensity_corrected*bkg_spec.iloc[0:32,0].values
        min_intensity = np.min(delta_spec.values)
        min_corrected_intensity = np.min(delta_spec_corrected.values)
        if min_corrected_intensity < 0:
            if min_intensity < 0:
                delta_spec_avg = np.average(delta_spec, axis = 0)
                min_index = np.argmin(delta_spec_avg)
                min_sc_intensity = delta_spec.iloc[:,min_index]
                min_cell_index = np.argmin(min_sc_intensity)
                scale_factor = cell_info.values[min_cell_index,min_index]/(bkg_intensity*bkg_spec.values[min_index, 0])
                sam_tab_bkg_intensity.loc[i,'BkgIntensityFinal'] = bkg_intensity*scale_factor
            else:
                delta_spec_corrected_avg = np.average(delta_spec_corrected, axis = 0)
                min_index = np.argmin(delta_spec_corrected_avg)
                min_sc_intensity = delta_spec_corrected.iloc[:,min_index]
                min_cell_index = np.argmin(min_sc_intensity)
                scale_factor = cell_info.values[min_cell_index,min_index]/(bkg_intensity_corrected*bkg_spec.values[min_index, 0])
                sam_tab_bkg_intensity.loc[i,'BkgIntensityFinal'] = bkg_intensity_corrected*scale_factor
        else:
            sam_tab_bkg_intensity.loc[i,'BkgIntensityFinal'] = sam_tab_bkg_intensity.loc[i,'BkgIntensityCorrected']

    for i in range(sam_tab_bkg_intensity.shape[0]):
        if sam_tab_bkg_intensity.loc[i,'UVDye'] == 1:
            sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
            image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
            enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
            barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
            cell_info = pd.read_csv('{}/{}/{}_avgint.csv'.format(data_folder, sample, image_name), header = None, dtype = {132:str})
            avgint = cell_info.iloc[:,0:32].mean(axis = 0).values
            if i == 131:
                bkg_temp = baseline_als(avgint, 100, 0.01)
                bkg_final = avgint.copy()
                bkg_final[15:] = bkg_temp[15:]
            else:
                bkg_temp = baseline_als(avgint, 1, 0.01)
                bkg_final = avgint.copy()
                bkg_final[8:] = bkg_temp[8:]
            pd.DataFrame(bkg_final).to_csv('{}/{}/{}_bkg.csv'.format(data_folder, sample, image_name), index = None, header = None)
    return

@dask.delayed
def get_error_rate(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder):
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    # error_rate = 1 - np.sum(cell_info.iloc[:,100].values == barcode)/cell_info.shape[0]
    max_intensity_405 = np.max(cell_info.iloc[:,0:32].values, axis = 1)
    max_intensity_405_avg = np.average(max_intensity_405)
    max_intensity_405_std = np.std(max_intensity_405)
    max_intensity_488 = np.max(cell_info.iloc[:,32:55].values, axis = 1)
    max_intensity_488_avg = np.average(max_intensity_488)
    max_intensity_488_std = np.std(max_intensity_488)
    max_intensity_514 = np.max(cell_info.iloc[:,55:75].values, axis = 1)
    max_intensity_514_avg = np.average(max_intensity_514)
    max_intensity_514_std = np.std(max_intensity_514)
    max_intensity_561 = np.max(cell_info.iloc[:,75:89].values, axis = 1)
    max_intensity_561_avg = np.average(max_intensity_561)
    max_intensity_561_std = np.std(max_intensity_561)
    max_intensity_633 = np.max(cell_info.iloc[:,89:95].values, axis = 1)
    max_intensity_633_avg = np.average(max_intensity_633)
    max_intensity_633_std = np.std(max_intensity_633)
    condition_405 = (max_intensity_405 > max_intensity_405_avg - max_intensity_405_std)*(max_intensity_405 < max_intensity_405_avg + max_intensity_405_std)
    condition_488 = (max_intensity_488 > max_intensity_488_avg - max_intensity_488_std)*(max_intensity_488 < max_intensity_488_avg + max_intensity_488_std)
    condition_514 = (max_intensity_514 > max_intensity_514_avg - max_intensity_514_std)*(max_intensity_514 < max_intensity_514_avg + max_intensity_514_std)
    condition_561 = (max_intensity_561 > max_intensity_561_avg - max_intensity_561_std)*(max_intensity_561 < max_intensity_561_avg + max_intensity_561_std)
    condition_633 = (max_intensity_633 > max_intensity_633_avg - max_intensity_633_std)*(max_intensity_633 < max_intensity_633_avg + max_intensity_633_std)
    total_condition = condition_405*condition_488*condition_514*condition_561*condition_633
    cell_info_filtered = cell_info.loc[total_condition*1 == 1, :]
    barcode_consensus = []
    nn_dist = np.zeros((cell_info_filtered.shape[0], batch_size))
    for r in range(batch_size):
        cell_info_filename = '{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, r)
        if os.path.exists(cell_info_filename):
            cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, r), header = None, dtype = {100:str})
            cell_info_filtered = cell_info.loc[total_condition*1 == 1, :]
            barcode_consensus.append(cell_info_filtered.iloc[:,100].values)
            nn_dist[:,r] = cell_info_filtered.iloc[:,101].values
        else:
            barcode_consensus.append(np.repeat('0000000000', cell_info_filtered.shape[0]))
            nn_dist[:,r] = 1
    barcode_consensus = np.stack(barcode_consensus, axis = 0).transpose()
    nn_barcode = []
    nn_indices = np.argmin(nn_dist, axis = 1)
    for b in range(barcode_consensus.shape[0]):
        nn_barcode.append(barcode_consensus[b,nn_indices[b]])
    error_rate = 1 - np.sum(np.array(nn_barcode) == barcode)/cell_info_filtered.shape[0]
    cell_info_consensus = cell_info_filtered.copy()
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
    # barcode_hd = pd.DataFrame(nn_barcode).astype(str).iloc[:,0].apply(hamming2, args = (barcode,))
    # spec_heatmap[i,:] = np.average(cell_info.iloc[:,0:95].values, axis = 0)
    # for k in range(10):
    #     hamming_distance[i,k] = np.sum(barcode_hd == k)/barcode_hd.shape[0]
    return(np.array([error_rate, detection_limit]))

@dask.delayed
def get_hamming_distance(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder):
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
def get_spec_heatmap(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder):
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    spec_heatmap = np.average(cell_info.iloc[:,0:95].values, axis = 0)
    return(spec_heatmap)

def summarize_reference_experiment(sam_tab_bkg_filename, data_folder, n_workers):
    sam_tab_bkg_intensity = pd.read_csv(sam_tab_bkg_filename)
    barcode_error_rate = pd.DataFrame(np.zeros((1023,2)), index = np.arange(1023), columns = ['error_rate','detection_limit'])
    spec_heatmap = np.zeros((sam_tab_bkg_intensity.shape[0], 95))
    hamming_distance = np.zeros((sam_tab_bkg_intensity.shape[0], 10))
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    barcode_error_rate_list = []
    hamming_distance_list = []
    spec_heatmap_list = []
    batch_size = 40
    for i in range(sam_tab_bkg_intensity.shape[0]):
        results = get_error_rate(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder)
        hamming_results = get_hamming_distance(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder)
        spec_heatmap = get_spec_heatmap(sam_tab_bkg_intensity, i, batch_size, nbit, data_folder)
        barcode_error_rate_list.append(results)
        hamming_distance_list.append(hamming_results)
        spec_heatmap_list.append(spec_heatmap)
    barcode_error_rate = dask.delayed(np.stack)(barcode_error_rate_list, axis = 0).compute()
    barcode_error_rate = pd.DataFrame(barcode_error_rate, columns = ['error_rate','detection_limit'])
    hamming_distance = dask.delayed(np.stack)(hamming_distance_list, axis = 0).compute()
    spec_heatmap = dask.delayed(np.stack)(spec_heatmap_list, axis = 0).compute()
    # 1023 reference spectra heatmap
    spec_heatmap_adjusted = spec_heatmap.copy()
    spec_heatmap_adjusted[:,0:95] = spec_heatmap_adjusted[:,0:95]/np.max(spec_heatmap_adjusted[:,0:95], axis = 1)[:,None]
    theme_color = 'black'
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3), cm_to_inches(6.25))
    gs = GridSpec(12, 3)
    ax = plt.subplot(gs[0:1,0:3])
    plt.hlines(0, 1, 34, colors = 'darkviolet', linewidth = 3)
    plt.hlines(0, 34, 56, colors = 'limegreen', linewidth = 3)
    plt.hlines(0, 56, 75, colors = 'yellowgreen', linewidth = 3)
    plt.hlines(0, 75, 88, colors = 'darkorange', linewidth = 3)
    plt.hlines(0, 88, 94, colors = 'red', linewidth = 3)
    plt.ylim(-0.1,0.6)
    plt.text(16, 0.2, '405', fontsize = 5, color = 'darkviolet', ha = 'center')
    plt.text(44, 0.2, '488', fontsize = 5, color = 'limegreen', ha = 'center')
    plt.text(65, 0.2, '514', fontsize = 5, color = 'yellowgreen', ha = 'center')
    plt.text(80, 0.2, '561', fontsize = 5, color = 'darkorange', ha = 'center')
    plt.text(87, 0.6, '633', fontsize = 5, color = 'red', ha = 'center')
    plt.text(35, 0.6, 'Excitaton Laser', fontsize = 6, ha = 'center')
    plt.xlim(0,95)
    plt.axis('off')
    ax = plt.subplot(gs[1:12,0:3])
    ax.imshow(spec_heatmap_adjusted, cmap = 'inferno')
    ax.set_aspect(0.21)
    ax.tick_params(direction = 'in', length = 2, color = 'white', labelsize = 6, pad = 2)
    plt.yticks([128, 256, 384, 512, 640, 768, 892], [128, 256, 384, 512, 640, 768, 892])
    plt.xlabel('Channel', color = theme_color, fontsize = 6, labelpad = 0)
    plt.ylabel('Barcode', color = theme_color, fontsize = 6, labelpad = 0)
    plt.subplots_adjust(left = 0.21,bottom = 0.08, right = 0.99, top = 0.97, hspace = 0.1, wspace = 0.15)
    plt.savefig('{}/ecoli_1023_reference.pdf'.format(data_folder), dpi = 300, transparent = True)
    plt.close()
    # 1023 spectra heatmap presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(9), cm_to_inches(16))
    gs = GridSpec(12, 3)
    ax = plt.subplot(gs[0:1,0:3])
    plt.hlines(0, 0, 32, colors = 'darkviolet', linewidth = 4)
    plt.hlines(0, 32, 55, colors = 'limegreen', linewidth = 4)
    plt.hlines(0, 55, 75, colors = 'yellowgreen', linewidth = 4)
    plt.hlines(0, 75, 89, colors = 'darkorange', linewidth = 4)
    plt.hlines(0, 89, 95, colors = 'red', linewidth = 4)
    plt.ylim(-0.1,0.6)
    plt.text(16, 0.2, '405', fontsize = font_size, color = 'darkviolet', ha = 'center')
    plt.text(44, 0.2, '488', fontsize = font_size, color = 'limegreen', ha = 'center')
    plt.text(65, 0.2, '514', fontsize = font_size, color = 'yellowgreen', ha = 'center')
    plt.text(82, 0.2, '561', fontsize = font_size, color = 'darkorange', ha = 'center')
    plt.text(97, 0.2, '633', fontsize = font_size, color = 'red', ha = 'center')
    plt.text(47, 0.6, 'Excitaton Laser', fontsize = 8, ha = 'center')
    plt.xlim(0,95)
    plt.axis('off')
    ax = plt.subplot(gs[1:12,0:3])
    ax.imshow(spec_heatmap, cmap = 'inferno')
    ax.set_aspect(0.2)
    ax.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    plt.yticks([128, 256, 384, 512, 640, 768, 892], [128, 256, 384, 512, 640, 768, 892])
    ax.spines['left'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    plt.xlabel('Channel', color = theme_color, fontsize = font_size)
    plt.ylabel('Barcode', color = theme_color, fontsize = font_size)
    plt.subplots_adjust(left = 0.24,bottom = 0.08, right = 0.92, top = 0.97, hspace = 0.1, wspace = 0.15)
    plt.savefig('{}/ecoli_1023_reference_presentation.svg'.format(data_folder), dpi = 300, transparent = True)
    plt.close()

    # error rate hamming distance decay
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(12), cm_to_inches(12))
    gs = GridSpec(6,4)
    ax = plt.subplot(gs[1:,:])
    for i in range(31):
        for j in range(32):
            if barcode_error_rate.detection_limit.values[i*32 + j] == 1:
                plt.plot(np.arange(i*15, i*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (0,0.5,1))
            else:
                plt.plot(np.arange(i*15, i*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (1,0.5,0))

    for j in range(31):
        if barcode_error_rate.detection_limit.values[i*32 + j] == 1:
            plt.plot(np.arange(31*15, 31*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (0,0.5,1))
        else:
            plt.plot(np.arange(31*15, 31*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (1,0.5,0))

    plt.plot((0,0), (0,0), label = 'error undetectable', color = (0,0.5,1))
    plt.plot((0,0), (0,0), label = 'error detectable', color = (1,0.5,0))
    plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.8,0.03))
    plt.axis('off')
    hamming_distance_above_dl = hamming_distance[barcode_error_rate.detection_limit.values == 0, :]
    ax = plt.subplot(gs[0,0])
    plt.plot(hamming_distance_above_dl[0,:], color = (1,0.5,0))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance_above_dl[0,1:])), color = theme_color, fontsize = 8)
    plt.xlabel(r'$d_{Hamming}$', fontsize = 8, color = theme_color)
    plt.ylabel('Abundance', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    ax = plt.subplot(gs[0,1])
    plt.plot(hamming_distance_above_dl[1,:], color = (1,0.5,0))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance_above_dl[1,1:])), color = theme_color, fontsize = 8)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    ax = plt.subplot(gs[0,2])
    plt.plot(hamming_distance_above_dl[2,:], color = (1,0.5,0))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance_above_dl[2,1:])), color = theme_color, fontsize = 8)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    plt.xticks([])
    plt.yticks([])
    ax = plt.subplot(gs[0,3])
    plt.plot(hamming_distance[0,:], color = (0,0.5,1))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance[32*30,1:])), color = theme_color, fontsize = 8)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left = 0.08, bottom = 0.06, right = 0.98, top = 0.98, hspace = 0.5)
    plt.savefig('{}/ecoli_1023_hamming_breakout.pdf'.format(data_folder), dpi = 300, transparent = True)
    # error rate hamming distance decay presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(16), cm_to_inches(16))
    gs = GridSpec(6,4)
    ax = plt.subplot(gs[1:,:])
    for i in range(31):
        for j in range(32):
            if barcode_error_rate.detection_limit.values[i*32 + j] == 1:
                plt.plot(np.arange(i*15, i*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (0,0.5,1))
            else:
                plt.plot(np.arange(i*15, i*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (1,0.5,0))

    for j in range(31):
        if barcode_error_rate.detection_limit.values[i*32 + j] == 1:
            plt.plot(np.arange(31*15, 31*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (0,0.5,1))
        else:
            plt.plot(np.arange(31*15, 31*15 + 10), 1.5*j + hamming_distance[i*32 + j, :], '-', color = (1,0.5,0))

    plt.plot((0,0), (0,0), label = 'error undetectable', color = (0,0.5,1))
    plt.plot((0,0), (0,0), label = 'error detectable', color = (1,0.5,0))
    l = plt.legend(frameon = False, fontsize = font_size, bbox_to_anchor = (0.8,0.03))
    for t in l.get_texts():
        t.set_color(theme_color)

    plt.axis('off')
    hamming_distance_above_dl = hamming_distance[barcode_error_rate.detection_limit.values == 0, :]
    ax = plt.subplot(gs[0,0])
    plt.plot(hamming_distance_above_dl[0,:], color = (1,0.5,0))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance_above_dl[0,1:])), color = theme_color, fontsize = font_size)
    plt.xlabel(r'$d_{Hamming}$', fontsize = font_size, color = theme_color)
    plt.ylabel('Abundance', fontsize = font_size, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    ax.spines['left'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    ax = plt.subplot(gs[0,1])
    plt.plot(hamming_distance_above_dl[1,:], color = (1,0.5,0))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance_above_dl[1,1:])), color = theme_color, fontsize = font_size)
    plt.xticks([])
    plt.yticks([])
    ax.spines['left'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    ax = plt.subplot(gs[0,2])
    plt.plot(hamming_distance_above_dl[2,:], color = (1,0.5,0))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance_above_dl[2,1:])), color = theme_color, fontsize = font_size)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    plt.xticks([])
    plt.yticks([])
    ax.spines['left'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    ax = plt.subplot(gs[0,3])
    plt.plot(hamming_distance[0,:], color = (0,0.5,1))
    plt.text(0.7, 0.5, r'{:03.1f}$\%$ error'.format(100*np.sum(hamming_distance[32*30,1:])), color = theme_color, fontsize = font_size)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = font_size)
    plt.xticks([])
    plt.yticks([])
    ax.spines['left'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    plt.subplots_adjust(left = 0.08, bottom = 0.06, right = 0.98, top = 0.98, hspace = 0.5)
    plt.savefig('{}/ecoli_1023_hamming_breakout_presentation.svg'.format(data_folder), dpi = 300, transparent = True)
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('summary_table_filename', type = str, help = 'Summary file path')
    args = parser.parse_args()
    data_dir = os.path.split(args.summary_table_filename)[0]
    sum_tab = pd.read_csv(args.summary_table_filename)
    summarize_reference_experiment(args.summary_table_filename, data_dir)

if __name__ == '__main__':
    main()
