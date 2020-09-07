import os
import re
import sys
import glob
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import color
from ete3 import NCBITaxa
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import Normalize
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from skimage import future
import dask
from dask.distributed import LocalCluster, Client
from matplotlib.gridspec import GridSpec

ncbi = NCBITaxa()

def cm_to_inches(length):
    return(length/2.54)

def get_taxon_lookup(probe_design_filename):
    probe_design_folder = os.path.split(probe_design_filename)[0]
    probes = pd.read_csv(probe_design_filename, dtype = {'code': str})
    ncbi = NCBITaxa()
    taxon_lookup = probes.loc[:,['target_taxon', 'code']].drop_duplicates()
    taxon_lookup['H'] = np.arange(0,1,1/taxon_lookup.shape[0])
    taxon_lookup['S'] = 1
    taxon_lookup['V'] = 1
    taxon_sciname = pd.DataFrame.from_dict(ncbi.get_taxid_translator(taxon_lookup.target_taxon.values), orient = 'index').reset_index()
    taxon_sciname.columns = ['target_taxon', 'sci_name']
    taxon_lookup = taxon_lookup.merge(taxon_sciname, on = 'target_taxon')
    taxon_lookup.to_csv('{}/taxon_color_lookup.csv'.format(probe_design_folder))
    return(taxon_lookup)

def get_lineage_at_desired_ranks(taxid, desired_ranks):
    'Retrieve lineage information at desired taxonomic ranks'
    # initiate an instance of the ncbi taxonomy database
    ncbi = NCBITaxa()
    # retrieve lineage information for each full length 16S molecule
    lineage = ncbi.get_lineage(taxid)
    lineage2ranks = ncbi.get_rank(lineage)
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())
    ranki = [ranks2lineage.get(x) for x in desired_ranks]
    ranks = [x if x is not None else 0 for x in ranki]
    return(ranks)

def remove_spurious_objects_and_debris(image_seg, image_identification, adjacency_seg, taxon_lookup, cell_info, sample):
    image_identification_filtered = image_identification.copy()
    for i in range(cell_info.shape[0]):
        cell_label = cell_info.loc[i, 'label']
        cell_area = cell_info.loc[i, 'area']
        # cell_prob = cell_info.loc[i, 'max_probability']
        # if (cell_area > 10000) or (cell_label in debris_labels) or (cell_prob <= 0.95):
        if (cell_area > 10000):
          cell_info.loc[i, 'type'] = 'debris'
          image_identification_filtered[image_seg == cell_label] = [0.5,0.5,0.5]
    # save_identification(image_identification_filtered, sample)
    cell_info_filtered = cell_info.loc[cell_info.type.values == 'cell',:].copy()
    # cell_info_filtered_filename = sample + '_cell_information_filtered.csv'
    # cell_info_filtered.to_csv(cell_info_filtered_filename, index = None)
    # avgint_filtered = avgint[cell_info.type.values == 'cell', :]
    # pd.DataFrame(avgint_filtered).to_csv('{}_avgint_filtered.csv'.format(sample), index = None)
    edge_map = skimage.filters.sobel(image_seg > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.code.values, columns = taxon_lookup.code.values)
    adjacency_matrix_filtered = adjacency_matrix.copy()
    for i in range(cell_info.shape[0]):
        edges = list(rag.edges(i+1))
        for e in edges:
          node_1 = e[0]
          node_2 = e[1]
          if (node_1 != 0) and (node_2 !=0):
              barcode_1 = cell_info.loc[cell_info.label == node_1-1,'cell_barcode']
              barcode_2 = cell_info.loc[cell_info.label == node_2-1,'cell_barcode']
              adjacency_matrix.loc[barcode_1, barcode_2] += 1
              if (cell_info.loc[node_1-1,'type'] == 'cell') and (cell_info.loc[node_2-1,'type'] == 'cell'):
                  adjacency_matrix_filtered.loc[barcode_1, barcode_2] += 1
    adjacency_filename = '{}_adjacency_matrix.csv'.format(sample)
    adjacency_filtered_filename = '{}_adjacency_matrix_filtered.csv'.format(sample)
    adjacency_matrix.to_csv(adjacency_filename)
    adjacency_matrix_filtered.to_csv(adjacency_filtered_filename)
    return(cell_info_filtered, image_identification_filtered)

def plot_barcode(cell_info_filtered, barcode):
    plt.plot(cell_info_filtered.iloc[cell_info_filtered.cell_barcode.values == barcode, 0:63].values.transpose(), color = (0,0.5,1), alpha = 0.1)
    plt.show()
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
    fig.savefig(segfilename, dpi = 300)
    plt.close()
    return

def get_taxa_barcode_sciname(probe_design_filename):
    probes = pd.read_csv(probe_design_filename, dtype = {'code':str})
    taxa_barcode = probes.loc[:,['target_taxon', 'code']].drop_duplicates()
    taxa_sciname = ncbi.get_taxid_translator(taxa_barcode.target_taxon.values)
    taxa_sciname_df = pd.DataFrame.from_dict(taxa_sciname, orient = 'index').reset_index()
    taxa_sciname_df.columns = ['target_taxon', 'sci_name']
    taxa_barcode_sciname = taxa_barcode.merge(taxa_sciname_df, on = 'target_taxon')
    taxa_barcode_sciname = taxa_barcode_sciname.rename(columns = {'code':'cell_barcode'})
    taxa_barcode_sciname_filename = re.sub('_full_length_probes.csv', '_taxa_barcode_sciname.csv', probe_design_filename)
    taxa_barcode_sciname.to_csv(taxa_barcode_sciname_filename, index = None)
    return(taxa_barcode_sciname)

def get_taxa_barcode_sciname_lineage(taxa_barcode_sciname):
    ranks = ['superkingdom', 'phylum', 'family', 'class', 'order', 'genus', 'species']
    for t in ranks:
        taxa_barcode_sciname[t] = 0
    for i in range(taxa_barcode_sciname.shape[0]):
        lineage = get_lineage_at_desired_ranks(taxa_barcode_sciname.loc[i,'target_taxon'], ranks)
        taxa_barcode_sciname.loc[i,ranks] = lineage
    return(taxa_barcode_sciname)

def analyze_cell_info(cell_info_filename, taxa_barcode_sciname):
    cell_info = pd.read_csv(cell_info_filename, dtype = {'cell_barcode':str})
    cell_info['max_intensity'] = cell_info.loc[:,['channel_{}'.format(i) for i in range(63)]].max(axis = 1)
    cell_info['type'] = 'debris'
    cell_info.loc[cell_info.max_intensity.values >= 0.1, 'type'] = 'cell'
    cell_snr = cell_info.groupby('cell_barcode').agg({'max_intensity': ['std', 'mean', 'count']})
    cell_snr.columns = cell_snr.columns.droplevel(0)
    cell_snr = cell_snr.sort_values(by = ['mean', 'count'], ascending = [False, False])
    cell_snr = cell_snr.merge(taxa_barcode_sciname, on = 'cell_barcode', how = 'left')
    cell_snr_summary_filename = re.sub('_cell_information_consensus.csv', '_cell_snr_summary.csv', cell_info_filename)
    cell_snr.to_csv(cell_snr_summary_filename, index = None)
    return(cell_info)

def generate_identification_image(segmentation, cell_info_filtered, sample, taxon_lookup):
    cell_barcodes = cell_info_filtered.cell_barcode.drop_duplicates().reset_index().drop(columns = ['index'])
    image_identification = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    image_identification_barcode = np.zeros(segmentation.shape)
    for q in range(0, len(cell_barcodes)):
      cell_population = cell_info_filtered.loc[cell_info_filtered.cell_barcode.values == cell_barcodes.cell_barcode.values[q], :]
      cell_pop_barcode = int(cell_barcodes.cell_barcode.values[q], 2)
      for r in cell_population.index:
          image_identification_barcode[segmentation == cell_population.loc[r, 'label']] = cell_pop_barcode
          if (cell_barcodes.cell_barcode.values[q] in taxon_lookup.code.values):
              image_identification[segmentation == cell_population.loc[r, 'label'], :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code.values == cell_barcodes.cell_barcode.values[q], ['H', 'S', 'V']].values)
          else:
              pass
    return(image_identification)

def analyze_alignment_order_parameter(cell_info, image_seg, image_identification):
    grid_size = 50
    step_size = int(image_seg.shape[0]/grid_size)
    cell_info['q_eigenvalue'] = 0
    q_eigenvectors = np.zeros((step_size, step_size, 2))
    q_eigenvalue_heatmap = np.zeros((step_size, step_size))
    for i in range(step_size):
        for j in range(step_size):
            image_seg_local = image_seg[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            cell_labels = np.unique(image_seg_local[image_seg_local > 0])
            if cell_labels.shape[0] > 0:
                q_local = np.zeros((cell_labels.shape[0], cell_labels.shape[0], 2, 2))
                for ci in range(cell_labels.shape[0]):
                    for cj in range(cell_labels.shape[0]):
                        cli = cell_labels[ci]
                        clj = cell_labels[cj]
                        theta_i = cell_info.loc[cell_info.label.values == cli, 'orientation'].values[0]
                        theta_j = cell_info.loc[cell_info.label.values == clj, 'orientation'].values[0]
                        ni = np.array([np.cos(theta_i), np.sin(theta_i)])
                        nj = np.array([np.cos(theta_j), np.sin(theta_j)])
                        q_ci_cj = (1/2)*(3*np.outer(ni, nj) - 1)
                        q_local[ci, cj, :, :] = q_ci_cj
                q_average = np.average(q_local, axis = (0,1))
                q_eigenvalue, q_eigen_vectors = np.linalg.eig(q_average)
                q_eigenvalue_heatmap[i, j] = np.max(np.abs(q_eigenvalue))
                q_eigenvectors[i,j,:] = q_eigen_vectors[:, np.argmax(np.abs(q_eigenvalue))]
                for cii in range(cell_labels.shape[0]):
                    cell_info.loc[cell_info.label.values == cii, 'q_eigenvalue'] = q_eigenvalue_heatmap[i,j]
    return(q_eigenvalue_heatmap, q_eigenvectors, cell_info)

def analyze_spatial_adjacency_network(segmentation, adjacency_seg, cell_info, taxa_barcode_sciname, sample):
    edge_map = skimage.filters.sobel(image_seg > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.code.values, columns = taxon_lookup.code.values)
    adjacency_matrix_filtered = adjacency_matrix.copy()
    for i in range(cell_info.shape[0]):
        edges = list(rag.edges(i+1))
        for e in edges:
          node_1 = e[0]
          node_2 = e[1]
          if (node_1 != 0) and (node_2 !=0):
              barcode_1 = cell_info.loc[cell_info.label == node_1-1,'cell_barcode']
              barcode_2 = cell_info.loc[cell_info.label == node_2-1,'cell_barcode']
              adjacency_matrix.loc[barcode_1, barcode_2] += 1
              if (cell_info.loc[node_1-1,'type'] == 'cell') and (cell_info.loc[node_2-1,'type'] == 'cell'):
                  adjacency_matrix_filtered.loc[barcode_1, barcode_2] += 1
    adjacency_filename = '_adjacency_matrix.csv'.format(sample)
    adjacency_filtered_filename = '_adjacency_matrix_filtered.csv'.format(sample)
    adjacency_matrix.to_csv(adjacency_filename)
    adjacency_matrix_filtered.to_csv(adjacency_filtered_filename)
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

def analyze_oral_diversity(image_tab_filename):
    # Shannon diversity time series
    image_tab = pd.read_csv(image_tab_filename)
    for i in range(image_tab.shape[0]):
        sample = image_tab.loc[i, 'SAMPLE']
        image_name = image_tab.loc[i, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        cell_info = cell_info.loc[cell_info.spectral_centroid_distance.values < 1, :]
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab.loc[i,'CELL_NUMBER'] = cell_info.shape[0]

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4), cm_to_inches(3.5))
    image_tab_nc = image_tab.loc[image_tab.CELL_NUMBER.values > 500, :]
    time_points = image_tab_nc.SAMPLING_TIME.drop_duplicates()
    shannon_diversity_list = [image_tab_nc.loc[image_tab_nc.SAMPLING_TIME.values == tp, 'SHANNON_DIVERSITY'].values for tp in time_points]
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    bp = plt.boxplot(shannon_diversity_list, positions = np.arange(7), patch_artist = True, flierprops = flierprops)
    for b in bp['boxes']:
        b.set_facecolor('dodgerblue')

    day_list = image_tab_nc.SAMPLING_DAY_REFERENCE.drop_duplicates().values
    xlabel_list = [int(t/30) for t in day_list]
    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.xticks(np.arange(7), xlabel_list)
    plt.xlabel('Time [Month]', fontsize = 8, color = theme_color)
    plt.ylabel('Shannon Diversity', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.18, bottom = 0.23, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_shannon_diversity_time_series.pdf'.format(data_folder), dpi = 300)

    # alpha diversity as a function of patch size and time
    n_habitat = 100
    image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
    beta_diversity_patch_time_series = []
    for nc in np.arange(10,200,10):
        n_cell = nc
        beta_diversity_time_series = []
        for t in image_tab_sc.SAMPLING_TIME.drop_duplicates().values:
            image_tab_sub = image_tab_sc.loc[image_tab_sc.SAMPLING_TIME.values == t, :].reset_index().drop(columns = 'index')
            beta_diversity_image = []
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
                barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
                barcode_abundance.columns = ['cell_barcode', 'abundance']
                barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                p_i = barcode_abundance.relative_abundance.values
                p_i = p_i[p_i > 0]
                shannon_diversity = -np.sum(p_i*np.log(p_i))
                beta_diversity_list = []
                for j in range(n_habitat):
                    reference_point = 2000*np.random.random(2)
                    cell_distance = (cell_info.centroid_x.values - reference_point[0])**2 + (cell_info.centroid_y.values - reference_point[1])**2
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    barcode_abundance = cell_info_sub.cell_barcode.value_counts().reset_index()
                    barcode_abundance.columns = ['cell_barcode', 'abundance']
                    barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                    p_ij = barcode_abundance.relative_abundance.values
                    p_ij = p_ij[p_ij > 0]
                    alpha_diversity = -np.sum(p_ij*np.log(p_ij))
                    beta_diversity_list.append(shannon_diversity/alpha_diversity)
                beta_diversity_patch = np.average(beta_diversity_list)
                beta_diversity_image.append(beta_diversity_patch)
            beta_diversity_time_series.append(beta_diversity_image)
        beta_diversity_patch_time_series.append(beta_diversity_time_series)

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(3.5))
    time_points = image_tab.SAMPLING_TIME.drop_duplicates()
    cmap = cm.get_cmap('coolwarm')
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    meanprops = dict(linestyle='none', linewidth = 0)
    medianprops = dict(linestyle='none', linewidth = 0)
    # plt.plot(beta_diversity_list)
    for i in np.arange(19):
        bp = plt.boxplot(beta_diversity_patch_time_series[i], positions = np.arange(0, 2.2*7, 2.2) - 1 + 0.1*i, patch_artist = True, flierprops = flierprops,
                    widths = 0.3, showcaps = False, showfliers = False, meanline = False, medianprops = medianprops)
        for b in bp['boxes']:
            b.set_facecolor(cmap(i/20))
            b.set_edgecolor((0.4,0.4,0.4))
            b.set_linewidth(0.25)
        for w in bp['whiskers']:
            w.set_linewidth(0.25)
            w.set_color((0.4,0.4,0.4))

    day_list = image_tab.SAMPLING_DAY_REFERENCE.drop_duplicates().values
    xlabel_list = [int(t/30) for t in day_list]
    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.xticks(np.arange(0, 2.2*7, 2.2), xlabel_list)
    plt.xlabel('Time [month]', fontsize = 8, color = theme_color, labelpad = 0)
    plt.ylabel(r'$\beta$ diversity', fontsize = 8, color = theme_color, labelpad = 0)
    cbaxes = fig.add_axes([0.65, 0.9, 0.2, 0.05])
    norm = Normalize(vmin = 10, vmax = 200)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize=6, color = theme_color)
    cb.set_ticks([10,100,200])
    cb.set_label(r'$N_{cells}$', fontsize = 6, color = theme_color, labelpad = 0)
    plt.subplots_adjust(left = 0.18, bottom = 0.2, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_beta_diversity_patch_size_time_series.pdf'.format(data_folder), dpi = 300)

    for nc in [20,40,60,80,100,120,140,160,180,200]:
        n_pairs = 200
        n_cell = nc
        bray_curtis_list = []
        image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
        for t in image_tab_sc.SAMPLING_TIME.drop_duplicates().values:
            bray_curtis_tp = []
            image_tab_sub = image_tab_sc.loc[image_tab_sc.SAMPLING_TIME.values == t, :].reset_index().drop(columns = 'index')
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                bray_curtis_image = np.zeros((n_pairs, 2))
                cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name))
                for j in range(n_pairs):
                    rp_1 = 2000*np.random.random(2)
                    cell_distance = (cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_1 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_1_barcode = cell_info_sub_1.cell_barcode.value_counts().reset_index()
                    cell_info_sub_1_barcode.columns = ['abundance_1', 'cell_barcode']
                    rp_2 = 2000*np.random.random(2)
                    cell_distance = (cell_info.centroid_x.values - rp_2[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_2 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_2_barcode = cell_info_sub_2.cell_barcode.value_counts().reset_index()
                    cell_info_sub_2_barcode.columns = ['abundance_2', 'cell_barcode']
                    cell_info_merge = cell_info_sub_1_barcode.merge(cell_info_sub_2_barcode, on = 'cell_barcode', how = 'outer').fillna(0)
                    coa_1_x = np.sum(cell_info_sub_1.centroid_x.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_1_y = np.sum(cell_info_sub_1.centroid_y.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_2_x = np.sum(cell_info_sub_2.centroid_x.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    coa_2_y = np.sum(cell_info_sub_2.centroid_y.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    bray_curtis_image[j, 0] = np.sqrt((coa_1_x - coa_2_x)**2 + (coa_1_y - coa_2_y)**2)
                    bray_curtis_image[j, 1] = scipy.spatial.distance.braycurtis(cell_info_merge.abundance_1.values, cell_info_merge.abundance_2.values)
                bray_curtis_tp.append(bray_curtis_image)
            bray_curtis_list.append(np.concatenate(bray_curtis_tp, axis = 0))
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(3.5))
        cmap = cm.get_cmap('tab10')
        color_list = [cmap(i/7) for i in range(7)]
        time_points = image_tab.SAMPLING_TIME.drop_duplicates()
        day_list = image_tab.SAMPLING_DAY_REFERENCE.drop_duplicates().values
        label_list = ['M{}'.format(int(t/30)) for t in day_list]
        for i in range(7):
            bcd = bray_curtis_list[i]
            bcd_sorted = bcd[bcd[:,0].argsort()]
            bcd_smoothed = scipy.signal.savgol_filter(bcd_sorted[:,1], window_length = 41, polyorder = 3)
            bcd_smoothed = np.clip(bcd_smoothed, 0, 1)
            plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = color_list[i], label = label_list[i], alpha = 0.5)

        plt.tick_params(direction = 'in', length = 2, labelsize = 8)
        plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.18,0.64), ncol = 2)
        plt.xlabel(r'Intra-patch distance [$\mu$m]', fontsize = 8, color = theme_color, labelpad = 0)
        plt.ylabel('Bray-Curtis', fontsize = 8, color = theme_color, labelpad = 0)
        plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.98, top = 0.98)
        plt.savefig('{}/oral_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300)


    return

def analyze_gut_diversity_ciprofloxacin(image_tab_filename):

    # Shannon diversity time series
    image_tab = pd.read_csv(image_tab_gut_filename)
    for i in range(image_tab.shape[0]):
        sample = image_tab.loc[i, 'SAMPLE']
        image_name = image_tab.loc[i, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab.loc[i,'CELL_NUMBER'] = cell_info.shape[0]

    # Shannon diversity for slices from zstacks
    image_tab = pd.read_csv(image_tab_gut_filename)
    for i in range(image_tab.shape[0]):
        sample = image_tab.loc[i, 'SAMPLE']
        image_name = image_tab.loc[i, 'IMAGES']
        z = image_tab.loc[i,'ZSLICE']
        cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab.loc[i,'CELL_NUMBER'] = cell_info.shape[0]

    image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
    barcode_abundance_list = []
    treatments = image_tab_sc.TREATMENT.drop_duplicates()
    for t in treatments:
        image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
        sample = image_tab_sub.loc[0, 'SAMPLE']
        image_name = image_tab_sub.loc[0, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        barcode_abundance_full = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_full.columns = ['cell_barcode', 'abundance_0']
        for i in range(1, image_tab_sub.shape[0]):
            sample = image_tab_sub.loc[i, 'SAMPLE']
            image_name = image_tab_sub.loc[i, 'IMAGES']
            cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
            barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
            barcode_abundance.columns = ['cell_barcode', 'abundance_{}'.format(i)]
            barcode_abundance_full = barcode_abundance_full.merge(barcode_abundance, on = 'cell_barcode', how = 'outer')
        barcode_abundance_full = barcode_abundance_full.fillna(0)
        barcode_abundance_full['abundance_all'] = barcode_abundance_full.loc[:,['abundance_{}'.format(i) for i in range(image_tab_sub.shape[0])]].sum(axis = 1)
        barcode_abundance_full['relative_abundance'] = barcode_abundance_full.loc[:,'abundance_all'].values/barcode_abundance_full.abundance_all.sum()
        barcode_abundance_full = barcode_abundance_full.merge(taxa_barcode_sciname, on = 'cell_barcode', how = 'left')
        barcode_abundance_list.append(barcode_abundance_full)

    barcode_abundance_antibiotics = barcode_abundance_list[0].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_antibiotics.columns = ['taxid','abundance']
    barcode_abundance_control = barcode_abundance_list[1].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_control.columns = ['taxid','abundance']

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    antibiotics_bacteroides_abundance = barcode_abundance_antibiotics.loc[barcode_abundance_antibiotics.taxid.values == 976, 'abundance'].values[0]
    antibiotics_firmicutes_abundance = barcode_abundance_antibiotics.loc[barcode_abundance_antibiotics.taxid.values == 1239, 'abundance'].values[0]
    antibiotics_others_abundance = 1 - antibiotics_bacteroides_abundance - antibiotics_firmicutes_abundance
    control_bacteroides_abundance = barcode_abundance_control.loc[barcode_abundance_control.taxid.values == 976, 'abundance'].values[0]
    control_firmicutes_abundance = barcode_abundance_control.loc[barcode_abundance_control.taxid.values == 1239, 'abundance'].values[0]
    control_others_abundance = 1 - control_bacteroides_abundance - control_firmicutes_abundance
    plt.bar(0, control_bacteroides_abundance, bottom = control_others_abundance + control_firmicutes_abundance, width = 0.5, color = (0,0.5,1), label = 'Bacteroides')
    plt.bar(0, control_firmicutes_abundance, bottom = control_others_abundance, width = 0.5, color = (1,0.5,0), label = 'Firmicutes')
    plt.bar(0, control_others_abundance, width = 0.5, color = 'olivedrab', label = 'Other')
    plt.bar(1, antibiotics_others_abundance, width = 0.5, color = 'olivedrab')
    plt.bar(1, antibiotics_firmicutes_abundance, bottom = antibiotics_others_abundance, width = 0.5, color = (1,0.5,0))
    plt.bar(1, antibiotics_bacteroides_abundance, bottom = antibiotics_others_abundance + antibiotics_firmicutes_abundance, width = 0.5, color = (0,0.5,1))
    plt.xticks([0,1], ['Control', 'Ciprofloxacin'], rotation = 25, ha = 'right')
    plt.ylabel('Abundance', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    l = plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (1,0.75), title = 'Phylum')
    l.get_title().set_fontsize('8')
    plt.xlim(-0.5,1.5)
    plt.subplots_adjust(left = 0.25, bottom = 0.3, right = 0.55, top = 0.98)
    plt.savefig('{}/gut_phylum_abundance_treatment.pdf'.format(data_folder), dpi = 300)


    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3), cm_to_inches(4))
    image_tab_nc = image_tab.loc[image_tab.CELL_NUMBER.values > 500, :]
    treatments = image_tab_nc.TREATMENT.drop_duplicates().sort_values(ascending = False)
    shannon_diversity_list = [image_tab_nc.loc[image_tab_nc.TREATMENT.values == t, 'SHANNON_DIVERSITY'].values for t in treatments]
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    bp = plt.boxplot(shannon_diversity_list, positions = np.arange(2), widths = 0.5, patch_artist = True, flierprops = flierprops)
    for b in bp['boxes']:
        b.set_facecolor('dodgerblue')

    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.xticks(np.arange(2), ['Control', 'Ciprofloxacin'], rotation = 25, ha = 'right')
    plt.ylabel('Shannon Diversity', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.45, bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/gut_shannon_diversity_treatment.pdf'.format(data_folder), dpi = 300)

    # alpha diversity as a function of patch size and treatment
    n_habitat = 100
    image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
    beta_diversity_patch_treatment = []
    for nc in np.arange(10,200,10):
        n_cell = nc
        beta_diversity_treatment = []
        for t in image_tab_sc.TREATMENT.drop_duplicates().values:
            image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
            beta_diversity_image = []
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name))
                barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
                barcode_abundance.columns = ['cell_barcode', 'abundance']
                barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                p_i = barcode_abundance.relative_abundance.values
                p_i = p_i[p_i > 0]
                shannon_diversity = -np.sum(p_i*np.log(p_i))
                alpha_diversity_list = []
                for j in range(n_habitat):
                    reference_point = 2000*np.random.random(2)
                    cell_distance = (cell_info.centroid_x.values - reference_point[0])**2 + (cell_info.centroid_y.values - reference_point[1])**2
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    barcode_abundance = cell_info_sub.cell_barcode.value_counts().reset_index()
                    barcode_abundance.columns = ['cell_barcode', 'abundance']
                    barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                    p_ij = barcode_abundance.relative_abundance.values
                    p_ij = p_ij[p_ij > 0]
                    alpha_diversity = -np.sum(p_ij*np.log(p_ij))
                    alpha_diversity_list.append(alpha_diversity)
                alpha_diversity_patch = np.average(alpha_diversity_list)
                beta_diversity_image.append(shannon_diversity/alpha_diversity_patch)
            beta_diversity_treatment.append(beta_diversity_image)
        beta_diversity_patch_treatment.append(beta_diversity_treatment)

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4), cm_to_inches(4))
    treatments = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
    cmap = cm.get_cmap('coolwarm')
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    meanprops = dict(linestyle='none', linewidth = 0)
    medianprops = dict(linestyle='none', linewidth = 0)
    # plt.plot(beta_diversity_list)
    for i in np.arange(19):
        bp = plt.boxplot(beta_diversity_patch_treatment[i], positions = np.arange(0, 2.2*2, 2.2) - 1 + 0.1*i, patch_artist = True, flierprops = flierprops,
                    widths = 0.1, showcaps = False, showfliers = False, meanline = False, medianprops = medianprops)
        for b in bp['boxes']:
            b.set_facecolor(cmap(i/20))
            b.set_linewidth(0.25)
            b.set_edgecolor((0.4,0.4,0.4))
        for w in bp['whiskers']:
            w.set_linewidth(0.25)
            w.set_color((0.4,0.4,0.4))

    treatment_list = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
    plt.xticks(np.arange(0, 2.2*2, 2.2), ['Control', 'Ciprofloxacin'], rotation = 25, ha = 'right')
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.ylabel(r'$\beta$ Diversity', fontsize = 8, color = theme_color)
    plt.ylim(0.9,1.9)
    plt.xlim(-1.5, 4)
    cbaxes = fig.add_axes([0.6, 0.9, 0.3, 0.04])
    norm = Normalize(vmin = 10, vmax = 200)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize=8, color = theme_color)
    cb.set_ticks([10,100,200])
    cb.set_label(r'$N_{cells}$', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.32, bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/gut_beta_diversity_patch_size_treatment.pdf'.format(data_folder), dpi = 300)

    for nc in [20,40,60,80,100,120,140,160,180,200]:
        n_pairs = 200
        n_cell = nc
        bray_curtis_list = []
        image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
        for t in image_tab_sc.TREATMENT.drop_duplicates().values:
            bray_curtis_tp = []
            image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                bray_curtis_image = np.zeros((n_pairs, 2))
                cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
                for j in range(n_pairs):
                    rp_1 = 2000*np.random.random(2)
                    cell_distance = np.sqrt((cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_1[1])**2)
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_1 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_1_barcode = cell_info_sub_1.cell_barcode.value_counts().reset_index()
                    cell_info_sub_1_barcode.columns = ['cell_barcode', 'abundance_1']
                    rp_2 = 2000*np.random.random(2)
                    cell_distance = np.sqrt((cell_info.centroid_x.values - rp_2[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2)
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_2 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_2_barcode = cell_info_sub_2.cell_barcode.value_counts().reset_index()
                    cell_info_sub_2_barcode.columns = ['cell_barcode', 'abundance_2']
                    cell_info_merge = cell_info_sub_1_barcode.merge(cell_info_sub_2_barcode, on = 'cell_barcode', how = 'outer').fillna(0)
                    coa_1_x = np.sum(cell_info_sub_1.centroid_x.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_1_y = np.sum(cell_info_sub_1.centroid_y.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_2_x = np.sum(cell_info_sub_2.centroid_x.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    coa_2_y = np.sum(cell_info_sub_2.centroid_y.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    bray_curtis_image[j, 0] = np.sqrt((coa_1_x - coa_2_x)**2 + (coa_1_y - coa_2_y)**2)
                    bray_curtis_image[j, 1] = scipy.spatial.distance.braycurtis(cell_info_merge.abundance_1.values, cell_info_merge.abundance_2.values)
                bray_curtis_tp.append(bray_curtis_image)
            bray_curtis_list.append(np.concatenate(bray_curtis_tp, axis = 0))
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(4))
        cmap = cm.get_cmap('tab10')
        color_list = [(0,0.5,1),(1,0.5,0)]
        treatment = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
        label_list = ['Control', 'Ciprofloxacin']
        for i in range(2):
            bcd = bray_curtis_list[i]
            bcd_sorted = bcd[bcd[:,0].argsort()]
            bcd_smoothed = scipy.signal.savgol_filter(bcd_sorted[:,1], window_length = 41, polyorder = 3)
            bcd_smoothed = np.clip(bcd_smoothed, 0, 1)
            plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = color_list[i], label = label_list[i], alpha = 0.8)

        plt.tick_params(direction = 'in', length = 2, labelsize = 8)
        plt.ylim(-0.05, 0.38)
        plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.2,0.38), handlelength = 0.5)
        plt.xlabel('Intra-patch\n' + r'distance [$\mu$m]', fontsize = 8, color = theme_color)
        plt.ylabel('Bray-Curtis Dissimilarity', fontsize = 8, color = theme_color, position = (0.5,0.4))
        plt.subplots_adjust(left = 0.25, bottom = 0.3, right = 0.98, top = 0.98)
        plt.savefig('{}/gut_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300)

    return

def analyze_gut_diversity_clindamycin(image_tab_filename):

    # Shannon diversity for slices from zstacks
    image_tab = pd.read_csv(image_tab_gut_filename)
    for i in range(image_tab.shape[0]):
        sample = image_tab.loc[i, 'SAMPLE']
        image_name = image_tab.loc[i, 'IMAGES']
        z = image_tab.loc[i,'ZSLICE']
        cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab.loc[i,'CELL_NUMBER'] = cell_info.shape[0]

    image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 200, :]
    barcode_abundance_list = []
    treatments = image_tab_sc.TREATMENT.drop_duplicates()
    for t in treatments:
        image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
        sample = image_tab_sub.loc[0, 'SAMPLE']
        image_name = image_tab_sub.loc[0, 'IMAGES']
        z = image_tab_sub.loc[0,'ZSLICE']
        cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z), dtype = {'cell_barcode':str})
        barcode_abundance_full = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_full.columns = ['cell_barcode', 'abundance_0']
        for i in range(1, image_tab_sub.shape[0]):
            sample = image_tab_sub.loc[i, 'SAMPLE']
            image_name = image_tab_sub.loc[i, 'IMAGES']
            z = image_tab_sub.loc[i,'ZSLICE']
            cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z), dtype = {'cell_barcode':str})
            barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
            barcode_abundance.columns = ['cell_barcode', 'abundance_{}'.format(i)]
            barcode_abundance_full = barcode_abundance_full.merge(barcode_abundance, on = 'cell_barcode', how = 'outer')
        barcode_abundance_full = barcode_abundance_full.fillna(0)
        barcode_abundance_full['abundance_all'] = barcode_abundance_full.loc[:,['abundance_{}'.format(i) for i in range(image_tab_sub.shape[0])]].sum(axis = 1)
        barcode_abundance_full['relative_abundance'] = barcode_abundance_full.loc[:,'abundance_all'].values/barcode_abundance_full.abundance_all.sum()
        barcode_abundance_full = barcode_abundance_full.merge(taxa_barcode_sciname, on = 'cell_barcode', how = 'left')
        barcode_abundance_list.append(barcode_abundance_full)

    barcode_abundance_antibiotics = barcode_abundance_list[0].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_antibiotics.columns = ['taxid','abundance']
    barcode_abundance_control = barcode_abundance_list[1].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_control.columns = ['taxid','abundance']

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    antibiotics_bacteroides_abundance = barcode_abundance_antibiotics.loc[barcode_abundance_antibiotics.taxid.values == 976, 'abundance'].values[0]
    antibiotics_firmicutes_abundance = barcode_abundance_antibiotics.loc[barcode_abundance_antibiotics.taxid.values == 1239, 'abundance'].values[0]
    antibiotics_others_abundance = 1 - antibiotics_bacteroides_abundance - antibiotics_firmicutes_abundance
    control_bacteroides_abundance = barcode_abundance_control.loc[barcode_abundance_control.taxid.values == 976, 'abundance'].values[0]
    control_firmicutes_abundance = barcode_abundance_control.loc[barcode_abundance_control.taxid.values == 1239, 'abundance'].values[0]
    control_others_abundance = 1 - control_bacteroides_abundance - control_firmicutes_abundance
    plt.bar(0, control_bacteroides_abundance, bottom = control_others_abundance + control_firmicutes_abundance, width = 0.5, color = (0,0.5,1), label = 'Bacteroides')
    plt.bar(0, control_firmicutes_abundance, bottom = control_others_abundance, width = 0.5, color = (1,0.5,0), label = 'Firmicutes')
    plt.bar(0, control_others_abundance, width = 0.5, color = 'olivedrab', label = 'Other')
    plt.bar(1, antibiotics_others_abundance, width = 0.5, color = 'olivedrab')
    plt.bar(1, antibiotics_firmicutes_abundance, bottom = antibiotics_others_abundance, width = 0.5, color = (1,0.5,0))
    plt.bar(1, antibiotics_bacteroides_abundance, bottom = antibiotics_others_abundance + antibiotics_firmicutes_abundance, width = 0.5, color = (0,0.5,1))
    plt.xticks([0,1], ['Control', 'Clindamycin'], rotation = 25, ha = 'right')
    plt.ylabel('Abundance', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    l = plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (1,0.75), title = 'Phylum')
    l.get_title().set_fontsize('8')
    plt.xlim(-0.5,1.5)
    plt.subplots_adjust(left = 0.25, bottom = 0.3, right = 0.55, top = 0.98)
    plt.savefig('{}/gut_phylum_abundance_treatment_clindamycin.pdf'.format(data_folder), dpi = 300)


    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3), cm_to_inches(4))
    image_tab_nc = image_tab.loc[image_tab.CELL_NUMBER.values > 200, :]
    treatments = image_tab_nc.TREATMENT.drop_duplicates().sort_values(ascending = False)
    shannon_diversity_list = [image_tab_nc.loc[image_tab_nc.TREATMENT.values == t, 'SHANNON_DIVERSITY'].values for t in treatments]
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    bp = plt.boxplot(shannon_diversity_list, positions = np.arange(2), widths = 0.5, patch_artist = True, flierprops = flierprops)
    for b in bp['boxes']:
        b.set_facecolor('dodgerblue')

    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.xticks(np.arange(2), ['Control', 'Clindamycin'], rotation = 25, ha = 'right')
    plt.ylabel('Shannon Diversity', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.45, bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/gut_shannon_diversity_treatment_clindamycin.pdf'.format(data_folder), dpi = 300)

    # alpha diversity as a function of patch size and treatment
    n_habitat = 100
    image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 200, :]
    beta_diversity_patch_treatment = []
    for nc in np.arange(10,200,10):
        n_cell = nc
        beta_diversity_treatment = []
        for t in image_tab_sc.TREATMENT.drop_duplicates().values:
            image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
            beta_diversity_image = []
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                z = image_tab_sub.loc[i, 'ZSLICE']
                cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z))
                barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
                barcode_abundance.columns = ['cell_barcode', 'abundance']
                barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                p_i = barcode_abundance.relative_abundance.values
                p_i = p_i[p_i > 0]
                shannon_diversity = -np.sum(p_i*np.log(p_i))
                alpha_diversity_list = []
                for j in range(n_habitat):
                    reference_point = 2000*np.random.random(2)
                    cell_distance = (cell_info.centroid_x.values - reference_point[0])**2 + (cell_info.centroid_y.values - reference_point[1])**2
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    barcode_abundance = cell_info_sub.cell_barcode.value_counts().reset_index()
                    barcode_abundance.columns = ['cell_barcode', 'abundance']
                    barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                    p_ij = barcode_abundance.relative_abundance.values
                    p_ij = p_ij[p_ij > 0]
                    alpha_diversity = -np.sum(p_ij*np.log(p_ij))
                    alpha_diversity_list.append(alpha_diversity)
                alpha_diversity_patch = np.average(alpha_diversity_list)
                beta_diversity_image.append(shannon_diversity/alpha_diversity_patch)
            beta_diversity_treatment.append(beta_diversity_image)
        beta_diversity_patch_treatment.append(beta_diversity_treatment)

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4), cm_to_inches(4))
    treatments = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
    cmap = cm.get_cmap('coolwarm')
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    meanprops = dict(linestyle='none', linewidth = 0)
    medianprops = dict(linestyle='none', linewidth = 0)
    # plt.plot(beta_diversity_list)
    for i in np.arange(19):
        bp = plt.boxplot(beta_diversity_patch_treatment[i], positions = np.arange(0, 2.2*2, 2.2) - 1 + 0.1*i, patch_artist = True, flierprops = flierprops,
                    widths = 0.1, showcaps = False, showfliers = False, meanline = False, medianprops = medianprops)
        for b in bp['boxes']:
            b.set_facecolor(cmap(i/20))
            b.set_linewidth(0.25)
            b.set_edgecolor((0.4,0.4,0.4))
        for w in bp['whiskers']:
            w.set_linewidth(0.25)
            w.set_color((0.4,0.4,0.4))

    treatment_list = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
    plt.xticks(np.arange(0, 2.2*2, 2.2), ['Control', 'Clindamycin'], rotation = 25, ha = 'right')
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.ylabel(r'$\beta$ Diversity', fontsize = 8, color = theme_color)
    plt.ylim(0.9,1.9)
    plt.xlim(-1.5, 4)
    cbaxes = fig.add_axes([0.6, 0.9, 0.3, 0.04])
    norm = Normalize(vmin = 10, vmax = 200)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize=8, color = theme_color)
    cb.set_ticks([10,100,200])
    cb.set_label(r'$N_{cells}$', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.32, bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/gut_beta_diversity_patch_size_treatment_clindamycin.pdf'.format(data_folder), dpi = 300)

    for nc in [20,40,60,80,100,120,140,160,180,200]:
        n_pairs = 200
        n_cell = nc
        bray_curtis_list = []
        image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 200, :]
        for t in image_tab_sc.TREATMENT.drop_duplicates().values:
            bray_curtis_tp = []
            image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                z = image_tab_sub.loc[i, 'ZSLICE']
                bray_curtis_image = np.zeros((n_pairs, 2))
                cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z), dtype = {'cell_barcode':str})
                for j in range(n_pairs):
                    rp_1 = 2000*np.random.random(2)
                    cell_distance = np.sqrt((cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_1[1])**2)
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_1 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_1_barcode = cell_info_sub_1.cell_barcode.value_counts().reset_index()
                    cell_info_sub_1_barcode.columns = ['cell_barcode', 'abundance_1']
                    rp_2 = 2000*np.random.random(2)
                    cell_distance = np.sqrt((cell_info.centroid_x.values - rp_2[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2)
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_2 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_2_barcode = cell_info_sub_2.cell_barcode.value_counts().reset_index()
                    cell_info_sub_2_barcode.columns = ['cell_barcode', 'abundance_2']
                    cell_info_merge = cell_info_sub_1_barcode.merge(cell_info_sub_2_barcode, on = 'cell_barcode', how = 'outer').fillna(0)
                    coa_1_x = np.sum(cell_info_sub_1.centroid_x.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_1_y = np.sum(cell_info_sub_1.centroid_y.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_2_x = np.sum(cell_info_sub_2.centroid_x.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    coa_2_y = np.sum(cell_info_sub_2.centroid_y.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    bray_curtis_image[j, 0] = np.sqrt((coa_1_x - coa_2_x)**2 + (coa_1_y - coa_2_y)**2)
                    bray_curtis_image[j, 1] = scipy.spatial.distance.braycurtis(cell_info_merge.abundance_1.values, cell_info_merge.abundance_2.values)
                bray_curtis_tp.append(bray_curtis_image)
            bray_curtis_list.append(np.concatenate(bray_curtis_tp, axis = 0))
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(4))
        cmap = cm.get_cmap('tab10')
        color_list = [(0,0.5,1),(1,0.5,0)]
        treatment = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
        label_list = ['Control', 'Clindamycin']
        for i in range(2):
            bcd = bray_curtis_list[i]
            bcd_sorted = bcd[bcd[:,0].argsort()]
            bcd_smoothed = scipy.signal.savgol_filter(bcd_sorted[:,1], window_length = 41, polyorder = 3)
            bcd_smoothed = np.clip(bcd_smoothed, 0, 1)
            plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = color_list[i], label = label_list[i], alpha = 0.8)

        plt.tick_params(direction = 'in', length = 2, labelsize = 8)
        # plt.ylim(-0.05, 0.5)
        plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.35,0.38), handlelength = 0.5)
        plt.xlabel('Intra-patch\n' + r'distance [$\mu$m]', fontsize = 8, color = theme_color)
        plt.ylabel('Bray-Curtis Dissimilarity', fontsize = 8, color = theme_color, position = (0.5,0.4))
        plt.subplots_adjust(left = 0.25, bottom = 0.3, right = 0.98, top = 0.98)
        plt.savefig('{}/gut_bray_curtis_nc_{}_clindamycin.pdf'.format(data_folder, n_cell), dpi = 300)

    return

def analyze_gut_diversity_ciprofloxacin_clindamycin(image_tab_filename):
    data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging'
    taxa_barcode_sciname_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_1/simulation/DSGN0567/DSGN0567_primerset_B_barcode_selection_MostSimple_taxa_barcode_sciname.csv'
    taxa_barcode_sciname = pd.read_csv(taxa_barcode_sciname_filename, dtype = {'cell_barcode':str})
    image_tab_cipro_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_microbiome_5.csv'
    image_tab_clinda_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_microbiome_6.csv'
    # Shannon diversity ciprofloxacin
    image_tab_cipro = pd.read_csv(image_tab_cipro_filename)
    for i in range(image_tab_cipro.shape[0]):
        sample = image_tab_cipro.loc[i, 'SAMPLE']
        image_name = image_tab_cipro.loc[i, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab_cipro.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab_cipro.loc[i,'CELL_NUMBER'] = cell_info.shape[0]

    # Shannon diversity clindamycin
    image_tab_clinda = pd.read_csv(image_tab_clinda_filename)
    for i in range(image_tab_clinda.shape[0]):
        sample = image_tab_clinda.loc[i, 'SAMPLE']
        image_name = image_tab_clinda.loc[i, 'IMAGES']
        z = image_tab_clinda.loc[i,'ZSLICE']
        cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab_clinda.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab_clinda.loc[i,'CELL_NUMBER'] = cell_info.shape[0]

    image_tab_cipro_sc = image_tab_cipro.loc[image_tab_cipro.CELL_NUMBER.values >= 500, :]
    barcode_abundance_list = []
    treatments = image_tab_cipro_sc.TREATMENT.drop_duplicates()
    for t in treatments:
        image_tab_sub = image_tab_cipro_sc.loc[image_tab_cipro_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
        sample = image_tab_sub.loc[0, 'SAMPLE']
        image_name = image_tab_sub.loc[0, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        barcode_abundance_full = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_full.columns = ['cell_barcode', 'abundance_0']
        for i in range(1, image_tab_sub.shape[0]):
            sample = image_tab_sub.loc[i, 'SAMPLE']
            image_name = image_tab_sub.loc[i, 'IMAGES']
            cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
            barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
            barcode_abundance.columns = ['cell_barcode', 'abundance_{}'.format(i)]
            barcode_abundance_full = barcode_abundance_full.merge(barcode_abundance, on = 'cell_barcode', how = 'outer')
        barcode_abundance_full = barcode_abundance_full.fillna(0)
        barcode_abundance_full['abundance_all'] = barcode_abundance_full.loc[:,['abundance_{}'.format(i) for i in range(image_tab_sub.shape[0])]].sum(axis = 1)
        barcode_abundance_full['relative_abundance'] = barcode_abundance_full.loc[:,'abundance_all'].values/barcode_abundance_full.abundance_all.sum()
        barcode_abundance_full = barcode_abundance_full.merge(taxa_barcode_sciname, on = 'cell_barcode', how = 'left')
        barcode_abundance_list.append(barcode_abundance_full)

    barcode_abundance_ciprofloxacin = barcode_abundance_list[0].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_ciprofloxacin.columns = ['taxid','abundance']
    barcode_abundance_cipro_control = barcode_abundance_list[1].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_cipro_control.columns = ['taxid','abundance']

    image_tab_clinda_sc = image_tab_clinda.loc[image_tab_clinda.CELL_NUMBER.values >= 200, :]
    barcode_abundance_list = []
    treatments = image_tab_clinda_sc.TREATMENT.drop_duplicates()
    for t in treatments:
        image_tab_sub = image_tab_clinda_sc.loc[image_tab_clinda_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
        sample = image_tab_sub.loc[0, 'SAMPLE']
        image_name = image_tab_sub.loc[0, 'IMAGES']
        z = image_tab_sub.loc[0, 'ZSLICE']
        cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z), dtype = {'cell_barcode':str})
        barcode_abundance_full = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_full.columns = ['cell_barcode', 'abundance_0']
        for i in range(1, image_tab_sub.shape[0]):
            sample = image_tab_sub.loc[i, 'SAMPLE']
            image_name = image_tab_sub.loc[i, 'IMAGES']
            z = image_tab_sub.loc[i, 'ZSLICE']
            cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name, z), dtype = {'cell_barcode':str})
            barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
            barcode_abundance.columns = ['cell_barcode', 'abundance_{}'.format(i)]
            barcode_abundance_full = barcode_abundance_full.merge(barcode_abundance, on = 'cell_barcode', how = 'outer')
        barcode_abundance_full = barcode_abundance_full.fillna(0)
        barcode_abundance_full['abundance_all'] = barcode_abundance_full.loc[:,['abundance_{}'.format(i) for i in range(image_tab_sub.shape[0])]].sum(axis = 1)
        barcode_abundance_full['relative_abundance'] = barcode_abundance_full.loc[:,'abundance_all'].values/barcode_abundance_full.abundance_all.sum()
        barcode_abundance_full = barcode_abundance_full.merge(taxa_barcode_sciname, on = 'cell_barcode', how = 'left')
        barcode_abundance_list.append(barcode_abundance_full)

    barcode_abundance_clindamycin = barcode_abundance_list[0].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_clindamycin.columns = ['taxid','abundance']
    barcode_abundance_clinda_control = barcode_abundance_list[1].groupby('phylum')['relative_abundance'].agg('sum').reset_index()
    barcode_abundance_clinda_control.columns = ['taxid','abundance']

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    cipro_bacteroides_abundance = barcode_abundance_ciprofloxacin.loc[barcode_abundance_ciprofloxacin.taxid.values == 976, 'abundance'].values[0]
    cipro_firmicutes_abundance = barcode_abundance_ciprofloxacin.loc[barcode_abundance_ciprofloxacin.taxid.values == 1239, 'abundance'].values[0]
    cipro_others_abundance = 1 - cipro_bacteroides_abundance - cipro_firmicutes_abundance
    cipro_control_bacteroides_abundance = barcode_abundance_cipro_control.loc[barcode_abundance_cipro_control.taxid.values == 976, 'abundance'].values[0]
    cipro_control_firmicutes_abundance = barcode_abundance_cipro_control.loc[barcode_abundance_cipro_control.taxid.values == 1239, 'abundance'].values[0]
    cipro_control_others_abundance = 1 - cipro_control_bacteroides_abundance - cipro_control_firmicutes_abundance
    clinda_bacteroides_abundance = barcode_abundance_clindamycin.loc[barcode_abundance_clindamycin.taxid.values == 976, 'abundance'].values[0]
    clinda_firmicutes_abundance = barcode_abundance_clindamycin.loc[barcode_abundance_clindamycin.taxid.values == 1239, 'abundance'].values[0]
    clinda_others_abundance = 1 - clinda_bacteroides_abundance - clinda_firmicutes_abundance
    clinda_control_bacteroides_abundance = barcode_abundance_clinda_control.loc[barcode_abundance_clinda_control.taxid.values == 976, 'abundance'].values[0]
    clinda_control_firmicutes_abundance = barcode_abundance_clinda_control.loc[barcode_abundance_clinda_control.taxid.values == 1239, 'abundance'].values[0]
    clinda_control_others_abundance = 1 - clinda_control_bacteroides_abundance - clinda_control_firmicutes_abundance
    plt.bar(0, cipro_control_bacteroides_abundance, bottom = cipro_control_others_abundance + cipro_control_firmicutes_abundance, width = 0.5, color = (0,0.5,1), label = 'Bacteroidetes')
    plt.bar(0, cipro_control_firmicutes_abundance, bottom = cipro_control_others_abundance, width = 0.5, color = (1,0.5,0), label = 'Firmicutes')
    plt.bar(0, cipro_control_others_abundance, width = 0.5, color = 'olivedrab', label = 'Other')
    plt.bar(1, cipro_bacteroides_abundance, bottom = cipro_others_abundance + cipro_firmicutes_abundance, width = 0.5, color = (0,0.5,1))
    plt.bar(1, cipro_firmicutes_abundance, bottom = cipro_others_abundance, width = 0.5, color = (1,0.5,0))
    plt.bar(1, cipro_others_abundance, width = 0.5, color = 'olivedrab')
    plt.bar(2, clinda_control_bacteroides_abundance, bottom = clinda_control_others_abundance + clinda_control_firmicutes_abundance, width = 0.5, color = (0,0.5,1))
    plt.bar(2, clinda_control_firmicutes_abundance, bottom = clinda_control_others_abundance, width = 0.5, color = (1,0.5,0))
    plt.bar(2, clinda_control_others_abundance, width = 0.5, color = 'olivedrab')
    plt.bar(3, clinda_bacteroides_abundance, bottom = clinda_others_abundance + clinda_firmicutes_abundance, width = 0.5, color = (0,0.5,1))
    plt.bar(3, clinda_firmicutes_abundance, bottom = clinda_others_abundance, width = 0.5, color = (1,0.5,0))
    plt.bar(3, clinda_others_abundance, width = 0.5, color = 'olivedrab')
    plt.xticks([0,1,2,3], ['Control', 'Ciprofloxacin', 'Control', 'Clindamycin'], rotation = 30, ha = 'right', fontsize = 8)
    plt.ylabel('Abundance', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    l = plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (1,0.75), title = 'Phylum', handlelength = 1, handletextpad = 0.25)
    l.get_title().set_fontsize('6')
    plt.xlim(-0.5,3.5)
    plt.subplots_adjust(left = 0.2, bottom = 0.3, right = 0.67, top = 0.98)
    plt.savefig('{}/gut_phylum_abundance_ciprofloxacin_clindamycin.pdf'.format(data_folder), dpi = 300, transparent = True)


    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3), cm_to_inches(4))
    image_tab_nc = image_tab.loc[image_tab.CELL_NUMBER.values > 500, :]
    treatments = image_tab_nc.TREATMENT.drop_duplicates().sort_values(ascending = False)
    shannon_diversity_list = [image_tab_nc.loc[image_tab_nc.TREATMENT.values == t, 'SHANNON_DIVERSITY'].values for t in treatments]
    flierprops = dict(marker='.', markerfacecolor=(0.5,0,0), markersize=4, linestyle='none', markeredgewidth = 0)
    bp = plt.boxplot(shannon_diversity_list, positions = np.arange(2), widths = 0.5, patch_artist = True, flierprops = flierprops)
    for b in bp['boxes']:
        b.set_facecolor('dodgerblue')

    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.xticks(np.arange(2), ['Control', 'Ciprofloxacin'], rotation = 25, ha = 'right')
    plt.ylabel('Shannon Diversity', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.45, bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/gut_shannon_diversity_treatment.pdf'.format(data_folder), dpi = 300)

    # alpha diversity as a function of patch size and treatment
    n_habitat = 100
    image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
    beta_diversity_patch_treatment = []
    for nc in np.arange(10,200,10):
        n_cell = nc
        beta_diversity_treatment = []
        for t in image_tab_sc.TREATMENT.drop_duplicates().values:
            image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
            beta_diversity_image = []
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name))
                barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
                barcode_abundance.columns = ['cell_barcode', 'abundance']
                barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                p_i = barcode_abundance.relative_abundance.values
                p_i = p_i[p_i > 0]
                shannon_diversity = -np.sum(p_i*np.log(p_i))
                alpha_diversity_list = []
                for j in range(n_habitat):
                    reference_point = 2000*np.random.random(2)
                    cell_distance = (cell_info.centroid_x.values - reference_point[0])**2 + (cell_info.centroid_y.values - reference_point[1])**2
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    barcode_abundance = cell_info_sub.cell_barcode.value_counts().reset_index()
                    barcode_abundance.columns = ['cell_barcode', 'abundance']
                    barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
                    p_ij = barcode_abundance.relative_abundance.values
                    p_ij = p_ij[p_ij > 0]
                    alpha_diversity = -np.sum(p_ij*np.log(p_ij))
                    alpha_diversity_list.append(alpha_diversity)
                alpha_diversity_patch = np.average(alpha_diversity_list)
                beta_diversity_image.append(shannon_diversity/alpha_diversity_patch)
            beta_diversity_treatment.append(beta_diversity_image)
        beta_diversity_patch_treatment.append(beta_diversity_treatment)

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4), cm_to_inches(4))
    treatments = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
    cmap = cm.get_cmap('coolwarm')
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    meanprops = dict(linestyle='none', linewidth = 0)
    medianprops = dict(linestyle='none', linewidth = 0)
    # plt.plot(beta_diversity_list)
    for i in np.arange(19):
        bp = plt.boxplot(beta_diversity_patch_treatment[i], positions = np.arange(0, 2.2*2, 2.2) - 1 + 0.1*i, patch_artist = True, flierprops = flierprops,
                    widths = 0.1, showcaps = False, showfliers = False, meanline = False, medianprops = medianprops)
        for b in bp['boxes']:
            b.set_facecolor(cmap(i/20))
            b.set_linewidth(0.25)
            b.set_edgecolor((0.4,0.4,0.4))
        for w in bp['whiskers']:
            w.set_linewidth(0.25)
            w.set_color((0.4,0.4,0.4))

    treatment_list = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
    plt.xticks(np.arange(0, 2.2*2, 2.2), ['Control', 'Ciprofloxacin'], rotation = 25, ha = 'right')
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.ylabel(r'$\beta$ Diversity', fontsize = 8, color = theme_color)
    plt.ylim(0.9,1.9)
    plt.xlim(-1.5, 4)
    cbaxes = fig.add_axes([0.6, 0.9, 0.3, 0.04])
    norm = Normalize(vmin = 10, vmax = 200)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize=8, color = theme_color)
    cb.set_ticks([10,100,200])
    cb.set_label(r'$N_{cells}$', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.32, bottom = 0.3, right = 0.98, top = 0.98)
    plt.savefig('{}/gut_beta_diversity_patch_size_treatment.pdf'.format(data_folder), dpi = 300)

    for nc in [20,40,60,80,100,120,140,160,180,200]:
        n_pairs = 200
        n_cell = nc
        bray_curtis_list = []
        image_tab_sc = image_tab.loc[image_tab.CELL_NUMBER.values >= 500, :]
        for t in image_tab_sc.TREATMENT.drop_duplicates().values:
            bray_curtis_tp = []
            image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
            for i in range(image_tab_sub.shape[0]):
                sample = image_tab_sub.loc[i, 'SAMPLE']
                image_name = image_tab_sub.loc[i, 'IMAGES']
                bray_curtis_image = np.zeros((n_pairs, 2))
                cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
                for j in range(n_pairs):
                    rp_1 = 2000*np.random.random(2)
                    cell_distance = np.sqrt((cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_1[1])**2)
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_1 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_1_barcode = cell_info_sub_1.cell_barcode.value_counts().reset_index()
                    cell_info_sub_1_barcode.columns = ['cell_barcode', 'abundance_1']
                    rp_2 = 2000*np.random.random(2)
                    cell_distance = np.sqrt((cell_info.centroid_x.values - rp_2[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2)
                    cell_distance_index = np.argsort(cell_distance)
                    cell_info_sub_2 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                    cell_info_sub_2_barcode = cell_info_sub_2.cell_barcode.value_counts().reset_index()
                    cell_info_sub_2_barcode.columns = ['cell_barcode', 'abundance_2']
                    cell_info_merge = cell_info_sub_1_barcode.merge(cell_info_sub_2_barcode, on = 'cell_barcode', how = 'outer').fillna(0)
                    coa_1_x = np.sum(cell_info_sub_1.centroid_x.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_1_y = np.sum(cell_info_sub_1.centroid_y.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                    coa_2_x = np.sum(cell_info_sub_2.centroid_x.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    coa_2_y = np.sum(cell_info_sub_2.centroid_y.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                    bray_curtis_image[j, 0] = np.sqrt((coa_1_x - coa_2_x)**2 + (coa_1_y - coa_2_y)**2)
                    bray_curtis_image[j, 1] = scipy.spatial.distance.braycurtis(cell_info_merge.abundance_1.values, cell_info_merge.abundance_2.values)
                bray_curtis_tp.append(bray_curtis_image)
            bray_curtis_list.append(np.concatenate(bray_curtis_tp, axis = 0))
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(4))
        cmap = cm.get_cmap('tab10')
        color_list = [(0,0.5,1),(1,0.5,0)]
        treatment = image_tab.TREATMENT.drop_duplicates().sort_values(ascending = False)
        label_list = ['Control', 'Ciprofloxacin']
        for i in range(2):
            bcd = bray_curtis_list[i]
            bcd_sorted = bcd[bcd[:,0].argsort()]
            bcd_smoothed = scipy.signal.savgol_filter(bcd_sorted[:,1], window_length = 41, polyorder = 3)
            bcd_smoothed = np.clip(bcd_smoothed, 0, 1)
            plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = color_list[i], label = label_list[i], alpha = 0.8)

        plt.tick_params(direction = 'in', length = 2, labelsize = 8)
        plt.ylim(-0.05, 0.38)
        plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.2,0.38), handlelength = 0.5)
        plt.xlabel('Intra-patch\n' + r'distance [$\mu$m]', fontsize = 8, color = theme_color)
        plt.ylabel('Bray-Curtis Dissimilarity', fontsize = 8, color = theme_color, position = (0.5,0.4))
        plt.subplots_adjust(left = 0.25, bottom = 0.3, right = 0.98, top = 0.98)
        plt.savefig('{}/gut_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300)

    return


def compare_oral_gut_bray_curtis():
    image_tab_oral_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_microbiome_2.csv'
    image_tab_oral = pd.read_csv(image_tab_oral_filename)
    for i in range(image_tab_oral.shape[0]):
        sample = image_tab_oral.loc[i, 'SAMPLE']
        image_name = image_tab_oral.loc[i, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab_oral.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab_oral.loc[i,'CELL_NUMBER'] = cell_info.shape[0]
    n_pairs = 200
    n_cell = nc
    oral_bray_curtis_list = []
    image_tab_sc = image_tab_oral.loc[image_tab_oral.CELL_NUMBER.values >= 500, :]
    for t in image_tab_sc.SAMPLING_TIME.drop_duplicates().values:
        bray_curtis_tp = []
        image_tab_sub = image_tab_sc.loc[image_tab_sc.SAMPLING_TIME.values == t, :].reset_index().drop(columns = 'index')
        for i in range(image_tab_sub.shape[0]):
            sample = image_tab_sub.loc[i, 'SAMPLE']
            image_name = image_tab_sub.loc[i, 'IMAGES']
            bray_curtis_image = np.zeros((n_pairs, 2))
            cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name))
            for j in range(n_pairs):
                rp_1 = 2000*np.random.random(2)
                cell_distance = (cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2
                cell_distance_index = np.argsort(cell_distance)
                cell_info_sub_1 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                cell_info_sub_1_barcode = cell_info_sub_1.cell_barcode.value_counts().reset_index()
                cell_info_sub_1_barcode.columns = ['abundance_1', 'cell_barcode']
                rp_2 = 2000*np.random.random(2)
                cell_distance = (cell_info.centroid_x.values - rp_2[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2
                cell_distance_index = np.argsort(cell_distance)
                cell_info_sub_2 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                cell_info_sub_2_barcode = cell_info_sub_2.cell_barcode.value_counts().reset_index()
                cell_info_sub_2_barcode.columns = ['abundance_2', 'cell_barcode']
                cell_info_merge = cell_info_sub_1_barcode.merge(cell_info_sub_2_barcode, on = 'cell_barcode', how = 'outer').fillna(0)
                coa_1_x = np.sum(cell_info_sub_1.centroid_x.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                coa_1_y = np.sum(cell_info_sub_1.centroid_y.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                coa_2_x = np.sum(cell_info_sub_2.centroid_x.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                coa_2_y = np.sum(cell_info_sub_2.centroid_y.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                bray_curtis_image[j, 0] = np.sqrt((coa_1_x - coa_2_x)**2 + (coa_1_y - coa_2_y)**2)
                bray_curtis_image[j, 1] = scipy.spatial.distance.braycurtis(cell_info_merge.abundance_1.values, cell_info_merge.abundance_2.values)
            bray_curtis_tp.append(bray_curtis_image)
        oral_bray_curtis_list.append(np.concatenate(bray_curtis_tp, axis = 0))
    image_tab_gut_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_microbiome_analysis_5.csv'
    image_tab_gut = pd.read_csv(image_tab_gut_filename)
    for i in range(image_tab_gut.shape[0]):
        sample = image_tab_gut.loc[i, 'SAMPLE']
        image_name = image_tab_gut.loc[i, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name))
        barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance.columns = ['cell_barcode', 'abundance']
        barcode_abundance['relative_abundance'] = barcode_abundance.abundance.values/barcode_abundance.abundance.sum()
        p = barcode_abundance.relative_abundance.values
        p = p[p > 0]
        shannon_diversity = -np.sum(p*np.log(p))
        image_tab_gut.loc[i,'SHANNON_DIVERSITY'] = shannon_diversity
        image_tab_gut.loc[i,'CELL_NUMBER'] = cell_info.shape[0]
    n_pairs = 200
    n_cell = nc
    gut_bray_curtis_list = []
    image_tab_sc = image_tab_gut.loc[image_tab_gut.CELL_NUMBER.values >= 500, :]
    for t in image_tab_sc.TREATMENT.drop_duplicates().values:
        bray_curtis_tp = []
        image_tab_sub = image_tab_sc.loc[image_tab_sc.TREATMENT.values == t, :].reset_index().drop(columns = 'index')
        for i in range(image_tab_sub.shape[0]):
            sample = image_tab_sub.loc[i, 'SAMPLE']
            image_name = image_tab_sub.loc[i, 'IMAGES']
            bray_curtis_image = np.zeros((n_pairs, 2))
            cell_info = pd.read_csv('{}/{}/{}_cell_information_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
            for j in range(n_pairs):
                rp_1 = 2000*np.random.random(2)
                cell_distance = np.sqrt((cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_1[1])**2)
                cell_distance_index = np.argsort(cell_distance)
                cell_info_sub_1 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                cell_info_sub_1_barcode = cell_info_sub_1.cell_barcode.value_counts().reset_index()
                cell_info_sub_1_barcode.columns = ['cell_barcode', 'abundance_1']
                rp_2 = 2000*np.random.random(2)
                cell_distance = np.sqrt((cell_info.centroid_x.values - rp_2[0])**2 + (cell_info.centroid_y.values - rp_2[1])**2)
                cell_distance_index = np.argsort(cell_distance)
                cell_info_sub_2 = cell_info.iloc[cell_distance_index[0:n_cell], :]
                cell_info_sub_2_barcode = cell_info_sub_2.cell_barcode.value_counts().reset_index()
                cell_info_sub_2_barcode.columns = ['cell_barcode', 'abundance_2']
                cell_info_merge = cell_info_sub_1_barcode.merge(cell_info_sub_2_barcode, on = 'cell_barcode', how = 'outer').fillna(0)
                coa_1_x = np.sum(cell_info_sub_1.centroid_x.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                coa_1_y = np.sum(cell_info_sub_1.centroid_y.values*cell_info_sub_1.area.values)/cell_info_sub_1.area.sum()
                coa_2_x = np.sum(cell_info_sub_2.centroid_x.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                coa_2_y = np.sum(cell_info_sub_2.centroid_y.values*cell_info_sub_2.area.values)/cell_info_sub_2.area.sum()
                bray_curtis_image[j, 0] = np.sqrt((coa_1_x - coa_2_x)**2 + (coa_1_y - coa_2_y)**2)
                bray_curtis_image[j, 1] = scipy.spatial.distance.braycurtis(cell_info_merge.abundance_1.values, cell_info_merge.abundance_2.values)
            bray_curtis_tp.append(bray_curtis_image)
        gut_bray_curtis_list.append(np.concatenate(bray_curtis_tp, axis = 0))

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(3.5))
    cmap = cm.get_cmap('tab10')
    color_list = [cmap(i/7) for i in range(7)]
    time_points = image_tab_oral.SAMPLING_TIME.drop_duplicates()
    day_list = image_tab_oral.SAMPLING_DAY_REFERENCE.drop_duplicates().values
    label_list = ['M{}'.format(int(t/30)) for t in day_list]
    for i in range(7):
        bcd = oral_bray_curtis_list[i]
        bcd_sorted = bcd[bcd[:,0].argsort()]
        bcd_smoothed = scipy.signal.savgol_filter(bcd_sorted[:,1], window_length = 41, polyorder = 3)
        bcd_smoothed = np.clip(bcd_smoothed, 0, 1)
        plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = color_list[i], label = label_list[i], alpha = 0.5)

    for i in range(2):
        bcd = gut_bray_curtis_list[i]
        bcd_sorted = bcd[bcd[:,0].argsort()]
        bcd_smoothed = scipy.signal.savgol_filter(bcd_sorted[:,1], window_length = 41, polyorder = 3)
        bcd_smoothed = np.clip(bcd_smoothed, 0, 1)
        if i == 1:
            plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = (0.5,0.5,0.5), alpha = 0.5)
        else:
            plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = (0.5,0.5,0.5), label = 'mouse gut', alpha = 0.5)

    plt.ylim(-0.02, 1.2)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.05,0.72,0.8,0.2), ncol = 4, handlelength = 0.25, columnspacing = 0.5, handletextpad = 0.25)
    plt.xlabel(r'Intra-patch distance [$\mu$m]', fontsize = 8, color = theme_color, labelpad = 0)
    plt.ylabel('Bray-Curtis', fontsize = 8, color = theme_color, labelpad = 0)
    plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300)
    plt.close()

def measure_spatial_association_vs_abundance_by_treatment(sam_tab, taxon_lookup, data_folder):
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    taxon_lookup = taxon_lookup.rename(columns = {'code':'cell_barcode'})
    cipro_samples = sam_tab.loc[sam_tab.TREATMENT.values == 'ANTIBIOTICS',:]
    sample = cipro_samples.loc[0, 'SAMPLE']
    image_name = cipro_samples.loc[0, 'IMAGES']
    adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(cipro_samples)))
    association_matrix_fold_change_random = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(cipro_samples)))
    cell_info_cipro_all = []
    for s in range(len(cipro_samples)):
        print(s)
        image_seg = np.load('{}/{}/{}_seg.npy'.format(data_folder, sample, image_name))
        adjacency_seg = np.load('{}/{}/{}_adjacency_seg.npy'.format(data_folder, sample, image_name))
        image_identification = np.load('{}/{}/{}_identification.npy'.format(data_folder, sample, image_name))
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        cell_info_filtered = cell_info.loc[cell_info.type.values == 'cell',:]
        cell_info_cipro_all.append(cell_info_filtered)
        amrf_filename = '{}/{}/{}_adjacency_matrix_random_full.npy'.format(data_folder, sample, image_name)
        if not os.path.exists(amrf_filename):
            adjacency_matrix_random = []
            for k in range(500):
                adjacency_matrix_random_k = generate_random_spatial_adjacency_network(image_seg, adjacency_seg, cell_info_filtered, taxa_barcode_sciname)
                adjacency_matrix_random.append(adjacency_matrix_random_k)
            adjacency_matrix_random_full = dask.delayed(np.stack)(adjacency_matrix_random, axis = 2).compute()
            np.save(amrf_filename, adjacency_matrix_random_full)
        else:
            adjacency_matrix_random_full = np.load(amrf_filename)
        adjacency_matrix_random_avg = np.average(adjacency_matrix_random_full, axis = 2)
        adjacency_matrix_random_std = np.std(adjacency_matrix_random_full, axis = 2)
        association_matrix_fold_change_random[:,:,s] = (adjacency_matrix_filtered.values + 1)/(adjacency_matrix_random_avg + 1)

    control_samples = sam_tab.loc[sam_tab.TREATMENT.values == 'CONTROL', :].reset_index().drop(columns = 'index')
    sample = cipro_samples.loc[0, 'SAMPLE']
    image_name = cipro_samples.loc[0, 'IMAGES']
    adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_control_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(control_samples)))
    association_matrix_fold_change_control_random = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(control_samples)))
    cell_info_control_all = []
    for s in range(len(control_samples)):
        print(s)
        sample = control_samples.loc[s,'SAMPLE']
        image_name = control_samples.loc[s,'IMAGES']
        image_seg = np.load('{}/{}/{}_seg.npy'.format(data_folder, sample, image_name))
        adjacency_seg = np.load('{}/{}/{}_adjacency_seg.npy'.format(data_folder, sample, image_name))
        image_identification = np.load('{}/{}/{}_identification.npy'.format(data_folder, sample, image_name))
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_control_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        cell_info_filtered = cell_info.loc[cell_info.type.values == 'cell',:]
        cell_info_control_all.append(cell_info_filtered)
        amcrf_filename = '{}/{}/{}_adjacency_matrix_control_random_full.npy'.format(data_folder, sample, image_name)
        if not os.path.exists(amcrf_filename):
            adjacency_matrix_control_random = []
            for k in range(500):
                adjacency_matrix_control_random_k = generate_random_spatial_adjacency_network(image_seg, adjacency_seg, cell_info_filtered, taxa_barcode_sciname)
                adjacency_matrix_control_random.append(adjacency_matrix_control_random_k)
            adjacency_matrix_control_random_full = dask.delayed(np.stack)(adjacency_matrix_control_random, axis = 2).compute()
            np.save('{}/{}/{}_adjacency_matrix_control_random_full.npy'.format(data_folder, sample, image_name), adjacency_matrix_control_random_full)
        else:
            adjacency_matrix_control_random_full = np.load(amcrf_filename)
        adjacency_matrix_control_random_avg = np.average(adjacency_matrix_control_random_full, axis = 2)
        # adjacency_matrix_control_random_std = np.std(adjacency_matrix_control_random_full, axis = 2)
        association_matrix_fold_change_control_random[:,:,s] = (adjacency_matrix_filtered.values + 1)/(adjacency_matrix_control_random_avg + 1)


        # for i in range(len(taxa_list)):
        #     for j in range(len(taxa_list)):
        #         taxa_i = taxa_list[i]
        #         taxa_j = taxa_list[j]
        #         abundance_i = cell_info.loc[cell_info.cell_barcode.values == taxa_i, :].shape[0]
        #         abundance_j = cell_info.loc[cell_info.cell_barcode.values == taxa_j, :].shape[0]
        #         all_possible_association = abundance_i*abundance_j
        #         if all_possible_association > 0:
        #             association_probability_matrix[i,j,s] = adjacency_matrix_filtered.loc[taxa_i,taxa_j]/all_possible_association

        # for i in range(len(taxa_list)):
        #     for j in range(len(taxa_list)):
        #         taxa_i = taxa_list[i]
        #         taxa_j = taxa_list[j]
        #         abundance_i = cell_info.loc[cell_info.cell_barcode.values == taxa_i, :].shape[0]
        #         abundance_j = cell_info.loc[cell_info.cell_barcode.values == taxa_j, :].shape[0]
        #         all_possible_association = abundance_i*abundance_j
        #         if all_possible_association > 0:
        #             association_probability_control_matrix[i,j,s] = adjacency_matrix_filtered.loc[taxa_i,taxa_j]/all_possible_association

    # association_probability_fold_change = np.zeros((adjacency_matrix_filtered.shape))
    # association_probability_significance = np.zeros((adjacency_matrix_filtered.shape))
    # for i in range(len(taxa_list)):
    #     for j in range(len(taxa_list)):
    #         association_probability_ij = association_probability_matrix[i,j,:]
    #         association_probability_ij_control = association_probability_control_matrix[i,j,:]
    #         pcij = np.average(association_probability_ij_control)
    #         if (pcij > 0) & (i != j):
    #             association_probability_fold_change[i,j] = np.average(association_probability_ij)/np.average(association_probability_ij_control)
    #             statistics, pvalue = scipy.stats.ttest_ind(association_probability_ij, association_probability_ij_control)
    #             association_probability_significance[i,j] = pvalue

    cell_info_control_all_df = pd.concat(cell_info_control_all)
    control_barcode_abundance = cell_info_control_all_df.cell_barcode.value_counts().reset_index()
    control_barcode_abundance.columns = ['cell_barcode', 'cell_count']
    all_taxa_cell_count = taxon_lookup.merge(control_barcode_abundance, on = 'cell_barcode', how = 'left')
    all_taxa_cell_count = all_taxa_cell_count.fillna(0)
    xx, yy = np.meshgrid(all_taxa_cell_count.cell_count.sort_values(ascending = False).index, all_taxa_cell_count.cell_count.sort_values(ascending = False).index)
    association_matrix_fold_change_random_sorted = association_matrix_fold_change_random[xx,yy,:]
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    im = plt.imshow(np.log2(np.average(association_matrix_fold_change_random_sorted, axis = 2)), cmap = 'coolwarm')
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('Taxa', fontsize = 8)
    plt.ylabel('Taxa', fontsize = 8)
    cbaxes = fig.add_axes([0.15, 0.05, 0.02, 0.15])
    cbar = plt.colorbar(im, cax = cbaxes, orientation = 'vertical')
    cbar.ax.tick_params(labelsize = 6, direction = 'in', length = 3)
    cbar.set_ticks([0, 5])
    cbar.set_label(r'$\log_2$(FC)', color = 'black', fontsize = 6, labelpad = 1)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.98, top=0.98)
    plt.savefig('{}/association_fold_change_cipro_random.pdf'.format(data_folder), dpi = 300)
    plt.close()

    association_matrix_fold_change_control_random_sorted = association_matrix_fold_change_control_random[xx,yy,:]
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    im = plt.imshow(np.log2(np.average(association_matrix_fold_change_control_random_sorted, axis = 2)), cmap = 'coolwarm')
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('Taxa', fontsize = 8)
    plt.ylabel('Taxa', fontsize = 8)
    cbaxes = fig.add_axes([0.15, 0.05, 0.02, 0.15])
    cbar = plt.colorbar(im, cax = cbaxes, orientation = 'vertical')
    cbar.ax.tick_params(labelsize = 6, direction = 'in', length = 3)
    cbar.set_label(r'$\log_2$(FC)', color = 'black', fontsize = 6, labelpad = 1)
    cbar.set_ticks([0, 2])
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.98, top=0.98)
    plt.savefig('{}/association_fold_change_control_random.pdf'.format(data_folder), dpi = 300)
    plt.close()

    association_fc_comparison_sorted = np.average(association_matrix_fold_change_random_sorted, axis = 2)/np.average(association_matrix_fold_change_control_random_sorted, axis = 2)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    im = plt.imshow(np.log2(association_fc_comparison_sorted), cmap = 'coolwarm')
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('Taxa', fontsize = 8)
    plt.ylabel('Taxa', fontsize = 8)
    cbaxes = fig.add_axes([0.15, 0.05, 0.02, 0.15])
    cbar = plt.colorbar(im, cax = cbaxes, orientation = 'vertical')
    cbar.ax.tick_params(labelsize = 6, direction = 'in', length = 3)
    cbar.set_label(r'$\log_2$(FC)', color = 'black', fontsize = 6, labelpad = 1)
    cbar.set_ticks([-2, 2])
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.98, top=0.98)
    plt.savefig('{}/association_fold_change_cipro_random_vs_control_random.pdf'.format(data_folder), dpi = 300)
    plt.close()


    association_fold_change = np.zeros((adjacency_matrix_filtered.shape))
    association_fc_significance = np.zeros((adjacency_matrix_filtered.shape))
    association_nonzero_sorted = association_nonzero[xx,yy]
    for i in range(len(taxa_list)):
        for j in range(len(taxa_list)):
            association_fc_ij = association_matrix_fold_change_random_sorted[i,j,:]
            association_fc_ij_control = association_matrix_fold_change_control_random_sorted[i,j,:]
            fcij = np.average(association_fc_ij_control)
            # if (fcij > 0):
            association_fold_change[i,j] = np.average(association_fc_ij + 1)/np.average(association_fc_ij_control + 1)
            statistics, pvalue = scipy.stats.ttest_ind(association_fc_ij + 1, association_fc_ij_control + 1)
            association_fc_significance[i,j] = pvalue

    # cipro_control_fold_change = (np.average(adjacency_matrix_merge, axis = 2) + 1)/(np.average(adjacency_matrix_control_merge, axis = 2) + 1)
    association_fc = np.zeros((adjacency_matrix_filtered.shape))
    association_fc_significance = np.zeros((adjacency_matrix_filtered.shape))
    association_nonzero = (np.sum(adjacency_matrix_merge, axis = 2) > 0) + (np.sum(adjacency_matrix_control_merge, axis = 2) > 0)
    for i in range(len(taxa_list)):
        for j in range(len(taxa_list)):
            association_ij = association_matrix_fold_change_random_sorted[i,j,:]
            association_control_ij = association_matrix_fold_change_control_random_sorted[i,j,:]
            # pcij = np.average(association_probability_ij_control)
            # if (pcij > 0) & (i != j):
            association_fc[i,j] = np.average(association_ij)/np.average(association_control_ij)
            statistics, pvalue = scipy.stats.ttest_ind(association_ij, association_control_ij)
            association_fc_significance[i,j] = pvalue
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    plt.plot(np.log2(np.triu(association_fc*association_nonzero_sorted, k = 1)), -np.log10(np.triu(association_fc_significance*association_nonzero_sorted, k = 1)), 'o', color = (0,0.5,1), markersize = 4, markeredgewidth = 0, alpha = 0.8)
    # plt.xlim(-2.2,2.2)
    # plt.ylim(-0.2, 4.2)
    plt.hlines(-np.log10(0.05/n_hypothesis), -2, 2, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(-1, 0, 150, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(1, 0, 150, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.15)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('$\log_{2}$(SAP Fold Change)', fontsize = 8)
    plt.ylabel('-$\log_{2}$(p)', fontsize = 8)
    plt.savefig('{}/association_cipro_random_control_random_fold_change_significance.pdf'.format(data_folder), dpi = 300)
    plt.close()

    n_hypothesis = np.sum(association_nonzero > 0)/2
    significant_indices = np.array(np.where(association_fc_significance < 0.05/n_hypothesis)).transpose()
    labels_list = []
    positions_list = []
    ha_list = []
    colors_list = []
    fc_list = []
    for i in range(significant_indices.shape[0]):
        barcode_i_index = significant_indices[i,0]
        barcode_j_index = significant_indices[i,1]
        if barcode_i_index > barcode_j_index:
            apfc = association_fc[barcode_i_index, barcode_j_index]
            aps = association_fc_significance[barcode_i_index, barcode_j_index]
            if np.isfinite(np.abs(apfc)) > 0 and aps > 0 and np.abs(np.log2(apfc)) > 1:
                barcode_i = taxa_list[barcode_i_index]
                barcode_j = taxa_list[barcode_j_index]
                taxa_i = taxon_lookup.loc[taxon_lookup.cell_barcode.values == barcode_i, 'sci_name'].values[0]
                taxa_j = taxon_lookup.loc[taxon_lookup.cell_barcode.values == barcode_j, 'sci_name'].values[0]
                labels_list.append('{}-{}'.format(taxa_i, taxa_j))
                if np.log10(apfc) < 0:
                    position_x = np.log2(association_fc[barcode_i_index, barcode_j_index]) - 0.5
                    position_y = -np.log10(association_fc_significance[barcode_i_index, barcode_j_index]) + 0.05
                    positions_list.append([position_x, position_y])
                    ha_list.append('left')
                    colors_list.append((0,0.25,1))
                    fc_list.append(apfc)
                else:
                    position_x = np.log2(association_fc[barcode_i_index, barcode_j_index]) + 0.5
                    position_y = -np.log10(association_fc_significance[barcode_i_index, barcode_j_index]) + 0.05
                    positions_list.append([position_x, position_y])
                    ha_list.append('right')
                    colors_list.append((1,0.25,0))
                    fc_list.append(apfc)

    positions_list = [positions_list[k] for k in np.argsort(fc_list)]
    colors_list = [colors_list[k] for k in np.argsort(fc_list)]
    labels_list = [labels_list[k] for k in np.argsort(fc_list)]
    ha_list = [ha_list[k] for k in np.argsort(fc_list)]
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(12), cm_to_inches(6))
    gs = GridSpec(1,3)
    ax = plt.subplot(gs[0,0])
    plt.plot(np.log2(np.triu(association_fc*association_nonzero_sorted, k = 1)), -np.log10(np.triu(association_fc_significance*association_nonzero_sorted, k = 1)), 'o', color = (0,0.5,1), markersize = 4, markeredgewidth = 0, alpha = 0.8)
    # plt.xlim(-2.2,2.2)
    # plt.ylim(-0.2, 4.5)
    ax.hlines(-np.log10(0.05/n_hypothesis), -4, 4, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    ax.vlines(-1, 0, 150, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    ax.vlines(1, 0, 150, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    for i in range(len(labels_list)):
        pos_x = positions_list[i][0]
        pos_y = positions_list[i][1]
        ax.text(pos_x, pos_y, i+1, ha = ha_list[i], fontsize = 6)

    ax.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('$\log_{2}$(SAP Fold Change)', fontsize = 8)
    plt.ylabel('-$\log_{10}$(p)', fontsize = 8)
    ax = plt.subplot(gs[0,1:3])
    for i in range(len(labels_list)):
        ax.text(0, -i*10, '{}. {}'.format(i+1, labels_list[i]), color = colors_list[i], fontsize = 8)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.15)
    plt.ylim(-140,10)
    plt.xlim(-0.1,2.2)
    plt.axis('off')
    plt.savefig('{}/association_fold_change_significance_labeled.pdf'.format(data_folder), dpi = 300)
    plt.close()

    association_probability_fold_change_filtered = association_probability_fold_change.copy()
    association_probability_fold_change_filter = np.isnan(association_probability_fold_change).all(axis = 0)
    association_probability_fold_change_filtered = association_probability_fold_change_filtered[~association_probability_fold_change_filter,:]
    association_probability_fold_change_filtered = association_probability_fold_change_filtered[:,~association_probability_fold_change_filter]
    label_codes = taxa_list[~association_probability_fold_change_filter]
    tick_labels = [taxon_lookup.loc[taxon_lookup.cell_barcode.values == x, 'sci_name'].values[0] for x in label_codes.tolist()]
    cmap = matplotlib.cm.RdBu
    cmap.set_bad('black')
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(10),cm_to_inches(10))
    ax = plt.Axes(fig, [0.01, 0.3, 0.69, 0.69])
    fig.add_axes(ax)
    mappable = ax.imshow(np.log10(association_probability_fold_change_filtered), cmap = cmap)
    plt.xticks(np.arange(association_probability_fold_change_filtered.shape[0]), tick_labels, rotation = 90, style = 'italic')
    plt.yticks(np.arange(association_probability_fold_change_filtered.shape[1]), tick_labels, style = 'italic')
    plt.tick_params(direction = 'in', length = 0, labelsize = 6)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_right()
    cbaxes = inset_axes(ax, width="4%", height="25%", loc = 4, bbox_to_anchor = (0,-0.4,1.35,1.2), bbox_transform = ax.transAxes)
    cbar = plt.colorbar(mappable, cax=cbaxes, orientation = 'vertical')
    cbar.set_label(r'$\log_{10}$(FC)', color = 'black', fontsize = 8)
    cbar.ax.tick_params(labelsize = 8, direction = 'in', color = 'black', length = 3)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    plt.subplots_adjust(left = 0.01, bottom = 0.4, right = 0.6, top = 0.99)
    plt.savefig('{}/ciprofloxacin_differential_association_probability_matrix.pdf'.format(input_directory), dpi = 300, transparent = True)
    return

def main():
    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-c', '--cell_info', dest = 'cell_info_filename', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, default = '', help = 'Input normalized single cell spectra filename')
    args = parser.parse_args()
    print(args.cell_info_filename)
    sample = re.sub('_cell_information_consensus.csv', '', args.cell_info_filename)
    image_seg_filename = '{}_seg.npy'.format(sample)
    image_seg = np.load(image_seg_filename)
    adjacency_seg_filename = '{}_adjacency_seg.npy'.format(sample)
    adjacency_seg = np.load(adjacency_seg_filename)
    cell_info = pd.read_csv(args.cell_info_filename, dtype = {'cell_barcode':str})
    taxon_lookup = get_taxon_lookup(args.probe_design_filename)
    taxa_barcode_sciname = get_taxa_barcode_sciname(args.probe_design_filename)
    cell_info = analyze_cell_info(args.cell_info_filename, taxa_barcode_sciname)
    image_identification = generate_identification_image(image_seg, cell_info, sample, taxon_lookup)
    cell_info_filtered, image_identification_filtered = remove_spurious_objects_and_debris(image_seg, image_identification, adjacency_seg, taxon_lookup, cell_info, sample)
    # q_eigenvalue_heatmap, q_eigenvectors, cell_info = analyze_alignment_order_parameter(cell_info, segmentation, image_identification)
    # np.save('{}_q_eigenvalue.npy'.format(sample), q_eigenvalue_heatmap)
    # np.save('{}_q_eigenvectors.npy'.format(sample), q_eigenvectors)
    # cell_info_filtered = cell_info_filtered.merge(cell_info.loc[:,['label', 'q_eigenvalue']], on = 'label', how = 'left')
    cell_info_filtered.to_csv('{}_cell_information_consensus_filtered.csv'.format(sample), index = None)
    print('Saving identification image...')
    np.save('{}_identification.npy'.format(sample), image_identification_filtered)
    save_identification(image_identification_filtered, sample)
    # analyze_spatial_adjacency_network(image_seg, adjacency_seg, cell_info_filtered, taxa_barcode_sciname, sample)
    return

if __name__ == '__main__':
    main()
