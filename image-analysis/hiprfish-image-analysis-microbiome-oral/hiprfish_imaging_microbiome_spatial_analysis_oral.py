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
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 0.5

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
        if (cell_area > 10000):
          cell_info.loc[i, 'type'] = 'debris'
          image_identification_filtered[image_seg == cell_label] = [0,0,0]
    # save_identification(image_identification_filtered, sample)
    cell_info_filtered = cell_info.loc[cell_info.type.values == 'cell',:].copy()
    # cell_info_filtered_filename = sample + '_cell_information_filtered.csv'
    # cell_info_filtered.to_csv(cell_info_filtered_filename, index = None)
    # avgint_filtered = avgint[cell_info.type.values == 'cell', :]
    # pd.DataFrame(avgint_filtered).to_csv('{}_avgint_filtered.csv'.format(sample), index = None)
    edge_map = skimage.filters.sobel(image_seg > 0)
    rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
    adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.cell_barcode.values, columns = taxon_lookup.cell_barcode.values)
    adjacency_matrix_filtered = adjacency_matrix.copy()
    for i in range(cell_info.shape[0]):
        edges = list(rag.edges(i+1))
        for e in edges:
            node_1 = e[0]
            node_2 = e[1]
            if (node_1 != 0) and (node_2 !=0):
                barcode_1 = cell_info.loc[cell_info.label == node_1,'cell_barcode'].values[0]
                barcode_2 = cell_info.loc[cell_info.label == node_2,'cell_barcode'].values[0]
                adjacency_matrix.loc[barcode_1, barcode_2] += 1
                if (cell_info.loc[cell_info.label == node_1,'type'].values[0] == 'cell') and (cell_info.loc[cell_info.label == node_2,'type'].values[0] == 'cell'):
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
    cell_info.loc[(cell_info.max_intensity.values >= 0.1) & (cell_info.nn_dists.values <= 0.02), 'type'] = 'cell'
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
          if (cell_barcodes.cell_barcode.values[q] in taxon_lookup.cell_barcode.values):
              image_identification[segmentation == cell_population.loc[r, 'label'], :] = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.cell_barcode.values == cell_barcodes.cell_barcode.values[q], ['H', 'S', 'V']].values)
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

def analyze_oral_diversity(image_tab_filename, data_folder):
    theme_color = 'black'
    # Summary info
    image_tab = pd.read_csv(image_tab_filename)
    cell_info_all = []
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
        cell_info_all.append(cell_info)

    # Shannon diversity time series
    theme_color = 'black'
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3), cm_to_inches(2.5))
    image_tab_nc = image_tab.loc[image_tab.CELL_NUMBER.values > 500, :]
    time_points = image_tab_nc.SAMPLING_TIME.drop_duplicates()
    shannon_diversity_list = [image_tab_nc.loc[image_tab_nc.SAMPLING_TIME.values == tp, 'SHANNON_DIVERSITY'].values for tp in time_points]
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize = 6, linestyle='none', markeredgewidth = 0)
    capprops = dict(linewidth = 0.5)
    bp = plt.boxplot(shannon_diversity_list, positions = np.arange(7), patch_artist = True, flierprops = flierprops, capprops = capprops)
    for b in bp['boxes']:
        b.set_facecolor('dodgerblue')
        b.set_linewidth(0.5)
        b.set_edgecolor(theme_color)

    for w in bp['whiskers']:
        w.set_color(theme_color)
        w.set_linewidth(0.5)

    day_list = image_tab_nc.SAMPLING_DAY_REFERENCE.drop_duplicates().values
    xlabel_list = [int(t/30) for t in day_list]
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 6, colors = theme_color)
    plt.xticks(np.arange(7), xlabel_list)
    plt.xlim(-0.75,6.5)
    plt.xlabel('Time [Month]', fontsize = 6, color = theme_color, labelpad = 1)
    plt.ylabel('Shannon Diversity', fontsize = 6, color = theme_color, labelpad = 1)
    plt.subplots_adjust(left = 0.18, bottom = 0.23, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_shannon_diversity_time_series.pdf'.format(data_folder), dpi = 300)
    plt.close()

    # Shannon diversity time series presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(7))
    image_tab_nc = image_tab.loc[image_tab.CELL_NUMBER.values > 500, :]
    time_points = image_tab_nc.SAMPLING_TIME.drop_duplicates()
    shannon_diversity_list = [image_tab_nc.loc[image_tab_nc.SAMPLING_TIME.values == tp, 'SHANNON_DIVERSITY'].values for tp in time_points]
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    bp = plt.boxplot(shannon_diversity_list, positions = np.arange(7), patch_artist = True, flierprops = flierprops)
    for b in bp['boxes']:
        b.set_facecolor('dodgerblue')
        b.set_edgecolor(theme_color)

    for w in bp['whiskers']:
        w.set_color(theme_color)

    day_list = image_tab_nc.SAMPLING_DAY_REFERENCE.drop_duplicates().values
    xlabel_list = [int(t/30) for t in day_list]
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
    plt.xticks(np.arange(7), xlabel_list)
    plt.xlabel('Time [Month]', fontsize = font_size, color = theme_color)
    plt.ylabel('Shannon Diversity', fontsize = font_size, color = theme_color)
    plt.subplots_adjust(left = 0.18, bottom = 0.23, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_shannon_diversity_time_series_presentation.pdf'.format(data_folder), dpi = 300, transparent = True)
    plt.close()

    scipy.stats.f_oneway(shannon_diversity_list[0], shannon_diversity_list[1], shannon_diversity_list[2], shannon_diversity_list[3], shannon_diversity_list[4], shannon_diversity_list[5])
    # F_onewayResult(statistic=1.2793738618184372, pvalue=0.28573848815307745)

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
                cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
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
            beta_diversity_time_series.append(beta_diversity_image)
        beta_diversity_patch_time_series.append(beta_diversity_time_series)

    np.save('{}/oral_beta_diversity_patch_size_time_series_source_data.npy'.format(data_folder), beta_diversity_patch_time_series)
    # plot alpha diversity as a function of patch size and time
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3.25), cm_to_inches(2.5))
    time_points = image_tab.SAMPLING_TIME.drop_duplicates()
    cmap = cm.get_cmap('coolwarm')
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    meanprops = dict(linestyle='none', linewidth = 0)
    medianprops = dict(linestyle='none', linewidth = 0)
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
    plt.tick_params(direction = 'in', length = 2, labelsize = 6)
    plt.xticks(np.arange(0, 2.2*7, 2.2), xlabel_list)
    plt.xlabel('Time [month]', fontsize = 6, color = theme_color, labelpad = 2)
    plt.ylabel(r'$\beta$ diversity', fontsize = 6, color = theme_color, labelpad = 0)
    cbaxes = fig.add_axes([0.65, 0.9, 0.25, 0.05])
    norm = Normalize(vmin = 10, vmax = 200)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize = 5, color = theme_color, width = 0.5, pad = 1)
    cb.set_ticks([10,100,200])
    cb.set_label(r'$N_{cells}$', fontsize = 5, color = theme_color, labelpad = 0)
    plt.subplots_adjust(left = 0.22, bottom = 0.24, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_beta_diversity_patch_size_time_series.pdf'.format(data_folder), dpi = 300)
    plt.close()

    # plot alpha diversity as a function of patch size and time presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(11), cm_to_inches(7))
    time_points = image_tab.SAMPLING_TIME.drop_duplicates()
    cmap = cm.get_cmap('coolwarm')
    flierprops = dict(marker='.', markerfacecolor='darkorange', markersize=8, linestyle='none', markeredgewidth = 0)
    meanprops = dict(linestyle='none', linewidth = 0)
    medianprops = dict(linestyle='none', linewidth = 0)
    # plt.plot(beta_diversity_list)
    for i in np.arange(19):
        bp = plt.boxplot(beta_diversity_patch_time_series[i], positions = np.arange(0, 2.2*7, 2.2) - 1 + 0.1*i, patch_artist = True, flierprops = flierprops,
                    widths = 0.6, showcaps = False, showfliers = False, meanline = False, medianprops = medianprops)
        for b in bp['boxes']:
            b.set_facecolor(cmap(i/20))
            b.set_edgecolor((0.5,0.5,0.5))
            b.set_linewidth(0.5)
        for w in bp['whiskers']:
            w.set_linewidth(0.5)
            w.set_color((0.5,0.5,0.5))

    day_list = image_tab.SAMPLING_DAY_REFERENCE.drop_duplicates().values
    xlabel_list = [int(t/30) for t in day_list]
    plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
    plt.xticks(np.arange(0, 2.2*7, 2.2), xlabel_list)
    plt.xlabel('Time [month]', fontsize = font_size, color = theme_color, labelpad = 0)
    plt.ylabel(r'$\beta$ diversity', fontsize = font_size, color = theme_color, labelpad = 0)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    cbaxes = fig.add_axes([0.65, 0.9, 0.2, 0.05])
    norm = Normalize(vmin = 10, vmax = 200)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation = 'horizontal')
    cbaxes.tick_params(direction = 'in', length = 2, labelsize = font_size, color = theme_color)
    cb.set_ticks([10,100,200])
    cb.set_label(r'$N_{cells}$', fontsize = font_size, color = theme_color, labelpad = 0)
    plt.subplots_adjust(left = 0.18, bottom = 0.2, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_beta_diversity_patch_size_time_series_presentation.pdf'.format(data_folder), dpi = 300, transparent = True)
    plt.close()

    # calculate bray-curtis
    nc = 200
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
            cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name))
            for j in range(n_pairs):
                rp_1 = 2000*np.random.random(2)
                cell_distance = (cell_info.centroid_x.values - rp_1[0])**2 + (cell_info.centroid_y.values - rp_1[1])**2
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

    # bray-curtis intra-patch distance
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(3), cm_to_inches(2.5))
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

    plt.tick_params(direction = 'in', length = 2, labelsize = 6)
    plt.legend(frameon = False, fontsize = 5, handlelength = 1, bbox_to_anchor = (0.24,0.64), ncol = 2, columnspacing = 0)
    plt.xlabel(r'Intra-patch distance [$\mu$m]', fontsize = 6, color = theme_color, labelpad = 0)
    plt.axes().xaxis.set_label_coords(0.4,-0.2)
    plt.ylabel('Bray-Curtis', fontsize = 6, color = theme_color, labelpad = 0)
    plt.subplots_adjust(left = 0.22, bottom = 0.24, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300)
    plt.close()

    # bray-curtis intra-patch distance presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(9), cm_to_inches(7))
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
        plt.plot(bcd_sorted[:,0]*0.07, bcd_smoothed, color = color_list[i], label = label_list[i], alpha = 0.8)

    plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
    l = plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.4,0.5), ncol = 2)
    for t in l.get_texts():
        t.set_color(theme_color)

    plt.xlabel(r'Intra-patch distance [$\mu$m]', fontsize = font_size, color = theme_color)
    plt.ylabel('Bray-Curtis', fontsize = font_size, color = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.subplots_adjust(left = 0.15, bottom = 0.2, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_bray_curtis_nc_{}_presentation.svg'.format(data_folder, n_cell), dpi = 300, transparent = True)
    plt.close()


    return

def measure_oral_spatial_association_vs_random(sam_tab, taxon_lookup, data_folder):
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    taxon_lookup = pd.read_csv('/workdir/hs673/Runs/V1/Samples/HIPRFISH_7/simulation/DSGN0673/taxon_color_lookup.csv', dtype = {'code':str})
    taxon_lookup = taxon_lookup.rename(columns = {'code':'cell_barcode'})
    taxa_barcode_sciname = pd.read_csv('/workdir/hs673/Runs/V1/Samples/HIPRFISH_7/simulation/DSGN0673/DSGN0673_primerset_C_barcode_selection_MostSimple_taxa_barcode_sciname.csv', dtype = {'cell_barcode':str})
    sampling_times = sam_tab.SAMPLING_TIME.drop_duplicates()
    sample = sam_tab.loc[0, 'SAMPLE']
    image_name = sam_tab.loc[0, 'IMAGES']
    adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], sam_tab.shape[0]))
    association_matrix_random = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], sam_tab.shape[0]))
    cell_info_all = []
    for s in range(sam_tab.shape[0]):
        print(s)
        sample = sam_tab.loc[s,'SAMPLE']
        image_name = sam_tab.loc[s,'IMAGES']
        image_seg = np.load('{}/{}/{}_seg.npy'.format(data_folder, sample, image_name))
        adjacency_seg = np.load('{}/{}/{}_adjacency_seg.npy'.format(data_folder, sample, image_name))
        image_identification = np.load('{}/{}/{}_identification.npy'.format(data_folder, sample, image_name))
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        cell_info_filtered = cell_info.loc[cell_info.type.values == 'cell',:]
        cell_info_all.append(cell_info_filtered)
        amrf_filename = '{}/{}/{}_adjacency_matrix_random_full.npy'.format(data_folder, sample, image_name)
        if not os.path.exists(amrf_filename):
            adjacency_matrix_random = []
            for k in range(100):
                adjacency_matrix_random_k = generate_random_spatial_adjacency_network(image_seg, adjacency_seg, cell_info_filtered, taxa_barcode_sciname)
                adjacency_matrix_random.append(adjacency_matrix_random_k)
            adjacency_matrix_random_full = dask.delayed(np.stack)(adjacency_matrix_random, axis = 2).compute()
            np.save(amrf_filename, adjacency_matrix_random_full)
        else:
            adjacency_matrix_random_full = np.load(amrf_filename)
        adjacency_matrix_random_avg = np.average(adjacency_matrix_random_full, axis = 2)
        adjacency_matrix_random_std = np.std(adjacency_matrix_random_full, axis = 2)
        association_matrix_random[:,:,s] = adjacency_matrix_random_avg

    adjacency_matrix_list = []
    adjacency_matrix_list_by_image = []
    adjacency_matrix_random = pd.DataFrame(np.zeros((taxa_barcode_sciname.shape[0], taxa_barcode_sciname.shape[0])), index = taxa_barcode_sciname.cell_barcode.values, columns = taxa_barcode_sciname.cell_barcode.values)
    adjacency_matrix_random_by_image = []
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
            adjacency_matrix_random_full = np.load('{}/{}/{}_adjacency_matrix_random_full.npy'.format(data_folder, sample, image_name))
            adjacency_matrix_random.loc[:,:] = adjacency_matrix_random.values + np.average(adjacency_matrix_random_full, axis = 2)
            adjacency_matrix_list_by_image.append(adjacency_matrix.values)
            adjacency_matrix_random_by_image.append(np.average(adjacency_matrix_random_full, axis = 2))
        adjacency_matrix_list.append(adjacency_matrix_tp)
    adjacency_matrix_stack = np.stack(adjacency_matrix_list, axis = 2)
    adjacency_matrix_mask = np.prod(adjacency_matrix_stack > 0, axis = 2)
    adj_filtered = adjacency_matrix_mask[:,np.sum(adjacency_matrix_mask, axis = 0) > 0]
    adj_filtered = adj_filtered[np.sum(adj_filtered, axis = 1) > 0,:]
    network_distances = np.zeros((sampling_times.shape[0] + 1, sampling_times.shape[0] + 1))
    # random_network = np.random.random(adjacency_matrix_list[0].shape)
    # random_network_norm = pd.DataFrame(random_network/np.max(random_network))
    # adjacency_matrix_list.append(random_network_norm)
    adjacency_matrix_list.append(adjacency_matrix_random)
    for i in range(network_distances.shape[0]):
        for j in range(network_distances.shape[0]):
            adj_i = adjacency_matrix_list[i].values
            adj_j = adjacency_matrix_list[j].values
            adj_i_log = np.log(adj_i + 1)
            adj_j_log = np.log(adj_j + 1)
            network_distances[i,j] = np.nansum((adj_i_log - adj_j_log)**2)

    # spatial network comparison
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(2.5), cm_to_inches(2.75))
    im = plt.imshow(np.log(network_distances), cmap = 'coolwarm')
    plt.xticks(np.arange(8), ['0', '3', '9', '12', '15', '21', '27', 'Rd'])
    plt.yticks(np.arange(8), ['0', '3', '9', '12', '15', '21', '27', 'Rd'])
    plt.xlabel('Time [month]', fontsize = 6, color = theme_color, labelpad = 2)
    plt.ylabel('Time [month]', fontsize = 6, color = theme_color, labelpad = 0)
    plt.axes().xaxis.tick_top()
    plt.axes().xaxis.set_label_position('top')
    plt.tick_params(direction = 'in', length = 2, labelsize = 5, width = 0.5, pad = 1, colors = theme_color)
    cax = fig.add_axes([0.4, 0.08, 0.4, 0.04])
    cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cax.tick_params(direction = 'in', length = 2, labelsize = 5, color = theme_color)
    network_distances_nonzero = network_distances[network_distances>0]
    text_loc_y = (np.nanmin(np.log(network_distances_nonzero)) + np.nanmax(np.log(network_distances_nonzero)))/2
    cb.ax.text(np.nanmin(np.log(network_distances_nonzero)) - 0.2, text_loc_y, 'low', ha = 'right', va = 'center', fontsize = 5)
    cb.ax.text(np.nanmax(np.log(network_distances_nonzero)) + 0.2, text_loc_y, 'high',ha = 'left', va = 'center', fontsize = 5)
    cb.set_ticks([])
    cb.outline.set_edgecolor(theme_color)
    cb.set_label('Network Difference', fontsize = 5, color = theme_color, labelpad = 1)
    plt.subplots_adjust(left = 0.18, bottom = 0.16, right = 0.98, top = 0.84)
    plt.savefig('{}/oral_spatial_adjacency_network_time_series_comparison.pdf'.format(data_folder), dpi = 300, transparent = True)

    # spatial network example
    theme_color = 'white'
    font_size = 12
    taxa_label_font_size = 10
    top_n_keep = 20
    adjacency_matrix_by_image = np.stack(adjacency_matrix_list_by_image, axis = 2)
    adjacency_matrix_random_by_image = np.stack(adjacency_matrix_random_by_image, axis = 2)
    for i in range(adjacency_matrix_by_image.shape[2]):
        sample = sam_tab.loc[i,'SAMPLE']
        image_name = sam_tab.loc[i,'IMAGES']
        ranking_order = np.argsort(np.sum(np.abs(np.log2(adjacency_matrix_by_image[:,:,i] + 1)), axis = 0))[::-1]
        taxon_labels = taxa_barcode_sciname.sci_name.values
        xx, yy = np.meshgrid(ranking_order, ranking_order)
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(13.5), cm_to_inches(13.5))
        im = plt.imshow(np.log10(adjacency_matrix_by_image[:,:,i]+1)[xx,yy][:top_n_keep,:top_n_keep], cmap = 'coolwarm')
        plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
        plt.axes().spines['left'].set_color(theme_color)
        plt.axes().spines['bottom'].set_color(theme_color)
        plt.axes().spines['top'].set_color(theme_color)
        plt.axes().spines['right'].set_color(theme_color)
        plt.xticks(np.arange(top_n_keep), taxon_labels[ranking_order][:top_n_keep], color = theme_color, fontsize = taxa_label_font_size, rotation = -90)
        plt.yticks(np.arange(top_n_keep), taxon_labels[ranking_order][:top_n_keep], color = theme_color, fontsize = taxa_label_font_size)
        plt.axes().yaxis.tick_right()
        cax = fig.add_axes([0.82, 0.15, 0.1, 0.02])
        cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
        cax.tick_params(direction = 'in', length = 2, labelsize = taxa_label_font_size, colors = theme_color)
        network_distances_nonzero = network_distances[network_distances>0]
        cb.outline.set_edgecolor(theme_color)
        cb.set_label(r'$\log_{10}$(SAM + 1)', fontsize = taxa_label_font_size, color = theme_color)
        plt.subplots_adjust(left = 0.02, bottom = 0.35, right = 0.65, top = 0.98)
        plt.savefig('{}/{}/{}_top_{}_sam_heatmap.pdf'.format(data_folder, sample, image_name, top_n_keep), dpi = 300, transparent = True)
        plt.close()
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(13.5), cm_to_inches(13.5))
        im = plt.imshow(np.log10(adjacency_matrix_random_by_image[:,:,i]+1)[xx,yy][:top_n_keep,:top_n_keep], cmap = 'coolwarm')
        plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
        plt.axes().spines['left'].set_color(theme_color)
        plt.axes().spines['bottom'].set_color(theme_color)
        plt.axes().spines['top'].set_color(theme_color)
        plt.axes().spines['right'].set_color(theme_color)
        plt.xticks(np.arange(top_n_keep), taxon_labels[ranking_order][:top_n_keep], color = theme_color, fontsize = taxa_label_font_size, rotation = -90)
        plt.yticks(np.arange(top_n_keep), taxon_labels[ranking_order][:top_n_keep], color = theme_color, fontsize = taxa_label_font_size)
        plt.axes().yaxis.tick_right()
        cax = fig.add_axes([0.82, 0.15, 0.1, 0.02])
        cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
        cax.tick_params(direction = 'in', length = 2, labelsize = taxa_label_font_size, colors = theme_color)
        network_distances_nonzero = network_distances[network_distances>0]
        cb.outline.set_edgecolor(theme_color)
        cb.set_label(r'$\log_{10}$(SAM + 1)', fontsize = taxa_label_font_size, color = theme_color)
        plt.subplots_adjust(left = 0.02, bottom = 0.35, right = 0.65, top = 0.98)
        plt.savefig('{}/{}/{}_top_{}_sam_random_heatmap.pdf'.format(data_folder, sample, image_name, top_n_keep), dpi = 300, transparent = True)
        plt.close()


    # spatial network comparison presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(7), cm_to_inches(8))
    im = plt.imshow(np.log(network_distances), cmap = 'coolwarm')
    plt.xticks(np.arange(8), ['0', '3', '9', '12', '15', '21', '27', 'Rd'])
    plt.yticks(np.arange(8), ['0', '3', '9', '12', '15', '21', '27', 'Rd'])
    plt.xlabel('Time [month]', fontsize = font_size, color = theme_color, labelpad = 2)
    plt.ylabel('Time [month]', fontsize = font_size, color = theme_color, labelpad = 0)
    plt.axes().xaxis.tick_top()
    plt.axes().xaxis.set_label_position('top')
    plt.tick_params(direction = 'in', length = 2, labelsize = font_size, colors = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    cax = fig.add_axes([0.4, 0.1, 0.4, 0.04])
    cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cax.tick_params(direction = 'in', length = 2, labelsize = font_size, color = theme_color)
    network_distances_nonzero = network_distances[network_distances>0]
    text_loc_y = (np.nanmin(np.log(network_distances_nonzero)) + np.nanmax(np.log(network_distances_nonzero)))/2
    cb.ax.text(np.nanmin(np.log(network_distances_nonzero)) - 0.2, text_loc_y, 'low', ha = 'right', va = 'center', fontsize = font_size, color = theme_color)
    cb.ax.text(np.nanmax(np.log(network_distances_nonzero)) + 0.2, text_loc_y, 'high',ha = 'left', va = 'center', fontsize = font_size, color = theme_color)
    cb.set_ticks([])
    cb.outline.set_edgecolor(theme_color)
    cb.set_label('Network Difference', fontsize = font_size, color = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.18, right = 0.98, top = 0.84)
    plt.savefig('{}/oral_spatial_adjacency_network_time_series_comparison_presentation.pdf'.format(data_folder), dpi = 300, transparent = True)

    return


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
        cell_info = pd.read_csv('{}/{}/{}_z_{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name))
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
                cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
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
        # plt.ylim(-0.05, 0.3)
        plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.2,0.38), handlelength = 0.5)
        plt.xlabel('Intra-patch\n' + r'distance [$\mu$m]', fontsize = 8, color = theme_color)
        plt.ylabel('Bray-Curtis', fontsize = 8, color = theme_color, position = (0.5,0.4))
        plt.subplots_adjust(left = 0.25, bottom = 0.3, right = 0.98, top = 0.98)
        plt.savefig('{}/gut_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300, transparent = True)

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

    plt.ylim(-0.02, 1.4)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8)
    plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.15,0.6,0.8,0.2), ncol = 3, handlelength = 0.25, columnspacing = 0.5, handletextpad = 0.25)
    plt.xlabel(r'Intra-patch distance [$\mu$m]', fontsize = 8, color = theme_color, labelpad = 0)
    plt.ylabel('Bray-Curtis', fontsize = 8, color = theme_color, labelpad = 0)
    plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.98, top = 0.98)
    plt.savefig('{}/oral_bray_curtis_nc_{}.pdf'.format(data_folder, n_cell), dpi = 300)
    plt.close()

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
    taxon_lookup = taxon_lookup.rename(columns = {'code':'cell_barcode'})
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
