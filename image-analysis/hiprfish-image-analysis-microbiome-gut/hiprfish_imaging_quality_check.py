import numpy as np
import pandas as pd
import argparse
import glob
import re
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import matplotlib

def cm_to_inches(x):
    return(x/2.54)

def pair_correlation(distance, boundary_distance, dr, nr, A):
    minimum_boundary_distance = np.min(boundary_distance, axis = 1)
    n_cell_total = distance.shape[0]
    g = np.zeros(nr)
    for k in range(1, nr+1):
        total_count = 0
        for i in range(distance.shape[0]):
            if k*dr < minimum_boundary_distance[i]:
                total_count += distance[i, (distance[i, :] > (k-1)*dr) & (distance[i, :] < k*dr)].shape[0]
            else:
                angle = np.arccos(minimum_boundary_distance[i]/(k*dr))
                total_count += ((2*np.pi-2*angle)/(2*np.pi))*distance[i, (distance[i, :] > (k-1)*dr) & (distance[i, :] < k*dr)].shape[0]
        g[k-1] = A*total_count/(n_cell_total*n_cell_total*2*np.pi*k*dr*dr)
    return(g)

def measure_pair_correlation_by_treatment(sam_tab, nr, taxon_lookup, input_directory):
    cipro_samples = sam_tab.loc[sam_tab.Treatment.values == 'Ciprofloxacin', 'Sample'].values
    sample = cipro_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['barcode', 'abundance_0']
    for s in range(1, len(cipro_samples)):
        sample = cipro_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance = barcode_abundance.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].mean(axis = 1)
    barcode_abundance['std_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].std(axis = 1)
    # barcode_list = barcode_abundance.loc[barcode_abundance.average_abundance.values > 50, :].barcode.drop_duplicates()
    barcode_list = barcode_abundance.barcode.drop_duplicates()
    g_matrix_dict = {}
    for b in barcode_list:
        g_matrix = pd.DataFrame(np.zeros((len(cipro_samples), nr)), index = cipro_samples, columns = np.arange(nr), dtype = float)
        for s in range(len(cipro_samples)):
            sample = cipro_samples[s]
            cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
            cells = cell_info.iloc[cell_info['cell_barcode'].values == b].reset_index()
            distance = np.zeros((cells.shape[0], cells.shape[0]))
            boundary_distance = np.zeros((cells.shape[0], 4))
            print('{}, {}, {}'.format(b, s, cells.shape[0]))
            if cells.shape[0] > 0:
                for i in range(cells.shape[0]):
                    for j in range(cells.shape[0]):
                        x1 = cells.loc[i, 'centroid_x']
                        x2 = cells.loc[j, 'centroid_x']
                        y1 = cells.loc[i, 'centroid_y']
                        y2 = cells.loc[j, 'centroid_y']
                        distance[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                for i in range(cells.shape[0]):
                    x1 = cells.loc[i, 'centroid_x']
                    y1 = cells.loc[i, 'centroid_y']
                    boundary_distance[i,0] = np.sqrt((x1 - 0)**2)
                    boundary_distance[i,1] = np.sqrt((x1 - 1000)**2)
                    boundary_distance[i,2] = np.sqrt((y1 - 0)**2)
                    boundary_distance[i,3] = np.sqrt((y1 - 1000)**2)
                minimum_boundary_distance = np.min(boundary_distance, axis = 1)
                g_matrix.loc[sample,:] = pair_correlation(distance, boundary_distance, 2, nr, 1000*1000)
        g_matrix_dict.update({b:g_matrix})
    control_samples = sam_tab.loc[sam_tab.Treatment.values == 'Control', 'Sample'].values
    sample = control_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance_control = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance_control.columns = ['barcode', 'abundance_0']
    for s in range(1, len(control_samples)):
        sample = control_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance_control = barcode_abundance_control.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance_control['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].mean(axis = 1)
    barcode_abundance_control['std_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].std(axis = 1)
    # barcode_list_control = barcode_abundance.loc[barcode_abundance.average_abundance.values > 50, :].barcode.drop_duplicates()
    barcode_list_control = barcode_abundance_control.barcode.drop_duplicates()
    g_control_matrix_dict = {}
    for b in barcode_list_control:
        g_control_matrix = pd.DataFrame(np.zeros((len(control_samples), nr)), index = control_samples, columns = np.arange(nr), dtype = float)
        for s in range(len(control_samples)):
            sample = control_samples[s]
            cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
            cells = cell_info.iloc[cell_info['cell_barcode'].values == b].reset_index()
            distance = np.zeros((cells.shape[0], cells.shape[0]))
            boundary_distance = np.zeros((cells.shape[0], 4))
            print('{}, {}, {}'.format(b, s, cells.shape[0]))
            if cells.shape[0] > 0:
                for i in range(cells.shape[0]):
                    for j in range(cells.shape[0]):
                        x1 = cells.loc[i, 'centroid_x']
                        x2 = cells.loc[j, 'centroid_x']
                        y1 = cells.loc[i, 'centroid_y']
                        y2 = cells.loc[j, 'centroid_y']
                        distance[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                for i in range(cells.shape[0]):
                    x1 = cells.loc[i, 'centroid_x']
                    y1 = cells.loc[i, 'centroid_y']
                    boundary_distance[i,0] = np.sqrt((x1 - 0)**2)
                    boundary_distance[i,1] = np.sqrt((x1 - 1000)**2)
                    boundary_distance[i,2] = np.sqrt((y1 - 0)**2)
                    boundary_distance[i,3] = np.sqrt((y1 - 1000)**2)
                minimum_boundary_distance = np.min(boundary_distance, axis = 1)
                g_control_matrix.loc[sample,:] += pair_correlation(distance, boundary_distance, 2, nr, 1000*1000)
        g_control_matrix_dict.update({b:g_control_matrix})
    barcode_intersection_list = list(set(barcode_list).intersection(barcode_list_control))
    barcode_abundance_merge = barcode_abundance.loc[:,['barcode', 'average_abundance', 'std_abundance']].merge(barcode_abundance_control.loc[:,['barcode', 'average_abundance', 'std_abundance']], on = 'barcode', how = 'outer')
    barcode_abundance_merge.columns = ['barcode', 'average_abundance_ciprofloxacin', 'std_abundance_ciprofloxacin', 'average_abundance_control', 'std_abundance_control']
    for b in barcode_intersection_list:
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
        g_matrix = g_matrix_dict[b]
        g_control_matrix = g_control_matrix_dict[b]
        g_matrix_avg = g_matrix.mean(axis = 0)
        g_matrix_lqt = g_matrix.quantile(0.25, axis = 0)
        g_matrix_uqt = g_matrix.quantile(0.75, axis = 0)
        g_control_matrix_avg = g_control_matrix.mean(axis = 0)
        g_control_matrix_lqt = g_control_matrix.quantile(0.25, axis = 0)
        g_control_matrix_uqt = g_control_matrix.quantile(0.75, axis = 0)
        plt.plot(np.arange(0, 200, 2)*0.07, g_matrix_avg, label = 'Ciprofloxacin', color = (0, 0.5, 1), alpha = 0.8)
        plt.fill_between(np.arange(0, 200, 2)*0.07,g_matrix_lqt, g_matrix_uqt, color = (0,0.5,1), alpha = 0.5)
        plt.plot(np.arange(0, 200, 2)*0.07, g_control_matrix_avg, label = 'Control', color = (1, 0.5, 0), alpha = 0.8)
        plt.fill_between(np.arange(0, 200, 2)*0.07, g_control_matrix_lqt, g_control_matrix_uqt, color = (1,0.5,0), alpha = 0.5)
        # plt.text(0.2, 0.8, '{}'.format(taxon_lookup.loc[taxon_lookup.cell_barcode.values == b, 'sci_name'].values[0]), transform = plt.axes().transAxes, fontsize = 8)
        plt.text(0.2, 0.6, '{0:s}\n$N_t$ = {1:.1f}\n$N_c$ = {2:.1f}'.format(taxon_lookup.loc[taxon_lookup.cell_barcode.values == b, 'sci_name'].values[0], barcode_abundance_merge.loc[barcode_abundance_merge.barcode.values == b, 'average_abundance_ciprofloxacin'].values[0], barcode_abundance_merge.loc[barcode_abundance_merge.barcode.values == b, 'average_abundance_control'].values[0]), transform = plt.axes().transAxes, fontsize = 6)
        plt.xlabel(r'Distance [$\mu$m]', fontsize = 8)
        plt.ylabel('Pair correlation', fontsize = 8)
        plt.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.25)
        plt.tick_params(direction = 'in', labelsize = 8)
        plt.savefig('{}/{}_pair_correlation_comparison.png'.format(input_directory, b), dpi = 300)
        plt.close()
    return

def measure_barcode_abundace_correlation(sam_tab, taxon_lookup, input_directory):
    cipro_samples = sam_tab.loc[sam_tab.Treatment.values == 'Ciprofloxacin', 'Sample'].values
    sample = cipro_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['barcode', 'abundance_0']
    for s in range(1, len(cipro_samples)):
        sample = cipro_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance = barcode_abundance.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].mean(axis = 1)
    barcode_abundance['std_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].std(axis = 1)
    control_samples = sam_tab.loc[sam_tab.Treatment.values == 'Control', 'Sample'].values
    sample = control_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance_control = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance_control.columns = ['barcode', 'abundance_0']
    for s in range(1, len(control_samples)):
        sample = control_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance_control = barcode_abundance_control.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance_control['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].mean(axis = 1)
    barcode_abundance_control['std_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].std(axis = 1)
    barcode_abundance_merge = barcode_abundance.loc[:,['barcode', 'average_abundance', 'std_abundance']].merge(barcode_abundance_control.loc[:,['barcode', 'average_abundance', 'std_abundance']], on = 'barcode', how = 'outer')
    barcode_abundance_merge.columns = ['barcode', 'average_abundance_ciprofloxacin', 'std_abundance_ciprofloxacin', 'average_abundance_control', 'std_abundance_control']
    barcode_abundance_merge = barcode_abundance_merge.fillna(0.1)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(12.5), cm_to_inches(10))
    plt.errorbar(barcode_abundance_merge.average_abundance_control.values, barcode_abundance_merge.average_abundance_ciprofloxacin.values, yerr = barcode_abundance_merge.std_abundance_ciprofloxacin.values, xerr = barcode_abundance_merge.std_abundance_control.values, fmt = 'o', color = (0,0.5,1), ecolor = 'black', capsize = 3)
    plt.plot([0.1,2000], [0.1,2000], '--', color = 'gray', linewidth = 1)
    plt.plot([0.1,200], [1,2000], '--', color = 'gray', linewidth = 1)
    plt.plot([1,2000], [0.1,200], '--', color = 'gray', linewidth = 1)
    plt.xscale('log')
    plt.yscale('log')
    for i in range(barcode_abundance_merge.shape[0]):
        barcode = barcode_abundance_merge.loc[i, 'barcode']
        label = taxon_lookup.loc[taxon_lookup.cell_barcode.values == barcode, 'sci_name'].values[0]
        label_x = barcode_abundance_merge.average_abundance_control.values[i]
        label_y = barcode_abundance_merge.average_abundance_ciprofloxacin.values[i]
        plt.text(label_x, label_y, label, fontsize = 8)
    plt.axes().set_aspect('equal')
    plt.xlabel('Control Abundance')
    plt.ylabel('Ciprofloxacin Abundance')
    plt.subplots_adjust(left = 0.12, bottom = 0.15, right = 0.85, top = 0.95)
    plt.tick_params(labelsize = 8, direction = 'in')
    plt.savefig('{}/abundance_comparison.png'.format(input_directory), dpi = 300)

def measure_barcode_abundace_correlation_by_section(sam_tab, taxon_lookup, input_directory):
    section_1_samples = sam_tab.loc[sam_tab.Section.values == 1, 'Sample'].values
    sample = section_1_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['barcode', 'abundance_0']
    for s in range(1, len(section_1_samples)):
        sample = section_1_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance = barcode_abundance.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(section_1_samples))]].mean(axis = 1)
    barcode_abundance['lqt_abundance'] = np.percentile(barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(section_1_samples))]], 25, axis = 1)
    barcode_abundance['uqt_abundance'] = np.percentile(barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(section_1_samples))]], 75, axis = 1)
    section_2_samples = sam_tab.loc[sam_tab.Section.values == 2, 'Sample'].values
    sample = section_2_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance_control = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance_control.columns = ['barcode', 'abundance_0']
    for s in range(1, len(section_2_samples)):
        sample = section_2_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance_control = barcode_abundance_control.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance_control['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(section_2_samples))]].mean(axis = 1)
    barcode_abundance_control['lqt_abundance'] = np.percentile(barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(section_2_samples))]], 25, axis = 1)
    barcode_abundance_control['uqt_abundance'] = np.percentile(barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(section_2_samples))]], 75, axis = 1)
    barcode_abundance_merge = barcode_abundance.loc[:,['barcode', 'average_abundance', 'lqd_abundance', 'uqt_abundance']].merge(barcode_abundance_control.loc[:,['barcode', 'average_abundance', 'lqt_abundance', 'uqt_abundance']], on = 'barcode', how = 'outer')
    barcode_abundance_merge.columns = ['barcode', 'average_abundance_section_1', 'lqt_abundance_section_1', 'uqt_abundance_section_1', 'average_abundance_section_2', 'lqt_abundance_section_2', 'uqt_abundance_section_2']
    barcode_abundance_merge = barcode_abundance_merge.fillna(0.1)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(12.5), cm_to_inches(10))
    plt.errorbar(barcode_abundance_merge.average_abundance_section_1.values, barcode_abundance_merge.average_abundance_section_2.values, yerr = [barcode_abundance_merge.lqt_abundance_section_2.values, barcode_abundance_merge.uqt_abundance_section_2.values], xerr = [barcode_abundance_merge.lqt_abundance_section_1.values, barcode_abundance_merge.uqt_abundance_section_2.values], fmt = 'o', color = (0,0.5,1), ecolor = 'black', capsize = 3, alpha = 0.5)
    plt.plot([0.1,2000], [0.1,2000], '--', color = 'gray', linewidth = 1)
    plt.plot([0.1,200], [1,2000], '--', color = 'gray', linewidth = 1)
    plt.plot([1,2000], [0.1,200], '--', color = 'gray', linewidth = 1)
    plt.xscale('log')
    plt.yscale('log')
    # for i in range(barcode_abundance_merge.shape[0]):
    #     barcode = barcode_abundance_merge.loc[i, 'barcode']
    #     label = taxon_lookup.loc[taxon_lookup.cell_barcode.values == barcode, 'sci_name'].values[0]
    #     label_x = barcode_abundance_merge.average_abundance_section_1.values[i]
    #     label_y = barcode_abundance_merge.average_abundance_section_2.values[i]
    #     plt.text(label_x, label_y, label, fontsize = 8)
    plt.axes().set_aspect('equal')
    plt.xlabel('Section 1 Abundance')
    plt.ylabel('Section 2 Abundance')
    plt.subplots_adjust(left = 0.12, bottom = 0.15, right = 0.85, top = 0.95)
    plt.tick_params(labelsize = 8, direction = 'in')
    plt.savefig('{}/abundance_comparison_by_section.png'.format(input_directory), dpi = 300)

def measure_pair_correlation(sample, nr, taxon_lookup):
    cell_info = pd.read_csv('{}_cell_information_filtered.csv'.format(sample), dtype = {'cell_barcode': str})
    barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['barcode', 'abundance']
    barcode_list = barcode_abundance.loc[barcode_abundance.abundance.values > 50, :].barcode.drop_duplicates()
    g_matrix = pd.DataFrame(index = barcode_list, columns = np.arange(nr))
    for b in barcode_list:
        cells = cell_info.iloc[cell_info['cell_barcode'].values == b].reset_index()
        distance = np.zeros((cells.shape[0], cells.shape[0]))
        boundary_distance = np.zeros((cells.shape[0], 4))
        print(cells.shape[0])
        for i in range(cells.shape[0]):
            for j in range(cells.shape[0]):
                x1 = cells.loc[i, 'centroid_x']
                x2 = cells.loc[j, 'centroid_x']
                y1 = cells.loc[i, 'centroid_y']
                y2 = cells.loc[j, 'centroid_y']
                distance[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        for i in range(cells.shape[0]):
            x1 = cells.loc[i, 'centroid_x']
            y1 = cells.loc[i, 'centroid_y']
            boundary_distance[i,0] = np.sqrt((x1 - 0)**2)
            boundary_distance[i,1] = np.sqrt((x1 - 1000)**2)
            boundary_distance[i,2] = np.sqrt((y1 - 0)**2)
            boundary_distance[i,3] = np.sqrt((y1 - 1000)**2)
        # minimum_boundary_distance = np.min(boundary_distance, axis = 1)
        g_matrix.loc[b,:] = pair_correlation(distance, boundary_distance, 2, nr, 1000*1000)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(6))
    for i in range(g_matrix.shape[0]):
        plt.plot(g_matrix.iloc[i,0:50]*0.07, label = taxon_lookup.loc[taxon_lookup.cell_barcode.values == g_matrix.index[i], 'sci_name'].values[0], alpha = 0.8)
    plt.xlabel(r'Distance [$\mu$m]', fontsize = 8)
    plt.ylabel('Pair correlation', fontsize = 8)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.15)
    plt.legend(fontsize = 8, frameon = False)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.savefig('{}_pair_correlation.png'.format(sample), dpi = 300)
    plt.close()
    return(g_matrix)

def measure_barcode_epithelial_distance_by_treatment(sam_tab, taxon_lookup, input_directory):
    cipro_samples = sam_tab.loc[sam_tab.Treatment.values == 'Ciprofloxacin', 'Sample'].values
    sample = cipro_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['barcode', 'abundance_0']
    for s in range(1, len(cipro_samples)):
        sample = cipro_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance = barcode_abundance.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].mean(axis = 1)
    barcode_abundance['std_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(cipro_samples))]].std(axis = 1)
    barcode_list = barcode_abundance.barcode.drop_duplicates()
    epithelial_boundary_distance_dict = {}
    for b in barcode_list:
        epithelial_boundary_distance = np.zeros((len(cipro_samples), 99))
        for s in range(len(cipro_samples)):
            sample = cipro_samples[s]
            cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
            cells = cell_info.iloc[cell_info['cell_barcode'].values == b].reset_index()
            values, bins = np.histogram(cells.epithelial_distance.values, bins = np.arange(0,1000,10))
            epithelial_boundary_distance[s, :] = values
        epithelial_boundary_distance_dict.update({b:epithelial_boundary_distance})
    control_samples = sam_tab.loc[sam_tab.Treatment.values == 'Control', 'Sample'].values
    sample = control_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance_control = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance_control.columns = ['barcode', 'abundance_0']
    for s in range(1, len(control_samples)):
        sample = control_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance_control = barcode_abundance_control.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance_control['average_abundance'] = barcode_abundance_control.loc[:,['abundance_{}'.format(k) for k in range(len(control_samples))]].mean(axis = 1)
    barcode_abundance_control['std_abundance'] = barcode_abundance_control.loc[:,['abundance_{}'.format(k) for k in range(len(control_samples))]].std(axis = 1)
    barcode_list_control = barcode_abundance_control.barcode.drop_duplicates()
    epithelial_boundary_distance_control_dict = {}
    for b in barcode_list_control:
        epithelial_boundary_distance = np.zeros((len(cipro_samples), 99))
        for s in range(len(control_samples)):
            sample = control_samples[s]
            cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
            cells = cell_info.iloc[cell_info['cell_barcode'].values == b].reset_index()
            values, bins = np.histogram(cells.epithelial_distance.values, bins = np.arange(0,1000,10))
            epithelial_boundary_distance[s, :] = values
        epithelial_boundary_distance_control_dict.update({b:epithelial_boundary_distance})
    barcode_intersection_list = list(set(barcode_list).intersection(barcode_list_control))
    for b in barcode_intersection_list:
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
        epithelial_boundary_distance = epithelial_boundary_distance_dict[b]
        epithelial_boundary_distance_control = epithelial_boundary_distance_control_dict[b]
        epithelial_boundary_distance_avg = np.mean(epithelial_boundary_distance, axis = 0)
        epithelial_boundary_distance_lqt = np.percentile(epithelial_boundary_distance, 25, axis = 0)
        epithelial_boundary_distance_uqt = np.percentile(epithelial_boundary_distance, 75, axis = 0)
        epithelial_boundary_distance_control_avg = np.mean(epithelial_boundary_distance_control, axis = 0)
        epithelial_boundary_distance_control_lqt = np.percentile(epithelial_boundary_distance_control, 25, axis = 0)
        epithelial_boundary_distance_control_uqt = np.percentile(epithelial_boundary_distance_control, 75, axis = 0)
        plt.plot(np.arange(0, 990, 10)*0.07, epithelial_boundary_distance_avg, label = 'Ciprofloxacin', color = (0, 0.5, 1), alpha = 0.8)
        plt.fill_between(np.arange(0, 990, 10)*0.07,epithelial_boundary_distance_lqt, epithelial_boundary_distance_uqt, color = (0,0.5,1), alpha = 0.5)
        plt.plot(np.arange(0, 990, 10)*0.07, epithelial_boundary_distance_control_avg, label = 'Control', color = (1, 0.5, 0), alpha = 0.8)
        plt.fill_between(np.arange(0, 990, 10)*0.07, epithelial_boundary_distance_control_lqt, epithelial_boundary_distance_control_uqt, color = (1,0.5,0), alpha = 0.5)
        plt.xlabel(r'Distance to Mucosa [$\mu$m]', fontsize = 8)
        plt.ylabel('Frequency', fontsize = 8)
        plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.25)
        plt.tick_params(direction = 'in', labelsize = 8)
        plt.savefig('{}/{}_epithelial_distance.png'.format(input_directory, b), dpi = 300)
        plt.close()
    return

def measure_epithelial_distance_by_treatment(sam_tab, taxon_lookup, input_directory):
    cipro_samples = sam_tab.loc[sam_tab.Treatment.values == 'Ciprofloxacin', 'Sample'].values
    sample = cipro_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    epithelial_distance = np.zeros((len(cipro_samples), 99))
    for s in range(len(cipro_samples)):
        sample = cipro_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        values, bins = np.histogram(cell_info.epithelial_distance.values, bins = np.arange(0,1000,10))
        epithelial_distance[s,:] = values
    control_samples = sam_tab.loc[sam_tab.Treatment.values == 'Control', 'Sample'].values
    sample = control_samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    epithelial_distance_control = np.zeros((len(control_samples), 99))
    for s in range(len(control_samples)):
        sample = control_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        values, bins = np.histogram(cell_info.epithelial_distance.values, bins = np.arange(0,1000,10))
        epithelial_distance_control[s,:] = values
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
    epithelial_distance_avg = np.mean(epithelial_distance, axis = 0)
    epithelial_distance_lqt = np.percentile(epithelial_distance, 25, axis = 0)
    epithelial_distance_uqt = np.percentile(epithelial_distance, 75, axis = 0)
    epithelial_distance_control_avg = np.mean(epithelial_distance_control, axis = 0)
    epithelial_distance_control_lqt = np.percentile(epithelial_distance_control, 25, axis = 0)
    epithelial_distance_control_uqt = np.percentile(epithelial_distance_control, 75, axis = 0)
    plt.plot(np.arange(0, 990, 10)*0.07, epithelial_distance_avg, label = 'Ciprofloxacin', color = (0, 0.5, 1), alpha = 0.8)
    plt.fill_between(np.arange(0, 990, 10)*0.07,epithelial_distance_lqt, epithelial_distance_uqt, color = (0,0.5,1), alpha = 0.5)
    plt.plot(np.arange(0, 990, 10)*0.07, epithelial_distance_control_avg, label = 'Control', color = (1, 0.5, 0), alpha = 0.8)
    plt.fill_between(np.arange(0, 990, 10)*0.07, epithelial_distance_control_lqt, epithelial_distance_control_uqt, color = (1,0.5,0), alpha = 0.5)
    plt.xlabel(r'Distance to Mucosa [$\mu$m]', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.25)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.savefig('{}/ciprofloxacin_vs_control_epithelial_distance.png'.format(input_directory), dpi = 300)
    plt.close()
    return

def measure_spatial_association_vs_abundance_by_treatment(sam_tab, taxon_lookup, data_folder):
    cipro_samples = sam_tab.loc[sam_tab.TREATMENT.values == 'ANTIBIOTICS',:]
    sample = cipro_samples.loc[0, 'SAMPLE']
    image_name = cipro_samples.loc[0, 'IMAGES']
    adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(cipro_samples)))
    association_probability_matrix = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(cipro_samples)))
    for s in range(len(cipro_samples)):
        sample = cipro_samples.loc[s, 'SAMPLE']
        image_name = cipro_samples.loc[s, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        for i in range(len(taxa_list)):
            for j in range(len(taxa_list)):
                taxa_i = taxa_list[i]
                taxa_j = taxa_list[j]
                abundance_i = cell_info.loc[cell_info.cell_barcode.values == taxa_i, :].shape[0]
                abundance_j = cell_info.loc[cell_info.cell_barcode.values == taxa_j, :].shape[0]
                all_possible_association = abundance_i*abundance_j
                if all_possible_association > 0:
                    association_probability_matrix[i,j,s] = adjacency_matrix_filtered.loc[taxa_i,taxa_j]/all_possible_association
    control_samples = sam_tab.loc[sam_tab.TREATMENT.values == 'CONTROL', :].reset_index().drop(columns = 'index')
    sample = cipro_samples.loc[0, 'SAMPLE']
    image_name = cipro_samples.loc[0, 'IMAGES']
    adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_control_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(control_samples)))
    association_probability_control_matrix = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(control_samples)))
    for s in range(len(control_samples)):
        sample = control_samples.loc[s,'SAMPLE']
        image_name = control_samples.loc[s,'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        adjacency_matrix_filtered = pd.read_csv('{}/{}/{}_adjacency_matrix_filtered.csv'.format(data_folder, sample, image_name), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_control_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        for i in range(len(taxa_list)):
            for j in range(len(taxa_list)):
                taxa_i = taxa_list[i]
                taxa_j = taxa_list[j]
                abundance_i = cell_info.loc[cell_info.cell_barcode.values == taxa_i, :].shape[0]
                abundance_j = cell_info.loc[cell_info.cell_barcode.values == taxa_j, :].shape[0]
                all_possible_association = abundance_i*abundance_j
                if all_possible_association > 0:
                    association_probability_control_matrix[i,j,s] = adjacency_matrix_filtered.loc[taxa_i,taxa_j]/all_possible_association
    association_probability_fold_change = np.zeros((adjacency_matrix_filtered.shape))
    association_probability_significance = np.zeros((adjacency_matrix_filtered.shape))
    for i in range(len(taxa_list)):
        for j in range(len(taxa_list)):
            association_probability_ij = association_probability_matrix[i,j,:]
            association_probability_ij_control = association_probability_control_matrix[i,j,:]
            pcij = np.average(association_probability_ij_control)
            if (pcij > 0) & (i != j):
                association_probability_fold_change[i,j] = np.average(association_probability_ij)/np.average(association_probability_ij_control)
                statistics, pvalue = scipy.stats.ttest_ind(association_probability_ij, association_probability_ij_control)
                association_probability_significance[i,j] = pvalue
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    plt.plot(np.log10(np.triu(association_probability_fold_change, k = 1)), -np.log10(np.triu(association_probability_significance, k = 1)), 'o', color = (0,0.5,1), markersize = 4, markeredgewidth = 0, alpha = 0.8)
    plt.xlim(-2.2,2.2)
    plt.ylim(-0.2, 4.2)
    plt.hlines(-np.log10(0.05), -2, 2, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(-1, 0, 6, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(1, 0, 6, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.15)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('$\log_{10}$(SAP Fold Change)', fontsize = 8)
    plt.ylabel('-$\log_{10}$(p)', fontsize = 8)
    plt.savefig('{}/association_probability_fold_change_significance.png'.format(input_directory), dpi = 300)
    plt.close()

    n_hypothesis = np.sum(association_probability_significance > 0)/2
    significant_indices = np.array(np.where(association_probability_significance < 0.05/n_hypothesis)).transpose()
    labels_list = []
    positions_list = []
    ha_list = []
    for i in range(significant_indices.shape[0]):
        barcode_i_index = significant_indices[i,0]
        barcode_j_index = significant_indices[i,1]
        if barcode_i_index > barcode_j_index:
            apfc = association_probability_fold_change[barcode_i_index, barcode_j_index]
            aps = association_probability_significance[barcode_i_index, barcode_j_index]
            if np.isfinite(np.abs(apfc)) > 0 and aps > 0 and np.abs(np.log10(apfc)) > 1:
                barcode_i = taxa_list[barcode_i_index]
                barcode_j = taxa_list[barcode_j_index]
                taxa_i = taxon_lookup.loc[taxon_lookup.code.values == barcode_i, 'sci_name'].values[0]
                taxa_j = taxon_lookup.loc[taxon_lookup.code.values == barcode_j, 'sci_name'].values[0]
                labels_list.append('{}-{}'.format(taxa_i, taxa_j))
                if np.log10(apfc) < 0:
                    position_x = np.log10(association_probability_fold_change[barcode_i_index, barcode_j_index]) - 0.5
                    position_y = -np.log10(association_probability_significance[barcode_i_index, barcode_j_index]) + 0.05
                    positions_list.append([position_x, position_y])
                    ha_list.append('left')
                else:
                    position_x = np.log10(association_probability_fold_change[barcode_i_index, barcode_j_index]) + 0.5
                    position_y = -np.log10(association_probability_significance[barcode_i_index, barcode_j_index]) + 0.05
                    positions_list.append([position_x, position_y])
                    ha_list.append('right')
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8.5), cm_to_inches(8.5))
    plt.plot(np.log10(np.triu(association_probability_fold_change, k = 1)), -np.log10(np.triu(association_probability_significance, k = 1)), 'o', color = (0,0.5,1), markersize = 4, markeredgewidth = 0, alpha = 0.8)
    plt.xlim(-2.2,2.2)
    # plt.ylim(-0.2, 4.5)
    plt.hlines(-np.log10(0.05/n_hypothesis), -2, 2, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(-1, 0, 7, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(1, 0, 7, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    for i in range(len(labels_list)):
        pos_x = positions_list[i][0]
        pos_y = positions_list[i][1]
        plt.text(pos_x, pos_y, labels_list[i], ha = ha_list[i], fontsize = 6)

    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.15)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('$\log_{10}$(SAP Fold Change)', fontsize = 8)
    plt.ylabel('-$\log_{10}$(p)', fontsize = 8)
    plt.savefig('{}/association_probability_fold_change_significance_labeled.png'.format(data_folder), dpi = 300)
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

def measure_spatial_association_vs_abundance_by_treatment_presentation(sam_tab, taxon_lookup, input_directory):
    cipro_samples = sam_tab.loc[sam_tab.Treatment.values == 'Ciprofloxacin', 'Sample'].values
    sample = cipro_samples[0]
    adjacency_matrix_filtered = pd.read_csv('{}/{}_adjacency_matrix_filtered.csv'.format(input_directory, sample), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(cipro_samples)))
    association_probability_matrix = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(cipro_samples)))
    for s in range(len(cipro_samples)):
        sample = cipro_samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode':str})
        adjacency_matrix_filtered = pd.read_csv('{}/{}_adjacency_matrix_filtered.csv'.format(input_directory, sample), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        for i in range(len(taxa_list)):
            for j in range(len(taxa_list)):
                taxa_i = taxa_list[i]
                taxa_j = taxa_list[j]
                abundance_i = cell_info.loc[cell_info.cell_barcode.values == taxa_i, :].shape[0]
                abundance_j = cell_info.loc[cell_info.cell_barcode.values == taxa_j, :].shape[0]
                all_possible_association = abundance_i*abundance_j
                if all_possible_association > 0:
                    association_probability_matrix[i,j,s] = adjacency_matrix_filtered.loc[taxa_i,taxa_j]/all_possible_association
    control_samples = sam_tab.loc[sam_tab.Treatment.values == 'Control', 'Sample'].values
    sample = control_samples[0]
    adjacency_matrix_filtered = pd.read_csv('{}/{}_adjacency_matrix_filtered.csv'.format(input_directory, sample), dtype = {0:str}).set_index('Unnamed: 0')
    adjacency_matrix_control_merge = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(control_samples)))
    association_probability_control_matrix = np.zeros((adjacency_matrix_filtered.shape[0], adjacency_matrix_filtered.shape[1], len(control_samples)))
    for s in range(len(control_samples)):
        sample = control_samples[s]
        adjacency_matrix_filtered = pd.read_csv('{}/{}_adjacency_matrix_filtered.csv'.format(input_directory, sample), dtype = {0:str}).set_index('Unnamed: 0')
        adjacency_matrix_control_merge[:,:,s] = adjacency_matrix_filtered.values
        taxa_list = adjacency_matrix_filtered.columns.values
        for i in range(len(taxa_list)):
            for j in range(len(taxa_list)):
                taxa_i = taxa_list[i]
                taxa_j = taxa_list[j]
                abundance_i = cell_info.loc[cell_info.cell_barcode.values == taxa_i, :].shape[0]
                abundance_j = cell_info.loc[cell_info.cell_barcode.values == taxa_j, :].shape[0]
                all_possible_association = abundance_i*abundance_j
                if all_possible_association > 0:
                    association_probability_control_matrix[i,j,s] = adjacency_matrix_filtered.loc[taxa_i,taxa_j]/all_possible_association
    association_probability_fold_change = np.zeros((adjacency_matrix_filtered.shape))
    association_probability_significance = np.zeros((adjacency_matrix_filtered.shape))
    for i in range(len(taxa_list)):
        for j in range(len(taxa_list)):
            association_probability_ij = association_probability_matrix[i,j,:]
            association_probability_ij_control = association_probability_control_matrix[i,j,:]
            association_probability_fold_change[i,j] = np.average(association_probability_ij)/np.average(association_probability_ij_control)
            statistics, pvalue = scipy.stats.ttest_ind(association_probability_ij, association_probability_ij_control)
            association_probability_significance[i,j] = pvalue
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    plt.plot(np.log10(np.triu(association_probability_fold_change, k = 1)), -np.log10(np.triu(association_probability_significance, k = 1)), 'o', color = (0,0.5,1), markersize = 4, markeredgewidth = 0, alpha = 0.8)
    plt.xlim(-2.2,2.2)
    plt.ylim(-0.2, 4.2)
    plt.hlines(-np.log10(0.05), -2, 2, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(-1, 0, 6, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(1, 0, 6, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.15)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.xlabel('$\log_{10}$(SAP Fold Change)', fontsize = 8)
    plt.ylabel('-$\log_{10}$(p)', fontsize = 8)
    plt.savefig('{}/association_probability_fold_change_significance_presentation.png'.format(input_directory), dpi = 300)
    plt.close()

    significant_indices = np.array(np.where(association_probability_significance < 0.0001)).transpose()
    labels_list = []
    positions_list = []
    ha_list = []
    for i in range(significant_indices.shape[0]):
        barcode_i_index = significant_indices[i,0]
        barcode_j_index = significant_indices[i,1]
        if barcode_i_index > barcode_j_index:
            apfc = association_probability_fold_change[barcode_i_index, barcode_j_index]
            aps = association_probability_significance[barcode_i_index, barcode_j_index]
            if np.isfinite(np.abs(apfc)) > 0 and aps > 0 and np.abs(np.log10(apfc)) > 1:
                barcode_i = taxa_list[barcode_i_index]
                barcode_j = taxa_list[barcode_j_index]
                taxa_i = taxon_lookup.loc[taxon_lookup.cell_barcode.values == barcode_i, 'sci_name'].values[0]
                taxa_j = taxon_lookup.loc[taxon_lookup.cell_barcode.values == barcode_j, 'sci_name'].values[0]
                labels_list.append('{}-{}'.format(taxa_i, taxa_j))
                if np.log10(apfc) < 0:
                    position_x = np.log10(association_probability_fold_change[barcode_i_index, barcode_j_index]) - 0.5
                    position_y = -np.log10(association_probability_significance[barcode_i_index, barcode_j_index]) + 0.05
                    positions_list.append([position_x, position_y])
                    ha_list.append('left')
                else:
                    position_x = np.log10(association_probability_fold_change[barcode_i_index, barcode_j_index]) + 0.5
                    position_y = -np.log10(association_probability_significance[barcode_i_index, barcode_j_index]) + 0.05
                    positions_list.append([position_x, position_y])
                    ha_list.append('right')
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8.5), cm_to_inches(8.5))
    plt.plot(np.log10(np.triu(association_probability_fold_change, k = 1)), -np.log10(np.triu(association_probability_significance, k = 1)), 'o', color = (0,0.5,1), markersize = 4, markeredgewidth = 0, alpha = 0.8)
    plt.xlim(-2.2,2.2)
    plt.axes().spines['left'].set_color('white')
    plt.axes().spines['bottom'].set_color('white')
    plt.axes().spines['right'].set_color('white')
    plt.axes().spines['top'].set_color('white')
    # plt.ylim(-0.2, 4.5)
    plt.hlines(-np.log10(0.0001), -2, 2, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(-1, 0, 7, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    plt.vlines(1, 0, 7, linestyles = '--', color = (1,0.5,0), linewidth = 1)
    for i in range(len(labels_list)):
        pos_x = positions_list[i][0]
        pos_y = positions_list[i][1]
        plt.text(pos_x, pos_y, labels_list[i], ha = ha_list[i], fontsize = 6, color = 'white')

    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.15)
    plt.tick_params(direction = 'in', labelsize = 8, colors = 'white')
    plt.xlabel('$\log_{10}$(SAP Fold Change)', fontsize = 8, color = 'white')
    plt.ylabel('-$\log_{10}$(p)', fontsize = 8, color = 'white')
    plt.savefig('{}/association_probability_fold_change_significance_labeled_presentation.png'.format(input_directory), dpi = 300, transparent = True)
    plt.close()

    association_probability_fold_change_filtered = association_probability_fold_change.copy()
    association_probability_fold_change_filter = np.isnan(association_probability_fold_change).all(axis = 0)
    association_probability_fold_change_filtered = association_probability_fold_change_filtered[~association_probability_fold_change_filter,:]
    association_probability_fold_change_filtered = association_probability_fold_change_filtered[:,~association_probability_fold_change_filter]
    label_codes = taxa_list[~association_probability_fold_change_filter]
    tick_labels = [taxon_lookup.loc[taxon_lookup.cell_barcode.values == x, 'sci_name'].values[0] for x in label_codes.tolist()]
    cmap = matplotlib.cm.RdBu
    cmap.set_bad('grey')
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(10),cm_to_inches(10))
    ax = plt.Axes(fig, [0.01, 0.3, 0.69, 0.69])
    fig.add_axes(ax)
    mappable = ax.imshow(np.log10(association_probability_fold_change_filtered), cmap = cmap)
    plt.xticks(np.arange(association_probability_fold_change_filtered.shape[0]), tick_labels, rotation = 90, style = 'italic')
    plt.yticks(np.arange(association_probability_fold_change_filtered.shape[1]), tick_labels, style = 'italic')
    plt.tick_params(direction = 'in', length = 0, labelsize = 6, colors = 'white')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_right()
    cbaxes = inset_axes(ax, width="4%", height="25%", loc = 4, bbox_to_anchor = (0,-0.4,1.35,1.2), bbox_transform = ax.transAxes)
    cbar = plt.colorbar(mappable, cax=cbaxes, orientation = 'vertical')
    cbar.set_label(r'$\log_{10}$(FC)', color = 'white', fontsize = 8)
    cbar.ax.tick_params(labelsize = 8, direction = 'in', colors = 'white', length = 3)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    plt.subplots_adjust(left = 0.01, bottom = 0.4, right = 0.6, top = 0.99)
    plt.savefig('{}/ciprofloxacin_differential_association_probability_matrix_presentation.pdf'.format(input_directory), dpi = 300, transparent = True)
    return

def measure_max_intensity_distribution(sample):
    cell_info = pd.read_csv('{}_cell_information_filtered.csv'.format(sample), dtype = {'cell_barcode': str})
    avgint = pd.read_csv('{}_avgint.csv'.format(sample), dtype = float)
    max_intensity = np.max(avgint.values.astype(float), axis = 1)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(3.5))
    plt.hist(max_intensity, bins = np.arange(0,1,0.01), color = 'orange')
    plt.xlabel('Maximum Intensity [A.U.]', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.25)
    plt.tick_params(direction = 'in', labelsize = 8)
    plt.savefig('{}_max_intensity_histogram.png'.format(sample), dpi = 300)
    plt.close()
    return

def measure_max_intensity_vs_abundance(sample, taxon_lookup):
    cell_info = pd.read_csv('{}_cell_information_filtered.csv'.format(sample), dtype = {'cell_barcode': str})
    avgint = pd.read_csv('{}_avgint.csv'.format(sample), dtype = float)
    cell_info['max_intensity'] = np.max(avgint.values.astype(float), axis = 1)
    code_max_intensity = cell_info.groupby(['cell_barcode'])['max_intensity'].mean().reset_index()
    code_max_intensity.columns = ['cell_barcode', 'max_intensity']
    code_abundance = cell_info.groupby(['cell_barcode'])['max_intensity'].count().reset_index()
    code_abundance.columns = ['cell_barcode', 'abundance']
    max_intensity_abundance = code_max_intensity.merge(code_abundance, on = 'cell_barcode')
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(3.5))
    for i in range(max_intensity_abundance.shape[0]):
        code = max_intensity_abundance.cell_barcode.values[i]
        color = taxon_lookup.loc[taxon_lookup.cell_barcode.values == code, ['H', 'S', 'V']]
        rgb = tuple(hsv_to_rgb(color.values)[0])
        plt.plot(max_intensity_abundance.loc[i, 'abundance'], max_intensity_abundance.loc[i, 'max_intensity'], 'o', color = rgb, markersize = 2)
    plt.xscale('log')
    plt.xlabel('Abundance', fontsize = 8)
    plt.ylabel('Max Intensity [A.U.]', fontsize = 8)
    plt.xlim(0.1, 2000)
    plt.ylim(-0.05,1.05)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)
    plt.minorticks_off()
    plt.savefig('{}_max_intensity_vs_abundance.png'.format(sample), dpi = 300)
    plt.close()
    return

def measure_svm_probability(s):
    cell_info = pd.read_csv('{}_cell_information_filtered.csv'.format(s), dtype = {'cell_barcode': str})
    probability = np.max(cell_info.iloc[:,68:68+47-1].values.astype(float), axis = 1)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(3.5))
    plt.hist(probability, bins = np.arange(0,1,0.02), color = (0,0.5,1))
    plt.xlabel('Probability', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2)
    plt.subplots_adjust(left=0.25, right=0.96, top=0.96, bottom=0.25)
    plt.savefig('{}_probability_distribution.png'.format(s), dpi = 300)
    plt.close()
    return

def measure_svm_probability_vs_max_intensity(s):
    cell_info = pd.read_csv('{}_cell_information_filtered.csv'.format(s), dtype = {67: str})
    avgint = pd.read_csv('{}_avgint.csv'.format(s), dtype = float)
    cell_info['probability'] = np.max(cell_info.iloc[:,68:68+47-1].values.astype(float), axis = 1)
    cell_info['max_intensity'] = np.max(avgint.values.astype(float), axis = 1)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(3.5))
    plt.plot(cell_info.max_intensity.values, cell_info.probability.values, 'o', markersize = 1, color = (0,0.5,1), alpha = 0.5)
    plt.xlabel('Max Intensity [A.U.]', fontsize = 8)
    plt.ylabel('Probability', fontsize = 8)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2)
    plt.subplots_adjust(left=0.25, right=0.96, top=0.96, bottom=0.25)
    plt.savefig('{}_probability_vs_max_intensity.png'.format(s), dpi = 300)
    plt.close()
    return

def measure_svm_probability_vs_area(s):
    cell_info = pd.read_csv('{}_cell_information_filtered.csv'.format(s), dtype = {'cell_barcode': str})
    avgint = pd.read_csv('{}_avgint.csv'.format(s), dtype = float)
    cell_info['probability'] = np.max(cell_info.iloc[:,68:68+47-1].values.astype(float), axis = 1)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(3.5))
    plt.plot(cell_info.area.values, cell_info.probability.values, 'o', markersize = 2, color = (0,0.5,1), alpha = 0.5)
    plt.xlabel('Area', fontsize = 8)
    plt.ylabel('Probability', fontsize = 8)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.25)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2)
    plt.savefig('{}_probability_vs_area.png'.format(s), dpi = 300)
    plt.close()
    return

def measure_barcode_spectra(sam_tab, taxon_lookup, input_directory):
    samples = sam_tab.Sample.values
    sample = samples[0]
    cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
    barcode_abundance = cell_info.cell_barcode.value_counts().reset_index()
    barcode_abundance.columns = ['barcode', 'abundance_0']
    cell_info_list = []
    avgint_list = []
    for s in range(1, len(samples)):
        sample = samples[s]
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(input_directory, sample), dtype = {'cell_barcode': str})
        avgint = pd.read_csv('{}/{}_avgint_filtered.csv'.format(input_directory, sample))
        cell_info_list.append(cell_info)
        avgint_list.append(avgint)
        barcode_abundance_temp = cell_info.cell_barcode.value_counts().reset_index()
        barcode_abundance_temp.columns = ['barcode', 'abundance_{}'.format(s)]
        barcode_abundance = barcode_abundance.merge(barcode_abundance_temp, on = 'barcode', how = 'outer')
    barcode_abundance['average_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(samples))]].mean(axis = 1)
    barcode_abundance['std_abundance'] = barcode_abundance.loc[:,['abundance_{}'.format(k) for k in range(len(samples))]].std(axis = 1)
    barcode_list = barcode_abundance.barcode.drop_duplicates()
    for bc in barcode_list:
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
        for i in range(len(cell_info_list)):
            cell_info = cell_info_list[i]
            avgint = avgint_list[i]
            spec = avgint.loc[cell_info.cell_barcode.values == bc, :]
            plt.plot(spec.values.transpose(), '-', color = (0,0.5,1), alpha = 0.5)
            plt.text(55, 0.8, bc, fontsize = 8)
    plt.axes().set_aspect('equal')
    plt.xlabel('Channel')
    plt.ylabel('Intensity')
    plt.subplots_adjust(left = 0.12, bottom = 0.15, right = 0.85, top = 0.95)
    plt.tick_params(labelsize = 8, direction = 'in')
    plt.savefig('{}/abundance_comparison.png'.format(input_directory), dpi = 300)

def cross_sample_comparison(sam_tab, data_folder, taxon_lookup):
    samples = sam_tab.SAMPLE.values
    cell_info_list = []
    for s in range(samples.shape[0]):
        sample = sam_tab.loc[s, 'SAMPLE']
        image_name = sam_tab.loc[s, 'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode': str})
        cell_info_list.append(cell_info)
    taxa_max_intensity_df = cell_info_list[0].groupby(['cell_barcode'])['max_intensity'].mean().reset_index()
    taxa_max_intensity_df.columns = ['cell_barcode', 'max_intensity_0']
    for i in range(1, len(samples)):
        taxa_max_intensity = cell_info_list[i].groupby(['cell_barcode'])['max_intensity'].mean().reset_index()
        taxa_max_intensity.columns = ['cell_barcode', 'max_intensity_{}'.format(i)]
        taxa_max_intensity_df = taxa_max_intensity_df.merge(taxa_max_intensity, on = 'cell_barcode', how = 'outer')
    # taxa_max_intensity_df.fillna(0, inplace = True)
    taxa_max_intensity_lookup = taxa_max_intensity_df.merge(taxon_lookup, on = 'cell_barcode', how = 'left')
    taxa_max_intensity_lookup.to_csv('{}/taxa_max_intensity_lookup.csv'.format(data_folder), header = None, index = None)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(18), cm_to_inches(18))
    # taxa_max_intensity_df_gp = taxa_max_intensity_lookup.loc[taxa_max_intensity_lookup.gram_stain.values == 'Positive', :]
    # taxa_max_intensity_df_gn = taxa_max_intensity_lookup.loc[taxa_max_intensity_lookup.gram_stain.values == 'Negative', :]
    # taxa_max_intensity_df_gv = taxa_max_intensity_lookup.loc[taxa_max_intensity_lookup.gram_stain.values == 'Variable', :]
    gs = GridSpec(taxa_max_intensity_df.shape[1] - 1, taxa_max_intensity_df.shape[1] - 1)
    for i in range(taxa_max_intensity_df.shape[1] - 1):
        for j in range(i, taxa_max_intensity_df.shape[1] - 1):
            i_na_index = taxa_max_intensity_df.iloc[:,i+1].isna()
            j_na_index = taxa_max_intensity_df.iloc[:,j+1].isna()
            na_index = i_na_index | j_na_index
            ax = plt.subplot(gs[i,j])
            ax.plot(taxa_max_intensity_df.iloc[:,i+1][~na_index], taxa_max_intensity_df.iloc[:,j+1][~na_index], 'o', color = (0,0.5,1), markersize = 1)
            ax.plot(taxa_max_intensity_df.iloc[:,i+1][~i_na_index & j_na_index], np.zeros(taxa_max_intensity_df.iloc[:,i+1][~i_na_index & j_na_index].shape[0]), 'o', color = (1,0.5,0), markersize = 1)
            ax.plot(np.zeros(taxa_max_intensity_df.iloc[:,j+1][~j_na_index & i_na_index].shape[0]), taxa_max_intensity_df.iloc[:,j+1][~j_na_index & i_na_index], 'o', color = (1,0.5,0), markersize = 1)
            ax.plot([0,1], [0,1], '--', color = 'gray', linewidth = 1)
            ax.tick_params(labelsize = 6, direction = 'in', length = 2, bottom = False, left = False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.xlim(-0.05,1.05)
            plt.ylim(-0.05,1.05)
    for i in range(taxa_max_intensity_df.shape[1] - 1):
        ax = plt.subplot(gs[0,i])
        ax.set_title(sam_tab.Mouse_treatment.values[i], fontsize = 8, rotation = 90, va = 'bottom')
    for i in range(taxa_max_intensity_df.shape[1] - 1):
        ax = plt.subplot(gs[i, taxa_max_intensity_df.shape[1] - 2])
        ax.get_yaxis().set_visible(True)
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(sam_tab.Mouse_treatment.values[i], fontsize = 8, rotation = 0, ha = 'left')
    ax = plt.subplot(gs[0,0])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    plt.subplots_adjust(left=0.02, right=0.85, top=0.85, bottom=0.02)
    plt.savefig('{}/taxa_max_intensity_correlation.png'.format(input_directory), dpi = 300)
    plt.close()
    intensity_pearson_matrix = np.zeros((len(cell_info_list),len(cell_info_list)))
    intensity_pval_matrix = np.zeros((len(cell_info_list),len(cell_info_list)))
    for i in range(taxa_max_intensity_df.shape[1] - 1):
        for j in range(taxa_max_intensity_df.shape[1] - 1):
            i_na_index = taxa_max_intensity_df.iloc[:,i+1].isna()
            j_na_index = taxa_max_intensity_df.iloc[:,j+1].isna()
            na_index = i_na_index | j_na_index
            pearson, p_val = scipy.stats.pearsonr(taxa_max_intensity_df.iloc[:,i+1][~na_index], taxa_max_intensity_df.iloc[:,j+1][~na_index])
            intensity_pearson_matrix[i,j] = pearson
            intensity_pval_matrix [i,j]= p_val
    pd.DataFrame(intensity_pearson_matrix).to_csv('{}/intensity_pearson_matrix.csv'.format(data_folder), header = None, index = None)
    pd.DataFrame(intensity_pval_matrix).to_csv('{}/intensity_pval_matrix.csv'.format(data_folder), header = None, index = None)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    ax = plt.Axes(fig, [0.005,0.05,0.8,0.8])
    fig.add_axes(ax)
    mappable = ax.imshow(intensity_pearson_matrix, cmap = 'inferno')
    # ax.xaxis.tick_top()
    # plt.xticks(np.arange(intensity_pearson_matrix.shape[0]), sam_tab.Mouse_treatment.values, rotation = 90, fontsize = 8)
    # ax.yaxis.tick_right()
    # plt.yticks(np.arange(intensity_pearson_matrix.shape[0]), sam_tab.Mouse_treatment.values, fontsize = 8)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tick_params(direction = 'in', length = 2)
    cbaxes = inset_axes(ax, width="4%", height="40%", loc = 5, bbox_to_anchor = (0.5,0.05,0.7,0.7), bbox_transform = ax.transAxes)
    cbar = plt.colorbar(mappable, cax = cbaxes, orientation = 'vertical')
    cbar.ax.tick_params(labelsize = 8, direction = 'in', length = 3)
    cbar.set_label(r'$\rho_{pearson}$', color = 'black', fontsize = 8)
    cbar.ax.yaxis.tick_right()
    cbar.ax.yaxis.set_label_position('right')
    plt.savefig('{}/taxa_max_intensity_pearson_correlation.png'.format(input_directory), dpi = 300)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(18), cm_to_inches(18))
    ax = plt.Axes(fig, [0.005,0.05,0.7,0.7])
    fig.add_axes(ax)
    mappable = ax.imshow(-np.log10(intensity_pval_matrix), cmap = 'inferno')
    ax.xaxis.tick_top()
    plt.xticks(np.arange(intensity_pearson_matrix.shape[0]), sam_tab.Mouse_treatment.values, rotation = 90, fontsize = 8)
    ax.yaxis.tick_right()
    plt.yticks(np.arange(intensity_pearson_matrix.shape[0]), sam_tab.Mouse_treatment.values, fontsize = 8)
    plt.tick_params(direction = 'in', length = 2)
    cbaxes = inset_axes(ax, width="4%", height="40%", loc = 5, bbox_to_anchor = (0.5,0.05,0.95,0.95), bbox_transform = ax.transAxes)
    cbar = plt.colorbar(mappable, cax = cbaxes, orientation = 'vertical')
    cbar.set_label(r'$-\log_{10}$(p)', color = 'black', fontsize = 8)
    cbar.ax.tick_params(labelsize = 8, direction = 'in', length = 3)
    cbar.ax.yaxis.tick_right()
    cbar.ax.yaxis.set_label_position('right')
    plt.savefig('{}/taxa_max_intensity_pearson_pval.png'.format(input_directory), dpi = 300)
    return

def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('sample_table', type = str, help = 'Input folder containing images of biological samples')
    parser.add_argument('input_directory', type = str, help = 'Input folder containing images of biological samples')
    parser.add_argument('taxon_lookup', type = str, help = 'Input folder containing images of biological samples')
    args = parser.parse_args()
    taxon_lookup = pd.read_csv(args.taxon_lookup, dtype = {'code': str})
    taxon_lookup = taxon_lookup.rename(columns = {'code':'cell_barcode'})
    sam_tab = pd.read_csv(args.sample_table)
    sam_tab['NCells'] = 0
    sam_tab['NTaxa'] = 0
    for i in range(sam_tab.shape[0]):
        sample = sam_tab.loc[i, 'Sample']
        cell_info = pd.read_csv('{}/{}_cell_information_filtered.csv'.format(args.input_directory, sample), dtype = {'cell_barcode': str})
        sam_tab.loc[i, 'NCells'] = cell_info.shape[0]
        sam_tab.loc[i, 'NTaxa'] = cell_info['cell_barcode'].drop_duplicates().shape[0]
    sam_tab['Mouse_treatment'] = sam_tab['Mouse'].astype(str) + sam_tab['Location'] + 'FOV' + sam_tab['FOV'].astype(str)
    sam_tab.to_csv(re.sub('.csv', '_results.csv', args.sample_table))
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(18), cm_to_inches(8))
    plt.plot(np.arange(sam_tab.Mouse_treatment.shape[0]), sam_tab.NCells.values, 'o', color = (1,0.5,0))
    plt.tick_params(labelsize = 8, direction = 'in')
    plt.xticks(np.arange(sam_tab.shape[0]), sam_tab.Mouse_treatment.values, rotation = 90)
    plt.ylabel('Number of Cells')
    plt.ylim(0, sam_tab.NCells.max() + 400)
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.35)
    plt.savefig('{}/number_of_cells_vs_mouse_treatment.png'.format(args.input_directory), dpi = 300)

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(18), cm_to_inches(8))
    plt.plot(np.arange(sam_tab.Mouse_treatment.shape[0]), sam_tab.NTaxa.values, 'o', color = (1,0.5,0))
    plt.tick_params(labelsize = 8, direction = 'in')
    plt.xticks(np.arange(sam_tab.shape[0]), sam_tab.Mouse_treatment.values, rotation = 90)
    plt.ylabel('Number of Taxa')
    plt.ylim(0, sam_tab.NTaxa.max() + 2)
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.35)
    plt.savefig('{}/number_of_taxa_vs_mouse_treatment.png'.format(args.input_directory), dpi = 300)
    # cross_sample_comparison(sam_tab, args.input_directory, taxon_lookup)
    # measure_barcode_abundace_correlation_by_section(sam_tab, taxon_lookup, args.input_directory)
    # for s in sam_tab.Sample.values:
    #     measure_svm_probability('{}/{}'.format(args.input_directory, s))
    #     measure_svm_probability_vs_max_intensity('{}/{}'.format(args.input_directory, s))
    #     measure_svm_probability_vs_area('{}/{}'.format(args.input_directory, s))
    #     measure_max_intensity_distribution('{}/{}'.format(args.input_directory, s))
    #     measure_max_intensity_vs_abundance('{}/{}'.format(args.input_directory, s), taxon_lookup)
    #     measure_pair_correlation('{}/{}'.format(args.input_directory, s), 100, taxon_lookup)
    # measure_pair_correlation_by_treatment(sam_tab, 100, taxon_lookup, args.input_directory)
    # measure_epithelial_distance_by_treatment(sam_tab, taxon_lookup, args.input_directory)
    # measure_barcode_abundace_correlation(sam_tab, taxon_lookup, args.input_directory)
    # measure_spatial_association_vs_abundance_by_treatment(sam_tab, taxon_lookup, args.input_directory)
    measure_spatial_association_vs_abundance_by_treatment_presentation(sam_tab, taxon_lookup, args.input_directory)
    return

if __name__ == '__main__':
    main()






















#xxxx
