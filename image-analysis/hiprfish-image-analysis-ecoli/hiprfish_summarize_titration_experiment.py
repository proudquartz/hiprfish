
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
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib
matplotlib.params['axes.linewidth'] = 0.5

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def cm_to_inches(length):
    return(length/2.54)

def bootstrap_estimate_mean(df, nb):
    df_fov = df.filter(regex = 'FOV*')
    bs_mean = np.mean(bootstrap(df_fov.values, nb, bootfunc = np.mean))
    bs_lq = np.percentile(bootstrap(df_fov.values, nb, bootfunc = np.mean), 25)
    bs_uq = np.percentile(bootstrap(df_fov.values, nb, bootfunc = np.mean),75)
    return(pd.Series({'BinaryBarcode': df.BinaryBarcode, 'BootstrapMean': bs_mean, 'BootstrapLower': bs_lq, 'BootstrapUpper': bs_uq}))

def plot_abundance_correlation(sum_tab_filename, sum_tab, input_tab, m):
    abundance = sum_tab.drop(columns = ['Barcodes'])
    mean_absolute_abundance = abundance.sum(axis = 1)
    sum_tab['MeasuredAbundance'] = mean_absolute_abundance/np.sum(mean_absolute_abundance)
    sum_tab = sum_tab.merge(input_tab, how = 'left', on = 'Barcodes').fillna(0)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4.5),cm_to_inches(4.5))
    plt.plot(sum_tab.Concentration.values*1000, sum_tab.MeasuredAbundance.values*1000, '.', markersize = 2, color = 'blue', alpha = 0.5)
    plt.plot(sum_tab.Concentration.values*1000, sum_tab.MeasuredAbundance.values*1000, '.', markersize = 2, color = 'orange', alpha = 0.5)
    plt.xlabel(r'Input Abundance$\times 10^{3}$', fontsize = 8)
    plt.ylabel(r'Measured Abundance$\times 10^{3}$', fontsize = 8)
    lim_max = np.maximum(np.max(sum_tab.Concentration), np.max(sum_tab.MeasuredAbundance))*1.05
    abundance_correlation_filename = re.sub('_abundance.csv', '_abundance_correlation.tif', sum_tab_filename)
    plt.plot([0,lim_max*1000], [0,lim_max*1000], 'w--', alpha = 0.8, linewidth = 0.5)
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 8, color = 'black', labelcolor = 'black')
    plt.tight_layout(pad = 0)
    plt.axes().set_aspect('equal')
    plt.axes().spines['bottom'].set_color('black')
    plt.axes().spines['left'].set_color('black')
    gross_error_rate = sum_tab.loc[sum_tab.Concentration.values == 0].MeasuredAbundance.sum()
    plt.xlim(-0.5,lim_max*1000)
    plt.ylim(-0.5,lim_max*1000)
    plt.savefig(abundance_correlation_filename, dpi = 300, format = 'tiff', transparent = True)
    plt.close()

def plot_abundance_correlation_all(data_dir):
    files = glob.glob('{}/images_table_mix_*_results_abundance.csv'.format(data_dir))
    fig_abundance = plt.figure(0)
    fig_fp = plt.figure(1)
    fig_abundance.set_size_inches(cm_to_inches(2.8),cm_to_inches(2.8))
    fig_fp.set_size_inches(cm_to_inches(4),cm_to_inches(3))
    plt.figure(0)
    color_list = ['darkviolet', 'navy', 'fuchsia', 'red', 'limegreen', 'gold', 'darkorange', 'dodgerblue']
    for i in range(len(files)):
        filename = files[i]
        sum_tab = pd.read_csv(filename)
        mix_id = int(re.sub('mix_', '', re.search('mix_[0-9]*', filename).group(0)))
        input_tab_filename = '{}/hiprfish_1023_mix_{}.csv'.format(data_dir, str(mix_id))
        input_tab = pd.read_csv(input_tab_filename)
        abundance = sum_tab.drop(columns = ['Barcodes'])
        mean_absolute_abundance = abundance.sum(axis = 1)
        ul_absolute_abundance = np.percentile(abundance.values, 75, axis = 1)
        ll_absolute_abundance = np.percentile(abundance.values, 25, axis = 1)
        sum_tab['MeasuredAbundance'] = mean_absolute_abundance/np.sum(mean_absolute_abundance)
        sum_tab['ULAbundance'] = ul_absolute_abundance/np.sum(mean_absolute_abundance)
        sum_tab['LLAbundance'] = ll_absolute_abundance/np.sum(mean_absolute_abundance)
        sum_tab = sum_tab.merge(input_tab, how = 'left', on = 'Barcodes').fillna(0)
        sum_tab_fp = sum_tab.loc[sum_tab.Concentration.values == 0]
        plt.figure(0)
        plt.plot(sum_tab.Concentration.values*1000, sum_tab.MeasuredAbundance.values*1000, '.', markersize = 4, alpha = 0.5, color = color_list[i], markeredgewidth = 0)
        sum_tab_trim = sum_tab[sum_tab.Concentration != 0]
        slope, intercept, r_value, p_value, std_err = linregress(sum_tab_trim.Concentration.values, sum_tab_trim.MeasuredAbundance.values)
        gross_error_rate = sum_tab.loc[sum_tab.Concentration.values == 0].MeasuredAbundance.sum()
        plt.figure(1)
        plt.hist(sum_tab_fp.MeasuredAbundance.values*1000, bins = 100, alpha = 0.2)
    plt.figure(0)
    plt.xlabel(r'Input$\times 10^{3}$', fontsize = 6, color = 'black', labelpad = 0)
    plt.ylabel(r'Measured$\times 10^{3}$', fontsize = 6, color = 'black', labelpad = 1)
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 6, labelcolor = 'black', color = 'black', pad = 2)
    lim_max = np.maximum(np.max(sum_tab.Concentration), np.max(sum_tab.MeasuredAbundance))*1.05
    abundance_correlation_filename = '{}/abundance_correlation_all.pdf'.format(data_dir)
    abundance_fp_filename = '{}/abundance_false_positive_histogram.pdf'.format(data_dir)
    plt.plot([0,17.5], [0,17.5], '--', color = 'black', alpha = 0.8, linewidth = 0.5)
    plt.xlim(-0.5,17.5)
    plt.ylim(-0.5,17.5)
    plt.subplots_adjust(left=0.22, bottom=0.2, right=0.99, top=0.99)
    plt.axes().set_aspect('equal')
    plt.savefig(abundance_correlation_filename, dpi = 300, transparent = True)
    plt.close()
    plt.figure(1)
    plt.yscale('log')
    plt.xlabel(r'Measured Abundance$\times 10^{3}$', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 8)
    plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(abundance_fp_filename, dpi = 300, transparent = True)
    plt.close()

def plot_abundance_correlation_all_presentation(data_dir):
    files = glob.glob('{}/images_table_mix_*_results_abundance.csv'.format(data_dir))
    fig_abundance = plt.figure(0)
    fig_fp = plt.figure(1)
    fig_abundance.set_size_inches(cm_to_inches(4*2.54),cm_to_inches(4*2.54))
    fig_fp.set_size_inches(cm_to_inches(4),cm_to_inches(3))
    plt.figure(0)
    color_list = ['darkviolet', 'navy', 'fuchsia', 'red', 'limegreen', 'gold', 'darkorange', 'dodgerblue']
    for i in range(len(files)):
        filename = files[i]
        sum_tab = pd.read_csv(filename)
        mix_id = int(re.sub('mix_', '', re.search('mix_[0-9]*', filename).group(0)))
        input_tab_filename = '{}/hiprfish_1023_mix_{}.csv'.format(data_dir, str(mix_id))
        input_tab = pd.read_csv(input_tab_filename)
        abundance = sum_tab.drop(columns = ['Barcodes'])
        mean_absolute_abundance = abundance.sum(axis = 1)
        ul_absolute_abundance = np.percentile(abundance.values, 75, axis = 1)
        ll_absolute_abundance = np.percentile(abundance.values, 25, axis = 1)
        sum_tab['MeasuredAbundance'] = mean_absolute_abundance/np.sum(mean_absolute_abundance)
        sum_tab['ULAbundance'] = ul_absolute_abundance/np.sum(mean_absolute_abundance)
        sum_tab['LLAbundance'] = ll_absolute_abundance/np.sum(mean_absolute_abundance)
        sum_tab = sum_tab.merge(input_tab, how = 'left', on = 'Barcodes').fillna(0)
        sum_tab_fp = sum_tab.loc[sum_tab.Concentration.values == 0]
        plt.figure(0)
        plt.plot(sum_tab.Concentration.values*1000, sum_tab.MeasuredAbundance.values*1000, '.', markersize = 8, alpha = 0.5, color = color_list[i], markeredgewidth = 0)
        sum_tab_trim = sum_tab[sum_tab.Concentration != 0]
        slope, intercept, r_value, p_value, std_err = linregress(sum_tab_trim.Concentration.values, sum_tab_trim.MeasuredAbundance.values)
        gross_error_rate = sum_tab.loc[sum_tab.Concentration.values == 0].MeasuredAbundance.sum()
        print(gross_error_rate)
        plt.figure(1)
        plt.hist(sum_tab_fp.MeasuredAbundance.values*1000, bins = 100, alpha = 0.2)
    plt.figure(0)
    plt.xlabel(r'Input$\times 10^{3}$', fontsize = 12, color = 'white')
    plt.ylabel(r'Measured$\times 10^{3}$', fontsize = 12, color = 'white', labelpad = 1)
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 12, labelcolor = 'white', colors= 'white')
    lim_max = np.maximum(np.max(sum_tab.Concentration), np.max(sum_tab.MeasuredAbundance))*1.05
    abundance_correlation_filename = '{}/abundance_correlation_all_presentation.pdf'.format(data_dir)
    abundance_fp_filename = '{}/abundance_false_positive_histogram.pdf'.format(data_dir)
    plt.plot([0,17.5], [0,17.5], '--', color = 'white', alpha = 1, linewidth = 1)
    plt.xlim(-0.5,17.5)
    plt.ylim(-0.5,17.5)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.98)
    plt.axes().set_aspect('equal')
    plt.axes().spines['top'].set_color('white')
    plt.axes().spines['right'].set_color('white')
    plt.axes().spines['bottom'].set_color('white')
    plt.axes().spines['left'].set_color('white')
    plt.savefig(abundance_correlation_filename, dpi = 300, transparent = True)
    plt.figure(1)
    plt.yscale('log')
    plt.xlabel(r'Measured Abundance$\times 10^{3}$', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 8)
    plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(abundance_fp_filename, dpi = 300, transparent = True)
    plt.close()

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('data_dir', type = str, help = 'Input folder containing point spread function measurements')
    parser.add_argument('-m', '--mix', dest = 'mix_id', nargs = '*', default = [], type = str, help = '')
    args = parser.parse_args()
    if not args.mix_id:
        plot_abundance_correlation_all(args.data_dir)
        plot_abundance_correlation_all_presentation(args.data_dir)
    else:
        for m in args.mix_id:
            summary_table_filename = '{}/images_table_mix_{}_results_abundance.csv'.format(args.data_dir, m)
            input_table_filename = '{}/hiprfish_1023_mix_{}.csv'.format(args.data_dir, m)
            sum_tab = pd.read_csv(summary_table_filename)
            input_tab = pd.read_csv(input_table_filename)
            plot_abundance_correlation(summary_table_filename, sum_tab, input_tab, m)

if __name__ == '__main__':
    main()
