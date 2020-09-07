
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
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 0.5
###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def cm_to_inches(length):
    return(length/2.54)

def plot_mean_abundance_distribution(sum_tab_filename, sum_tab):
    abundance = sum_tab.drop(columns = ['Barcodes'])
    mean_absolute_abundance = abundance.sum(axis = 1)
    mean_relative_abundance = mean_absolute_abundance/np.sum(mean_absolute_abundance)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(2.8),cm_to_inches(2.8))
    plt.hist(mean_relative_abundance*1000, bins = 100, color = (0,0.5,1), alpha = 0.75)
    plt.vlines(1/1023*1000, 0, 80, color = 'black', linewidth = 0.5)
    plt.xlabel(r'Abundance $\times10^{3}$', fontsize = 6, labelpad = 0)
    plt.ylabel('Frequency', fontsize = 6, labelpad = 0)
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 6, color = 'black', labelcolor = 'black', pad = 2)
    plt.text(1.5, 25, r'$\frac{1}{2^{10}-1}\times 10^3$', fontsize = 6)
    plt.ylim(0,75)
    plt.subplots_adjust(left=0.22, bottom=0.2, right=0.98, top=0.98)
    mean_abundance_histogram_filename = re.sub('.csv', '_mean_relative_abundance_histogram.pdf', sum_tab_filename)
    plt.savefig(mean_abundance_histogram_filename, dpi = 300, transparent = True)
    plt.close()

def plot_mean_abundance_distribution_presentation(sum_tab_filename, sum_tab):
    abundance = sum_tab.drop(columns = ['Barcodes'])
    mean_absolute_abundance = abundance.sum(axis = 1)
    mean_relative_abundance = mean_absolute_abundance/np.sum(mean_absolute_abundance)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4*2.54),cm_to_inches(4*2.54))
    plt.hist(mean_relative_abundance*1000, bins = 100, color = (0,0.5,1), alpha = 1)
    plt.vlines(1/1023*1000, 0, 80, color = 'white', linewidth = 1)
    plt.xlabel(r'Abundance $\times10^{3}$', fontsize = 12, color = 'white')
    plt.ylabel('Frequency', fontsize = 12, color = 'white')
    plt.tick_params(direction = 'in', width = 0.5, length = 2, labelsize = 12, colors = 'white', labelcolor = 'white')
    plt.axes().spines['bottom'].set_color('white')
    plt.axes().spines['left'].set_color('white')
    plt.axes().spines['top'].set_color('white')
    plt.axes().spines['right'].set_color('white')
    plt.text(2, 25, r'$\frac{1}{2^{10}-1}\times 10^3$', fontsize = 12, color = 'white')
    plt.ylim(0,75)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.98)
    mean_abundance_histogram_filename = re.sub('.csv', '_mean_relative_abundance_histogram_presentation.pdf', sum_tab_filename)
    plt.savefig(mean_abundance_histogram_filename, dpi = 300, transparent = True)
    plt.close()

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
