
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import skimage
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def cm_to_inches(x):
    return(x/2.54)

def summarize_photon_counting_results(filename, theme_color):
    sample = re.sub('_average_photon_counts.csv', '', filename)
    average_photon_counts = pd.read_csv(filename)
    channel_names = ['495', '504', '513', '522', '531', '539', '548', '557', '566', '575', '584', '593', '602', '611', '620', '628', '637', '646', '655', '664', '673', '682', '691']
    color_list = [hsv_to_rgb((0.5-0.5*i/23, 1, 1)) for i in range(23)]
    # photon counting per pixel
    flierprops = dict(marker='o', markerfacecolor= (0,0.5,1), markeredgewidth = 0, markersize=2, alpha = 0.2, linestyle='none')
    capprops = dict(color = theme_color)
    whiskerprops = dict(color = theme_color)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
    bp = plt.boxplot([average_photon_counts.iloc[:,i] for i in range(23)], patch_artist = True, labels = channel_names, flierprops = flierprops, capprops = capprops, whiskerprops = whiskerprops)
    for i in range(23):
        patch = bp['boxes'][i]
        patch.set_facecolor(color_list[i])

    plt.xticks(fontsize = 6, rotation = 90, ha = 'center')
    plt.xlabel('Emission Wavelength [nm]', color = theme_color, fontsize = 8)
    plt.ylabel('Photon Counts', color = theme_color, fontsize = 8, position = (0, 0.48))
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 6, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.16, bottom = 0.35, right = 0.98, top = 0.98)
    apc_filename = '{}_average_photon_counts.pdf'.format(sample)
    plt.savefig(apc_filename, dpi = 300, transparent = True)
    plt.close()
    # photon counting per pixel presentation
    theme_color = 'white'
    font_size = 12
    color_list = [hsv_to_rgb((0.5-0.5*i/23, 1, 1)) for i in range(23)]
    flierprops = dict(marker='o', markerfacecolor= (0,0.5,1), markeredgewidth = 0, markersize=2, alpha = 0.2, linestyle='none')
    capprops = dict(color = theme_color)
    whiskerprops = dict(color = theme_color)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(4))
    bp = plt.boxplot([average_photon_counts.iloc[:,i] for i in range(23)], patch_artist = True, labels = channel_names, flierprops = flierprops, capprops = capprops, whiskerprops = whiskerprops)
    for i in range(23):
        patch = bp['boxes'][i]
        patch.set_facecolor(color_list[i])

    plt.xticks(fontsize = 8, rotation = 90, ha = 'center')
    plt.xlabel('Emission Wavelength [nm]', color = theme_color, fontsize = font_size)
    plt.ylabel('Photon Counts', color = theme_color, fontsize = font_size, position = (0, 0.35))
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.35, right = 0.98, top = 0.98)
    apc_filename = '{}_average_photon_counts_presentation.svg'.format(sample)
    plt.savefig(apc_filename, dpi = 300, transparent = True)
    plt.close()

    total_photon_counts = pd.read_csv('{}_total_photon_counts.csv'.format(sample))
    channel_names = ['495', '504', '513', '522', '531', '539', '548', '557', '566', '575', '584', '593', '602', '611', '620', '628', '637', '646', '655', '664', '673', '682', '691']
    color_list = [hsv_to_rgb((0.5-0.5*i/23, 1, 1)) for i in range(23)]
    flierprops = dict(marker='o', markerfacecolor= (0,0.5,1), markeredgewidth = 0, markersize=2, alpha = 0.2, linestyle='none')
    capprops = dict(color = theme_color)
    whiskerprops = dict(color = theme_color)
    # total photon counts
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
    bp = plt.boxplot([total_photon_counts.iloc[:,i]/1000 for i in range(23)], patch_artist = True, labels = channel_names, flierprops = flierprops, capprops = capprops, whiskerprops = whiskerprops)
    for i in range(23):
        patch = bp['boxes'][i]
        patch.set_facecolor(color_list[i])

    plt.xticks(fontsize = 6, rotation = 90)
    plt.yticks([0,1,2,3], ['0', '1K', '2K', '3K'])
    plt.xlabel('Emission Wavelength [nm]', color = theme_color, fontsize = 8)
    plt.ylabel('Photon Counts ', color = theme_color, fontsize = 8)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.ylim(-0.5, 3.3)
    plt.tick_params(labelsize = 6, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.16, bottom = 0.35, right = 0.98, top = 0.98)
    tpc_filename = '{}_total_photon_counts.pdf'.format(sample)
    plt.savefig(tpc_filename, dpi = 300, transparent = True)
    plt.close()
    # total photon counts presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(4))
    bp = plt.boxplot([total_photon_counts.iloc[:,i]/1000 for i in range(23)], patch_artist = True, labels = channel_names, flierprops = flierprops, capprops = capprops, whiskerprops = whiskerprops)
    for i in range(23):
        patch = bp['boxes'][i]
        patch.set_facecolor(color_list[i])

    plt.xticks(fontsize = 12, rotation = 90)
    plt.yticks([0,1,2,3], ['0', '1K', '2K', '3K'])
    plt.xlabel('Emission Wavelength [nm]', color = theme_color, fontsize = font_size)
    plt.ylabel('Photon Counts ', color = theme_color, fontsize = font_size, position = (0, 0.35))
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.ylim(-0.5, 3.3)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.35, right = 0.98, top = 0.98)
    tpc_filename = '{}_total_photon_counts_presentation.svg'.format(sample)
    plt.savefig(tpc_filename, dpi = 300, transparent = True)
    plt.close()
    apc_spectral_average = np.average(average_photon_counts, axis = 0)
    apc_snr = apc_spectral_average/np.sqrt(apc_spectral_average)
    # SNR pixel
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(6))
    plt.plot(np.sqrt(3)*apc_snr, '--o', color = (0,0.5,1))
    channel_names = ['495', '504', '513', '522', '531', '539', '548', '557', '566', '575', '584', '593', '602', '611', '620', '628', '637', '646', '655', '664', '673', '682', '691']
    plt.xticks(np.arange(23), channel_names, rotation = 90)
    plt.xlabel('Emission Wavelength [nm]', color = theme_color)
    plt.ylabel('SNR', color = theme_color)
    plt.yscale('log')
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.tick_params(which = 'minor', labelsize = 8, direction = 'in', colors = theme_color)
    plt.subplots_adjust(left = 0.18, bottom = 0.25, right = 0.98, top = 0.98)
    snr_filename = '{}_snr.pdf'.format(sample)
    plt.savefig(snr_filename, dpi = 300, transparent = True)
    plt.close()
    # average SNR total
    apc_spectral_integrated = np.sum(average_photon_counts, axis = 1)
    apc_spectral_integrated_snr = apc_spectral_integrated/np.sqrt(apc_spectral_integrated)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    plt.hist(apc_spectral_integrated_snr, bins = 50, color = (0,0.5,1), histtype = 'step')
    plt.xlabel('SNR', color = theme_color, fontsize = 8)
    plt.ylabel('Frequency', color = theme_color, fontsize = 8)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.98, top = 0.98)
    apc_si_filename = '{}_apc_integrated_snr_histogram.pdf'.format(sample)
    plt.savefig(apc_si_filename, dpi = 300, transparent = True)
    tpc_spectral_integrated = np.sum(total_photon_counts, axis = 1)
    tpc_spectral_integrated_snr = tpc_spectral_integrated/np.sqrt(tpc_spectral_integrated)
    # total SNR
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
    plt.hist(tpc_spectral_integrated_snr, bins = 50, color = (0,0.5,1), histtype = 'step')
    plt.xlabel('SNR', color = theme_color, fontsize = 8)
    plt.ylabel('Frequency', color = theme_color, fontsize = 8)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.26, right = 0.98, top = 0.98)
    tpc_si_filename = '{}_tpc_integrated_snr_histogram.pdf'.format(sample)
    plt.savefig(tpc_si_filename, dpi = 300, transparent = True)
    plt.close()
    # total SNR presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(4))
    plt.hist(tpc_spectral_integrated_snr, bins = 50, color = (0,0.5,1), histtype = 'step')
    plt.xlabel('SNR', color = theme_color, fontsize = font_size)
    plt.ylabel('Frequency', color = theme_color, fontsize = font_size)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.26, right = 0.98, top = 0.98)
    tpc_si_filename = '{}_tpc_integrated_snr_histogram_presentation.svg'.format(sample)
    plt.savefig(tpc_si_filename, dpi = 300, transparent = True)
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('filename', type = str, help = 'Image filenames')
    parser.add_argument('theme_color', type = str, help = 'Image filenames')
    args = parser.parse_args()
    summarize_photon_counting_results(args.filename, args.theme_color)
    return

if __name__ == '__main__':
    main()
