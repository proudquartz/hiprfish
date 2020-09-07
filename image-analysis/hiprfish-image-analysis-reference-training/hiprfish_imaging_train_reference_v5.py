
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import umap
import glob
import numba
import scipy
import joblib
import argparse
import matplotlib
import numpy as np
import pandas as pd
from sklearn import svm
from scipy import ndimage
from sklearn import preprocessing
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dask.dataframe as dd
import dask
from datetime import datetime
from scipy import interpolate
from dask.distributed import Client, LocalCluster

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
numba.config.NUMBA_NUM_THREADS = 16

def cm_to_inches(x):
    return(x/2.54)

def plot_umap(umap_transform):
    embedding_df = pd.DataFrame(umap_transform.embedding_)
    embedding_df['numeric_barcode'] = training_data_compute.code.apply(int, args = (2,)).values
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6),cm_to_inches(6))
    cmap = matplotlib.cm.get_cmap('jet')
    delta = 1/127
    color_list = [cmap(i*delta) for i in range(127)]
    for i in range(127):
        enc = i+1
        emd = embedding_df.loc[embedding_df.numeric_barcode.values == enc,:]
        plt.plot(emd.iloc[:,0], emd.iloc[:,1], 'o', alpha = 0.5, color = color_list[i], markersize = 1, rasterized = True)

    plt.axes().set_aspect('equal')
    plt.xlabel('UMAP 1', fontsize = 8, color = 'black')
    plt.ylabel('UMAP 2', fontsize = 8, color = 'black', labelpad = -1)
    plt.axes().spines['bottom'].set_color('black')
    plt.axes().spines['top'].set_color('black')
    plt.axes().spines['left'].set_color('black')
    plt.axes().spines['right'].set_color('black')
    plt.axes().tick_params(direction = 'in', labelsize = 8, colors = 'black')
    plt.subplots_adjust(left = 0.16, bottom = 0.13, right = 0.97, top = 0.97)
    plt.show()
    # plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_1023_reference_08_18_2018/reference_umap_visualization.pdf', dpi = 300, transparent = True)
    # plt.close()
    return

def plot_coefficients(coefficients,relevant_fluorophores, theme_color):
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
    cmap = cm.get_cmap('Spectral')
    lambda_min = 400
    lambda_max = 700
    wavelength_list = [488, 546, 561, 610, 647, 532, 514]
    for i in np.where(relevant_fluorophores)[0]:
        plt.hist(coefficients[:,i], color = cmap((wavelength_list[i] - lambda_min)/(lambda_max - lambda_min)), bins = 100, histtype = 'step')
    plt.xlabel('Transfer Coefficient', color = theme_color, fontsize = 8)
    plt.ylabel('Frequency', color = theme_color, fontsize = 8)
    plt.tick_params(direction = 'in', color = theme_color, length = 2)
    plt.subplots_adjust(left = 0.22, bottom = 0.2, right = 0.98, top = 0.98)
    plt.show()
    return

def analyze_transfer_coefficients():
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        indices = [0,23,43,57,63]
        for exc in range(4):
            relevant_fluorophores = numeric_code_list*excitation_matrix[exc, :]
            coefficients_fo = np.zeros((simulation_per_code, 7))
            coefficients_so = np.zeros((simulation_per_code, 7))
            for i in range(simulation_per_code):
                coefficients_fo[i,:] = np.dot(fret_transfer_matrix[i,:,:], relevant_fluorophores)*relevant_fluorophores
                coefficients_so[i,:] = np.dot(fret_transfer_matrix[i,:,:], np.abs(coefficients_fo[i,:] - relevant_fluorophores))*relevant_fluorophores
    return

@numba.njit()
def channel_cosine_intensity_7b_v2(x, y):
    check = np.sum(np.abs(x[63:67] - y[63:67]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[63] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
        else:
            cos_weight_1 = 1.0
            for i in range(0,23):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1 = 1.0
            else:
                cos_dist_1 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[64] == 0:
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
        else:
            cos_weight_2 = 1.0
            for i in range(23,43):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2 = 1.0
            else:
                cos_dist_2 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[65] == 0:
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
        else:
            cos_weight_3 = 1.0
            for i in range(43,57):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3 = 1.0
            else:
                cos_dist_3 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[66] == 0:
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
        else:
            cos_weight_4 = 1.0
            for i in range(57,63):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4 = 1.0
            else:
                cos_dist_4 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        cos_dist = 0.5*(cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4)/(cos_weight_1 + cos_weight_2 + cos_weight_3 + cos_weight_4)
    else:
        cos_dist = 1
    return(cos_dist)

@numba.njit()
def channel_cosine_combined_7b_v2(x, y):
    check = np.sum(np.abs(x[126:130] - y[126:130]))
    if check < 0.01:
        if x[126] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
        else:
            cos_weight_1 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(0,23):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1a = 1.0
            else:
                cos_dist_1a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(63,86):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1b = 1.0
            else:
                cos_dist_1b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_1 = (cos_dist_1a + cos_dist_1b)/2
        if x[127] == 0:
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
        else:
            cos_weight_2 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(23,43):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2a = 1.0
            else:
                cos_dist_2a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(86,106):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2b = 1.0
            else:
                cos_dist_2b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_2 = (cos_dist_2a + cos_dist_2b)/2
        if x[128] == 0:
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
        else:
            cos_weight_3 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(43,57):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3a = 1.0
            else:
                cos_dist_3a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(106,120):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3b = 1.0
            else:
                cos_dist_3b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_3 = (cos_dist_3a + cos_dist_3b)/2
        if x[129] == 0:
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
        else:
            cos_weight_4 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(57,63):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4a = 1.0
            else:
                cos_dist_4a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(120,126):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4b = 1.0
            else:
                cos_dist_4b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_4 = (cos_dist_4a + cos_dist_4b)/2
        cos_dist = 0.5*(cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4)/(cos_weight_1 + cos_weight_2 + cos_weight_3 + cos_weight_4)
    else:
        cos_dist = 1
    return(cos_dist)

@numba.njit()
def channel_cosine_intensity_v2(x, y):
    check = np.sum(np.abs(x[95:100] - y[95:100]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[95] == 0:
            cos_dist_1 = 0.0
            cos_weight_1 = 0.0
        else:
            cos_weight_1 = 1.0
            for i in range(0,32):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1 = 1.0
            else:
                cos_dist_1 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[96] == 0:
            cos_dist_2 = 0.0
            cos_weight_2 = 0.0
        else:
            cos_weight_2 = 1.0
            for i in range(32,55):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2 = 1.0
            else:
                cos_dist_2 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[97] == 0:
            cos_dist_3 = 0.0
            cos_weight_3 = 0.0
        else:
            cos_weight_3 = 1.0
            for i in range(55,75):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3 = 1.0
            else:
                cos_dist_3 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[98] == 0:
            cos_dist_4 = 0.0
            cos_weight_4 = 0.0
        else:
            cos_weight_4 = 1.0
            for i in range(75,89):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4 = 1.0
            else:
                cos_dist_4 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[99] == 0:
            cos_dist_5 = 0.0
            cos_weight_5 = 0.0
        else:
            cos_weight_5 = 1.0
            for i in range(89,95):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_5 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_5 = 1.0
            else:
                cos_dist_5 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/(cos_weight_1 + cos_weight_2 + cos_weight_3 + cos_weight_4 + cos_weight_5)
    else:
        cos_dist = 1.0
    return(cos_dist)

@numba.njit()
def channel_cosine_combined_v2(x, y):
    check = np.sum(np.abs(x[190:195] - y[190:195]))
    if check < 0.01:
        if x[190] == 0:
            cos_dist_1 = 0.0
            cos_weight_1 = 0.0
        else:
            cos_weight_1 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(0,32):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1a = 1.0
            else:
                cos_dist_1a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(95,127):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1b = 1.0
            else:
                cos_dist_1b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_1 = (cos_dist_1a + cos_dist_1b)/2
        if x[191] == 0:
            cos_dist_2 = 0.0
            cos_weight_2 = 0.0
        else:
            cos_weight_2 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(32,55):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2a = 1.0
            else:
                cos_dist_2a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(127,150):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2b = 1.0
            else:
                cos_dist_2b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_2 = (cos_dist_2a + cos_dist_2b)/2
        if x[192] == 0:
            cos_dist_3 = 0.0
            cos_weight_3 = 0.0
        else:
            cos_weight_3 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(55,75):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3a = 1.0
            else:
                cos_dist_3a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(150,170):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3b = 1.0
            else:
                cos_dist_3b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_3 = (cos_dist_3a + cos_dist_3b)/2
        if x[193] == 0:
            cos_dist_4 = 0.0
            cos_weight_4 = 0.0
        else:
            cos_weight_4 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(75,89):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4a = 1.0
            else:
                cos_dist_4a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(170,184):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4b = 1.0
            else:
                cos_dist_4b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_4 = (cos_dist_4a + cos_dist_4b)/2
        if x[194] == 0:
            cos_dist_5 = 0.0
            cos_weight_5 = 0.0
        else:
            cos_weight_5 = 1.0
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(89,95):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_5a = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_5a = 1.0
            else:
                cos_dist_5a = 1.0 - (result / np.sqrt(norm_x * norm_y))
            result = 0.0
            norm_x = 0.0
            norm_y = 0.0
            for i in range(184,190):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_5b = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_5b = 1.0
            else:
                cos_dist_5b = 1.0 - (result / np.sqrt(norm_x * norm_y))
            cos_dist_5 = (cos_dist_5a + cos_dist_5b)/2
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/(cos_weight_1 + cos_weight_2 + cos_weight_3 + cos_weight_4 + cos_weight_5)
    else:
        cos_dist = 1.0
    return(cos_dist)

def get_qec(fret_folder):
    gaasp_qe = pd.read_table('{}/GaAsP_QE.txt'.format(fret_folder), header = None)
    lambda_min = 410
    d_lambda = 8.9
    n_channels = 32
    lambda_max = lambda_min + d_lambda*n_channels
    lambda_bins = np.arange(lambda_min, lambda_max, d_lambda)
    lambda_bin_labels = np.searchsorted(lambda_bins, gaasp_qe.iloc[:,0].values[(gaasp_qe.iloc[:,0].values >= lambda_min) & (gaasp_qe.iloc[:,0].values <= lambda_max)], "right")
    lambda_bin_labels -= 1
    gaasp_qe_downsampled = ndimage.mean(gaasp_qe.iloc[:,1].values[(gaasp_qe.iloc[:,0].values >= lambda_min) & (gaasp_qe.iloc[:,0].values <= lambda_max)], labels=lambda_bin_labels, index=np.arange(0,lambda_bin_labels.max()+1))
    qec = np.zeros(95)
    for exc in range(5):
        qec[channel_indices[exc]:channel_indices[exc+1]] = gaasp_qe_downsampled[synthetic_channel_indices[exc]:32]
    return(qec)

def get_downsampled_synthetic_spectra(spec_synthetic, lambda_min, d_lambda, n_channels, exc):
    synthetic_channel_indices = [0, 3, 9, 17]
    spec_synthetic_norm = np.zeros(spec_synthetic.shape[0])
    spec_synthetic_norm[spec_synthetic.Emission.values > 0] = spec_synthetic.Emission.values[spec_synthetic.Emission.values > 0]
    spec_synthetic_norm[spec_synthetic.Emission.values <= 0] = np.min(spec_synthetic.Emission.values)
    spec_synthetic_norm = spec_synthetic_norm - np.min(spec_synthetic_norm)
    spec_synthetic_norm = spec_synthetic_norm/np.max(spec_synthetic_norm)
    spec_synthetic.loc[:,'Emission'] = spec_synthetic_norm
    lambda_max = lambda_min + d_lambda*n_channels
    lambda_bins = np.arange(lambda_min, lambda_max, d_lambda)
    lambda_bin_labels = np.searchsorted(lambda_bins, spec_synthetic.Wavelength.values[(spec_synthetic.Wavelength.values >= lambda_min) & (spec_synthetic.Wavelength.values <= lambda_max)], "right")
    lambda_bin_labels -= 1
    spec_synthetic_downsampled = ndimage.mean(spec_synthetic.Emission.values[(spec_synthetic.Wavelength.values >= lambda_min) & (spec_synthetic.Wavelength.values <= lambda_max)], labels=lambda_bin_labels, index=np.arange(0,lambda_bin_labels.max()+1))
    if np.max(spec_synthetic_downsampled) > 0.0:
        spec_synthetic_downsampled = spec_synthetic_downsampled/np.max(spec_synthetic_downsampled)
    return(spec_synthetic_downsampled)

def get_downsampled_synthetic_spectra_sum(spec_synthetic, lambda_min, d_lambda, n_channels, exc):
    spec_synthetic_norm = spec_synthetic.Emission.values - np.min(spec_synthetic.Emission.values)
    spec_synthetic_norm = spec_synthetic_norm/np.max(spec_synthetic_norm)
    synthetic_channel_indices = [0, 3, 9, 17]
    lambda_max = lambda_min + d_lambda*n_channels
    lambda_bins = np.arange(lambda_min, lambda_max, d_lambda)
    lambda_bin_labels = np.searchsorted(lambda_bins, spec_synthetic.Wavelength.values[(spec_synthetic.Wavelength.values >= lambda_min) & (spec_synthetic.Wavelength.values <= lambda_max)], "right")
    lambda_bin_labels -= 1
    spec_synthetic_downsampled = ndimage.sum(spec_synthetic.Emission.values[(spec_synthetic.Wavelength.values >= lambda_min) & (spec_synthetic.Wavelength.values <= lambda_max)], labels=lambda_bin_labels, index=np.arange(0,lambda_bin_labels.max()+1))
    if np.max(spec_synthetic_downsampled) > 0.0:
        spec_synthetic_downsampled = spec_synthetic_downsampled/np.max(spec_synthetic_downsampled)
    return(spec_synthetic_downsampled)

def get_excitation_max(channel_max):
    if (channel_max >= 0) & (channel_max < 23):
        exc_max = 0
    elif (channel_max >= 23) & (channel_max < 43):
        exc_max = 1
    elif (channel_max >= 43) & (channel_max < 57):
        exc_max = 2
    elif (channel_max >= 57) & (channel_max < 63):
        exc_max = 3
    return(exc_max)

def get_excitation_max_10b(channel_max):
    if (channel_max >= 0) & (channel_max < 32):
        exc_max = 0
    elif (channel_max >= 32) & (channel_max < 55):
        exc_max = 1
    elif (channel_max >= 55) & (channel_max < 75):
        exc_max = 2
    elif (channel_max >= 75) & (channel_max < 89):
        exc_max = 3
    elif (channel_max >= 89) & (channel_max < 95):
        exc_max = 4
    return(exc_max)

def plot_synthetic_spectra_characterizations(reference_folder, spec_merge_list, cost_matrix_full, theme_color):
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(25), cm_to_inches(15))
    gs = GridSpec(7, 8)
    for i in range(7):
        for j in range(4):
            ax = plt.subplot(gs[i,j])
            if np.max(cost_matrix_full[i,j,:,:] > 0):
                ax.imshow(np.log10(cost_matrix_full[i,j,:,:]), cmap = 'RdBu')
                # plt.xticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/20,d_ticks), np.arange(4+d_ticks/20,4+n_ticks*d_ticks/20+d_ticks/20, d_ticks/20))
                # plt.yticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(420+d_ticks/2,420+n_ticks*d_ticks/2+d_ticks/2, d_ticks/2))
                plt.xticks([])
                plt.yticks([])
                plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
                ax.spines['left'].set_color(theme_color)
                ax.spines['bottom'].set_color(theme_color)
                ax.spines['right'].set_color(theme_color)
                ax.spines['top'].set_color(theme_color)
            else:
                ax.imshow(np.log10(cost_matrix_full[i,j,:,:]), cmap = 'RdBu')
                plt.tick_params(direction = 'in', length = 0, colors = theme_color, grid_alpha = 0)
                plt.xticks([])
                plt.yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
    for j in range(4):
        ax = plt.subplot(gs[0,j])
        plt.title(excitation_lasers[j], fontsize = 8, color = theme_color)
    for i in range(7):
        ax = plt.subplot(gs[i,0])
        plt.text(-225,150,fluorophores_text_list[i], fontsize = 8, color = theme_color, ha = 'right')
    n_ticks = 5
    d_ticks = 60
    ax = plt.subplot(gs[4,0])
    plt.yticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(l_min_list[0]+d_ticks/2,l_min_list[0]+n_ticks*d_ticks/2+d_ticks/2, d_ticks/2, dtype = int))
    # plt.ylabel(r'$\lambda_{min}$ [nm]', color = theme_color, fontsize = 8)
    ax = plt.subplot(gs[4,1])
    plt.yticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(l_min_list[1]+d_ticks/2,l_min_list[1]+n_ticks*d_ticks/2+d_ticks/2, d_ticks/2, dtype = int))
    # plt.ylabel(r'$\lambda_{min}$ [nm]', color = theme_color, fontsize = 8)
    ax = plt.subplot(gs[6,2])
    plt.xticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(4+d_ticks/20,4+n_ticks*d_ticks/20+d_ticks_x/20, d_ticks/20, dtype = int))
    plt.yticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(l_min_list[2]+d_ticks/2,l_min_list[2]+n_ticks*d_ticks/2+d_ticks/2, d_ticks/2, dtype = int))
    plt.xlabel(r'$\Delta\lambda$ [nm]', color = theme_color, fontsize = 8)
    plt.ylabel(r'$\lambda_{min}$ [nm]', color = theme_color, fontsize = 8)
    ax = plt.subplot(gs[6,3])
    plt.yticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(l_min_list[3]+d_ticks/2,l_min_list[3]+n_ticks*d_ticks/2+d_ticks/2, d_ticks/2, dtype = int))
    for i in range(7):
        ax = plt.subplot(gs[i,4:6])
        spec_i = spec_merge_list[i]
        spec_m = spec_i.loc[spec_i.Source.values == 'Measured', :]
        spec_s = spec_i.loc[spec_i.Source.values == 'Simulated', :]
        plt.plot(spec_i.Channel.values, spec_i.Intensity.values, '-', color = (0.5,0.5,0.5), label = 'Experimental Measured')
        plt.plot(spec_m.Channel.values, spec_m.Intensity.values, 'o', color = (0,0.5,1), markersize = 3, label = 'Synthetic Measured')
        plt.plot(spec_s.Channel.values, spec_s.Intensity.values, 'o', color = (1,0.5,0), markersize = 3, label = 'Synthetic Simulated')
        plt.plot(spec_avg[i][32:], '--', color = (0.5,0,1))
        plt.xticks([])
        plt.yticks([])
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        plt.ylim(-0.05, 1.05)
        ax = plt.subplot(gs[i,6:8])
        spec_i = spec_merge_list[i]
        plt.plot(spec_i.Channel.values, 1000*(spec_avg[i][32:] - spec_i.Intensity.values), 'o', color = (0,0.5,1), markersize = 3)
        plt.subplots_adjust(left = 0.25, bottom = 0.2, right = 0.98, top = 0.98)
        plt.xticks([])
        plt.yticks([])
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.ylim(-12,12)
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
    ax = plt.subplot(gs[6,4:6])
    plt.xticks(np.arange(15,60,15))
    plt.yticks(np.arange(0,1,0.2))
    plt.xlabel('Channels', fontsize = 8, color = theme_color)
    plt.ylabel('Intensity', fontsize = 8, color = theme_color)
    l = plt.legend(frameon = False, fontsize = 6, loc = 2)
    for t in l.get_texts():
        t.set_color(theme_color)
    ax = plt.subplot(gs[6,6:8])
    plt.xticks(np.arange(15,60,15))
    plt.yticks(np.arange(-8,16,8))
    plt.text(-13.5,12,r'$\times10^{-3}$', fontsize = 8, color = theme_color)
    plt.xlabel('Channels', fontsize = 8, color = theme_color)
    plt.ylabel('Residuals', fontsize = 8, color = theme_color, labelpad = 3.5)
    plt.subplots_adjust(left = 0.15, bottom = 0.08, right = 0.98, top = 0.95, wspace = 0.75, hspace = 0.2)
    plt.savefig('{}/fluorophore_synthetic_cost.pdf'.format(fret_folder), dpi = 300, transparent = True)
    for i in range(7):
        fig = plt.figure(2)
        fig.set_size_inches(cm_to_inches(6), cm_to_inches(5))
        plt.imshow(np.log10(cost_matrix_full[0,1,:,:]), cmap = 'RdBu')
        n_ticks = 5
        d_ticks = 60
        plt.xticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/20,d_ticks), np.arange(4+d_ticks/20,4+n_ticks*d_ticks/20+d_ticks/20, d_ticks/20))
        plt.yticks(np.arange(0+d_ticks/2,d_ticks*n_ticks+d_ticks/2,d_ticks), np.arange(420+d_ticks/2,420+n_ticks*d_ticks/2+d_ticks/2, d_ticks/2))
        plt.xlabel(r'$\Delta\lambda$ [nm]', color = theme_color, fontsize = 8)
        plt.ylabel(r'$\lambda_{min}$ [nm]', color = theme_color, fontsize = 8)
        plt.text(100,250, fluorophores_text_list[i], color = 'white', fontsize = 8)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.axes().spines['left'].set_color(theme_color)
        plt.axes().spines['bottom'].set_color(theme_color)
        plt.axes().spines['right'].set_color(theme_color)
        plt.axes().spines['top'].set_color(theme_color)
        plt.subplots_adjust(left = 0.25, bottom = 0.2, right = 0.98, top = 0.98)
        plt.savefig('{}/fluorophore_{}_synthetic_cost.pdf'.format(reference_folder, fluorophores_text_list[i]), dpi = 300, transparent = True)
        plt.close()
    optimal_synthetic_spec_full = []
    for i in range(7):
        spec_measured_avg = spec_avg[i][32:]
        spec_measured_std = spec_std[i][32:]
        spec_synthetic_downsampled_optimal_full = np.zeros(63)
        for exc in range(4):
            channel_max = np.max(spec_measured_avg[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            spec_synthetic_downsampled_optimal_full[measured_channel_indices[exc]:measured_channel_indices[exc+1]] = channel_max*optimal_synthetic_spec[i][synthetic_channel_indices[exc]:23]
        optimal_synthetic_spec_full.append(spec_synthetic_downsampled_optimal_full)
    for i in range(7):
        spec_measured_avg = spec_avg[i][32:]
        spec_measured_std = spec_std[i][32:]
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(6), cm_to_inches(4))
        plt.plot(spec_measured_avg, '-o', color = (0,0.5,1), markersize = 3)
        plt.plot(optimal_synthetic_spec_full[i], '-o', color = (1,0.5,0), markersize = 3)
        plt.fill_between(np.arange(63), spec_measured_avg - spec_measured_std, spec_measured_avg + spec_measured_std, color = (0,0.5,1), alpha = 0.5)
        plt.xlabel('Channels', fontsize = 8, color = theme_color)
        plt.ylabel('Intensity', fontsize = 8, color = theme_color)
        plt.axes().spines['left'].set_color(theme_color)
        plt.axes().spines['bottom'].set_color(theme_color)
        plt.axes().spines['right'].set_color(theme_color)
        plt.axes().spines['top'].set_color(theme_color)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.98, top = 0.98)
        plt.savefig('{}/fluorophore_{}_synthetic_measured_spec_comparison.pdf'.format(reference_folder, fluorophores_text_list[i]), dpi = 300, transparent = True)
        plt.close()
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(5))
    plt.plot(optimal_downsampling_parameters[:,0], 'o', color = (0,0.5,1), markeredgewidth = 0)
    plt.xticks(np.arange(7), fluorophores_text_list, color = theme_color, fontsize = 8, rotation = 90)
    plt.ylabel(r'$\lambda_{min}$ [nm]', color = theme_color, fontsize = 8)
    plt.ylim(np.min(optimal_downsampling_parameters[:,0]) - 10, np.max(optimal_downsampling_parameters[:,0]) + 10)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.55, right = 0.98, top = 0.98)
    plt.savefig('{}/fluorophore_optimal_lambda_min.pdf'.format(reference_folder), dpi = 300, transparent = True)
    plt.close()
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(5))
    plt.plot(optimal_downsampling_parameters[:,1], 'o', color = (0,0.5,1), markeredgewidth = 0)
    plt.xticks(np.arange(7), fluorophores_text_list, color = theme_color, fontsize = 8, rotation = 90)
    plt.ylabel(r'$\Delta\lambda$ [nm]', color = theme_color, fontsize = 8)
    plt.ylim(np.min(optimal_downsampling_parameters[:,1]) - 1, np.max(optimal_downsampling_parameters[:,1]) + 1)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.55, right = 0.98, top = 0.98)
    plt.savefig('{}/fluorophore_optimal_delta_lambda.pdf'.format(reference_folder), dpi = 300, transparent = True)
    plt.close()
    return

def extract_uv_bkg_spec(reference_folder, fret_folder):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9,5,4,1,10,8,2,3,6,7]
    measured_channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [9, 12, 18, 26]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    lmin = 405
    dl = 8.9
    spec_synthetic_downsampled_list = []
    for f in range(10):
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[f])))
        spec_synthetic_downsampled = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = lmin, d_lambda = dl, n_channels = 32, exc = 0)
        spec_synthetic_downsampled_list.append(spec_synthetic_downsampled)
    coefficients = [0, 0, 0, 0.02, 0.01, 0.01, 0.012, 0.03, 0.01, 0.004]
    uv_bkg_list = []
    for i in range(3,10):
        uv_bkg = spec_avg[i][0:32] - coefficients[i]*spec_synthetic_downsampled_list[i]
        uv_bkg_list.append(uv_bkg)
    uv_bkg_avg = np.average(np.stack(uv_bkg_list, axis = 1), axis = 1)
    uv_bkg_avg_norm = uv_bkg/np.max(uv_bkg)
    uv_bkg_filename = '{}/ultraviolet_average_background.csv'.format(fret_folder)
    pd.DataFrame(uv_bkg_avg_norm).to_csv(uv_bkg_filename, index = None, header = None)
    return

def reference_spectra_background_removal(reference_folder, fret_folder, data_folder):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9,5,4,1,10,8,2,3,6,7]
    measured_channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [9, 12, 18, 26]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    bkg_spec = pd.read_csv('{}/ultraviolet_average_background.csv'.format(fret_folder), header = None).iloc[0:32,0].values
    sam_tab = pd.read_csv('{}/images_table_1023_reference_bkg.csv'.format(data_folder))
    spec_avg_bkg_filtered_list = []
    for f in range(len(barcode_list)):
        sample = sam_tab.loc[sam_tab.IMAGES.values == image_name, 'SAMPLE'].values[0]
        image_name = '08_18_2018_enc_{}'.format(barcode_list[f])
        bkg_intensity = sam_tab.loc[sam_tab.IMAGES.values == image_name, 'BkgFactorCorrected'].values[0]
        image_name_adjacent = sam_tab.loc[sam_tab.IMAGES.values == image_name, 'SampleAdjacent'].values[0]
        if f == 1:
            image_name_adjacent = '08_18_2018_enc_1'
        bkg_spec = pd.read_csv('{}/{}/{}_bkg.csv'.format(data_folder, sample, image_name_adjacent), header = None).iloc[0:32,0].values
        spec_avg_bkg_filtered = spec_avg[f].copy()
        spec_avg_bkg_filtered[0:32] = np.abs(spec_avg_bkg_filtered[0:32] - bkg_intensity*bkg_spec)
        spec_avg_bkg_filtered_list.append(spec_avg_bkg_filtered)
        pd.DataFrame(spec_avg_bkg_filtered).to_csv('{}/08_18_2018_enc_{}_avgint_bkg_filtered.csv'.format(fret_folder,barcode_list[f]), index = None, header = None)
    return

def reference_spectra_background_removal_full(reference_folder, fret_folder, data_folder):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9,5,4,1,10,8,2,3,6,7]
    measured_channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [9, 12, 18, 26]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [pd.read_csv(f, header = None).values for f in files]
    bkg_spec = pd.read_csv('{}/ultraviolet_average_background.csv'.format(fret_folder), header = None).iloc[0:32,0].values
    sam_tab = pd.read_csv('{}/images_table_1023_reference_bkg.csv'.format(data_folder))
    spec_avg_bkg_filtered_list = []
    for f in range(len(barcode_list)):
        sample = sam_tab.loc[sam_tab.IMAGES.values == image_name, 'SAMPLE'].values[0]
        image_name = '08_18_2018_enc_{}'.format(barcode_list[f])
        bkg_intensity = sam_tab.loc[sam_tab.IMAGES.values == image_name, 'BkgFactorCorrected'].values[0]
        image_name_adjacent = sam_tab.loc[sam_tab.IMAGES.values == image_name, 'SampleAdjacent'].values[0]
        if f == 1:
            image_name_adjacent = '08_18_2018_enc_1'
        bkg_spec = pd.read_csv('{}/{}/{}_bkg.csv'.format(data_folder, sample, image_name_adjacent), header = None).iloc[0:32,0].values
        spec_avg_bkg_filtered = spec_avg[f].copy()
        spec_avg_bkg_filtered[:,0:32] = np.abs(spec_avg_bkg_filtered[:,0:32] - bkg_intensity*bkg_spec)
        spec_avg_bkg_filtered_list.append(spec_avg_bkg_filtered)
        pd.DataFrame(spec_avg_bkg_filtered).to_csv('{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder,barcode_list[f]), index = None, header = None)
    return

def generate_synthetic_reference_spectra(reference_folder, fret_folder):
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    fluorophores = [1,10,8,2,3,6,7]
    measured_channel_indices = [0,23,43,57,63]
    synthetic_channel_indices = [0, 3, 9, 17]
    l_min_list = [420, 460, 500, 570]
    excitation_lasers = ['488 nm', '514 nm', '561nm', '633nm']
    fluorophores_text_list = ['Alexa488', 'Alexa532', 'DyLight 510', 'Alexa 546', 'Rhodamine Red X', 'Atto 610', 'Alexa 647']
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    spec_std = [np.std(pd.read_csv(f, header = None), axis = 0) for f in files]
    optimal_downsampling_parameters = np.zeros((7,4,2))
    cost_matrix_size = 300
    cost_matrix_full = np.zeros((7,4,cost_matrix_size,cost_matrix_size))
    for f in range(7):
        print('Fitting fluorophore {} {}...'.format(f, fluorophores_text_list[f]))
        spec_measured = spec_avg[f][32:]
        channel_max = np.argmax(spec_measured)
        exc_max = get_excitation_max(channel_max)
        relevant_exc_channels = [exc_max - 1 + i for i in range(3) if exc_max - 1 + i >= 0 and exc_max - 1 + i <= 3]
        if f == 3 or f == 4:
            relevant_exc_channels.insert(0, 0)
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[f])))
        for exc in relevant_exc_channels:
            cost_matrix = np.zeros((cost_matrix_size, cost_matrix_size))
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            spec_measured_norm = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]/np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            for i in np.arange(cost_matrix_size):
                for j in np.arange(cost_matrix_size):
                    lmin = l_min_list[exc] + i/2
                    dl = 4 + j/20
                    spec_synthetic_downsampled = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = lmin, d_lambda = dl, n_channels = n_channels, exc = exc)
                    if spec_synthetic_downsampled.shape[0] < n_channels:
                        spec_synthetic_downsampled = np.append(spec_synthetic_downsampled, np.repeat(spec_synthetic_downsampled[-1], n_channels - spec_synthetic_downsampled.shape[0]))
                    cost_matrix[i,j] = np.sum((spec_measured_norm - spec_synthetic_downsampled)**2)
            io, jo = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            optimal_downsampling_parameters[f, exc, :] = l_min_list[exc] + io/2, 4 + jo/20
            cost_matrix_full[f,exc,:,:]= cost_matrix
    spec_merge_list = []
    for i in range(7):
        spec_merge = pd.DataFrame(columns = ['Channel', 'Source', 'Intensity'])
        spec_merge['Channel'] = np.arange(0,63)
        spec_merge['Intensity'] = np.zeros(63)
        spec_merge['Source'] = 'Measured'
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        spec_measured = spec_avg[i][32:]
        for exc in range(4):
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            measured_intensity_magnitude = np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            if (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude < 0.1):
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, exc, 0], d_lambda = optimal_downsampling_parameters[i, exc, 1], n_channels = n_channels, exc = exc)
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
            elif (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude > 0.1):
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]
            elif (optimal_downsampling_parameters[i, exc, 0] == 0) & (exc <= 1):
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 2, 0], d_lambda = optimal_downsampling_parameters[i, 2, 1], n_channels = 14, exc = exc)
                spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
        spec_merge_list.append(spec_merge)
    for i in range(7):
        spec_merge_list[i].to_csv('{}/R{}_synthetic_spectra_7b.csv'.format(fret_folder, fluorophores[i]), index = None)
    return

def generate_synthetic_reference_spectra_10b(reference_folder, fret_folder):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9,5,4,1,10,8,2,3,6,7]
    measured_channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [9, 12, 18, 26]
    l_min_list = [360, 420, 460, 500, 570]
    excitation_lasers = ['405 nm', '488 nm', '514 nm', '561nm', '633nm']
    fluorophores_text_list = ['Alexa 405', 'Pacific Blue', 'Pacific Green', 'Alexa488', 'Alexa532', 'DyLight 510', 'Alexa 546', 'Rhodamine Red X', 'Atto 610', 'Alexa 647']
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    optimal_downsampling_parameters = np.zeros((10,5,2))
    cost_matrix_size = 300
    cost_matrix_full = np.zeros((10,5,cost_matrix_size,cost_matrix_size))
    for f in range(10):
        print('Fitting fluorophore {} {}...'.format(f, fluorophores_text_list[f]))
        spec_measured = spec_avg[f]
        channel_max = np.argmax(spec_measured)
        exc_max = get_excitation_max_10b(channel_max)
        relevant_exc_channels = [exc_max - 1 + i for i in range(3) if exc_max - 1 + i >= 0 and exc_max - 1 + i <= 4]
        if f == 6 or f == 7:
            relevant_exc_channels.insert(1, 1)
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[f])))
        for exc in relevant_exc_channels:
            cost_matrix = np.zeros((cost_matrix_size, cost_matrix_size))
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            spec_measured_norm = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]/np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            for i in np.arange(cost_matrix_size):
                for j in np.arange(cost_matrix_size):
                    lmin = l_min_list[exc] + i/2
                    dl = 4 + j/20
                    spec_synthetic_downsampled = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = lmin, d_lambda = dl, n_channels = n_channels, exc = exc)
                    if spec_synthetic_downsampled.shape[0] < n_channels:
                        spec_synthetic_downsampled = np.append(spec_synthetic_downsampled, np.repeat(spec_synthetic_downsampled[-1], n_channels - spec_synthetic_downsampled.shape[0]))
                    cost_matrix[i,j] = np.sum((spec_measured_norm - spec_synthetic_downsampled)**2)
            io, jo = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            optimal_downsampling_parameters[f, exc, :] = l_min_list[exc] + io/2, 4 + jo/20
            cost_matrix_full[f,exc,:,:]= cost_matrix
    # Adjust parameters for Alexa488 under 405 excitation because of the UV background
    optimal_downsampling_parameters[3,0,0] = 405
    optimal_downsampling_parameters[3,0,1] = 8.9
    spec_merge_list = []
    for i in range(10):
        spec_merge = pd.DataFrame(columns = ['Channel', 'Source', 'Intensity'])
        spec_merge['Channel'] = np.arange(0,95)
        spec_merge['Intensity'] = np.zeros(95)
        spec_merge['Source'] = 'Measured'
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        spec_measured = spec_avg[i]
        for exc in range(5):
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            measured_intensity_magnitude = np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            if (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude < 0.1):
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, exc, 0], d_lambda = optimal_downsampling_parameters[i, exc, 1], n_channels = n_channels, exc = exc)
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
            elif (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude > 0.1):
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]
            elif (optimal_downsampling_parameters[i, exc, 0] == 0) & (exc <= 1):
                if optimal_downsampling_parameters[i, 2, 0] > 0:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 2, 0], d_lambda = optimal_downsampling_parameters[i, 2, 1], n_channels = 20, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
                else:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 3, 0], d_lambda = optimal_downsampling_parameters[i, 3, 1], n_channels = 14, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
        spec_merge_list.append(spec_merge)
    for i in range(10):
        spec_merge_list[i].to_csv('{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]), index = None)
    optimal_downsampling_parameters[3,0,0] = 405
    optimal_downsampling_parameters[3,0,1] = 8.9
    spec_merge_shift_list = []
    optimal_downsampling_parameters[:,:,0][optimal_downsampling_parameters[:,:,0] > 0] += 5
    for i in range(10):
        spec_merge = pd.DataFrame(columns = ['Channel', 'Source', 'Intensity'])
        spec_merge['Channel'] = np.arange(0,95)
        spec_merge['Intensity'] = np.zeros(95)
        spec_merge['Source'] = 'Measured'
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        spec_measured = spec_avg[i]
        for exc in range(5):
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            measured_intensity_magnitude = np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            if optimal_downsampling_parameters[i, exc, 0] > 0:
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, exc, 0], d_lambda = optimal_downsampling_parameters[i, exc, 1], n_channels = n_channels, exc = exc)
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
            elif (optimal_downsampling_parameters[i, exc, 0] == 0) & (exc <= 1):
                if optimal_downsampling_parameters[i, 2, 0] > 0:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 2, 0], d_lambda = optimal_downsampling_parameters[i, 2, 1], n_channels = 20, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
                else:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 3, 0], d_lambda = optimal_downsampling_parameters[i, 3, 1], n_channels = 14, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
        spec_merge_shift_list.append(spec_merge)
    for i in range(10):
        spec_merge_shift_list[i].to_csv('{}/R{}_synthetic_spectra_shifted.csv'.format(fret_folder, fluorophores[i]), index = None)
    return

def generate_synthetic_reference_spectra_10b_bkg_filtered(reference_folder, fret_folder):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9,5,4,1,10,8,2,3,6,7]
    measured_channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [9, 12, 18, 26]
    l_min_list = [360, 420, 460, 500, 570]
    excitation_lasers = ['405 nm', '488 nm', '514 nm', '561nm', '633nm']
    fluorophores_text_list = ['Alexa 405', 'Pacific Blue', 'Pacific Green', 'Alexa488', 'Alexa532', 'DyLight 510', 'Alexa 546', 'Rhodamine Red X', 'Atto 610', 'Alexa 647']
    files = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered.csv'.format(fret_folder, b) for b in barcode_list]
    spec_avg = [pd.read_csv(f, header = None).iloc[0:95,0].values for f in files]
    optimal_downsampling_parameters = np.zeros((10,5,2))
    cost_matrix_size = 300
    cost_matrix_full = np.zeros((10,5,cost_matrix_size,cost_matrix_size))
    for f in range(10):
        print('Fitting fluorophore {} {}...'.format(f, fluorophores_text_list[f]))
        spec_measured = spec_avg[f]
        channel_max = np.argmax(spec_measured)
        exc_max = get_excitation_max_10b(channel_max)
        relevant_exc_channels = [exc_max - 1 + i for i in range(3) if exc_max - 1 + i >= 0 and exc_max - 1 + i <= 4]
        if f == 6 or f == 7:
            relevant_exc_channels.insert(1, 1)
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[f])))
        for exc in relevant_exc_channels:
            cost_matrix = np.zeros((cost_matrix_size, cost_matrix_size))
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            spec_measured_norm = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]/np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            for i in np.arange(cost_matrix_size):
                for j in np.arange(cost_matrix_size):
                    lmin = l_min_list[exc] + i/2
                    dl = 4 + j/20
                    spec_synthetic_downsampled = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = lmin, d_lambda = dl, n_channels = n_channels, exc = exc)
                    if spec_synthetic_downsampled.shape[0] < n_channels:
                        spec_synthetic_downsampled = np.append(spec_synthetic_downsampled, np.repeat(spec_synthetic_downsampled[-1], n_channels - spec_synthetic_downsampled.shape[0]))
                    cost_matrix[i,j] = np.sum((spec_measured_norm - spec_synthetic_downsampled)**2)
            io, jo = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            optimal_downsampling_parameters[f, exc, :] = l_min_list[exc] + io/2, 4 + jo/20
            cost_matrix_full[f,exc,:,:]= cost_matrix
    # Adjust parameters for Alexa488 under 405 excitation because of the UV background
    optimal_downsampling_parameters[3,0,0] = 405
    optimal_downsampling_parameters[3,0,1] = 8.9
    spec_merge_list = []
    for i in range(10):
        spec_merge = pd.DataFrame(columns = ['Channel', 'Source', 'Intensity'])
        spec_merge['Channel'] = np.arange(0,95)
        spec_merge['Intensity'] = np.zeros(95)
        spec_merge['Source'] = 'Measured'
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        spec_measured = spec_avg[i]
        for exc in range(5):
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            measured_intensity_magnitude = np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            if (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude < 0.1):
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, exc, 0], d_lambda = optimal_downsampling_parameters[i, exc, 1], n_channels = n_channels, exc = exc)
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
            elif (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude > 0.1):
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]
            elif (optimal_downsampling_parameters[i, exc, 0] == 0) & (exc <= 1):
                if optimal_downsampling_parameters[i, 2, 0] > 0:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 2, 0], d_lambda = optimal_downsampling_parameters[i, 2, 1], n_channels = 20, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
                else:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 3, 0], d_lambda = optimal_downsampling_parameters[i, 3, 1], n_channels = 14, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
        spec_merge_list.append(spec_merge)
    for i in range(10):
        spec_merge_list[i].to_csv('{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]), index = None)
    np.save('{}/optimal_downsampling_parameters.npy'.format(fret_folder), optimal_downsampling_parameters)
    return

def generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9,5,4,1,10,8,2,3,6,7]
    measured_channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [9, 12, 18, 26]
    l_min_list = [360, 420, 460, 500, 570]
    excitation_lasers = ['405 nm', '488 nm', '514 nm', '561nm', '633nm']
    fluorophores_text_list = ['Alexa 405', 'Pacific Blue', 'Pacific Green', 'Alexa488', 'Alexa532', 'DyLight 510', 'Alexa 546', 'Rhodamine Red X', 'Atto 610', 'Alexa 647']
    files = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered.csv'.format(fret_folder, b) for b in barcode_list]
    spec_avg = [pd.read_csv(f, header = None).iloc[0:95,0].values for f in files]
    optimal_downsampling_parameters = np.load('{}/optimal_downsampling_parameters.npy'.format(fret_folder))
    optimal_downsampling_parameters[3,0,0] = 405
    optimal_downsampling_parameters[3,0,1] = 8.9
    spec_merge_shift_list = []
    spectral_r_shift = [0, 7, 7, 0, 0, 0, 0, 0, 3, 0]
    spectral_b_shift = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(10):
        for exc in range(5):
            if optimal_downsampling_parameters[i,exc,0] > 0:
                optimal_downsampling_parameters[i,exc,0] += shift[i]*((spectral_r_shift[i] - spectral_b_shift[i])*np.random.random() + spectral_b_shift[i])
    for i in range(10):
        spec_merge = pd.DataFrame(columns = ['Channel', 'Source', 'Intensity'])
        spec_merge['Channel'] = np.arange(0,95)
        spec_merge['Intensity'] = np.zeros(95)
        spec_merge['Source'] = 'Measured'
        spec_synthetic = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        spec_measured = spec_avg[i]
        for exc in range(5):
            n_channels = measured_channel_indices[exc + 1] - measured_channel_indices[exc]
            measured_intensity_magnitude = np.max(spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]])
            if (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude < 0.05):
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, exc, 0], d_lambda = optimal_downsampling_parameters[i, exc, 1], n_channels = n_channels, exc = exc)
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
            elif (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude > 0.05) & (shift[i] > 0):
                spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, exc, 0], d_lambda = optimal_downsampling_parameters[i, exc, 1], n_channels = n_channels, exc = exc)
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
            elif (optimal_downsampling_parameters[i, exc, 0] > 0) & (measured_intensity_magnitude > 0.05) & (shift[i] == 0):
                spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Intensity'] = spec_measured[measured_channel_indices[exc]:measured_channel_indices[exc+1]]
            elif (optimal_downsampling_parameters[i, exc, 0] == 0) & (exc <= 1):
                if optimal_downsampling_parameters[i, 2, 0] > 0:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 2, 0], d_lambda = optimal_downsampling_parameters[i, 2, 1], n_channels = 20, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
                else:
                    spec_merge.loc[measured_channel_indices[exc]:measured_channel_indices[exc+1]-1, 'Source'] = 'Simulated'
                    spec_synthetic_downsampled_optimal = get_downsampled_synthetic_spectra(spec_synthetic, lambda_min = optimal_downsampling_parameters[i, 3, 0], d_lambda = optimal_downsampling_parameters[i, 3, 1], n_channels = 14, exc = exc)
                    spec_merge.loc[measured_channel_indices[exc+1]-spec_synthetic_downsampled_optimal.shape[0]:measured_channel_indices[exc+1]-1, 'Intensity'] = measured_intensity_magnitude*spec_synthetic_downsampled_optimal
        spec_merge_shift_list.append(spec_merge)
    return(spec_merge_shift_list)

def calculate_fret_distance_term(data_folder, dlm, dum):
    files = glob.glob(data_folder + '/*_excitation.csv')
    samples = [re.sub('_excitation.csv', '', os.path.basename(file)) for file in files]
    forster_distance = np.zeros((7,7))
    fret_transfer_distance_term_matrix = np.zeros((7,7))
    kappa_squared = 2/3
    ior = 1.4
    NA = 6.022e23
    Qd = 1
    prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
    molar_extinction_coefficient = [81000, 50000, 270000, 144000, 120000, 112000, 73000]
    fluorescence_quantum_yield = [0.61, 0.5, 0.33, 0.33, 0.5, 0.79, 0.92]
    fluorophores = [10,8,7,6,3,2,1]
    for i in range(7):
        for j in range(7):
            if i != j:
                distance = dlm + (dum - dlm)*np.random.random()
                fi = pd.read_csv('{}/R{}_excitation.csv'.format(data_folder, str(fluorophores[i])))
                fj = pd.read_csv('{}/R{}_excitation.csv'.format(data_folder, str(fluorophores[j])))
                emission_max_i = np.argmax(fi.Emission.values)
                emission_max_j = np.argmax(fj.Emission.values)
                if emission_max_i < emission_max_j:
                    fi_norm = np.clip(fi.Emission.values/fi.Emission.sum(), 0, 1)
                    fj_norm = np.clip(fj.Excitation.values/fj.Excitation.max(), 0, 1)
                    j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                    forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[j]*fluorescence_quantum_yield[i], 1/6)
                else:
                    fi_norm = np.clip(fi.Excitation.values/fi.Excitation.max(), 0, 1)
                    fj_norm = np.clip(fj.Emission.values/fj.Emission.sum(), 0, 1)
                    j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                    forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[i]*fluorescence_quantum_yield[j], 1/6)
                fret_transfer_distance_term_matrix[i,j] = (forster_distance[i,j]/distance)**6
    return(fret_transfer_distance_term_matrix)

def calculate_fret_distance_term_v2(data_folder, dlm, dum):
    files = glob.glob(data_folder + '/*_excitation.csv')
    samples = [re.sub('_excitation.csv', '', os.path.basename(file)) for file in files]
    forster_distance = np.zeros((7,7))
    fret_transfer_distance_term_matrix = np.zeros((7,7))
    kappa_squared = 2/3
    ior = 1.4
    NA = 6.022e23
    Qd = 1
    prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
    molar_extinction_coefficient = [81000, 50000, 270000, 144000, 120000, 112000, 73000]
    fluorescence_quantum_yield = [0.61, 0.5, 0.33, 0.33, 0.5, 0.79, 0.92]
    # DyLight 510 and RRX quantum yield are estimated at 0.5
    fluorophores = [10,8,7,6,3,2,1]
    for i in range(7):
        for j in range(7):
            if i != j:
                distance = dlm + (dum - dlm)*np.random.random()
                fi = pd.read_csv('{}/R{}_excitation.csv'.format(data_folder, str(fluorophores[i])))
                fj = pd.read_csv('{}/R{}_excitation.csv'.format(data_folder, str(fluorophores[j])))
                emission_max_i = np.argmax(fi.Emission.values)
                emission_max_j = np.argmax(fj.Emission.values)
                fi_norm = np.clip(fi.Emission.values/fi.Emission.sum(), 0, 1)
                fj_norm = np.clip(fj.Excitation.values/fj.Excitation.max(), 0, 1)
                j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[j]*fluorescence_quantum_yield[i], 1/6)
                fret_transfer_distance_term_matrix[i,j] = (forster_distance[i,j]/distance)**6
    return(forster_distance)

def calculate_forster_distance_7b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield):
    forster_distance = np.zeros((7,7))
    kappa_squared = 2/3
    ior = 1.52
    NA = 6.022e23
    Qd = 1
    prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
    # DyLight 510 and RRX quantum yield are estimated at 0.5
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    for i in range(7):
        for j in range(7):
            if i != j:
                fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
                fj = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[j])))
                emission_max_i = np.argmax(fi.Emission.values)
                emission_max_j = np.argmax(fj.Emission.values)
                fi_norm = np.clip(fi.Emission.values/fi.Emission.sum(), 0, 1)
                fj_norm = np.clip(fj.Excitation.values/fj.Excitation.max(), 0, 1)
                j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[j]*fluorescence_quantum_yield[i], 1/6)
    return(forster_distance)

def calculate_forster_distance_10b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield):
    forster_distance = np.zeros((10,10))
    kappa_squared = 2/3
    ior = 1.52
    NA = 6.022e23
    Qd = 1
    prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    for i in range(10):
        for j in range(10):
            if i != j:
                fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
                fj = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[j])))
                emission_max_i = np.argmax(fi.Emission.values)
                emission_max_j = np.argmax(fj.Emission.values)
                fi_norm = np.clip(fi.Emission.values/fi.Emission.sum(), 0, 1)
                fj_norm = np.clip(fj.Excitation.values/fj.Excitation.max(), 0, 1)
                j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[j]*fluorescence_quantum_yield[i], 1/6)
    return(forster_distance)

def calculate_fret_number_7b(forster_distance_matrix, dlm, dum):
    fret_number_matrix = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            if i != j:
                distance = dlm + (dum - dlm)*np.random.random()
                fret_number_matrix[i,j] = (forster_distance[i,j]/distance)**2
    return(fret_number_matrix)

def calculate_fret_number_10b(forster_distance_matrix, dlm, dum):
    fret_number_matrix = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if i != j:
                distance = dlm + (dum - dlm)*np.random.random()
                fret_number_matrix[i,j] = (forster_distance[i,j]/distance)**2
    return(fret_number_matrix)

def calculate_excitation_efficiency_matrix_7b(reference_folder):
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 23, 43, 57, 63]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    # DyLight 510 and RRX quantum yield are estimated at 0.5
    excitation_efficiency_matrix = np.zeros((7,4))
    for i in range(7):
        spec = np.average(pd.read_csv(files[i]), axis = 0)
        for exc in range(4):
            excitation_efficiency_matrix[i, exc] = np.max(spec[32+channel_indices[exc]: 32+channel_indices[exc+1]])
    return(excitation_efficiency_matrix)

def calculate_excitation_efficiency_matrix_10b(reference_folder):
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 32, 55, 75, 89, 95]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    # DyLight 510 and RRX quantum yield are estimated at 0.5
    excitation_efficiency_matrix = np.zeros((10,5))
    for i in range(10):
        spec = np.average(pd.read_csv(files[i]), axis = 0)
        for exc in range(5):
            excitation_efficiency_matrix[i, exc] = np.max(spec[channel_indices[exc]: channel_indices[exc+1]])
    return(excitation_efficiency_matrix)

def convert_barcode_to_10b(bc7b):
    bits = list(bc7b.astype(str))
    bits.insert(4, '0')
    bits.insert(4, '0')
    bits.insert(1, '0')
    bc10b = ''.join(bits)
    return(bc10b)

def calculate_extinction_coefficient_direct_excitation_7b(fret_folder, molar_extinction_coefficient):
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 23, 43, 57, 63]
    excitation_wavelength = [488, 514, 561, 633]
    extinction_coefficient_matrix = np.zeros((7,4))
    for i in range(7):
        fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        max_excitation = fi.loc[np.argmax(fi.Excitation.values), 'Excitation']
        for exc in range(4):
            extinction_coefficient_matrix[i, exc] = molar_extinction_coefficient[i]*fi.loc[fi.Wavelength.values == excitation_wavelength[exc], 'Excitation'].values[0]/max_excitation
    return(extinction_coefficient_matrix)

def calculate_extinction_coefficient_direct_excitation_10b(fret_folder, molar_extinction_coefficient):
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 32, 55, 75, 89, 95]
    excitation_wavelength = [405, 488, 514, 561, 633]
    extinction_coefficient_matrix = np.zeros((10,5))
    for i in range(10):
        fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
        max_excitation = fi.loc[np.argmax(fi.Excitation.values), 'Excitation']
        for exc in range(5):
            extinction_coefficient_matrix[i, exc] = molar_extinction_coefficient[i]*fi.loc[fi.Wavelength.values == excitation_wavelength[exc], 'Excitation'].values[0]/max_excitation
    return(extinction_coefficient_matrix)

def calculate_extinction_coefficient_fret_7b(fret_folder, molar_extinction_coefficient):
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 23, 43, 57, 63]
    excitation_wavelength = [488, 514, 561, 633]
    # assuming conjugation to DNA moves Alexa 647 QY from 270000 down to 60%
    extinction_coefficient_fret_matrix = np.zeros((7,7))
    for i in range(7):
        for j in range(i+1, 7):
            fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
            fj = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[j])))
            max_donor_emission_wavelength = fi.loc[np.argmax(fi.Emission.values), 'Wavelength']
            max_acceptor_excitation = fj.loc[np.argmax(fj.Excitation.values), 'Excitation']
            extinction_coefficient_fret_matrix[i, j] = molar_extinction_coefficient[j]*fj.loc[fj.Wavelength.values == int(max_donor_emission_wavelength), 'Excitation'].values[0]/max_acceptor_excitation
    return(extinction_coefficient_fret_matrix)

def calculate_extinction_coefficient_fret_10b(fret_folder, molar_extinction_coefficient):
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 32, 55, 75, 89, 95]
    excitation_wavelength = [405, 488, 514, 561, 633]
    extinction_coefficient_fret_matrix = np.zeros((10,10))
    for i in range(10):
        for j in range(i+1, 10):
            fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[i])))
            fj = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[j])))
            max_donor_emission_wavelength = fi.loc[np.argmax(fi.Emission.values), 'Wavelength']
            max_acceptor_excitation = fj.loc[np.argmax(fj.Excitation.values), 'Excitation']
            extinction_coefficient_fret_matrix[i, j] = molar_extinction_coefficient[j]*fj.loc[fj.Wavelength.values == int(max_donor_emission_wavelength), 'Excitation'].values[0]/max_acceptor_excitation
    return(extinction_coefficient_fret_matrix)

def analyze_absorption_probability():
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    channel_indices = [0, 23, 43, 57, 63]
    excitation_wavelength = [488, 514, 561, 633]
    molar_extinction_coefficient = [73000, 81000, 50000, 112000, 120000, 150000, 270000]
    absorption_efficiency_true = np.zeros((concentration.shape[0],2))
    absorption_efficiency_approximate = np.zeros((concentration.shape[0],2))
    exc = 3
    for i in range(concentration.shape[0]):
        for j in range(2):
            fluorophore_index = j + 5
            fi = pd.read_csv('{}/R{}_excitation.csv'.format(fret_folder, str(fluorophores[fluorophore_index])))
            max_excitation = fi.loc[np.argmax(fi.Excitation.values), 'Excitation']
            molar_extinction_at_excitation = molar_extinction_coefficient[fluorophore_index]*fi.loc[fi.Wavelength.values == excitation_wavelength[exc], 'Excitation'].values[0]/max_excitation
            absorption_efficiency_approximate[i, j] = molar_extinction_at_excitation
            absorption_efficiency_true[i, j] = 1 - 10**(-depth*molar_extinction_at_excitation*concentration[i])
    relative_absorption_efficiency_approximate = absorption_efficiency_approximate/np.sum(absorption_efficiency_approximate, axis = 1)[:,None]
    relative_absorption_efficiency_true = absorption_efficiency_true/np.sum(absorption_efficiency_true, axis = 1)[:,None]
    return

def analyze_residual_spc(reference_folder, enc, exc):
    spc_list = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    min_distance = np.zeros((len(spc_list), 20))
    best_simulated_spec_list = np.zeros((len(spc_list), 20, 23))
    for s in range(len(spc_list)):
        for r in range(20):
            spc = spc_list[s]
            print('Trying spc = {}, replicate {}...'.format(spc, r))
            barcode_list = [1, 512, 128, 2, 4, 32, 64]
            files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
            spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
            spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
            channel_indices = [0,23,43,57,63]
            synthetic_channel_indices = [0, 3, 9, 17]
            nbit = 7
            nchannels = 63
            simulation_per_code = spc
            dlm = 5
            dum = 15
            labeling_density = [1, 1, 2, 0.5, 2, 1, 1]
            fluorophore_list = [6, 0, 1, 5, 4, 3, 2]
            excitation_efficiency_matrix = calculate_excitation_efficiency_matrix(reference_folder)
            forster_distance = calculate_forster_distance(fret_folder)
            fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
            for i in range(simulation_per_code):
                fret_number_matrix[i,:,:] = calculate_fret_number(forster_distance, dlm, dum)
            code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
            numeric_code_list = np.array([int(a) for a in list(code)])
            simulated_spectra = np.zeros((simulation_per_code, nchannels))
            indices = [0,23,43,57,63]
            relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
            coefficients = np.zeros((simulation_per_code, 7))
            absorption_probability_direct_excitation = extinction_coefficient_matrix[:,exc]*relevant_fluorophores/np.sum(extinction_coefficient_matrix[:,exc]*relevant_fluorophores)
            for i in range(simulation_per_code):
                f_sensitized = np.zeros(7)
                fret_efficiency_out = np.zeros(7)
                for j in range(7):
                    omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:])
                    fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
                    f_fret_sensitized = 0
                    for k in range(j):
                        omega_total = np.sum(fret_number_matrix[i, k, k+1:]*relevant_fluorophores[k+1:])
                        absorption_probability_fret = extinction_coefficient_fret_matrix[k,:]*relevant_fluorophores/np.sum(extinction_coefficient_fret_matrix[k,:]*relevant_fluorophores)
                        if omega_total > 0:
                            f_fret_sensitized += (fret_number_matrix[i, k, j]/omega_total)*relevant_fluorophores[k]*absorption_probability_fret[j]*f_sensitized[k]*fret_efficiency_out[k]
                    f_sensitized[j] = absorption_probability_direct_excitation[j]*excitation_efficiency_matrix[j, exc] + f_fret_sensitized
                    coefficients[i,j] = f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]
            simulated_spectra_list = [coefficients[:,k][:,None]*optimal_synthetic_spec[fluorophore_list[k]][None,:] for k in range(nbit)]
            test = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
            test_norm = test/np.max(test,axis=1)[:,None]
            distance = np.zeros(spc)
            for i in range(spc):
                distance[i] = np.sum((cell_avg_spec_norm[0:23] - test_norm[i,:])**2)
            min_distance[s,r] = np.min(distance)
            best_simulated_spec_list[s, r, :] = test_norm[np.argmin(distance),:]
    np.save('{}/spc_vs_min_distance_matrix.npy'.format(fret_folder), min_distance)
    np.save('{}/best_simulated_spec_list.npy'.format(fret_folder), best_simulated_spec_list)
    ### plot min error box plot
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(5))
    flierprops = dict(marker = '.', markerfacecolor = (1,0.5,0), markersize = 6, markeredgewidth = 0, alpha = 0.8)
    bp = plt.boxplot([min_distance[i,:]*100 for i in range(len(spc_list))], positions = [np.log10(s) for s in spc_list], notch = False, widths = 0.2, flierprops = flierprops)
    for b in bp['boxes']:
        b.set_color((0,0.5,1))
    for w in bp['whiskers']:
        w.set_color((0,0.5,1))
    for c in bp['caps']:
        c.set_color((0,0.5,1))
    for m in bp['medians']:
        m.set_color((1,0.5,0))
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.xlim(np.log10(10), np.log10(100000))
    plt.xticks(np.arange(1,6), [r'$10^{}$'.format(i) for i in range(1,6)], fontsize = 8, color = theme_color)
    plt.xlabel('Simulation Per Barcode', fontsize = 8, color = theme_color)
    plt.ylabel(r'$\sum$($f_m$ - $f_s$)$^2$', fontsize = 8, color = theme_color)
    plt.text(np.log10(3), np.max(min_distance)*100 + 0.3, r'$\times10^{-2}$', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.95, top = 0.9)
    plt.savefig('{}/min_error_barcode_1111111_vs_spc.pdf'.format(reference_folder), dpi = 300, transparent = True)
    ### plot spec vs. cost heatmap
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(4))
    im = plt.imshow(min_distance*1000, cmap = 'RdBu')
    plt.yticks(np.arange(10), ['50', '100', '200', '500', '1K', '2K', '5K', '10K', '20K', '50K'])
    plt.xlabel('Simulations ID', fontsize = 8, color = theme_color)
    plt.ylabel('SPC', fontsize = 8, color = theme_color)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.yticks(np.arange(10))
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    divider = make_axes_locatable(plt.axes())
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    cbar = plt.colorbar(im, cax = cax, orientation = 'vertical')
    cbar.ax.tick_params(direction = 'in', length = 1, labelsize = 8, colors = theme_color)
    cbar.outline.set_edgecolor(theme_color)
    cbar.set_label(r'Cost$\times10^3$', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.15, bottom = 0.15, right = 0.85, top = 0.98)
    plt.savefig('{}/spc_vs_cost_heatmap.pdf'.format(fret_folder), dpi = 300, transparent = True)
    plt.close()
    ### plot min error spec comparison
    cmap = cm.get_cmap('RdYlBu')
    max_min_distance = np.max(min_distance)
    min_min_distance = np.min(min_distance)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(25), cm_to_inches(9))
    gs = GridSpec(9,10)
    for s in range(len(spc_list)):
        best_simulated_spec_norm = best_simulated_spec_list[s, np.argmin(min_distance[s,:]), :]
        median_simulated_spec_norm = best_simulated_spec_list[s, np.argsort(min_distance[s,:])[min_distance.shape[1]//2], :]
        worst_simulated_spec_norm = best_simulated_spec_list[s, np.argmax(min_distance[s,:]), :]
        min_distance_s = np.min(min_distance[s,:])
        median_distance_s = np.median(min_distance[s,:])
        max_distance_s = np.max(min_distance[s,:])
        ax = plt.subplot(gs[0:2, s])
        plt.plot(np.arange(23), cell_avg_spec_norm[0:23])
        plt.fill_between(np.arange(23), cell_avg_spec_norm[0:23] - cell_std_spec_norm[0:23], cell_avg_spec_norm[0:23] + cell_std_spec_norm[0:23], color = (0,0.5,1), alpha = 0.5)
        plt.plot(best_simulated_spec_norm, color = (1,0.5,0))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        plt.ylim(-0.05, 1.25)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        ax = plt.subplot(gs[2, s])
        plt.plot(np.arange(23), best_simulated_spec_norm - cell_avg_spec_norm[0:23], color = cmap((min_distance_s - min_min_distance)/(max_min_distance - min_min_distance)))
        plt.fill_between(np.arange(23), best_simulated_spec_norm - cell_avg_spec_norm[0:23] + cell_std_spec_norm[0:23], best_simulated_spec_norm - cell_avg_spec_norm[0:23] - cell_std_spec_norm[0:23], color = cmap((min_distance_s - min_min_distance)/(max_min_distance - min_min_distance)), alpha = 0.8)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.yticks([-0.1, 0, 0.1])
        plt.ylim(-0.15, 0.15)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax = plt.subplot(gs[3:5, s])
        plt.plot(np.arange(23), cell_avg_spec_norm[0:23])
        plt.fill_between(np.arange(23), cell_avg_spec_norm[0:23] - cell_std_spec_norm[0:23], cell_avg_spec_norm[0:23] + cell_std_spec_norm[0:23], color = (0,0.5,1), alpha = 0.5)
        plt.plot(median_simulated_spec_norm, color = (1,0.5,0))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        plt.ylim(-0.05, 1.25)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        ax = plt.subplot(gs[5, s])
        plt.plot(np.arange(23), median_simulated_spec_norm - cell_avg_spec_norm[0:23], color = cmap((median_distance_s - min_min_distance)/(max_min_distance - min_min_distance)))
        plt.fill_between(np.arange(23), median_simulated_spec_norm - cell_avg_spec_norm[0:23] + cell_std_spec_norm[0:23], median_simulated_spec_norm - cell_avg_spec_norm[0:23] - cell_std_spec_norm[0:23], color = cmap((median_distance_s - min_min_distance)/(max_min_distance - min_min_distance)), alpha = 0.8)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.yticks([-0.1, 0, 0.1])
        plt.ylim(-0.15, 0.15)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax = plt.subplot(gs[6:8, s])
        plt.plot(np.arange(23), cell_avg_spec_norm[0:23])
        plt.fill_between(np.arange(23), cell_avg_spec_norm[0:23] - cell_std_spec_norm[0:23], cell_avg_spec_norm[0:23] + cell_std_spec_norm[0:23], color = (0,0.5,1), alpha = 0.5)
        plt.plot(best_simulated_spec_norm, color = (1,0.5,0))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        plt.ylim(-0.05, 1.25)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        ax = plt.subplot(gs[8, s])
        plt.plot(np.arange(23), worst_simulated_spec_norm - cell_avg_spec_norm[0:23], color = cmap((max_distance_s - min_min_distance)/(max_min_distance - min_min_distance)))
        plt.fill_between(np.arange(23), worst_simulated_spec_norm - cell_avg_spec_norm[0:23] + cell_std_spec_norm[0:23], worst_simulated_spec_norm - cell_avg_spec_norm[0:23] - cell_std_spec_norm[0:23], color = cmap((max_distance_s - min_min_distance)/(max_min_distance - min_min_distance)), alpha = 0.8)
        plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
        plt.yticks([-0.1, 0, 0.1])
        plt.ylim(-0.15, 0.15)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
    ax = plt.subplot(gs[6:8, 9])
    plt.ylabel('Intensity', fontsize = 8, color = theme_color, labelpad = 10)
    plt.yticks([0.0, 0.5, 1.0], [' 0.0', ' 0.5', ' 1.0'], ha = 'right')
    ax.tick_params(axis = 'y', pad = 18)
    ax.yaxis.tick_right()
    ax = plt.subplot(gs[8, 9])
    plt.xlabel('Channel', fontsize = 8, color = theme_color)
    plt.ylabel(r'$\delta$', fontsize = 8, color = theme_color, labelpad = 2)
    ax.yaxis.tick_right()
    plt.xticks([0, 10, 20], [0, 10, 20], fontsize = 8)
    plt.yticks([-0.1, 0.0, 0.1], ['-0.1', '0.0', '0.1'], fontsize = 8, ha = 'right')
    ax.tick_params(axis = 'y', pad = 18)
    spc_text_list = ['50', '100', '200', '500', '1K', '2K', '5K', '10K', '20K', '50K']
    for i in range(10):
        ax = plt.subplot(gs[0:2, i])
        ax.set_title('SPC = {}'.format(spc_text_list[i]), fontsize = 8, color = theme_color)
    ax = plt.subplot(gs[0:2,0])
    plt.text(-10, 0.25, r'Minimum', fontsize = 8, color = theme_color, ha = 'center')
    ax = plt.subplot(gs[3:5,0])
    plt.text(-10, 0.25, r'Median', fontsize = 8, color = theme_color, ha = 'center')
    ax = plt.subplot(gs[6:8,0])
    plt.text(-10, 0.25, r'Maximum', fontsize = 8, color = theme_color, ha = 'center')
    plt.subplots_adjust(left = 0.08, bottom = 0.12, right = 0.95, top = 0.98)
    plt.savefig('{}/min_error_spec_comparison_1111111.pdf'.format(fret_folder), dpi = 300, transparent = True)
    return

def plot_forster_distance(forster_distance, reference_folder):
    fluorophore_list = ['Alexa 488', 'Alexa 532', 'DyLight 511', 'Alexa 546', 'Rhodamind Red X', 'Atto 610', 'Alexa 647']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(7))
    im = plt.imshow(forster_distance, cmap = 'RdBu')
    plt.xticks(np.arange(7), fluorophore_list, color = theme_color, fontsize = 8, rotation = 90)
    plt.yticks(np.arange(7), fluorophore_list, color = theme_color, fontsize = 8)
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    divider = make_axes_locatable(plt.axes())
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    cbar = plt.colorbar(im, cax = cax, orientation = 'vertical')
    cbar.ax.tick_params(direction = 'in', length = 1, labelsize = 8, colors = theme_color)
    cbar.outline.set_edgecolor(theme_color)
    cbar.set_label(r'Forster Distance [nm]', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.35, bottom = 0.35, right = 0.85, top = 0.98)
    plt.savefig('{}/forster_distance.pdf'.format(fret_folder), dpi = 300, transparent = True)
    return

def plot_spectral_overlap(fret_folder):
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    measured_spectra_filenames = ['{}/R{}_excitation.csv'.format(fret_folder, fluorophores[i]) for i in range(7)]
    spec = [pd.read_csv(f) for f in measured_spectra_filenames]
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(20), cm_to_inches(20))
    gs = GridSpec(7,7)
    for i in range(7):
        for j in range(7):
            ax = plt.subplot(gs[i,j])
            plt.plot(spec[i].Wavelength.values, spec[i].Emission.values/np.max(spec[i].Emission.values), color = (0, 0.5, 1))
            plt.plot(spec[j].Wavelength.values, spec[j].Excitation.values/np.max(spec[j].Excitation.values), color = (1, 0.5, 0))
            plt.xlim(400,700)
            plt.xticks([])
            plt.yticks([])
            plt.axes().spines['left'].set_color(theme_color)
            plt.axes().spines['bottom'].set_color(theme_color)
            plt.axes().spines['right'].set_color(theme_color)
            plt.axes().spines['top'].set_color(theme_color)
    return

def load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b_v4(reference_folder, fret_folder, spc):
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    fluorescence_quantum_yield = [0.92, 0.61, 0.5, 0.79, 0.5, 0.7, 0.33]
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra_7b.csv'.format(fret_folder, fluorophores[i]) for i in range(7)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose())[32:,32:] for f in measured_spectra_filenames]
    channel_indices = [0,23,43,57,63]
    synthetic_channel_indices = [0, 3, 9, 17]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    dlm = 5
    dum = 20
    labeling_density = [1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [6, 0, 1, 5, 4, 3, 2]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation(fret_folder)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret(fret_folder)
    forster_distance = calculate_forster_distance(fret_folder)
    fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
    distance_list = np.zeros(simulation_per_code)
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number(forster_distance, dlm, dum)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1]])
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    training_data_low488 = pd.DataFrame()
    training_data_low514 = pd.DataFrame()
    training_data_low561 = pd.DataFrame()
    training_data_low488561 = pd.DataFrame()
    for enc in range(1, 128):
        print(enc)
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        indices = [0,23,43,57,63]
        for exc in range(4):
            relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
            coefficients = np.zeros((simulation_per_code, 7))
            extinction_coefficient_direct_excitation_total = np.sum(extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores)
            if extinction_coefficient_direct_excitation_total > 0:
                absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores/extinction_coefficient_direct_excitation_total
            else:
                absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores
            for i in range(simulation_per_code):
                f_sensitized = np.zeros(7)
                fret_efficiency_out = np.zeros(7)
                exciton_saturation = np.random.random()
                absorption_probability_direct_excitation_adjusted = (1 - exciton_saturation)*absorption_probability_direct_excitation + exciton_saturation*np.repeat(1/7, 7)
                distance = dlm + (dum - dlm)*np.random.random()
                if exc == 1:
                    quenching = np.exp(-2*(1/6)*np.arange(7)*(distance - dlm)/(dum - dlm))
                else:
                    quenching = np.ones(7)
                for j in range(7):
                    omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:]*quenching[j+1:])
                    fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
                    f_fret_sensitized = 0
                    for k in range(j):
                        omega_total = np.sum(fret_number_matrix[i, k, k+1:]*relevant_fluorophores[k+1:]*quenching[k+1:])
                        extinction_fret_total = np.sum(extinction_coefficient_fret[k,:]*relevant_fluorophores*quenching)
                        if extinction_fret_total > 0:
                            absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores*quenching/extinction_fret_total
                        else:
                            absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores*quenching
                        absorption_probability_fret_adjusted = (1 - exciton_saturation)*absorption_probability_fret + exciton_saturation*np.repeat(1/7, 7)
                        if omega_total > 0:
                            f_fret_sensitized += (fret_number_matrix[i, k, j]/omega_total)*relevant_fluorophores[k]*absorption_probability_fret_adjusted[j]*f_sensitized[k]*fret_efficiency_out[k]*quenching[k]
                    f_sensitized[j] = absorption_probability_direct_excitation_adjusted[j]*excitation_efficiency_matrix[j, exc]*quenching[j] + f_fret_sensitized
                    coefficients[i,j] = f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]
            spec_list = [np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,channel_indices[exc]:channel_indices[exc+1]] for k in range(nbit)]
            spec_list_abs = [np.abs(s) for s in spec_list]
            spec_list_norm = [s/np.max(s, axis = 1)[:,None] for s in spec_list_abs]
            simulated_spectra_list = [coefficients[:,k][:,None]*spec_list_norm[k] for k in range(nbit)]
            simulated_spectra[0:simulation_per_code,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale = [0.05, 0.1, 0.25, 0.35]
        for k in range(0,4):
            error_coefficient = error_scale[k] + (1-error_scale[k])*np.random.random(simulated_spectra_norm.shape[0])
            max_intensity = np.max(simulated_spectra_norm[:,indices[k]:indices[k+1]], axis = 1)
            max_intensity_error_simulation = error_coefficient*max_intensity
            error_coefficient[max_intensity_error_simulation < error_scale[k]] = 1
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = error_coefficient[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1]
        ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,23,43,57,63]
        for k in range(0,4):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (error_scale[k]*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm['c1'] = 0
        ss_norm['c2'] = 0
        ss_norm['c3'] = 0
        ss_norm['c4'] = 0
        ss_norm['code'] = '{}_error'.format(code)
        training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,23,43,57,63]
        if (numeric_code_list[6] == 0) & (numeric_code_list[0] | numeric_code_list[1]):
            error_scale_488 = 0.015
            simulated_spectra_norm[:,indices[0]:indices[1]] = (error_scale_488 + (error_scale[0] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[0]:indices[1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = 0
            ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1]
            ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
            ss_norm['code'] = '{}'.format(code)
            training_data_low488 = training_data_low488.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,23,43,57,63]
        if (numeric_code_list[4] or numeric_code_list[5]) & (numeric_code_list[6] == 0) & (numeric_code_list[0] == 0) & (numeric_code_list[1] == 0):
            error_scale_514 = 0.02
            simulated_spectra_norm[:,indices[1]:indices[2]] = (error_scale_514 + (error_scale[1] - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[1]:indices[2]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
            ss_norm['c2'] = 0
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
            ss_norm['code'] = '{}'.format(code)
            training_data_low514 = training_data_low514.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,23,43,57,63]
        if (numeric_code_list[1] == 1) & (numeric_code_list[4] == 0) & (numeric_code_list[5] == 0):
            error_scale_561 = 0.1
            simulated_spectra_norm[:,indices[2]:indices[3]] = (error_scale_561 + (error_scale[2] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[2]:indices[3]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
            ss_norm['code'] = '{}'.format(code)
            training_data_low561 = training_data_low561.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,23,43,57,63]
        if (numeric_code_list[1] == 1) & (numeric_code_list[4] == 0) & (numeric_code_list[5] == 0) & (numeric_code_list[6] == 0):
            error_scale_488 = 0.015
            error_scale_561 = 0.1
            simulated_spectra_norm[:,indices[0]:indices[1]] = (error_scale_488 + (error_scale[0] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[0]:indices[1]]
            simulated_spectra_norm[:,indices[2]:indices[3]] = (error_scale_561 + (error_scale[2] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[2]:indices[3]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
            ss_norm['code'] = '{}'.format(code)
            training_data_low488561 = training_data_low488561.append(ss_norm, ignore_index = True)

    training_data_full = pd.concat([training_data, training_data_negative, training_data_low488, training_data_low514, training_data_low561, training_data_low488561])
    training_data = pd.concat([training_data, training_data_low488, training_data_low514, training_data_low561, training_data_low488561])
    scaler_full = preprocessing.StandardScaler().fit(training_data_full.values[:,0:63])
    training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:63])
    umap_transform = umap.UMAP(n_neighbors = 400, min_dist = 0.001, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67], y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
    clf[0].fit(training_data_full.values[:,[3, 7, 11]], training_data_full.c1.values)
    clf[1].fit(training_data_full.values[:,[24, 27, 29, 31, 33]], training_data_full.c2.values)
    clf[2].fit(training_data_full.values[:,[43, 47]], training_data_full.c3.values)
    clf[3].fit(training_data_full.values[:,[57, 60]], training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 1)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(scaler_full, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_scaler.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return(training_data_full)

def load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_10b_v4(reference_folder, fret_folder, spc):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    fluorescence_quantum_yield = [0.5, 0.5, 0.5, 0.92, 0.61, 0.5, 0.79, 0.5, 0.7, 0.33]
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    dlm = 5
    dum = 15
    labeling_density = [1, 1, 1, 1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_10b(fret_folder)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_10b(fret_folder)
    forster_distance = calculate_forster_distance_10b(fret_folder)
    fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
    distance_list = np.zeros(simulation_per_code)
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    for enc in range(1, 10):
        print(enc)
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        indices = [0,32,55,75,89,95]
        for exc in range(5):
            relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
            coefficients = np.zeros((simulation_per_code, 10))
            extinction_coefficient_direct_excitation_total = np.sum(extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores)
            if extinction_coefficient_direct_excitation_total > 0:
                absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores/extinction_coefficient_direct_excitation_total
            else:
                absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores
            for i in range(simulation_per_code):
                f_sensitized = np.zeros(10)
                fret_efficiency_out = np.zeros(10)
                exciton_saturation = np.random.random()
                absorption_probability_direct_excitation_adjusted = (1 - exciton_saturation)*absorption_probability_direct_excitation + exciton_saturation*np.repeat(1/10, 10)
                distance = dlm + (dum - dlm)*np.random.random()
                if exc <= 2:
                    quenching = np.ones(10)
                    quenching[3:] = np.exp(-2*(1/6)*np.arange(7)*(distance - dlm)/(dum - dlm))
                else:
                    quenching = np.ones(10)
                for j in range(10):
                    omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:])
                    fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
                    f_fret_sensitized = 0
                    for k in range(j):
                        omega_total = np.sum(fret_number_matrix[i, k, k+1:]*relevant_fluorophores[k+1:])
                        extinction_fret_total = np.sum(extinction_coefficient_fret[k,:]*relevant_fluorophores)
                        if extinction_fret_total > 0:
                            absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores/extinction_fret_total
                        else:
                            absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores
                        absorption_probability_fret_adjusted = (1 - exciton_saturation)*absorption_probability_fret + exciton_saturation*np.repeat(1/10, 10)
                        if omega_total > 0:
                            f_fret_sensitized += (fret_number_matrix[i, k, j]/omega_total)*relevant_fluorophores[k]*absorption_probability_fret_adjusted[j]*f_sensitized[k]*fret_efficiency_out[k]
                    f_sensitized[j] = absorption_probability_direct_excitation_adjusted[j]*excitation_efficiency_matrix[j, exc] + f_fret_sensitized
                    coefficients[i,j] = f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]*quenching[j]
            spec_list = [np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,channel_indices[exc]:channel_indices[exc+1]] for k in range(nbit)]
            spec_list_abs = [np.abs(s) for s in spec_list]
            spec_list_norm = [s/np.max(s, axis = 1)[:,None] for s in spec_list_abs]
            simulated_spectra_list = [coefficients[:,k][:,None]*spec_list_norm[k] for k in range(nbit)]
            simulated_spectra[0:simulation_per_code,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale = [0.05, 0.05, 0.1, 0.25, 0.35]
        for k in range(0,5):
            error_coefficient = error_scale[k] + (1-error_scale[k])*np.random.random(simulated_spectra_norm.shape[0])
            max_intensity = np.max(simulated_spectra_norm[:,indices[k]:indices[k+1]], axis = 1)
            max_intensity_error_simulation = error_coefficient*max_intensity
            error_coefficient[max_intensity_error_simulation < error_scale[k]] = 1
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = error_coefficient[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2]
        ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm['c6'] = numeric_code_list[1]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (error_scale[k]*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm['c0'] = 0
        ss_norm['c1'] = 0
        ss_norm['c2'] = 0
        ss_norm['c3'] = 0
        ss_norm['c4'] = 0
        ss_norm['code'] = '{}_error'.format(code)
        training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)

    training_data_full = pd.concat([training_data, training_data_negative])
    training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:95])
    umap_transform = umap.UMAP(n_neighbors = 200, min_dist = 0.001, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:100], y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
    clf[0].fit(training_data_full.values[:,[2, 5, 11]], training_data_full.c0.values)
    clf[1].fit(training_data_full.values[:,[35, 39, 43]], training_data_full.c1.values)
    clf[2].fit(training_data_full.values[:,[56, 59, 61, 63, 65]], training_data_full.c2.values)
    clf[3].fit(training_data_full.values[:,[75, 79]], training_data_full.c3.values)
    clf[4].fit(training_data_full.values[:,[89, 92]], training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 1)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(scaler_full, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_10b_scaler.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_10b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_10b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transform_biofilm_10b.pkl'.format(reference_folder, str(spc)))
    return(training_data_full)

def load_training_data_simulate_excitation_adjusted_normalized_violet_derivative_umap_transformed(reference_folder, learning_mode, spc):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    fluorescence_quantum_yield = [0.5, 0.5, 0.5, 0.92, 0.61, 0.5, 0.79, 0.5, 0.7, 0.33]
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    dlm = 5
    dum = 15
    labeling_density = [1, 1, 1, 1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_10b(fret_folder)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_10b(fret_folder)
    forster_distance = calculate_forster_distance_10b(fret_folder)
    fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
    distance_list = np.zeros(simulation_per_code)
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    for enc in range(1, 16):
        print(enc)
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        indices = [0,32,55,75,89,95]
        for exc in range(5):
            relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
            coefficients = np.zeros((simulation_per_code, 10))
            extinction_coefficient_direct_excitation_total = np.sum(extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores)
            if extinction_coefficient_direct_excitation_total > 0:
                absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores/extinction_coefficient_direct_excitation_total
            else:
                absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores
            for i in range(simulation_per_code):
                f_sensitized = np.zeros(10)
                fret_efficiency_out = np.zeros(10)
                exciton_saturation = np.random.random()
                absorption_probability_direct_excitation_adjusted = (1 - exciton_saturation)*absorption_probability_direct_excitation + exciton_saturation*np.repeat(1/10, 10)
                distance = dlm + (dum - dlm)*np.random.random()
                if exc <= 2:
                    quenching = np.ones(10)
                    quenching[3:] = np.exp(-2*(1/6)*np.arange(7)*(distance - dlm)/(dum - dlm))
                else:
                    quenching = np.ones(10)
                for j in range(10):
                    omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:])
                    fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
                    f_fret_sensitized = 0
                    for k in range(j):
                        omega_total = np.sum(fret_number_matrix[i, k, k+1:]*relevant_fluorophores[k+1:])
                        extinction_fret_total = np.sum(extinction_coefficient_fret[k,:]*relevant_fluorophores)
                        if extinction_fret_total > 0:
                            absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores/extinction_fret_total
                        else:
                            absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores
                        absorption_probability_fret_adjusted = (1 - exciton_saturation)*absorption_probability_fret + exciton_saturation*np.repeat(1/10, 10)
                        if omega_total > 0:
                            f_fret_sensitized += (fret_number_matrix[i, k, j]/omega_total)*relevant_fluorophores[k]*absorption_probability_fret_adjusted[j]*f_sensitized[k]*fret_efficiency_out[k]
                    f_sensitized[j] = absorption_probability_direct_excitation_adjusted[j]*excitation_efficiency_matrix[j, exc] + f_fret_sensitized
                    coefficients[i,j] = f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]*quenching[j]
            spec_list = [np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,channel_indices[exc]:channel_indices[exc+1]] for k in range(nbit)]
            spec_list_abs = [np.abs(s) for s in spec_list]
            spec_list_norm = [s/np.max(s, axis = 1)[:,None] for s in spec_list_abs]
            simulated_spectra_list = [coefficients[:,k][:,None]*spec_list_norm[k] for k in range(nbit)]
            simulated_spectra[0:simulation_per_code,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale = [0.1, 0.05, 0.1, 0.25, 0.35]
        for k in range(0,5):
            error_coefficient = error_scale[k] + (1-error_scale[k])*np.random.random(simulated_spectra_norm.shape[0])
            max_intensity = np.max(simulated_spectra_norm[:,indices[k]:indices[k+1]], axis = 1)
            max_intensity_error_simulation = error_coefficient*max_intensity
            error_coefficient[max_intensity_error_simulation < error_scale[k]] = 1
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = error_coefficient[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2]
        ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm['c5'] = numeric_code_list[1]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (error_scale[k]*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm['c0'] = 0
        ss_norm['c1'] = 0
        ss_norm['c2'] = 0
        ss_norm['c3'] = 0
        ss_norm['c4'] = 0
        ss_norm['c5'] = 0
        ss_norm['code'] = '{}_error'.format(code)
        training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)

    training_data_full = pd.concat([training_data, training_data_negative])
    umap_transform = umap.UMAP(n_neighbors = 200, min_dist = 0.001, metric = channel_cosine_intensity_violet_derivative_v2).fit(training_data.iloc[:,0:132].values, y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(6)]
    clf[0].fit(training_data_full.values[:,[2, 5, 11]], training_data_full.c0.values)
    clf[1].fit(training_data_full.values[:,[35, 39, 43]], training_data_full.c1.values)
    clf[2].fit(training_data_full.values[:,[56, 59, 61, 63, 65]], training_data_full.c2.values)
    clf[3].fit(training_data_full.values[:,[75, 79]], training_data_full.c3.values)
    clf[4].fit(training_data_full.values[:,[89, 92]], training_data_full.c4.values)
    clf[5].fit(training_data.iloc[:,95:126].values, training_data.c5.values)
    clf_umap = svm.SVC(C = 10, gamma = 1)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_synthetic_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic.pkl'.format(reference_folder, str(spc)))
    return

def calculate_ensemble_efficiency(omega):
    e = 0
    for j in range(200):
        e += ((-scipy.special.gamma(2/3)*omega/np.pi)**j)*scipy.special.gamma(j/3+1)/scipy.special.factorial(j)
    return(1-e)

def get_quenching(exc):
    if exc == 2:
        quenching = np.ones(10)
        quenching[3:] = np.exp(-2*(1/6)*np.arange(7)*np.random.random())
    else:
        quenching = np.ones(10)
    return(quenching)

def get_absorption_probability_direct_excitation(extinction_coefficient_direct_excitation, relevant_fluorophores, labeling_density, exc):
    extinction_coefficient_direct_excitation_total = np.sum(extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores*labeling_density)
    if extinction_coefficient_direct_excitation_total > 0:
        absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores*labeling_density/extinction_coefficient_direct_excitation_total
    else:
        absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores*labeling_density
    return(absorption_probability_direct_excitation)

def get_absorption_probability_fret(extinction_coefficient_fret, relevant_fluorophores, labeling_density, k):
    extinction_fret_total = np.sum(extinction_coefficient_fret[k,:]*relevant_fluorophores*labeling_density)
    if extinction_fret_total > 0:
        absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores*labeling_density/extinction_fret_total
    else:
        absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores*labeling_density
    return(absorption_probability_fret)

def get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield):
    relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
    coefficients = np.zeros((simulation_per_code,7))
    if np.sum(relevant_fluorophores) > 0:
        absorption_probability_direct_excitation = get_absorption_probability_direct_excitation(extinction_coefficient_direct_excitation, relevant_fluorophores, labeling_density, exc)
        for i in range(simulation_per_code):
            f_sensitized = np.zeros(7)
            fret_efficiency_out = np.zeros(7)
            exciton_saturation = np.random.random()
            absorption_probability_direct_excitation_adjusted = (1 - exciton_saturation)*absorption_probability_direct_excitation + exciton_saturation*(1/np.sum(relevant_fluorophores))*np.repeat(1, 7)*relevant_fluorophores
            quenching = get_quenching(exc)
            for j in range(7):
                omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:])
                fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
                f_fret_sensitized = 0
                for k in range(j):
                    omega_total = np.sum(fret_number_matrix[i, k, k+1:]*labeling_density[k+1:]*relevant_fluorophores[k+1:])
                    absorption_probability_fret = get_absorption_probability_fret(extinction_coefficient_fret, relevant_fluorophores, labeling_density, k)
                    absorption_probability_fret_adjusted = (1 - exciton_saturation)*absorption_probability_fret + exciton_saturation*(1/np.sum(relevant_fluorophores))*np.repeat(1, 7)*relevant_fluorophores
                    if omega_total > 0:
                        f_fret_sensitized += (fret_number_matrix[i, k, j]*labeling_density[j]*relevant_fluorophores[j]/omega_total)*absorption_probability_fret_adjusted[j]*relevant_fluorophores[k]*f_sensitized[k]*fret_efficiency_out[k]
                f_sensitized[j] = absorption_probability_direct_excitation_adjusted[j]*excitation_efficiency_matrix[j, exc] + f_fret_sensitized
                coefficients[i,j] = (fluorescence_quantum_yield[j] + (1-fluorescence_quantum_yield[j])*np.random.random())*f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]
    return(coefficients)

def get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield):
    relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
    coefficients = np.zeros((simulation_per_code,10))
    if np.sum(relevant_fluorophores) > 0:
        absorption_probability_direct_excitation = get_absorption_probability_direct_excitation(extinction_coefficient_direct_excitation, relevant_fluorophores, labeling_density, exc)
        for i in range(simulation_per_code):
            f_sensitized = np.zeros(10)
            fret_efficiency_out = np.zeros(10)
            exciton_saturation = np.random.random()
            absorption_probability_direct_excitation_adjusted = (1 - exciton_saturation)*absorption_probability_direct_excitation + exciton_saturation*(1/np.sum(relevant_fluorophores))*np.repeat(1, 10)*relevant_fluorophores
            quenching = get_quenching(exc)
            for j in range(10):
                omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:])
                fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
                f_fret_sensitized = 0
                for k in range(j):
                    omega_total = np.sum(fret_number_matrix[i, k, k+1:]*labeling_density[k+1:]*relevant_fluorophores[k+1:])
                    absorption_probability_fret = get_absorption_probability_fret(extinction_coefficient_fret, relevant_fluorophores, labeling_density, k)
                    absorption_probability_fret_adjusted = (1 - exciton_saturation)*absorption_probability_fret + exciton_saturation*(1/np.sum(relevant_fluorophores))*np.repeat(1, 10)*relevant_fluorophores
                    if omega_total > 0:
                        f_fret_sensitized += (fret_number_matrix[i, k, j]*labeling_density[j]*relevant_fluorophores[j]/omega_total)*absorption_probability_fret_adjusted[j]*relevant_fluorophores[k]*f_sensitized[k]*fret_efficiency_out[k]
                f_sensitized[j] = absorption_probability_direct_excitation_adjusted[j]*excitation_efficiency_matrix[j, exc] + f_fret_sensitized
                coefficients[i,j] = (fluorescence_quantum_yield[j] + (1-fluorescence_quantum_yield[j])*np.random.random())*f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]
                # if j == 0:
                #     coefficients[i,j] = (0.3 + 0.7*np.random.random())*(fluorescence_quantum_yield[j] + (1-fluorescence_quantum_yield[j])*np.random.random())*f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]*quenching[j]
                # else:
                #     coefficients[i,j] = (fluorescence_quantum_yield[j] + (1-fluorescence_quantum_yield[j])*np.random.random())*f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]*quenching[j]
    return(coefficients)

def get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels):
    relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc,:]
    coefficients = np.zeros((simulation_per_code,10))
    if np.sum(relevant_fluorophores) > 0:
        sim_emission_strength = excitation_efficiency_matrix[:,exc][None,:]*(plm[None,:] + (1 - plm)[None,:]*np.random.random((simulation_per_code, 10)))
        coefficients = relevant_fluorophores*sim_emission_strength
    return(coefficients)

@dask.delayed
def simulate_spectra_direct(filename, bkg_spectra_filename, bkg_intensity, simulation_per_code, nbit):
    enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', filename).group(0)))
    spectra = pd.read_csv(filename, header = None)
    bkg_spec = pd.read_csv(bkg_spectra_filename, header = None)
    spectra_filtered = np.zeros((spectra.shape[0], 95))
    spectra_filtered[:,0:95] = spectra.iloc[:,0:95].copy().values
    spectra_filtered[:,0:32] = spectra_filtered[:,0:32] - (bkg_intensity*bkg_spec.iloc[0:32,0])[None,:]
    spectra_mean = np.average(spectra_filtered, axis = 0)
    spectra_cov = np.cov(spectra_filtered.transpose())
    simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    indices = [0,32,55,75,89,95]
    for k in range(0,5):
        simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
    simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
    # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
    # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
    ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
    code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
    numeric_code_list = [int(a) for a in list(code)]
    ss_norm.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
    ss_norm.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
    ss_norm.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
    ss_norm['code'] = code
    # Low 488
    if (numeric_code_list[9] == 0) & (numeric_code_list[0] | numeric_code_list[2] | numeric_code_list[7] | numeric_code_list[8]):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_488 = 0.02
        simulated_spectra_norm[:,indices[1]:indices[2]] = (error_scale_488 + (0.4 - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[1]:indices[2]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488.loc[:,'c1'] = 0
        ss_norm_low488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488['code'] = '{}'.format(code)
    else:
        ss_norm_low488 = pd.DataFrame(columns = ss_norm.columns)
    # Low 514
    if (numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[9]) & (numeric_code_list[0] == 0) & (numeric_code_list[2] == 0):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_514 = 0.02
        simulated_spectra_norm[:,indices[2]:indices[3]] = (error_scale_514 + (0.4 - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[2]:indices[3]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c2'] = 0
        ss_norm_low514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low514['code'] = '{}'.format(code)
    else:
        ss_norm_low514 = pd.DataFrame(columns = ss_norm.columns)
    # Low 561
    if (numeric_code_list[2] or numeric_code_list[0]) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_561 = 0.02
        simulated_spectra_norm[:,indices[3]:indices[4]] = (error_scale_561 + (0.4 - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[3]:indices[4]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low561.loc[:,'c3'] = 0
        ss_norm_low561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low561['code'] = '{}'.format(code)
    else:
        ss_norm_low561 = pd.DataFrame(columns = ss_norm.columns)
    # Low 633
    if (numeric_code_list[2] == 1) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0) & (numeric_code_list[9] == 0):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_488 = 0.02
        error_scale_561 = 0.02
        simulated_spectra_norm[:,indices[1]:indices[2]] = (error_scale_488 + (0.4 - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[1]:indices[2]]
        simulated_spectra_norm[:,indices[3]:indices[4]] = (error_scale_561 + (0.4 - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[3]:indices[4]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488561['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488561['c1'] = 0
        ss_norm_low488561['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488561['c3'] = 0
        ss_norm_low488561['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488561['code'] = '{}'.format(code)
    else:
        ss_norm_low488561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 633
    if (numeric_code_list[3] or numeric_code_list[4]):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[4]:indices[5]] = simulated_spectra_norm[:,indices[4]:indices[5]]/np.max(simulated_spectra_norm[:,indices[4]:indices[5]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,indices[4]:indices[5]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.7*spec_temp[spec_temp > 0]
        spec_sat = spec_temp*(1-saturation_weight) + saturation_weight*saturation_ceiling
        simulated_spectra_norm[:,indices[4]:indices[5]] = spec_sat
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat633['code'] = '{}'.format(code)
    else:
        ss_norm_sat633 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 405
    if numeric_code_list[1] + numeric_code_list[5] + numeric_code_list[6] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(32):
            delta = 0.4*(p/31)
            damping[:,p] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[0]:indices[1]] = simulated_spectra_norm[:,indices[0]:indices[1]]/np.max(simulated_spectra_norm[:,indices[0]:indices[1]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret405 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret405.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret405.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret405.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret405.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret405.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret405['code'] = '{}'.format(code)
    else:
        ss_norm_fret405 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 488-514-561
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(23):
            delta = 0.5*(p/22)
            damping[:,p + 32] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[1]:indices[2]] = simulated_spectra_norm[:,indices[1]:indices[2]]/np.max(simulated_spectra_norm[:,indices[1]:indices[2]], axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(20):
            delta = 0.5*(p/19)
            damping[:,p + 55] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[2]:indices[3]] = simulated_spectra_norm[:,indices[2]:indices[3]]/np.max(simulated_spectra_norm[:,indices[2]:indices[3]], axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(14):
            delta = 0.5*(p/13)
            damping[:,p + 75] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[3]:indices[4]] = simulated_spectra_norm[:,indices[3]:indices[4]]/np.max(simulated_spectra_norm[:,indices[3]:indices[4]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret488514561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret488514561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret488514561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret488514561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret488514561.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret488514561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret488514561['code'] = '{}'.format(code)
    else:
        ss_norm_fret488514561 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 633
    if numeric_code_list[3] + numeric_code_list[4] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(6):
            delta = 0.4*(p/5)
            damping[:,p] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[4]:indices[5]] = simulated_spectra_norm[:,indices[4]:indices[5]]/np.max(simulated_spectra_norm[:,indices[4]:indices[5]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret633['code'] = '{}'.format(code)
    else:
        ss_norm_fret633 = pd.DataFrame(columns = ss_norm.columns)
    ss_all = pd.concat([ss_norm, ss_norm_low488, ss_norm_low514, ss_norm_low561, ss_norm_low488561, ss_norm_sat633,
                        ss_norm_fret405, ss_norm_fret488514561, ss_norm_fret633])
    return(ss_all)

@dask.delayed
def simulate_spectra_direct_2(filename, bkg_spectra_filename, bkg_intensity, simulation_per_code, nbit):
    enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', filename).group(0)))
    spectra = pd.read_csv(filename, header = None)
    bkg_spec = pd.read_csv(bkg_spectra_filename, header = None)
    spectra_filtered = np.zeros((spectra.shape[0], 95))
    spectra_filtered[:,0:95] = spectra.iloc[:,0:95].copy().values
    spectra_filtered[:,0:32] = spectra_filtered[:,0:32] - (bkg_intensity*bkg_spec.iloc[0:32,0])[None,:]
    spectra_mean = np.average(spectra_filtered, axis = 0)
    spectra_cov = np.cov(spectra_filtered.transpose())
    simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    indices = [0,32,55,75,89,95]
    for k in range(0,5):
        simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
    simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
    # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
    # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
    ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
    code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
    numeric_code_list = [int(a) for a in list(code)]
    ss_norm.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
    ss_norm.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
    ss_norm.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
    ss_norm['code'] = code
    # Low 488
    if (numeric_code_list[9] == 0) & (numeric_code_list[0] | numeric_code_list[2] | numeric_code_list[7] | numeric_code_list[8]):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_488 = 0.02
        simulated_spectra_norm[:,indices[1]:indices[2]] = (error_scale_488 + (0.4 - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[1]:indices[2]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488.loc[:,'c1'] = 0
        ss_norm_low488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488['code'] = '{}'.format(code)
    else:
        ss_norm_low488 = pd.DataFrame(columns = ss_norm.columns)
    # Low 514
    if (numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[9]) & (numeric_code_list[0] == 0) & (numeric_code_list[2] == 0):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_514 = 0.02
        simulated_spectra_norm[:,indices[2]:indices[3]] = (error_scale_514 + (0.4 - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[2]:indices[3]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c2'] = 0
        ss_norm_low514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low514['code'] = '{}'.format(code)
    else:
        ss_norm_low514 = pd.DataFrame(columns = ss_norm.columns)
    # Low 561
    if (numeric_code_list[2] or numeric_code_list[0]) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_561 = 0.02
        simulated_spectra_norm[:,indices[3]:indices[4]] = (error_scale_561 + (0.4 - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[3]:indices[4]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low561.loc[:,'c3'] = 0
        ss_norm_low561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low561['code'] = '{}'.format(code)
    else:
        ss_norm_low561 = pd.DataFrame(columns = ss_norm.columns)
    # Low 633
    if (numeric_code_list[2] == 1) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0) & (numeric_code_list[9] == 0):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        # violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        # ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        error_scale_488 = 0.02
        error_scale_561 = 0.02
        simulated_spectra_norm[:,indices[1]:indices[2]] = (error_scale_488 + (0.4 - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[1]:indices[2]]
        simulated_spectra_norm[:,indices[3]:indices[4]] = (error_scale_561 + (0.4 - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[3]:indices[4]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488561['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488561['c1'] = 0
        ss_norm_low488561['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488561['c3'] = 0
        ss_norm_low488561['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488561['code'] = '{}'.format(code)
    else:
        ss_norm_low488561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 633
    if (numeric_code_list[3] or numeric_code_list[4]):
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[4]:indices[5]] = simulated_spectra_norm[:,indices[4]:indices[5]]/np.max(simulated_spectra_norm[:,indices[4]:indices[5]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,indices[4]:indices[5]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.7*spec_temp[spec_temp > 0]
        spec_sat = spec_temp*(1-saturation_weight) + saturation_weight*saturation_ceiling
        simulated_spectra_norm[:,indices[4]:indices[5]] = spec_sat
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat633['code'] = '{}'.format(code)
    else:
        ss_norm_sat633 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 405 - red shift
    if numeric_code_list[1] + numeric_code_list[5] + numeric_code_list[6] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(32):
            delta = 0.4*(p/31)
            damping[:,p] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[0]:indices[1]] = simulated_spectra_norm[:,indices[0]:indices[1]]/np.max(simulated_spectra_norm[:,indices[0]:indices[1]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_red_405 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_red_405.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_red_405.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_405.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_405.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_red_405.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_red_405['code'] = '{}'.format(code)
    else:
        ss_norm_fret_red_405 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 405 - blue shift
    if numeric_code_list[1] + numeric_code_list[5] + numeric_code_list[6] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(32):
            delta = 0.4*((32 - p)/31)
            damping[:,p] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[0]:indices[1]] = simulated_spectra_norm[:,indices[0]:indices[1]]/np.max(simulated_spectra_norm[:,indices[0]:indices[1]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_blue_405 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_blue_405.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_blue_405.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_405.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_405.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_blue_405.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_blue_405['code'] = '{}'.format(code)
    else:
        ss_norm_fret_blue_405 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 488 - red shift
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(23):
            delta = 0.5*(p/22)
            damping[:,p + 32] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[1]:indices[2]] = simulated_spectra_norm[:,indices[1]:indices[2]]/np.max(simulated_spectra_norm[:,indices[1]:indices[2]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_red_488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_red_488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_red_488.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_red_488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_red_488['code'] = '{}'.format(code)
    else:
        ss_norm_fret_red_488 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 488 - blue shift
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(23):
            delta = 0.5*((23 - p)/22)
            damping[:,p + 32] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[1]:indices[2]] = simulated_spectra_norm[:,indices[1]:indices[2]]/np.max(simulated_spectra_norm[:,indices[1]:indices[2]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_blue_488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_blue_488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_blue_488.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_blue_488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_blue_488['code'] = '{}'.format(code)
    else:
        ss_norm_fret_blue_488 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 514 - red shift
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(20):
            delta = 0.5*(p/19)
            damping[:,p + 55] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[2]:indices[3]] = simulated_spectra_norm[:,indices[2]:indices[3]]/np.max(simulated_spectra_norm[:,indices[2]:indices[3]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_red_514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_red_514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_red_514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_514.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_red_514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_red_514['code'] = '{}'.format(code)
    else:
        ss_norm_fret_red_514 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 514 - blue shift
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(20):
            delta = 0.5*((20 - p)/19)
            damping[:,p + 55] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[2]:indices[3]] = simulated_spectra_norm[:,indices[2]:indices[3]]/np.max(simulated_spectra_norm[:,indices[2]:indices[3]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_blue_514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_blue_514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_blue_514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_514.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_blue_514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_blue_514['code'] = '{}'.format(code)
    else:
        ss_norm_fret_blue_514 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 561 - red shift
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(14):
            delta = 0.5*(p/13)
            damping[:,p + 75] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[3]:indices[4]] = simulated_spectra_norm[:,indices[3]:indices[4]]/np.max(simulated_spectra_norm[:,indices[3]:indices[4]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_red_561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_red_561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_red_561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_561.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_red_561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_red_561['code'] = '{}'.format(code)
    else:
        ss_norm_fret_red_561 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 561 - blue shift
    if numeric_code_list[9] + numeric_code_list[2] + numeric_code_list[0] + numeric_code_list[7] + numeric_code_list[8] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(14):
            delta = 0.5*((14 - p)/13)
            damping[:,p + 75] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[3]:indices[4]] = simulated_spectra_norm[:,indices[3]:indices[4]]/np.max(simulated_spectra_norm[:,indices[3]:indices[4]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_blue_561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_blue_561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_blue_561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_561.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_blue_561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_blue_561['code'] = '{}'.format(code)
    else:
        ss_norm_fret_blue_561 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 633 - red shift
    if numeric_code_list[3] + numeric_code_list[4] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(6):
            delta = 0.4*(p/5)
            damping[:,p] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[4]:indices[5]] = simulated_spectra_norm[:,indices[4]:indices[5]]/np.max(simulated_spectra_norm[:,indices[4]:indices[5]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_red_633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_red_633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_red_633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_red_633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_red_633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_red_633['code'] = '{}'.format(code)
    else:
        ss_norm_fret_red_633 = pd.DataFrame(columns = ss_norm.columns)
    # effective FRET 633 - blue shift
    if numeric_code_list[3] + numeric_code_list[4] > 1:
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        for p in range(6):
            delta = 0.4*((6 - p)/5)
            damping[:,p] = (1 - delta)  + delta*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,indices[4]:indices[5]] = simulated_spectra_norm[:,indices[4]:indices[5]]/np.max(simulated_spectra_norm[:,indices[4]:indices[5]], axis = 1)[:,None]
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_fret_blue_633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_fret_blue_633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_fret_blue_633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_fret_blue_633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_fret_blue_633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_fret_blue_633['code'] = '{}'.format(code)
    else:
        ss_norm_fret_blue_633 = pd.DataFrame(columns = ss_norm.columns)
    ss_all = pd.concat([ss_norm, ss_norm_low488, ss_norm_low514, ss_norm_low561, ss_norm_low488561, ss_norm_sat633,
                        ss_norm_fret_red_405, ss_norm_fret_blue_405,
                        ss_norm_fret_red_488, ss_norm_fret_blue_488,
                        ss_norm_fret_red_514, ss_norm_fret_blue_514,
                        ss_norm_fret_red_561, ss_norm_fret_blue_561,
                        ss_norm_fret_red_633, ss_norm_fret_blue_633])
    return(ss_all)

@dask.delayed
def simulate_spectra_7b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit):
    cov_scale = 1
    s_floor= 0.8
    s_ceiling = 1.2
    # Nominal
    simulated_spectra = np.zeros((simulation_per_code, nchannels))
    fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
    spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
    spec_synthetic_7b = spec_synthetic[3:]
    spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
    for exc in range(4):
        coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
        spec_list_norm = []
        for spec in spec_avg:
            if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
            else:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
        for s in range(simulation_per_code):
            cov_matrix = np.zeros((63,63))
            for i in range(7):
                for j in range(7):
                    if i == j:
                        # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                    else:
                        cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                        # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                        cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
            spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
            spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
            spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
            simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    scaling_matrix = np.zeros((simulated_spectra_norm.shape))
    scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
    scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
    scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
    scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
    simulated_spectra_adjusted_norm = simulated_spectra_norm
    ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
    ss_norm.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
    ss_norm.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
    ss_norm.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
    ss_norm.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
    ss_norm['code'] = code
    # Low 488
    if (numeric_code_list[6] == 0) & (numeric_code_list[0] | numeric_code_list[1] | numeric_code_list[4] | numeric_code_list[5]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488.loc[:,'c0'] = 0
        ss_norm_low488.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_low488.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_low488.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_low488['code'] = '{}'.format(code)
    else:
        ss_norm_low488 = pd.DataFrame(columns = ss_norm.columns)
    # Low 514
    if (numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[6]) & (numeric_code_list[0] == 0) & (numeric_code_list[1] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_514 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_514 + (error_scale[2] - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low514.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c1'] = 0
        ss_norm_low514.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_low514['code'] = '{}'.format(code)
    else:
        ss_norm_low514 = pd.DataFrame(columns = ss_norm.columns)
    # Low 561
    if (numeric_code_list[1] or numeric_code_list[0]) & (numeric_code_list[4] == 0) & (numeric_code_list[5] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low561.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_low561.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_low561.loc[:,'c2'] = 0
        ss_norm_low561.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_low561['code'] = '{}'.format(code)
    else:
        ss_norm_low561 = pd.DataFrame(columns = ss_norm.columns)
    # Low 488 561
    if (numeric_code_list[1] == 1) & (numeric_code_list[4] == 0) & (numeric_code_list[5] == 0) & (numeric_code_list[6] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488561['c0'] = 0
        ss_norm_low488561['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_low488561['c2'] = 0
        ss_norm_low488561['c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_low488561['code'] = '{}'.format(code)
    else:
        ss_norm_low488561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 488
    if (numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]/np.max(simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.7*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat488.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat488.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat488.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_sat488.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_sat488['code'] = '{}'.format(code)
    else:
        ss_norm_sat488 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 514
    if (numeric_code_list[1] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]/np.max(simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.7*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat514.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat514.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat514.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_sat514.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_sat514['code'] = '{}'.format(code)
    else:
        ss_norm_sat514 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 561
    if (numeric_code_list[4] or numeric_code_list[5]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]/np.max(simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.7*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat561.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat561.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat561.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_sat561.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_sat561['code'] = '{}'.format(code)
    else:
        ss_norm_sat561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 633
    if (numeric_code_list[2] or numeric_code_list[3]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]/np.max(simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.7*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat633.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat633.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_sat633.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_sat633.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_sat633['code'] = '{}'.format(code)
    else:
        ss_norm_sat633 = pd.DataFrame(columns = ss_norm.columns)
    # Damping 633
    if (numeric_code_list[2] == 1):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 7, 7))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_7b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_synthetic_7b = spec_synthetic[3:]
        spec_avg = [df.Intensity.values[32:] for df in spec_synthetic_7b]
        for exc in range(4):
            coefficients = get_channel_coefficients_7b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((63,63))
                for i in range(7):
                    for j in range(7):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                            # cov_matrix += np.cov(coefficients[s,i]*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            # cov_ij = np.cov(coefficients[s,i]*spec_matrices[i], coefficients[s,j]*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        damping = np.ones((simulated_spectra.shape[0],63))
        random = np.random.random(simulated_spectra.shape[0])
        damping[:,61] = 0.8 + 0.2*random
        damping[:,62] = 0.6 + 0.4*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:23] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,23:43] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,43:57] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,57:63] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm*scaling_matrix
        ss_norm_damp633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_damp633.loc[:,'c0'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_damp633.loc[:,'c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm_damp633.loc[:,'c2'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm_damp633.loc[:,'c3'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm_damp633['code'] = '{}'.format(code)
    else:
        ss_norm_damp633 = pd.DataFrame(columns = ss_norm.columns)
    ss_all = pd.concat([ss_norm, ss_norm_low488, ss_norm_low514, ss_norm_low561, ss_norm_low488561, ss_norm_sat488, ss_norm_sat514, ss_norm_sat561, ss_norm_sat633, ss_norm_damp633], axis = 0)
    return(ss_all)

@dask.delayed
def simulate_spectra_10b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit):
    cov_scale = 1
    s_floor= 0.8
    s_ceiling = 1.2
    # Nominal
    simulated_spectra = np.zeros((simulation_per_code, nchannels))
    fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
    spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
    spec_avg = [df.Intensity.values for df in spec_synthetic]
    for exc in range(5):
        spec_list_norm = []
        for spec in spec_avg:
            if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
            else:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
        coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
        for s in range(simulation_per_code):
            cov_matrix = np.zeros((95,95))
            for i in range(10):
                for j in range(10):
                    if i == j:
                        cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                    else:
                        cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                        cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
            spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
            spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
            spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
            simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    scaling_matrix = np.zeros((simulated_spectra_norm.shape))
    scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
    scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
    scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
    scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
    scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
    simulated_spectra_adjusted_norm = simulated_spectra_norm
    ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
    ss_norm.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
    ss_norm.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
    ss_norm.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
    ss_norm['code'] = code
    # Low 488
    if (numeric_code_list[9] == 0) & (numeric_code_list[0] | numeric_code_list[2] | numeric_code_list[7] | numeric_code_list[8]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488.loc[:,'c1'] = 0
        ss_norm_low488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488['code'] = '{}'.format(code)
    else:
        ss_norm_low488 = pd.DataFrame(columns = ss_norm.columns)
    # Low 514
    if (numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[9]) & (numeric_code_list[0] == 0) & (numeric_code_list[2] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_514 = 0.02
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = (error_scale_514 + (error_scale[2] - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c2'] = 0
        ss_norm_low514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low514['code'] = '{}'.format(code)
    else:
        ss_norm_low514 = pd.DataFrame(columns = ss_norm.columns)
    # Low 561
    if (numeric_code_list[2] or numeric_code_list[0]) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low561.loc[:,'c3'] = 0
        ss_norm_low561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low561['code'] = '{}'.format(code)
    else:
        ss_norm_low561 = pd.DataFrame(columns = ss_norm.columns)
    # Low 488 561
    if (numeric_code_list[2] == 1) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0) & (numeric_code_list[9] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488561['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488561['c1'] = 0
        ss_norm_low488561['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488561['c3'] = 0
        ss_norm_low488561['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488561['code'] = '{}'.format(code)
    else:
        ss_norm_low488561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 405
    if (numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]/np.max(simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat405 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat405.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat405.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat405.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat405.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat405.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat405['code'] = '{}'.format(code)
    else:
        ss_norm_sat405 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 488
    if (numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]/np.max(simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat488.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat488['code'] = '{}'.format(code)
    else:
        ss_norm_sat488 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 514
    if (numeric_code_list[2] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]/np.max(simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat514.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat514['code'] = '{}'.format(code)
    else:
        ss_norm_sat514 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 561
    if (numeric_code_list[7] or numeric_code_list[8]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]/np.max(simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat561.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat561['code'] = '{}'.format(code)
    else:
        ss_norm_sat561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 633
    if (numeric_code_list[3] or numeric_code_list[4]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]] = simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]]/np.max(simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat633['code'] = '{}'.format(code)
    else:
        ss_norm_sat633 = pd.DataFrame(columns = ss_norm.columns)
    # Damping 633
    if (numeric_code_list[3] == 1):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        damping[:,93] = 0.8 + 0.2*random
        damping[:,94] = 0.6 + 0.4*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm*scaling_matrix
        ss_norm_damp633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_damp633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_damp633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_damp633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_damp633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_damp633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_damp633['code'] = '{}'.format(code)
    else:
        ss_norm_damp633 = pd.DataFrame(columns = ss_norm.columns)
    ss_all = pd.concat([ss_norm, ss_norm_low488, ss_norm_low514, ss_norm_low561, ss_norm_low488561, ss_norm_sat405, ss_norm_sat488, ss_norm_sat514, ss_norm_sat561, ss_norm_sat633, ss_norm_damp633], axis = 0)
    return(ss_all)

@dask.delayed
def simulate_spectra_10b_phenomenology(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, shift, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, nbit):
    cov_scale = 1
    s_floor= 0.8
    s_ceiling = 1.2
    # Nominal
    simulated_spectra = np.zeros((simulation_per_code, nchannels))
    spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
    spec_avg = [df.Intensity.values for df in spec_synthetic]
    for exc in range(5):
        coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
        spec_list_norm = []
        for spec in spec_avg:
            if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
            else:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
        for s in range(simulation_per_code):
            cov_matrix = np.zeros((95,95))
            for i in range(10):
                for j in range(10):
                    if i == j:
                        cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                    else:
                        cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                        cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
            spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
            spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
            spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
            simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    scaling_matrix = np.zeros((simulated_spectra_norm.shape))
    scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
    scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
    scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
    scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
    scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
    simulated_spectra_adjusted_norm = simulated_spectra_norm
    ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
    ss_norm.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
    ss_norm.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
    ss_norm.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
    ss_norm['code'] = code
    # Low 488
    if (numeric_code_list[9] == 0) & (numeric_code_list[0] | numeric_code_list[2] | numeric_code_list[7] | numeric_code_list[8]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488.loc[:,'c1'] = 0
        ss_norm_low488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488['code'] = '{}'.format(code)
    else:
        ss_norm_low488 = pd.DataFrame(columns = ss_norm.columns)
    # Low 514
    if (numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[9]) & (numeric_code_list[0] == 0) & (numeric_code_list[2] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_514 = 0.02
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = (error_scale_514 + (error_scale[2] - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c2'] = 0
        ss_norm_low514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low514['code'] = '{}'.format(code)
    else:
        ss_norm_low514 = pd.DataFrame(columns = ss_norm.columns)
    # Low 561
    if (numeric_code_list[2] or numeric_code_list[0]) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low561.loc[:,'c3'] = 0
        ss_norm_low561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low561['code'] = '{}'.format(code)
    else:
        ss_norm_low561 = pd.DataFrame(columns = ss_norm.columns)
    # Low 488 561
    if (numeric_code_list[2] == 1) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0) & (numeric_code_list[9] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_low488561['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488561['c1'] = 0
        ss_norm_low488561['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488561['c3'] = 0
        ss_norm_low488561['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488561['code'] = '{}'.format(code)
    else:
        ss_norm_low488561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 405
    if (numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]/np.max(simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat405 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat405.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat405.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat405.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat405.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat405.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat405['code'] = '{}'.format(code)
    else:
        ss_norm_sat405 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 488
    if (numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]/np.max(simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat488 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat488.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat488['code'] = '{}'.format(code)
    else:
        ss_norm_sat488 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 514
    if (numeric_code_list[2] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]/np.max(simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat514 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat514.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat514['code'] = '{}'.format(code)
    else:
        ss_norm_sat514 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 561
    if (numeric_code_list[7] or numeric_code_list[8]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]/np.max(simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat561 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat561.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat561['code'] = '{}'.format(code)
    else:
        ss_norm_sat561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 633
    if (numeric_code_list[3] or numeric_code_list[4]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]] = simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]]/np.max(simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_sat633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat633['code'] = '{}'.format(code)
    else:
        ss_norm_sat633 = pd.DataFrame(columns = ss_norm.columns)
    # Damping 633
    if (numeric_code_list[3] == 1):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        for exc in range(5):
            coefficients = get_channel_coefficients_10b_phenomenology(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, plm, exc, channel_indices, nchannels)
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_scale*cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        damping[:,93] = 0.8 + 0.2*random
        damping[:,94] = 0.6 + 0.4*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm*scaling_matrix
        ss_norm_damp633 = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm_damp633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_damp633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_damp633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_damp633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_damp633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_damp633['code'] = '{}'.format(code)
    else:
        ss_norm_damp633 = pd.DataFrame(columns = ss_norm.columns)
    ss_all = pd.concat([ss_norm, ss_norm_low488, ss_norm_low514, ss_norm_low561, ss_norm_low488561, ss_norm_sat405, ss_norm_sat488, ss_norm_sat514, ss_norm_sat561, ss_norm_sat633, ss_norm_damp633], axis = 0)
    return(ss_all)

def load_training_data_simulate_excitation_adjusted_normalized_violet_derivative_umap_transformed_parallel_bkg_unfiltered(reference_folder, learning_mode, spc):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    molar_extinction_coefficient = [35000, 46000, 35000, 73000, 81000, 50000, 112000, 120000, 0.1*144000, 0.12*270000]
    # Extinction coefficient estimated: Pacific Blue
    fluorescence_quantum_yield = [0.4, 0.9, 0.7, 0.92, 0.61, 0.5, 0.79, 0.5, 0.7, 0.33]
    # Estimated QY: 405 dyes, DyLight 510, and RRX
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra_bkg_unfiltered.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    dlm = 4
    dum = 25
    labeling_density = [1, 1, 1, 1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    qec = get_qec(fret_folder)
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_10b(fret_folder, molar_extinction_coefficient)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_10b(fret_folder, molar_extinction_coefficient)
    forster_distance = calculate_forster_distance_10b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield)
    fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
    distance_list = np.zeros(simulation_per_code)
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1)
    client = Client(cluster)
    ss_norm_list = []
    ss_norm_neg_list = []
    ss_norm_low488_list = []
    ss_norm_low514_list = []
    ss_norm_low561_list = []
    ss_norm_low488561_list = []
    for enc in range(1,1024):
        print(enc)
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        results = simulate_spectra(numeric_code_list, fluorophore_list, excitation_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, spec_avg, spec_cov, channel_indices, nchannels, code, qec)
        ss_norm_list.append(results[0])
        ss_norm_neg_list.append(results[1])
        ss_norm_low488_list.append(results[2])
        ss_norm_low514_list.append(results[3])
        ss_norm_low561_list.append(results[4])
        ss_norm_low488561_list.append(results[5])

    training_data = dask.delayed(pd.concat)(ss_norm_list, axis = 0)
    training_data_negative = dask.delayed(pd.concat)(ss_norm_neg_list, axis = 0)
    training_data_low488 = dask.delayed(pd.concat)(ss_norm_low488_list, axis = 0)
    training_data_low514 = dask.delayed(pd.concat)(ss_norm_low514_list, axis = 0)
    training_data_low561 = dask.delayed(pd.concat)(ss_norm_low561_list, axis = 0)
    training_data_488561 = dask.delayed(pd.concat)(ss_norm_low488561_list, axis = 0)
    training_data_compute = training_data.compute()
    training_data_negative_compute = training_data_negative.compute()
    training_data_low488_compute = training_data_low488.compute()
    training_data_low514_compute = training_data_low514.compute()
    training_data_low561_compute = training_data_low561.compute()
    training_data_low488561_compute = training_data_488561.compute()
    client.close()
    cluster.close()
    training_data_full = pd.concat([training_data_compute, training_data_negative_compute, training_data_low488_compute, training_data_low514_compute, training_data_low561_compute, training_data_low488561_compute])
    training_data_positive = pd.concat([training_data_compute, training_data_low488_compute, training_data_low514_compute, training_data_low561_compute, training_data_low488561_compute])
    umap_transform = umap.UMAP(n_neighbors = 20, min_dist = 0.001, metric = channel_cosine_intensity_violet_derivative_v2).fit(training_data_positive.iloc[:,0:132].values, y = training_data_positive.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(6)]
    clf[0].fit(training_data_full.values[:,[2, 5, 11]], training_data_full.c0.values.astype(int))
    clf[1].fit(training_data_full.values[:,[35, 39, 43]], training_data_full.c1.values.astype(int))
    clf[2].fit(training_data_full.values[:,[56, 59, 61, 63, 65]], training_data_full.c2.values.astype(int))
    clf[3].fit(training_data_full.values[:,[75, 79]], training_data_full.c3.values.astype(int))
    clf[4].fit(training_data_full.values[:,[89, 92]], training_data_full.c4.values.astype(int))
    clf[5].fit(training_data_full.iloc[:,95:126].values, training_data_full.c5.values.astype(int))
    clf_umap = svm.SVC(C = 10, gamma = 1)
    clf_umap.fit(umap_transform.embedding_, training_data_positive.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_synthetic_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_direct_simulation(reference_folder, image_tab, spc):
    image_tab = pd.read_csv(image_tab_filename)
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    for r in range(30,40):
        ss_list = []
        clf_umap_filename = '{}/reference_simulate_{}_direct_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r)
        if not os.path.exists(clf_umap_filename):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Current Time is {}, starting simulating replicate {}...'.format(current_time, r))
            for i in range(0, image_tab.shape[0]):
                image_name = image_tab.loc[i,'IMAGES']
                image_adjacent_name = image_tab.loc[i,'SampleAdjacent']
                bkg_intensity = image_tab.loc[i,'BkgFactorCorrected']
                spectra_filename = '{}/{}_avgint.csv'.format(reference_folder, image_name)
                bkg_spectra_filename = '{}/{}_bkg.csv'.format(reference_folder, image_adjacent_name)
                results = simulate_spectra_direct(spectra_filename, bkg_spectra_filename, bkg_intensity, simulation_per_code, nbit)
                ss_list.append(results)
            training_data = dask.delayed(pd.concat)(ss_list, axis = 0)
            training_data_compute = training_data.compute()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Current Time is {}, starting embedding replicate {}...'.format(current_time, r))
            umap_transform = umap.UMAP(n_neighbors = 30, min_dist = 0, n_components = 3, transform_queue_size = 16,
                                       metric = channel_cosine_intensity_v2, low_memory = False)
            umap_transform.fit(training_data_compute.iloc[:,0:100].values, y = training_data_compute.code.values)
            clf_umap = svm.SVC(C = 10, gamma = 1)
            clf_umap.fit(umap_transform.embedding_, training_data_compute.code.values)
            joblib.dump(clf_umap, '{}/reference_simulate_{}_direct_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            joblib.dump(umap_transform, '{}/reference_simulate_{}_direct_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time is {}, finished training replicate {}.".format(current_time, r))
        else:
            pass
    client.close()
    cluster.close()
    # clf = [svm.SVC(C = 10, gamma = 0.5) for i in range(6)]
    # clf[0].fit(training_data.iloc[:,0:32].values, training_data.c1.values)
    # clf[1].fit(training_data.iloc[:,32:55].values, training_data.c2.values)
    # clf[2].fit(training_data.iloc[:,55:75].values, training_data.c3.values)
    # clf[3].fit(training_data.iloc[:,75:89].values, training_data.c4.values)
    # clf[4].fit(training_data.iloc[:,89:95].values, training_data.c5.values)
    # clf[5].fit(training_data.iloc[:,95:126].values, training_data.c6.values)
    # joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_violet_derivative_umap_transformed_check_svc.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_phenomenology_simulation_parallel_10b(reference_folder, fret_folder, spc, r):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenamesd = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    synthetic_spectra_shifted_filenames = ['{}/R{}_synthetic_spectra_shifted.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    # spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    # spec_shift = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_shifted_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    # use 100 spectra for each barcode as some barcodes have less cells than others
    bkg_filtered_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder, b) for b in barcode_list]
    spec_matrices = [pd.read_csv(f, header = None).transpose().iloc[:,0:100] for f in bkg_filtered_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    error_scale = [0.15, 0.15, 0.15, 0.15, 0.15]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    plm = np.array([0.3, 0.3, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.3])
    shift_switch = 0
    shift = shift_switch*np.array([0,1,1,0,1,1,0,1,1,0])
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    for r in range(8,10):
        # if r < 50:
        #     shift_switch = 0
        #     shift = shift_switch*np.array([0,1,1,0,1,1,0,1,1,0])
        # else:
        #     shift_switch = 1
        #     shift = shift_switch*np.array([0,1,1,0,1,1,0,1,1,0])
        clf_umap_filename = '{}/reference_simulate_{}_phenomenology_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r)
        if not os.path.exists(clf_umap_filename):
            ss_all_list = []
            for enc in range(1,1024):
                code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
                numeric_code_list = np.array([int(a) for a in list(code)])
                results = simulate_spectra_10b_phenomenology(reference_folder, fret_folder, numeric_code_list,
                            fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code,
                            plm, shift, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, nbit)
                ss_all_list.append(results)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print('Current Time is {}, starting simulation...'.format(current_time))
            print('Current Time is {}, starting simulating replicate {}...'.format(current_time, r))
            training_data = dask.delayed(pd.concat)(ss_all_list, axis = 0)
            training_data_compute = training_data.compute()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Current Time is {}, starting embedding replicate {}...'.format(current_time, r))
            umap_transform = umap.UMAP(n_neighbors = 30, min_dist = 0, spread = 1, n_components = 3,
                                       set_op_mix_ratio = 1.0, repulsion_strength = 1.0, negative_sample_rate = 5,
                                       transform_queue_size = 16,
                                       metric = channel_cosine_intensity_v2)
            umap_transform.fit(training_data_compute.iloc[:,0:100].values, y = training_data_compute.code.values)
            clf_umap = svm.SVC(C = 10, gamma = 1)
            clf_umap.fit(umap_transform.embedding_, training_data_compute.code.values)
            # joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc.pkl'.format(reference_folder, str(spc)))
            # joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic.pkl'.format(reference_folder, str(spc)))
            joblib.dump(clf_umap, '{}/reference_simulate_{}_phenomenology_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            joblib.dump(umap_transform, '{}/reference_simulate_{}_phenomenology_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print("Current Time is {}, finished training.".format(current_time))
            print("Current Time is {}, finished training replicate {}.".format(current_time, r))
        else:
            pass
    client.close()
    cluster.close()
    return

def load_training_data_fret_simulation_parallel_7b(reference_folder, fret_folder, spc, r):
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    molar_extinction_coefficient = [73000, 81000, 50000, 112000, 120000, 144000, 270000]
    # Extinction coefficient estimated: Pacific Blue
    fluorescence_quantum_yield = [0.92, 0.61, 0.18, 0.79, 0.4, 0.7, 0.33]
    # Estimated QY: 405 dyes, DyLight 510, and RRX, Alexa 532 0.61 > 0.5
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(7)]
    # synthetic_spectra_shifted_filenames = ['{}/R{}_synthetic_spectra_shifted.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    # spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    # spec_shift = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_shifted_filenames]
    bkg_filtered_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder, b) for b in barcode_list]
    spec_matrices = [pd.read_csv(f, header = None).transpose().iloc[32:95,0:100] for f in bkg_filtered_spectra_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).values[:,32:].transpose()) for f in measured_spectra_filenames]
    channel_indices = [0,23,43,57,63]
    synthetic_channel_indices = [0, 3, 9, 17]
    error_scale = [0.15, 0.15, 0.15, 0.15]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    dlm = 3.5
    dum = 30
    shift_switch = 0
    shift = shift_switch*np.array([0,1,1,0,1,1,0,0,1,0])
    labeling_density = [1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [6, 0, 1, 5, 4, 3, 2]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_7b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_7b(fret_folder, molar_extinction_coefficient)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_7b(fret_folder, molar_extinction_coefficient)
    forster_distance = calculate_forster_distance_7b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1]])
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    for r in range(20):
        clf_umap_filename = '{}/reference_simulate_{}_synthetic_7b_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r)
        if not os.path.exists(clf_umap_filename):
            ss_all_list = []
            for enc in range(1,128):
                code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
                numeric_code_list = np.array([int(a) for a in list(code)])
                results = simulate_spectra_7b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit)
                ss_all_list.append(results)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print('Current Time is {}, starting simulation...'.format(current_time))
            print('Current Time is {}, starting simulating replicate {}...'.format(current_time, r))
            training_data = dask.delayed(pd.concat)(ss_all_list, axis = 0)
            training_data_compute = training_data.compute()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Current Time is {}, starting embedding replicate {}...'.format(current_time, r))
            umap_transform = umap.UMAP(n_neighbors = 30, min_dist = 0.0, spread = 1, n_components = 3,
                                       set_op_mix_ratio = 1.0, repulsion_strength = 1.0, negative_sample_rate = 5,
                                       transform_queue_size = 40,
                                       metric = channel_cosine_intensity_7b_v2)
            umap_transform.fit(training_data_compute.iloc[:,0:67].values, y = training_data_compute.code.values)
            clf_umap = svm.SVC(C = 10, gamma = 1)
            clf_umap.fit(umap_transform.embedding_, training_data_compute.code.values)
            # joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc.pkl'.format(reference_folder, str(spc)))
            # joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic.pkl'.format(reference_folder, str(spc)))
            joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_7b_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic_7b_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print("Current Time is {}, finished training.".format(current_time))
            print("Current Time is {}, finished training replicate {}.".format(current_time, r))
        else:
            pass
    client.close()
    cluster.close()
    return

def load_training_data_fret_simulation_parallel_10b(reference_folder, fret_folder, spc, r):
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    molar_extinction_coefficient = [35000, 46000, 50000, 73000, 81000, 50000, 112000, 120000, 0.4*144000, 270000]
    # Extinction coefficient estimated: Pacific Blue
    fluorescence_quantum_yield = [0.3, 0.9, 0.25, 0.92, 0.61, 0.2, 0.79, 0.4, 0.7, 0.33]
    # Estimated QY: 405 dyes, DyLight 510, and RRX
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    synthetic_spectra_shifted_filenames = ['{}/R{}_synthetic_spectra_shifted.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    # spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    # spec_shift = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_shifted_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    # use 100 spectra for each barcode as some barcodes have less cells than others
    bkg_filtered_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder, b) for b in barcode_list]
    spec_matrices = [pd.read_csv(f, header = None).transpose().iloc[:,0:100] for f in bkg_filtered_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    error_scale = [0.15, 0.15, 0.15, 0.15, 0.15]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    dlm = 3.5
    dum = 30
    shift_switch = 0
    shift = shift_switch*np.array([0,1,1,0,1,1,0,1,1,0])
    labeling_density = [1, 1, 1, 1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_10b(fret_folder, molar_extinction_coefficient)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_10b(fret_folder, molar_extinction_coefficient)
    forster_distance = calculate_forster_distance_10b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    for r in range(50,60):
        if r < 50:
            shift_switch = 0
            shift = shift_switch*np.array([0,1,1,0,1,1,0,1,1,0])
        else:
            shift_switch = 1
            shift = shift_switch*np.array([0,1,1,0,1,1,0,1,1,0])
        clf_umap_filename = '{}/reference_simulate_{}_synthetic_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r)
        if not os.path.exists(clf_umap_filename):
            ss_all_list = []
            for enc in range(1,1024):
                code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
                numeric_code_list = np.array([int(a) for a in list(code)])
                results = simulate_spectra_10b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit)
                ss_all_list.append(results)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print('Current Time is {}, starting simulation...'.format(current_time))
            print('Current Time is {}, starting simulating replicate {}...'.format(current_time, r))
            training_data = dask.delayed(pd.concat)(ss_all_list, axis = 0)
            training_data_compute = training_data.compute()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Current Time is {}, starting embedding replicate {}...'.format(current_time, r))
            umap_transform = umap.UMAP(n_neighbors = 30, min_dist = 0, spread = 1, n_components = 3,
                                       set_op_mix_ratio = 1.0, repulsion_strength = 1.0, negative_sample_rate = 5,
                                       transform_queue_size = 16,
                                       metric = channel_cosine_intensity_v2)
            umap_transform.fit(training_data_compute.iloc[:,0:100].values, y = training_data_compute.code.values)
            clf_umap = svm.SVC(C = 10, gamma = 1)
            clf_umap.fit(umap_transform.embedding_, training_data_compute.code.values)
            # joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc.pkl'.format(reference_folder, str(spc)))
            # joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic.pkl'.format(reference_folder, str(spc)))
            joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic_replicate_{}.pkl'.format(reference_folder, str(spc), r))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print("Current Time is {}, finished training.".format(current_time))
            print("Current Time is {}, finished training replicate {}.".format(current_time, r))
        else:
            pass
    client.close()
    cluster.close()
    return

def load_training_data_fret_simulation_parallel_7b_limited(reference_folder, fret_folder, spc, r, design_id):
    probe_design_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_1/simulation'
    design_id = 'DSGN0561'
    primerset = 'A'
    barcode_selection = 'MostSimple'
    probe_design_filename = '{}/{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(probe_design_folder, design_id, design_id, primerset, barcode_selection)
    probes = pd.read_csv(probe_design_filename, dtype = {'code': str})
    barcode_list = [1, 512, 128, 2, 4, 32, 64]
    fluorophores = [1, 10, 8, 2, 3, 6, 7]
    molar_extinction_coefficient = [73000, 81000, 50000, 112000, 120000, 144000, 270000]
    fluorescence_quantum_yield = [0.92, 0.61, 0.18, 0.79, 0.4, 0.7, 0.33]
    # Estimated QY: DyLight 510, and RRX
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(7)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    bkg_filtered_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder, b) for b in barcode_list]
    spec_matrices = [pd.read_csv(f, header = None).transpose().iloc[32:95,0:100] for f in bkg_filtered_spectra_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).values[:,32:].transpose()) for f in measured_spectra_filenames]
    channel_indices = [0,23,43,57,63]
    synthetic_channel_indices = [0, 3, 9, 17]
    error_scale = [0.15, 0.15, 0.15, 0.15]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    dlm = 3.5
    dum = 30
    shift_switch = 0
    shift = shift_switch*np.array([0,1,1,0,1,1,0,0,1,0])
    labeling_density = [1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [6, 0, 1, 5, 4, 3, 2]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_7b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_7b(fret_folder, molar_extinction_coefficient)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_7b(fret_folder, molar_extinction_coefficient)
    forster_distance = calculate_forster_distance_7b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1]])
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    for r in range(20):
        clf_umap_filename = '{}/reference_simulate_{}_{}_synthetic_7b_svc_replicate_{}.pkl'.format(reference_folder, str(spc), design_id, r)
        if not os.path.exists(clf_umap_filename):
            ss_all_list = []
            for enc in range(1,128):
                code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
                if code in probes.code.drop_duplicates().values:
                    numeric_code_list = np.array([int(a) for a in list(code)])
                    results = simulate_spectra_7b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit)
                    ss_all_list.append(results)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print('Current Time is {}, starting simulation...'.format(current_time))
            print('Current Time is {}, starting simulating replicate {}...'.format(current_time, r))
            training_data = dask.delayed(pd.concat)(ss_all_list, axis = 0)
            training_data_compute = training_data.compute()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Current Time is {}, starting embedding replicate {}...'.format(current_time, r))
            umap_transform = umap.UMAP(n_neighbors = 30, min_dist = 0.0, spread = 1, n_components = 3,
                                       set_op_mix_ratio = 1.0, repulsion_strength = 1.0, negative_sample_rate = 5,
                                       transform_queue_size = 40,
                                       metric = channel_cosine_intensity_7b_v2)
            umap_transform.fit(training_data_compute.iloc[:,0:67].values, y = training_data_compute.code.values)
            clf_umap = svm.SVC(C = 10, gamma = 1)
            clf_umap.fit(umap_transform.embedding_, training_data_compute.code.values)
            # joblib.dump(clf_umap, '{}/reference_simulate_{}_synthetic_svc.pkl'.format(reference_folder, str(spc)))
            # joblib.dump(umap_transform, '{}/reference_simulate_{}_synthetic.pkl'.format(reference_folder, str(spc)))
            joblib.dump(clf_umap, '{}/reference_simulate_{}_{}_synthetic_7b_svc_replicate_{}.pkl'.format(reference_folder, str(spc), design_id, r))
            joblib.dump(umap_transform, '{}/reference_simulate_{}_{}_synthetic_7b_replicate_{}.pkl'.format(reference_folder, str(spc), design_id, r))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            # print("Current Time is {}, finished training.".format(current_time))
            print("Current Time is {}, finished training replicate {}.".format(current_time, r))
        else:
            pass
    client.close()
    cluster.close()
    return

# def load_training_data_fret_simulation_parallel_7b_limited(reference_folder, fret_folder, spc, probe_design_filename):
#     barcode_list = [1, 512, 128, 2, 4, 32, 64]
#     fluorophores = [1, 10, 8, 2, 3, 6, 7]
#     fluorescence_quantum_yield = [0.92, 0.61, 0.18, 0.79, 0.5, 0.7, 0.33]
#     measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
#     synthetic_spectra_filenames = ['{}/R{}_synthetic_spectra.csv'.format(reference_folder, fluorophores[i]) for i in range(7)]
#     spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
#     spec_cov = [np.cov(pd.read_csv(f, header = None).transpose())[32:,32:] for f in measured_spectra_filenames]
#     probe_design_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_7/simulation/DSGN0673/DSGN0673_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
#     probes = pd.read_csv(probe_design_filename, dtype = {'code': str})
#     barcodes = probes.code.drop_duplicates()
#     channel_indices = [0,23,43,57,63]
#     synthetic_channel_indices = [0, 3, 9, 17]
#     nbit = 7
#     nchannels = 63
#     simulation_per_code = spc
#     dlm = 3.5
#     dum = 20
#     labeling_density = [1, 1, 2, 0.5, 2, 1, 1]
#     fluorophore_list = [6, 0, 1, 5, 4, 3, 2]
#     excitation_efficiency_matrix = calculate_excitation_efficiency_matrix(reference_folder)
#     extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation(fret_folder)
#     extinction_coefficient_fret = calculate_extinction_coefficient_fret(fret_folder)
#     forster_distance = calculate_forster_distance(fret_folder)
#     excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1],
#                                   [1, 1, 1, 1, 1, 1, 1],
#                                   [0, 0, 1, 1, 1, 1, 1],
#                                   [0, 0, 0, 0, 0, 1, 1]])
#     for enc in range(1, 128):
#         print(enc)
#         code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
#         if code in barcodes.values:
#             numeric_code_list = np.array([int(a) for a in list(code)])
#             simulated_spectra = np.zeros((simulation_per_code, nchannels))
#             indices = [0,23,43,57,63]
#             for exc in range(4):
#                 relevant_fluorophores = numeric_code_list[fluorophore_list]*excitation_matrix[exc, :]
#                 coefficients = np.zeros((simulation_per_code, 7))
#                 extinction_coefficient_direct_excitation_total = np.sum(extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores)
#                 if extinction_coefficient_direct_excitation_total > 0:
#                     absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores/extinction_coefficient_direct_excitation_total
#                 else:
#                     absorption_probability_direct_excitation = extinction_coefficient_direct_excitation[:,exc]*relevant_fluorophores
#                 for i in range(simulation_per_code):
#                     f_sensitized = np.zeros(7)
#                     fret_efficiency_out = np.zeros(7)
#                     for j in range(7):
#                         omega_ensemble = np.sum(fret_number_matrix[i, j, j+1:]*labeling_density[j+1:]*relevant_fluorophores[j+1:])
#                         fret_efficiency_out[j] = calculate_ensemble_efficiency(omega_ensemble)
#                         f_fret_sensitized = 0
#                         for k in range(j):
#                             omega_total = np.sum(fret_number_matrix[i, k, k+1:]*relevant_fluorophores[k+1:])
#                             extinction_fret_total = np.sum(extinction_coefficient_fret[k,:]*relevant_fluorophores)
#                             if extinction_fret_total > 0:
#                                 absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores/extinction_fret_total
#                             else:
#                                 absorption_probability_fret = extinction_coefficient_fret[k,:]*relevant_fluorophores
#                             if omega_total > 0:
#                                 f_fret_sensitized += (fret_number_matrix[i, k, j]/omega_total)*relevant_fluorophores[k]*absorption_probability_fret[j]*f_sensitized[k]*fret_efficiency_out[k]
#                         f_sensitized[j] = absorption_probability_direct_excitation[j]*excitation_efficiency_matrix[j, exc] + f_fret_sensitized
#                         coefficients[i,j] = f_sensitized[j]*(1 - fret_efficiency_out[j])*relevant_fluorophores[j]
#                 spec_list = [np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,channel_indices[exc]:channel_indices[exc+1]] for k in range(nbit)]
#                 spec_list_abs = [np.abs(s) for s in spec_list]
#                 spec_list_norm = [s/np.max(s, axis = 1)[:,None] for s in spec_list_abs]
#                 simulated_spectra_list = [coefficients[:,k][:,None]*spec_list_norm[k] for k in range(nbit)]
#                 simulated_spectra[0:simulation_per_code,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
#             simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
#             error_scale = [0.05, 0.1, 0.25, 0.35]
#             for k in range(0,4):
#                 error_coefficient = error_scale[k] + (1-error_scale[k])*np.random.random(simulated_spectra_norm.shape[0])
#                 max_intensity = np.max(simulated_spectra_norm[:,indices[k]:indices[k+1]], axis = 1)
#                 max_intensity_error_simulation = error_coefficient*max_intensity
#                 error_coefficient[max_intensity_error_simulation < error_scale[k]] = 1
#                 simulated_spectra_norm[:,indices[k]:indices[k+1]] = error_coefficient[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
#             simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
#             ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
#             ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
#             ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
#             ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1]
#             ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
#             ss_norm['code'] = code
#             training_data = training_data.append(ss_norm, ignore_index = True)
#             simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
#             indices = [0,23,43,57,63]
#             for k in range(0,4):
#                 simulated_spectra_norm[:,indices[k]:indices[k+1]] = (error_scale[k]*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
#             simulated_spectra_adjusted_norm = simulated_spectra_norm
#             ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
#             ss_norm['c1'] = 0
#             ss_norm['c2'] = 0
#             ss_norm['c3'] = 0
#             ss_norm['c4'] = 0
#             ss_norm['code'] = '{}_error'.format(code)
#             training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)
#             simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
#             indices = [0,23,43,57,63]
#             if (numeric_code_list[6] == 0) & (numeric_code_list[0] | numeric_code_list[1]):
#                 error_scale_488 = 0.015
#                 simulated_spectra_norm[:,indices[0]:indices[1]] = (error_scale_488 + (error_scale[0] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[0]:indices[1]]
#                 simulated_spectra_adjusted_norm = simulated_spectra_norm
#                 ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
#                 ss_norm['c1'] = 0
#                 ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
#                 ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5] or numeric_code_list[1]
#                 ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
#                 ss_norm['code'] = '{}'.format(code)
#                 training_data_low488 = training_data_low488.append(ss_norm, ignore_index = True)
#             simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
#             indices = [0,23,43,57,63]
#             if (numeric_code_list[4] or numeric_code_list[5]) & (numeric_code_list[6] == 0) & (numeric_code_list[0] == 0) & (numeric_code_list[1] == 0):
#                 error_scale_514 = 0.02
#                 simulated_spectra_norm[:,indices[1]:indices[2]] = (error_scale_514 + (error_scale[1] - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[1]:indices[2]]
#                 simulated_spectra_adjusted_norm = simulated_spectra_norm
#                 ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
#                 ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
#                 ss_norm['c2'] = 0
#                 ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
#                 ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
#                 ss_norm['code'] = '{}'.format(code)
#                 training_data_low514 = training_data_low514.append(ss_norm, ignore_index = True)
#             simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
#             indices = [0,23,43,57,63]
#             if (numeric_code_list[1] == 1) & (numeric_code_list[4] == 0) & (numeric_code_list[5] == 0):
#                 error_scale_561 = 0.1
#                 simulated_spectra_norm[:,indices[2]:indices[3]] = (error_scale_561 + (error_scale[2] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[2]:indices[3]]
#                 simulated_spectra_adjusted_norm = simulated_spectra_norm
#                 ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
#                 ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
#                 ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
#                 ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
#                 ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
#                 ss_norm['code'] = '{}'.format(code)
#                 training_data_low561 = training_data_low561.append(ss_norm, ignore_index = True)
#             simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
#             indices = [0,23,43,57,63]
#             if (numeric_code_list[1] == 1) & (numeric_code_list[4] == 0) & (numeric_code_list[5] == 0) & (numeric_code_list[6] == 0):
#                 error_scale_488 = 0.015
#                 error_scale_561 = 0.1
#                 simulated_spectra_norm[:,indices[0]:indices[1]] = (error_scale_488 + (error_scale[0] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[0]:indices[1]]
#                 simulated_spectra_norm[:,indices[2]:indices[3]] = (error_scale_561 + (error_scale[2] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[2]:indices[3]]
#                 simulated_spectra_adjusted_norm = simulated_spectra_norm
#                 ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
#                 ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[0]
#                 ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
#                 ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
#                 ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
#                 ss_norm['code'] = '{}'.format(code)
#                 training_data_low488561 = training_data_low488561.append(ss_norm, ignore_index = True)
#
#     training_data_full = pd.concat([training_data, training_data_negative, training_data_low488, training_data_low514, training_data_low561, training_data_low488561])
#     training_data = pd.concat([training_data, training_data_low488, training_data_low514, training_data_low561, training_data_low488561])
#     # scaler_full = preprocessing.StandardScaler().fit(training_data_full.values[:,0:63])
#     # training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:63])
#     umap_transform = umap.UMAP(n_neighbors = 20, min_dist = 0.001, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67], y = training_data.code.values)
#     # clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
#     # clf[0].fit(training_data_full.values[:,[3, 7, 11]], training_data_full.c1.values)
#     # clf[1].fit(training_data_full.values[:,[24, 27, 29, 31, 33]], training_data_full.c2.values)
#     # clf[2].fit(training_data_full.values[:,[43, 47]], training_data_full.c3.values)
#     # clf[3].fit(training_data_full.values[:,[57, 60]], training_data_full.c4.values)
#     clf_umap = svm.SVC(C = 10, gamma = 1)
#     clf_umap.fit(umap_transform.embedding_, training_data.code.values)
#     # joblib.dump(scaler_full, '{}/reference_simulate_{}_DSGN0673_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_scaler.pkl'.format(reference_folder, str(spc)))
#     # joblib.dump(clf, '{}/reference_simulate_{}_DSGN0673_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
#     joblib.dump(clf_umap, '{}/reference_simulate_{}_DSGN0673_synthetic_7b_svc_replicate_{}.pkl'.format(reference_folder, str(spc)))
#     joblib.dump(umap_transform, '{}/reference_simulate_{}_DSGN0673_synthetic_7b_replicate_{}.pkl'.format(reference_folder, str(spc)))
#     return(training_data_full)

@dask.delayed
def simulate_spectra_with_coefficient_10b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit):
    s_floor= 0.8
    s_ceiling = 1.2
    # Nominal
    simulated_spectra = np.zeros((simulation_per_code, nchannels))
    fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
    for i in range(simulation_per_code):
        fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
    spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
    spec_avg = [df.Intensity.values for df in spec_synthetic]
    coefficients_full = np.zeros((simulation_per_code, 50))
    for exc in range(5):
        spec_list_norm = []
        for spec in spec_avg:
            if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
            else:
                spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
        coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
        coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
        for s in range(simulation_per_code):
            cov_matrix = np.zeros((95,95))
            for i in range(10):
                for j in range(10):
                    if i == j:
                        cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                    else:
                        cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                        cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
            spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
            spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
            spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
            simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
    simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
    scaling_matrix = np.zeros((simulated_spectra_norm.shape))
    scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
    scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
    scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
    scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
    scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
    simulated_spectra_adjusted_norm = simulated_spectra_norm
    ss_norm = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
    ss_norm.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
    ss_norm.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
    ss_norm.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
    ss_norm.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
    ss_norm['code'] = code
    # Low 488
    if (numeric_code_list[9] == 0) & (numeric_code_list[0] | numeric_code_list[2] | numeric_code_list[7] | numeric_code_list[8]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_low488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488.loc[:,'c1'] = 0
        ss_norm_low488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488['code'] = '{}'.format(code)
    else:
        ss_norm_low488 = pd.DataFrame(columns = ss_norm.columns)
    # Low 514
    if (numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[9]) & (numeric_code_list[0] == 0) & (numeric_code_list[2] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_514 = 0.02
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = (error_scale_514 + (error_scale[2] - error_scale_514)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low514 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_low514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c2'] = 0
        ss_norm_low514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low514['code'] = '{}'.format(code)
    else:
        ss_norm_low514 = pd.DataFrame(columns = ss_norm.columns)
    # Low 561
    if (numeric_code_list[2] or numeric_code_list[0]) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low561 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_low561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_low561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low561.loc[:,'c3'] = 0
        ss_norm_low561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low561['code'] = '{}'.format(code)
    else:
        ss_norm_low561 = pd.DataFrame(columns = ss_norm.columns)
    # Low 488 561
    if (numeric_code_list[2] == 1) & (numeric_code_list[7] == 0) & (numeric_code_list[8] == 0) & (numeric_code_list[9] == 0):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        error_scale_488 = 0.02
        error_scale_561 = 0.02
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = (error_scale_488 + (error_scale[1] - error_scale_488)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = (error_scale_561 + (error_scale[3] - error_scale_561)*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_low488561 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_low488561['c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_low488561['c1'] = 0
        ss_norm_low488561['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_low488561['c3'] = 0
        ss_norm_low488561['c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_low488561['code'] = '{}'.format(code)
    else:
        ss_norm_low488561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 405
    if (numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]/np.max(simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[0]:channel_indices[1]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat405 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_sat405.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat405.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat405.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat405.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat405.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat405['code'] = '{}'.format(code)
    else:
        ss_norm_sat405 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 488
    if (numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]/np.max(simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[1]:channel_indices[2]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat488 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_sat488.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat488.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat488.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat488.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat488.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat488['code'] = '{}'.format(code)
    else:
        ss_norm_sat488 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 514
    if (numeric_code_list[2] or numeric_code_list[0]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]/np.max(simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[2]:channel_indices[3]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat514 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_sat514.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat514.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat514.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat514.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat514.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat514['code'] = '{}'.format(code)
    else:
        ss_norm_sat514 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 561
    if (numeric_code_list[7] or numeric_code_list[8]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]/np.max(simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[3]:channel_indices[4]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat561 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_sat561.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat561.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat561.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat561.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat561.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat561['code'] = '{}'.format(code)
    else:
        ss_norm_sat561 = pd.DataFrame(columns = ss_norm.columns)
    # Saturation 633
    if (numeric_code_list[3] or numeric_code_list[4]):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]] = simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]]/np.max(simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]], axis = 1)[:,None]
        spec_temp = simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]]
        saturation_weight = np.zeros(spec_temp.shape)
        saturation_ceiling = (1 + np.tanh(5*(spec_temp-0.5)))/2*(1 - spec_temp) + spec_temp
        saturation_weight[spec_temp > 0] = 0.8*spec_temp[spec_temp > 0]
        saturation_weight_adjusted = saturation_weight*np.random.random(saturation_weight.shape[0])[:,None]
        spec_sat = spec_temp*(1-saturation_weight_adjusted) + saturation_weight_adjusted*saturation_ceiling
        simulated_spectra_norm[:,channel_indices[4]:channel_indices[5]] = spec_sat
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm
        ss_norm_sat633 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_sat633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_sat633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_sat633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_sat633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_sat633['code'] = '{}'.format(code)
    else:
        ss_norm_sat633 = pd.DataFrame(columns = ss_norm.columns)
    # Damping 633
    if (numeric_code_list[3] == 1):
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        fret_number_matrix = np.zeros((simulation_per_code, 10, 10))
        for i in range(simulation_per_code):
            fret_number_matrix[i,:,:] = calculate_fret_number_10b(forster_distance, dlm, dum)
        spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
        spec_avg = [df.Intensity.values for df in spec_synthetic]
        coefficients_full = np.zeros((simulation_per_code, 50))
        for exc in range(5):
            coefficients = get_channel_coefficients_10b(numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, fret_number_matrix, labeling_density, channel_indices, nchannels, exc, fluorescence_quantum_yield)
            coefficients_full[:,exc*10:(exc+1)*10] = coefficients.copy()
            spec_list_norm = []
            for spec in spec_avg:
                if np.max(spec[channel_indices[exc]:channel_indices[exc+1]]) > 0:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]]/np.max(spec[channel_indices[exc]:channel_indices[exc+1]]))
                else:
                    spec_list_norm.append(spec[channel_indices[exc]:channel_indices[exc+1]])
            for s in range(simulation_per_code):
                cov_matrix = np.zeros((95,95))
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            cov_matrix += np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i])
                        else:
                            cov_ij = np.cov((coefficients[s,i]/excitation_efficiency_matrix[i,exc])*spec_matrices[i], (coefficients[s,j]/excitation_efficiency_matrix[j,exc])*spec_matrices[j])
                            cov_matrix += cov_ij[0:nchannels,nchannels:2*nchannels]
                spec_avg_simulated_list = [coefficients[s,k]*spec_list_norm[k] for k in range(nbit)]
                spec_avg_simulated = np.sum(np.stack(spec_avg_simulated_list, axis = 1), axis = 1)
                spec_simulated = np.random.multivariate_normal(spec_avg_simulated, cov_matrix[channel_indices[exc]:channel_indices[exc+1],channel_indices[exc]:channel_indices[exc+1]], 1)
                simulated_spectra[s,channel_indices[exc]:channel_indices[exc+1]] = np.abs(spec_simulated)
        damping = np.ones((simulated_spectra.shape[0],95))
        random = np.random.random(simulated_spectra.shape[0])
        damping[:,93] = 0.8 + 0.2*random
        damping[:,94] = 0.6 + 0.4*random
        simulated_spectra = simulated_spectra*damping
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        scaling_matrix = np.zeros((simulated_spectra_norm.shape))
        scaling_matrix[:,0:32] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 32, axis = 1)
        scaling_matrix[:,32:55] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 23, axis = 1)
        scaling_matrix[:,55:75] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 20, axis = 1)
        scaling_matrix[:,75:89] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 14, axis = 1)
        scaling_matrix[:,89:95] = np.repeat(s_floor + (s_ceiling - s_floor)*np.random.random(simulated_spectra_norm.shape[0])[:,None], 6, axis = 1)
        simulated_spectra_adjusted_norm = simulated_spectra_norm*scaling_matrix
        ss_norm_damp633 = pd.DataFrame(np.concatenate([simulated_spectra_adjusted_norm,coefficients_full], axis = 1))
        ss_norm_damp633.loc[:,'c0'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm_damp633.loc[:,'c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_damp633.loc[:,'c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0] or numeric_code_list[7] or numeric_code_list[8]
        ss_norm_damp633.loc[:,'c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm_damp633.loc[:,'c4'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm_damp633['code'] = '{}'.format(code)
    else:
        ss_norm_damp633 = pd.DataFrame(columns = ss_norm.columns)
    ss_all = pd.concat([ss_norm, ss_norm_low488, ss_norm_low514, ss_norm_low561, ss_norm_low488561, ss_norm_sat405, ss_norm_sat488, ss_norm_sat514, ss_norm_sat561, ss_norm_sat633, ss_norm_damp633], axis = 0)
    return(ss_all)

def generate_spectra_construction_examples_concatenation():
    reference_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_reference_02_20_2019'
    fret_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_fret'
    sam_tab_bkg_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_1023_reference_bkg.csv'
    data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging'
    spc = 100
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    molar_extinction_coefficient = [35000, 46000, 50000, 73000, 81000, 50000, 112000, 120000, 0.4*144000, 270000]
    # Extinction coefficient estimated: Pacific Blue
    fluorescence_quantum_yield = [0.3, 0.9, 0.25, 0.92, 0.61, 0.2, 0.79, 0.5, 0.7, 0.33]
    # Estimated QY: 405 dyes, DyLight 510, and RRX
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenamesd = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    synthetic_spectra_shifted_filenames = ['{}/R{}_synthetic_spectra_shifted.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    # spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    # spec_shift = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_shifted_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    # use 100 spectra for each barcode as some barcodes have less cells than others
    bkg_filtered_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder, b) for b in barcode_list]
    spec_matrices = [pd.read_csv(f, header = None).transpose().iloc[:,0:100] for f in bkg_filtered_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    error_scale = [0.15, 0.15, 0.15, 0.15, 0.15]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    dlm = 5
    dum = 30
    shift_switch = 0
    shift = shift_switch*np.array([0,1,1,0,1,1,0,0,1,0])
    labeling_density = [1, 1, 1, 1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_10b(fret_folder, molar_extinction_coefficient)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_10b(fret_folder, molar_extinction_coefficient)
    forster_distance = calculate_forster_distance_10b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    sam_tab_bkg_intensity = pd.read_csv(sam_tab_bkg_filename)
    i = 2
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    code = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    numeric_code_list = np.array([int(a) for a in list(code)])
    simulated_spectra = simulate_spectra_with_coefficient_10b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit)
    simulated_spectra_compute = simulated_spectra.compute()
    distances = np.zeros(simulated_spectra_compute.shape[0])
    for i in range(simulated_spectra_compute.shape[0]):
        distances[i] = channel_cosine_intensity_v2(cell_info.iloc[0,0:100].values.astype(float), np.concatenate([simulated_spectra_compute.iloc[i,0:95].values, simulated_spectra_compute.iloc[i,145:150].values]).astype(float))
    best_match_reference = simulated_spectra_compute.iloc[np.argmin(distances),0:95]
    if np.max(best_match_reference[0:32]) > 0:
        best_match_reference[0:32] = np.max(cell_info.iloc[0,0:32].values)*best_match_reference[0:32]/np.max(best_match_reference[0:32])
    if np.max(best_match_reference[32:55]) > 0:
        best_match_reference[32:55] = np.max(cell_info.iloc[0,32:55].values)*best_match_reference[32:55]/np.max(best_match_reference[32:55])
    if np.max(best_match_reference[55:75]) > 0:
        best_match_reference[55:75] = np.max(cell_info.iloc[0,55:75].values)*best_match_reference[55:75]/np.max(best_match_reference[55:75])
    if np.max(best_match_reference[75:89]) > 0:
        best_match_reference[75:89] = np.max(cell_info.iloc[0,75:89].values)*best_match_reference[75:89]/np.max(best_match_reference[75:89])
    if np.max(best_match_reference[89:95]) > 0:
        best_match_reference[89:95] = np.max(cell_info.iloc[0,89:95].values)*best_match_reference[89:95]/np.max(best_match_reference[89:95])
    # channel_names = ['414', '423', '432', '441', '450', '459', '468', '477', '486', '495',
    #                  '504', '513', '522', '531', '539', '548', '557', '566', '575', '584',
    #                  '593', '602', '611', '620', '628', '637', '646', '655', '664', '673', '682', '691']
    channel_names = ['414', '459', '504', '548', '593', '637', '682']
    color_list = ['darkviolet', 'chartreuse', 'yellowgreen', 'darkorange', 'red']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(8))
    gs = GridSpec(5, 1)
    channel_indices = [0,32,55,75,89,95]
    channel_start = [0, 9, 12, 18, 26]
    excitation_laser = ['405 nm', '488 nm', '514 nm', '561 nm', '633 nm']
    for i in range(5):
        ax = plt.subplot(gs[i,0])
        plt.plot(np.arange(channel_start[i], 32), cell_info.iloc[0,channel_indices[i]:channel_indices[i+1]].values, '-', color = color_list[i], alpha = 0.8)
        # plt.plot(np.arange(channel_start[i], 32), best_match_reference[channel_indices[i]:channel_indices[i+1]], '--', color = color_list[i], alpha = 0.8, label = 'Simulated')
        plt.ylim(-0.1,1.2)
        plt.xlim(-2,33)
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.text(22,0.8,excitation_laser[i], color = color_list[i], fontsize = 8)
        plt.xticks([])
        plt.yticks([])
        ax.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    ax = plt.subplot(gs[4,0])
    # l = plt.legend(frameon = False, loc = 2, prop={'size': 8})
    plt.xticks([0,5,10,15,20,25,30], channel_names, fontsize = 8, rotation = 90)
    plt.xlabel('Emission Wavelength [nm]', color = theme_color, fontsize = 8)
    plt.ylabel('Intensity', color = theme_color, fontsize = 8)
    plt.subplots_adjust(left = 0.12, bottom = 0.15, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_construction_example.pdf'.format(data_folder, sample), dpi = 300)
    plt.close()
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(4.5), cm_to_inches(2.5))
    channel_indices = [0,32,55,75,89,95]
    for i in range(5):
        plt.plot(np.arange(channel_indices[i], channel_indices[i+1]), cell_info.iloc[0,channel_indices[i]:channel_indices[i+1]].values, '-', color = color_list[i], alpha = 0.8)
        # plt.plot(np.arange(channel_indices[i], channel_indices[i+1]), best_match_reference[channel_indices[i]:channel_indices[i+1]], '--', color = color_list[i], alpha = 0.8)
        plt.ylim(-0.1,1.2)
        plt.xlim(-2,97)

    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.xlabel('Channels', color = theme_color, fontsize = 8)
    plt.ylabel('Intensity', color = theme_color, fontsize = 8)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.32, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_construction_example_concatenated.pdf'.format(data_folder, sample), dpi = 300)
    plt.close()

    # spectra concatenation presentation
    theme_color = 'white'
    font_size = 12
    channel_names = ['414', '459', '504', '548', '593', '637', '682']
    color_list = ['darkviolet', 'chartreuse', 'yellowgreen', 'darkorange', 'red']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(12))
    gs = GridSpec(5, 1)
    channel_indices = [0,32,55,75,89,95]
    channel_start = [0, 9, 12, 18, 26]
    excitation_laser = ['405 nm', '488 nm', '514 nm', '561 nm', '633 nm']
    for i in range(5):
        ax = plt.subplot(gs[i,0])
        plt.plot(np.arange(channel_start[i], 32), cell_info.iloc[0,channel_indices[i]:channel_indices[i+1]].values, '-', color = color_list[i], alpha = 0.8, linewidth = 2)
        # plt.plot(np.arange(channel_start[i], 32), best_match_reference[channel_indices[i]:channel_indices[i+1]], '--', color = color_list[i], alpha = 0.8, label = 'Simulated')
        plt.ylim(-0.1,1.2)
        plt.xlim(-2,33)
        ax.spines['left'].set_color(theme_color)
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.text(22,0.8,excitation_laser[i], color = color_list[i], fontsize = font_size)
        plt.xticks([])
        plt.yticks([])
        ax.tick_params(labelsize = font_size, direction = 'in', length = 2, colors = theme_color)
    ax = plt.subplot(gs[4,0])
    # l = plt.legend(frameon = False, loc = 2, prop={'size': 8})
    plt.xticks([0,5,10,15,20,25,30], channel_names, fontsize = font_size, rotation = 90)
    plt.xlabel('Emission Wavelength [nm]', color = theme_color, fontsize = font_size)
    plt.ylabel('Intensity', color = theme_color, fontsize = font_size)
    plt.subplots_adjust(left = 0.12, bottom = 0.15, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_construction_example_presentation.svg'.format(data_folder, sample), dpi = 300, transparent = True)
    plt.close()

    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(4))
    channel_indices = [0,32,55,75,89,95]
    for i in range(5):
        plt.plot(np.arange(channel_indices[i], channel_indices[i+1]), cell_info.iloc[0,channel_indices[i]:channel_indices[i+1]].values, '-', color = color_list[i], alpha = 0.8)
        # plt.plot(np.arange(channel_indices[i], channel_indices[i+1]), best_match_reference[channel_indices[i]:channel_indices[i+1]], '--', color = color_list[i], alpha = 0.8)
        plt.ylim(-0.1,1.2)
        plt.xlim(-2,97)

    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.xlabel('Channels', color = theme_color, fontsize = font_size)
    plt.ylabel('Intensity', color = theme_color, fontsize = font_size)
    plt.tick_params(labelsize = font_size, direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.2, bottom = 0.32, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_construction_example_concatenated_presentation.svg'.format(data_folder, sample), dpi = 300, transparent = True)
    plt.close()
    return

def generate_spectra_construction_examples():
    reference_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_reference_02_20_2019'
    fret_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_fret'
    sam_tab_bkg_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_1023_reference_bkg.csv'
    data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging'
    spc = 10
    barcode_list = [256, 16, 8, 1, 512, 128, 2, 4, 32, 64]
    fluorophores = [9, 5, 4, 1, 10, 8, 2, 3, 6, 7]
    molar_extinction_coefficient = [35000, 46000, 50000, 73000, 81000, 50000, 112000, 120000, 0.4*144000, 270000]
    # Extinction coefficient estimated: Pacific Blue
    fluorescence_quantum_yield = [0.3, 0.9, 0.25, 0.92, 0.61, 0.2, 0.79, 0.5, 0.7, 0.33]
    # Estimated QY: 405 dyes, DyLight 510, and RRX
    measured_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    synthetic_spectra_filenamesd = ['{}/R{}_synthetic_spectra.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    synthetic_spectra_shifted_filenames = ['{}/R{}_synthetic_spectra_shifted.csv'.format(fret_folder, fluorophores[i]) for i in range(10)]
    measured_spec_avg = [pd.read_csv(f, header = None) for f in measured_spectra_filenames]
    # spec_avg = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_filenames]
    # spec_shift = [pd.read_csv(f).Intensity.values for f in synthetic_spectra_shifted_filenames]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in measured_spectra_filenames]
    # use 100 spectra for each barcode as some barcodes have less cells than others
    bkg_filtered_spectra_filenames = ['{}/08_18_2018_enc_{}_avgint_bkg_filtered_full.csv'.format(fret_folder, b) for b in barcode_list]
    spec_matrices = [pd.read_csv(f, header = None).transpose().iloc[:,0:100] for f in bkg_filtered_spectra_filenames]
    channel_indices = [0,32,55,75,89,95]
    synthetic_channel_indices = [0, 9, 12, 18, 26]
    error_scale = [0.15, 0.15, 0.15, 0.15, 0.15]
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    dlm = 5
    dum = 30
    shift_switch = 0
    shift = shift_switch*np.array([0,1,1,0,1,1,0,0,1,0])
    labeling_density = [1, 1, 1, 1, 1, 2, 0.5, 2, 1, 1]
    fluorophore_list = [1, 5, 6, 9, 0, 2, 8, 7, 4, 3]
    excitation_efficiency_matrix = calculate_excitation_efficiency_matrix_10b(reference_folder)
    extinction_coefficient_direct_excitation = calculate_extinction_coefficient_direct_excitation_10b(fret_folder, molar_extinction_coefficient)
    extinction_coefficient_fret = calculate_extinction_coefficient_fret_10b(fret_folder, molar_extinction_coefficient)
    forster_distance = calculate_forster_distance_10b(fret_folder, molar_extinction_coefficient, fluorescence_quantum_yield)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    sam_tab_bkg_intensity = pd.read_csv(sam_tab_bkg_filename)
    i = 2
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    code = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    numeric_code_list = np.array([int(a) for a in list(code)])
    simulated_spectra = simulate_spectra_with_coefficient_10b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit)
    simulated_spectra_compute = simulated_spectra.compute()
    spectra_addition = simulated_spectra_compute.iloc[0,0:95].values
    coefficients = simulated_spectra_compute.iloc[0,95:145].values.reshape((5,10))
    spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
    spec_avg = [df.Intensity.values for df in spec_synthetic]
    color_list = ['dodgerblue', 'darkorange', 'olivedrab']
    # spectra peak barcode 2 example
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
    ci = 0
    for i in range(10):
        spec_channel_norm = spec_avg[i].copy()
        for exc in range(5):
            if np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]/np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]])
            else:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]
            spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = coefficients[exc,i]*spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]]
        if code[fluorophore_list[i]] == '1':
            plt.plot(np.arange(0,95), spec_channel_norm, '-', color = color_list[ci], alpha = 1, label = 'R{}'.format(fluorophores[i]))
            ci += 1

    plt.plot(np.max(coefficients)*spectra_addition + 0.05, color = (0.5,0.5,0.5), alpha = 1, label = 'Combined')
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.xlabel('Channels', color = theme_color, fontsize = 8)
    plt.ylabel('Intensity', color = theme_color, fontsize = 8)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.36, 0.4), handlelength = 0.5)
    plt.subplots_adjust(left = 0.175, bottom = 0.25, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_addition_distinct_peak_example.pdf'.format(data_folder, sample), dpi = 300)
    plt.close()

    # spectra peak barcode 2 example presentation
    theme_color = 'white'
    font_size = 12
    color_list = ['dodgerblue', 'darkorange', 'olivedrab']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(10), cm_to_inches(6))
    ci = 0
    for i in range(10):
        spec_channel_norm = spec_avg[i].copy()
        for exc in range(5):
            if np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]/np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]])
            else:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]
            spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = coefficients[exc,i]*spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]]
        if code[fluorophore_list[i]] == '1':
            plt.plot(np.arange(0,95), spec_channel_norm, '-', color = color_list[ci], alpha = 1, label = 'R{}'.format(fluorophores[i]))
            ci += 1

    plt.plot(np.max(coefficients)*spectra_addition + 0.05, color = (0.5,0.5,0.5), alpha = 1, label = 'Combined')
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.xlabel('Channels', color = theme_color, fontsize = font_size)
    plt.ylabel('Intensity', color = theme_color, fontsize = font_size)
    plt.tick_params(labelsize = font_size, direction = 'in', length = 2, colors = theme_color)
    l = plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.36, 0.4), handlelength = 0.5)
    for t in l.get_texts():
        t.set_color(theme_color)

    plt.subplots_adjust(left = 0.175, bottom = 0.25, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_addition_distinct_peak_example_presentation.svg'.format(data_folder, sample), dpi = 300, transparent = True)
    plt.close()


    i = 897
    sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
    image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
    enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
    code = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
    # cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
    cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
    numeric_code_list = np.array([int(a) for a in list(code)])
    simulated_spectra = simulate_spectra_with_coefficient_10b(reference_folder, fret_folder, numeric_code_list, fluorophore_list, excitation_matrix, excitation_efficiency_matrix, simulation_per_code, extinction_coefficient_direct_excitation, extinction_coefficient_fret, dlm, dum, shift, labeling_density, spec_cov, spec_matrices, channel_indices, nchannels, code, error_scale, fluorescence_quantum_yield, forster_distance, nbit)
    simulated_spectra_compute = simulated_spectra.compute()
    spectra_addition = simulated_spectra_compute.iloc[0,0:95].values
    coefficients = simulated_spectra_compute.iloc[0,95:145].values.reshape((5,10))
    spec_synthetic = generate_synthetic_reference_spectra_shift(reference_folder, fret_folder, shift)
    spec_avg = [df.Intensity.values for df in spec_synthetic]
    color_list = ['dodgerblue', 'darkorange', 'olivedrab', 'darkviolet']
    # spectra peak barcode 897 example
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6), cm_to_inches(3))
    ci = 0
    for i in range(10):
        spec_channel_norm = spec_avg[i].copy()
        for exc in range(5):
            if np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]/np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]])
            else:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]
            spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = coefficients[exc,i]*spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]]
        if code[fluorophore_list[i]] == '1':
            plt.plot(np.arange(0,95), spec_channel_norm, '-', color = color_list[ci], alpha = 1, label = 'R{}'.format(fluorophores[i]))
            ci += 1

    plt.plot(np.max(coefficients)*spectra_addition + 0.1, color = (0.5,0.5,0.5), alpha = 1, label = 'Combined')
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.xlabel('Channels', color = theme_color, fontsize = 8)
    plt.ylabel('Intensity', color = theme_color, fontsize = 8)
    plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = theme_color)
    plt.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.6, 0.35), handlelength = 0.5, ncol = 2, columnspacing = 0.5)
    plt.ylim(-0.02, 0.45)
    plt.subplots_adjust(left = 0.175, bottom = 0.25, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_addition_distinct_peak_example_2.pdf'.format(data_folder, sample), dpi = 300)
    plt.close()

    # spectra peak barcode 897 example presentation
    theme_color = 'white'
    font_size = 12
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(10), cm_to_inches(6))
    ci = 0
    for i in range(10):
        spec_channel_norm = spec_avg[i].copy()
        for exc in range(5):
            if np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]) > 0:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]/np.max(spec_avg[i][channel_indices[exc]:channel_indices[exc+1]])
            else:
                spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = spec_avg[i][channel_indices[exc]:channel_indices[exc+1]]
            spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]] = coefficients[exc,i]*spec_channel_norm[channel_indices[exc]:channel_indices[exc+1]]
        if code[fluorophore_list[i]] == '1':
            plt.plot(np.arange(0,95), spec_channel_norm, '-', color = color_list[ci], alpha = 1, label = 'R{}'.format(fluorophores[i]))
            ci += 1

    plt.plot(np.max(coefficients)*spectra_addition + 0.1, color = (0.5,0.5,0.5), alpha = 1, label = 'Combined')
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.xlabel('Channels', color = theme_color, fontsize = font_size)
    plt.ylabel('Intensity', color = theme_color, fontsize = font_size)
    plt.tick_params(labelsize = font_size, direction = 'in', length = 2, colors = theme_color)
    l = plt.legend(frameon = False, fontsize = 8, bbox_to_anchor = (0.6, 0.5), handlelength = 0.5, ncol = 2, columnspacing = 0.5)
    for t in l.get_texts():
        t.set_color(theme_color)

    plt.ylim(-0.02, 0.45)
    plt.subplots_adjust(left = 0.175, bottom = 0.25, right = 0.98, top = 0.98)
    plt.savefig('{}/{}/spectra_addition_distinct_peak_example_2_presentation.svg'.format(data_folder, sample), dpi = 300, transparent = True)
    plt.close()




    return

def generate_readout_probe_spectra():
    sam_tab_bkg_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_reference_02_20_2019'
    sam_tab_bkg_intensity = pd.read_csv(sam_tab_bkg_filename)
    data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_reference_02_20_2019'
    fluorophore_list = ['Alexa 488', 'Alexa 546', 'Rhodamine Red X', 'Pacific Green', 'Pacific Blue', 'Alexa 610', 'Alexa 647', 'DyLight 510 LS', 'Alexa 405', 'Alexa 532']
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(12.75), cm_to_inches(11.5))
    gs = GridSpec(5, 2)
    for i in range(10):
        pc, pr = np.divmod(i, 5)
        enc = 2**i
        spec = pd.read_csv('{}/08_18_2018_enc_{}_avgint.csv'.format(data_folder, enc), header = None)
        image_adjacent = sam_tab_bkg_intensity.loc[enc-1,'SampleAdjacent']
        bkg = pd.read_csv('{}/{}_bkg.csv'.format(data_folder, image_adjacent), header = None)
        spec_average = np.average(spec.values, axis = 0)
        spec_average[0:32] = spec_average[0:32] - bkg.iloc[:,0].values
        spec_std = np.std(spec.values, axis = 0)
        ax = plt.subplot(gs[pr, pc])
        ax.errorbar(np.arange(0,32), spec_average[0:32], yerr = spec_std[0:32], color = 'purple', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '405 nm')
        ax.errorbar(np.arange(32,55), spec_average[32:55], yerr = spec_std[32:55], color = 'chartreuse', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '488 nm')
        ax.errorbar(np.arange(55,75), spec_average[55:75], yerr = spec_std[55:75], color = 'yellowgreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '514 nm')
        ax.errorbar(np.arange(75,89), spec_average[75:89], yerr = spec_std[75:89], color = 'orange', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '561 nm')
        ax.errorbar(np.arange(89,95), spec_average[89:95], yerr = spec_std[89:95], color = 'red', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '633 nm')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.ylim(-0.1, 0.9)
        plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
        plt.text(0.12, 0.6, "Readout Probe {}\n{}".format(i+1, fluorophore_list[i]), transform = ax.transAxes, fontsize = 8, color = 'black')

    ax = plt.subplot(gs[4, 0])
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    ax.set_xticks([0,25,50,75])
    ax.set_xticklabels([0, 25, 50, 75])
    ax.set_yticks([0,0.5])
    ax.set_yticklabels([0,0.5])
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
    l = ax.legend(frameon = False, fontsize = 6, bbox_to_anchor = (0.6,0.3), labelspacing = 0)
    for t in l.get_texts():
        t.set_color('black')

    plt.xlabel('Channels [-]', fontsize = 8, color = 'black')
    plt.ylabel('Intensity [-]', fontsize = 8, color = 'black')
    plt.subplots_adjust(left = 0.08, right = 0.98, bottom = 0.08, top = 0.98, wspace = 0.2, hspace = 0.2)
    plt.savefig('{}/readout_probe_spectra.pdf'.format(data_folder), dpi = 300, transparent = True)
    return

def main():
    # parameters are hard coded within the function
    # load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b_limited('','','',2000)
    return

if __name__ == '__main__':
    main()
