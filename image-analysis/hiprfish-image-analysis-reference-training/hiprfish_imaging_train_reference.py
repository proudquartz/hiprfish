
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
import joblib
import argparse
import matplotlib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from matplotlib import pyplot as plt

rc('axes', linewidth = 1)

def convert_code_to_7b(code):
    bits = list(code)
    converted_code = ''.join([bits[i] for i in [0,2,3,4,7,8,9]])
    return(converted_code)

def cm_to_inches(x):
    return(x/2.54)

def plot_umap(umap_transform):
    embedding_df = pd.DataFrame(umap_transform.embedding_)
    embedding_df['numeric_barcode'] = training_data.code.apply(int, args = (2,))
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(6),cm_to_inches(6))
    cmap = matplotlib.cm.get_cmap('jet')
    delta = 1/1023
    color_list = [cmap(i*delta) for i in range(1023)]
    for i in range(1023):
        enc = i+1
        emd = embedding_df.loc[embedding_df.numeric_barcode.values == enc]
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
    plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_1023_reference_08_18_2018/reference_umap_visualization.pdf', dpi = 300, transparent = True)
    plt.close()
    return

def load_training_data_simulate_normalized(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_norm)
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    if learning_mode == 'SVM':
        clf = svm.SVC(C = 20, gamma = 0.5)
    elif learning_mode == 'RFC':
        clf = RandomForestClassifier()
    clf.fit(training_data.drop(columns = ['code'], axis = 1), training_data['code'])
    joblib.dump(clf, '{}/reference_simulate_{}_normalized.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_normalized_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_norm)
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 200).fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.5)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(neighbor, '{}/reference_simulate_{}_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_normalized_biofilm_select_umap_transformed(reference_folder, learning_mode, spc, taxon_lookup):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    code_list = taxon_lookup.code.apply(convert_code_to_10b).apply(int, args = (2,))
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        if enc in code_list.values:
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_norm)
            ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 200).fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.5)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(neighbor, '{}/reference_simulate_{}_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_normalized_differentiated_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        ss_derivative = np.diff(simulated_spectra_norm)
        ss_norm = pd.DataFrame(np.concatenate([simulated_spectra_norm, ss_derivative], axis = 1))
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP().fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.1)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        ss_norm = pd.DataFrame(simulated_spectra)
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    if learning_mode == 'SVM':
        clf = svm.SVC()
    elif learning_mode == 'RFC':
        clf = RandomForestClassifier()
    clf.fit(training_data.drop(columns = ['code'], axis = 1), training_data['code'])
    joblib.dump(clf, '{}/reference_simulate_{}.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_normalized_custom_kernel(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_norm)
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    if learning_mode == 'SVM':
        clf = svm.SVC(kernel = excitation_wise_inner_product)
    elif learning_mode == 'RFC':
        clf = RandomForestClassifier()
    clf.fit(training_data.drop(columns = ['code'], axis = 1), training_data['code'])
    joblib.dump(clf, '{}/reference_simulate_{}_normalized.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        simulated_spectra_norm_adjusted_list = [simulated_spectra_norm]
        for k in range(0,5):
            simulated_spectra_adjusted = simulated_spectra_norm.copy()
            simulated_spectra_adjusted[:,indices[k]:indices[k+1]] = (0.5+0.5*np.random.random(simulated_spectra_adjusted.shape[0]))[:,None]*simulated_spectra_adjusted[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_adjusted/np.max(simulated_spectra_adjusted, axis = 1)[:,None]
            simulated_spectra_norm_adjusted_list.append(simulated_spectra_adjusted_norm)
        ss_norm = pd.DataFrame(np.vstack(simulated_spectra_norm_adjusted_list))
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 200).fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.5)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

@numba.njit()
def channel_cosine_intensity(x, y):
    check = np.sum(np.abs(x[95:100] - y[95:100]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[95] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
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
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
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
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
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
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
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
            cos_weight_5 = 0.0
            cos_dist_5 = 0.0
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
        # cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/(cos_weight_1 + cos_weight_2 + cos_weight_3 + cos_weight_4 + cos_weight_5)
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/5
    else:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
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
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/5
    return(cos_dist)

@numba.njit()
def channel_cosine_intensity_violet_derivative(x, y):
    check = np.sum(np.abs(x[126:131] - y[126:131]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[126] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
            derivateive_cos_dist_1 = 0.0
            derivateive_cos_weight_1 = 0.0
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
            for i in range(95,126):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                derivative_cos_dist_1 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                derivative_cos_dist_1 = 1.0
            else:
                derivative_cos_dist_1 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[127] == 0:
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
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
        if x[128] == 0:
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
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
        if x[129] == 0:
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
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
        if x[130] == 0:
            cos_weight_5 = 0.0
            cos_dist_5 = 0.0
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
    else:
        derivative_cos_dist_1 = 1.0
        derivative_cos_weight_1 = 1.0
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
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
    cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/5
    return(cos_dist)

@numba.njit()
def channel_cosine_intensity_violet_derivative_v2(x, y):
    check = np.sum(np.abs(x[126:132] - y[126:132]))
    if check < 0.01:
        derivative_cos_dist_1 = 0.0
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[126] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
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
        if x[127] == 0:
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
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
        if x[128] == 0:
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
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
        if x[129] == 0:
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
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
        if x[130] == 0:
            cos_weight_5 = 0.0
            cos_dist_5 = 0.0
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
    else:
        derivative_cos_dist_1 = 1.0
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
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
    cos_dist = (derivative_cos_dist_1 + cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/6
    return(derivative_cos_dist_1 , cos_dist_1 , cos_dist_2 , cos_dist_3 , cos_dist_4 , cos_dist_5)

@numba.njit()
def channel_cosine_intensity_violet_derivative_vectorized(x, y):
    check = np.sum(np.abs(x[126:131] - y[126:131]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[126] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
            derivateive_cos_dist_1 = 0.0
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
            for i in range(95,126):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                derivative_cos_dist_1 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                derivative_cos_dist_1 = 1.0
            else:
                derivative_cos_dist_1 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[127] == 0:
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
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
        if x[128] == 0:
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
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
        if x[129] == 0:
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
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
        if x[130] == 0:
            cos_weight_5 = 0.0
            cos_dist_5 = 0.0
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
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/5
    else:
        derivative_cos_dist_1 = 1.0
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
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
        cos_dist = (derivative_cos_dist_1 + cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/6
    return(derivative_cos_dist_1 , cos_dist_1 , cos_dist_2 , cos_dist_3 , cos_dist_4 , cos_dist_5)

@numba.njit()
def channel_cosine_intensity_7b(x, y):
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
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4)/4
    else:
        cos_dist = 1
    return(cos_dist)

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
        cos_dist = 0.5*(cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4)/4
    else:
        cos_dist = 1
    return(cos_dist)

@numba.njit()
def channel_cosine_intensity_7b_v3(x, y):
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
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        mag_dist = 0.0
        for i in range(0,23):
            if norm_x == 0.0 and norm_y == 0.0:
                mag_dist = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                mag_dist = 1.0
            else:
                for i in range(0,23):
                    mag_dist += (x[i] - y[i])**2
        mag_dist = mag_dist/63
        cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + mag_dist)/5
    else:
        cos_dist = 1
    return(cos_dist)

@numba.njit()
def channel_chi_intensity_7b(x, y):
    check = np.sum(np.abs(x[63:67] - y[63:67]))
    if check < 0.01:
        result = 0.0
        for i in range(63):
            result += (x[i] - y[i])**2/(x[i] + y[i])
        result /= 63
    else:
        result = 1
    return(result)

@numba.njit()
def channel_cosine_intensity_normal(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
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
    cos_dist = (cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4 + cos_dist_5)/5
    return(cos_dist_1, cos_dist_2 , cos_dist_3 , cos_dist_4 , cos_dist_5)

@numba.njit()
def correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]
    mu_x /= x.shape[0]
    mu_y /= x.shape[0]
    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))

@numba.njit()
def channel_correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    for i in range(0,32):
        mu_x += x[i]
        mu_y += y[i]
    mu_x /= 32
    mu_y /= 32
    for i in range(0,32):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    if norm_x == 0.0 and norm_y == 0.0:
        corr_1 = 0.0
    elif dot_product == 0.0:
        corr_1 = 1.0
    else:
        corr_1 = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    for i in range(32,55):
        mu_x += x[i]
        mu_y += y[i]
    mu_x /= 23
    mu_y /= 23
    for i in range(32,55):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    if norm_x == 0.0 and norm_y == 0.0:
        corr_2 = 0.0
    elif dot_product == 0.0:
        corr_2 = 1.0
    else:
        corr_2 = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    for i in range(55,75):
        mu_x += x[i]
        mu_y += y[i]
    mu_x /= 20
    mu_y /= 20
    for i in range(55,75):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    if norm_x == 0.0 and norm_y == 0.0:
        corr_3 = 0.0
    elif dot_product == 0.0:
        corr_3 = 1.0
    else:
        corr_3 = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    for i in range(75,89):
        mu_x += x[i]
        mu_y += y[i]
    mu_x /= 14
    mu_y /= 14
    for i in range(75,89):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    if norm_x == 0.0 and norm_y == 0.0:
        corr_4 = 0.0
    elif dot_product == 0.0:
        corr_4 = 1.0
    else:
        corr_4 = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0
    for i in range(89,95):
        mu_x += x[i]
        mu_y += y[i]
    mu_x /= 6
    mu_y /= 6
    for i in range(89,95):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y
    if norm_x == 0.0 and norm_y == 0.0:
        corr_5 = 0.0
    elif dot_product == 0.0:
        corr_5 = 1.0
    else:
        corr_5 = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
    return(corr_1,corr_2,corr_3,corr_4,corr_5)

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        ss_norm['c1'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm['c3'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[8] or numeric_code_list[7]
        ss_norm['c4'] = numeric_code_list[7] or numeric_code_list[8]
        ss_norm['c5'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity).fit(training_data.iloc[:,0:100].values, y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 0.5) for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:32].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,32:55].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,55:75].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,75:89].values, training_data.c4.values)
    clf[4].fit(training_data.iloc[:,89:95].values, training_data.c5.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_violet_derivative_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        violet_derivative = np.diff(simulated_spectra_adjusted_norm[:,0:32], axis = 1)
        ss_norm = pd.DataFrame(np.concatenate((simulated_spectra_adjusted_norm, violet_derivative), axis = 1))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        ss_norm['c1'] = numeric_code_list[1] or numeric_code_list[5] or numeric_code_list[6]
        ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
        ss_norm['c3'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[8] or numeric_code_list[7]
        ss_norm['c4'] = numeric_code_list[7] or numeric_code_list[8]
        ss_norm['c5'] = numeric_code_list[3] or numeric_code_list[4]
        ss_norm['c6'] = numeric_code_list[1]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_violet_derivative_v2).fit(training_data.iloc[:,0:132].values, y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 0.5) for i in range(6)]
    clf[0].fit(training_data.iloc[:,0:32].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,32:55].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,55:75].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,75:89].values, training_data.c4.values)
    clf[4].fit(training_data.iloc[:,89:95].values, training_data.c5.values)
    clf[5].fit(training_data.iloc[:,95:126].values, training_data.c6.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_violet_derivative_umap_transformed_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_violet_derivative_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_violet_derivative_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed_biofilm_7b(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        if (numeric_code_list[6] == 0) and (numeric_code_list[5] == 0) and (numeric_code_list[1] == 0):
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)[:,32:95]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = 0
            ss_norm['c2'] = 0
            ss_norm['c3'] = 0
            ss_norm['c4'] = 0
            ss_norm['code'] = '{}_error'.format(code)
            training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    training_data_full = pd.concat([training_data, training_data_negative])
    clf = [svm.SVC(C = 10, gamma = 0.5) for i in range(4)]
    clf[0].fit(training_data_full.iloc[:,0:23].values, training_data_full.c1.values)
    clf[1].fit(training_data_full.iloc[:,23:43].values, training_data_full.c2.values)
    clf[2].fit(training_data_full.iloc[:,43:57].values, training_data_full.c3.values)
    clf[3].fit(training_data_full.iloc[:,57:63].values, training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_scaled_umap_transformed_biofilm_7b(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        if (numeric_code_list[6] == 0) and (numeric_code_list[5] == 0) and (numeric_code_list[1] == 0):
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)[:,32:95]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3+0.7*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8] or numeric_code_list[4]
            ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = 0
            ss_norm['c2'] = 0
            ss_norm['c3'] = 0
            ss_norm['c4'] = 0
            ss_norm['code'] = '{}_error'.format(code)
            training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)
    training_data_full = pd.concat([training_data, training_data_negative])
    scaler_full = preprocessing.StandardScaler().fit(training_data_full.values[:,0:63])
    training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:63])
    scaler = preprocessing.StandardScaler().fit(training_data.values[:,0:63])
    training_data_scaled = scaler.transform(training_data.values[:,0:63])
    training_data_scaled_full = np.zeros((training_data_scaled.shape[0], 67))
    training_data_scaled_full[:,0:63] = training_data_scaled
    training_data_scaled_full[:,63:67] = training_data.iloc[:,63:67].values
    clf_test = svm.SVC(C = 10, gamma = 0.1)
    clf_test.fit(training_data_scaled_full[:,0:63], training_data.code.values)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b_v2).fit(training_data_scaled_full[:,0:67], y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 0.1) for i in range(4)]
    clf[0].fit(training_data_full_scaled[:,0:23], training_data_full.c1.values)
    clf[1].fit(training_data_full_scaled[:,23:43], training_data_full.c2.values)
    clf[2].fit(training_data_full_scaled[:,43:57], training_data_full.c3.values)
    clf[3].fit(training_data_full_scaled[:,57:63], training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.1)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(scaler, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_scaler.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed_biofilm_7b_DSGN(reference_folder, learning_mode, probe_design_file, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    probes = pd.read_csv(probe_design_file, dtype = {'code' : str})
    design_id = 'DSGN0549'
    code_list = probes.code.unique()
    nbit = 10
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        code7b = convert_code_to_7b(code)
        numeric_code_list = [int(a) for a in list(code)]
        if (numeric_code_list[6] == 0) and (numeric_code_list[5] == 0) and (numeric_code_list[1] == 0) and (code7b in code_list):
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)[:,32:95]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.5+0.5*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = 0
            ss_norm['c2'] = 0
            ss_norm['c3'] = 0
            ss_norm['c4'] = 0
            ss_norm['code'] = '{}_error'.format(code)
            training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    training_data_full = pd.concat([training_data, training_data_negative])
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
    clf[0].fit(training_data_full.iloc[:,0:23].values, training_data_full.c1.values)
    clf[1].fit(training_data_full.iloc[:,23:43].values, training_data_full.c2.values)
    clf[2].fit(training_data_full.iloc[:,43:57].values, training_data_full.c3.values)
    clf[3].fit(training_data_full.iloc[:,57:63].values, training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 1)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc), design_id))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc), design_id))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_{}_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc), design_id))
    return

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed_error_threshold_biofilm_7b(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        if (numeric_code_list[6] == 0) and (numeric_code_list[5] == 0) and (numeric_code_list[1] == 0):
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)[:,32:95]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[8] or numeric_code_list[7]
            ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    clf = [svm.SVC() for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:23].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,23:43].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,43:57].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,57:63].values, training_data.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed_error_threshold_biofilm_7b_limited(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        if (numeric_code_list[6] == 0) and (numeric_code_list[5] == 0) and (numeric_code_list[1] == 0):
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)[:,32:95]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.4+0.6*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[8] or numeric_code_list[7]
            ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    clf = [svm.SVC() for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:23].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,23:43].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,43:57].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,57:63].values, training_data.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def calculate_fret_efficiency(data_folder, distance):
    files = glob.glob(data_folder + '/*_excitation.csv')
    samples = [re.sub('_excitation.csv', '', os.path.basename(file)) for file in files]
    forster_distance = np.zeros((7,7))
    fret_transfer_matrix = np.eye(7)
    kappa_squared = 2/3
    ior = 1.4
    NA = 6.022e23
    Qd = 1
    prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
    molar_extinction_coefficient = [73000, 112000, 120000, 144000, 270000, 50000, 81000]
    fluorescence_quantum_yield = [0.92, 0.79, 1, 0.33, 0.33, 1, 0.61]
    fluorophores = [10,8,7,6,3,2,1]
    for i in range(7):
        for j in range(7):
            if i != j:
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
                fret_transfer_matrix[i,j] = np.sign(emission_max_i - emission_max_j)*1/(1+(distance/forster_distance[i,j])**6)
    return(fret_transfer_matrix)

def load_training_data_simulate_reabsorption_umap_transformed_biofilm_7b(reference_folder, fret_folder, learning_mode, spc):
    barcode_list = [512, 128, 64, 32, 4, 2, 1]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    fret_transfer_matrix = calculate_fret_efficiency(fret_folder, 5)
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        coefficients = np.dot(fret_transfer_matrix, numeric_code_list)*numeric_code_list
        simulated_spectra_list = [coefficients[k]*np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,32:95] for k in range(nbit)]
        simulated_spectra = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,23,43,57,63]
        for k in range(0,4):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3+0.7*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
        ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    clf = [svm.SVC() for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:23].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,23:43].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,43:57].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,57:63].values, training_data.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_interaction_simulated_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_interaction_simulated_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_interaction_simulated_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_reabsorption_umap_transformed_limited_biofilm_7b(reference_folder, fret_folder, learning_mode, spc):
    barcode_list = [512, 128, 64, 32, 4, 2, 1]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    fret_transfer_matrix = calculate_fret_efficiency(fret_folder, 6)
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        if code in taxon_lookup.code.values:
            numeric_code_list = np.array([int(a) for a in list(code)])
            coefficients = np.dot(fret_transfer_matrix, numeric_code_list)*numeric_code_list
            simulated_spectra_list = [coefficients[k]*np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,32:95] for k in range(nbit)]
            simulated_spectra = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3+0.7*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    clf = [svm.SVC() for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:23].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,23:43].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,43:57].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,57:63].values, training_data.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_interaction_simulated_select_DSGN0524_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_interaction_simulated_select_DSGN0524_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_interaction_simulated_select_DSGN0524_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_biofilm_7b(reference_folder, fret_folder, learning_mode, spc):
    barcode_list = [512, 128, 64, 32, 4, 2, 1]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    fret_transfer_matrix = calculate_fret_efficiency(fret_folder, 6.5)
    excitation_matrix = np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        indices = [0,23,43,57,63]
        for exc in range(4):
            relevant_fluorophores = numeric_code_list*excitation_matrix[exc, :]
            coefficients = np.dot(fret_transfer_matrix, relevant_fluorophores)*relevant_fluorophores
            simulated_spectra_list = [coefficients[k]*np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,32:95] for k in range(nbit)]
            simulated_spectra[:,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)[:,indices[exc]:indices[exc+1]]
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        # for k in range(0,4):
        #     simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3+0.7*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
        ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
        ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
        ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    clf = [svm.SVC() for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:23].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,23:43].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,43:57].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,57:63].values, training_data.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_interaction_simulated_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_interaction_simulated_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_interaction_simulated_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b(reference_folder, fret_folder, learning_mode, spc):
    barcode_list = [512, 128, 64, 32, 4, 2, 1]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    fret_transfer_matrix = np.zeros((simulation_per_code, 7, 7))
    for i in range(simulation_per_code):
        fret_transfer_matrix[i,:,:] = calculate_fret_efficiency(fret_folder, 6 + 4*np.random.random())

    excitation_matrix = np.array([[1, 1, 0, 0, 1, 1, 1],
                                  [1, 1, 0, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = np.array([int(a) for a in list(code)])
        simulated_spectra = np.zeros((simulation_per_code, nchannels))
        indices = [0,23,43,57,63]
        if numeric_code_list[6] == 1:
            error_scale = [0.25, 0.1, 0.22, 0.45]
        else:
            error_scale = [0.05, 0.1, 0.22, 0.45]
        for exc in range(4):
            relevant_fluorophores = numeric_code_list*excitation_matrix[exc, :]
            coefficients = np.zeros((simulation_per_code, 7))
            for i in range(simulation_per_code):
                coefficients[i,:] = np.dot(fret_transfer_matrix[i,:,:], relevant_fluorophores)*relevant_fluorophores
            simulated_spectra_list = [coefficients[:,k][:,None]*np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,32:95] for k in range(nbit)]
            simulated_spectra[0:simulation_per_code,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)[:,indices[exc]:indices[exc+1]]
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
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
        ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
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

    training_data_full = pd.concat([training_data, training_data_negative])
    scaler_full = preprocessing.StandardScaler().fit(training_data_full.values[:,0:63])
    training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:63])
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67], y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
    clf[0].fit(training_data_full_scaled[:,0:23], training_data_full.c1.values)
    clf[1].fit(training_data_full_scaled[:,23:43], training_data_full.c2.values)
    clf[2].fit(training_data_full_scaled[:,43:57], training_data_full.c3.values)
    clf[3].fit(training_data_full_scaled[:,57:63], training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 1)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(scaler_full, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_scaler.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_interaction_simulated_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b_limited(reference_folder, fret_folder, learning_mode, spc):
    reference_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_1023_reference_08_18_2018'
    fret_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_fret'
    probe_design_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0567/DSGN0567_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
    barcode_list = [512, 128, 64, 32, 4, 2, 1]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    fret_transfer_matrix = np.zeros((simulation_per_code, 7, 7))
    probes = pd.read_csv(probe_design_filename, dtype = {'code': str})
    code_set = np.unique(probes.code.values)
    for i in range(simulation_per_code):
        fret_transfer_matrix[i,:,:] = calculate_fret_efficiency(fret_folder, 6 + 4*np.random.random())

    excitation_matrix = np.array([[1, 1, 0, 0, 1, 1, 1],
                                  [1, 1, 0, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        if code in code_set:
            numeric_code_list = np.array([int(a) for a in list(code)])
            simulated_spectra = np.zeros((simulation_per_code, nchannels))
            indices = [0,23,43,57,63]
            if numeric_code_list[6] == 1:
                error_scale = [0.25, 0.25, 0.35, 0.45]
            else:
                error_scale = [0.1, 0.25, 0.35, 0.45]
            for exc in range(4):
                relevant_fluorophores = numeric_code_list*excitation_matrix[exc, :]
                coefficients = np.zeros((simulation_per_code, 7))
                for i in range(simulation_per_code):
                    coefficients[i,:] = np.dot(fret_transfer_matrix[i,:,:], relevant_fluorophores)*relevant_fluorophores
                simulated_spectra_list = [coefficients[:,k][:,None]*np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,32:95] for k in range(nbit)]
                simulated_spectra[:,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)[:,indices[exc]:indices[exc+1]]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
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
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
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

    training_data_full = pd.concat([training_data, training_data_negative])
    scaler_full = preprocessing.StandardScaler().fit(training_data_full.values[:,0:63])
    training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:63])
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67], y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
    clf[0].fit(training_data_full_scaled[:,0:23], training_data_full.c1.values)
    clf[1].fit(training_data_full_scaled[:,23:43], training_data_full.c2.values)
    clf[2].fit(training_data_full_scaled[:,43:57], training_data_full.c3.values)
    clf[3].fit(training_data_full_scaled[:,57:63], training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 1, probability = True)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(scaler_full, '{}/reference_simulate_{}_DSGN0567_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_scaler.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf, '{}/reference_simulate_{}_DSGN0567_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_DSGN0567_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_DSGN0567_interaction_simulated_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_umap_transformed_biofilm_7b_limited(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        numeric_code_list = [int(a) for a in list(code)]
        subset_numeric_code = ''.join([str(numeric_code_list[x]) for x in [0,2,3,4,7,8,9]])
        if (numeric_code_list[6] == 0) and (numeric_code_list[5] == 0) and (numeric_code_list[1] == 0) and (subset_numeric_code in taxon_lookup.code.values):
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)[:,32:95]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.3+0.7*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[9] or numeric_code_list[2] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[9] or numeric_code_list[0] or numeric_code_list[2] or numeric_code_list[8] or numeric_code_list[7]
            ss_norm['c3'] = numeric_code_list[7] or numeric_code_list[8]
            ss_norm['c4'] = numeric_code_list[3] or numeric_code_list[4]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b).fit(training_data.iloc[:,0:67].values, y = training_data.code.values)
    clf = [svm.SVC() for i in range(5)]
    clf[0].fit(training_data.iloc[:,0:23].values, training_data.c1.values)
    clf[1].fit(training_data.iloc[:,23:43].values, training_data.c2.values)
    clf[2].fit(training_data.iloc[:,43:57].values, training_data.c3.values)
    clf[3].fit(training_data.iloc[:,57:63].values, training_data.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_select_DSGN0524_umap_transformed_biofilm_7b_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_select_DSGN0524_umap_transformed_biofilm_7b_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_select_DSGN0524_umap_transform_biofilm_7b.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_noise_free_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        simulated_spectra_norm = np.repeat(spectra_mean[None,:], 5**5, axis = 0)
        indices = [0,32,55,75,89,95]
        for k in range(0,5):
            simulated_spectra_norm[:,indices[k]:indices[k+1]] = (0.5+0.5*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
        simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
        ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        ss_norm['code'] = code
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP(n_neighbors = 200, metric = channel_cosine_intensity_normal).fit(training_data.drop(columns = ['code'], axis = 0), y = training_data.code.values)
    clf_umap = svm.SVC(C = 10, gamma = 0.5)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_check_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(clf_umap, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_excitation_adjusted_normalized_differentiated_umap_transformed(reference_folder, learning_mode, spc):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
        spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
        simulated_spectra = np.random.multivariate_normal(spectra_mean, spectra_cov, simulation_per_code)
        simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
        indices = [0,32,55,75,89,95]
        simulated_spectra_norm_adjusted_list = [simulated_spectra_norm]
        for k in range(0,5):
            simulated_spectra_adjusted = simulated_spectra_norm.copy()
            simulated_spectra_adjusted[:,indices[k]:indices[k+1]] = (0.7+0.3*np.random.random(simulated_spectra_adjusted.shape[0]))[:,None]*simulated_spectra_adjusted[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_adjusted/np.max(simulated_spectra_adjusted, axis = 1)[:,None]
            simulated_spectra_norm_adjusted_list.append(simulated_spectra_adjusted_norm)
        simulated_spectra_norm_all = np.vstack(simulated_spectra_norm_adjusted_list)
        ss_derivateive = np.diff(simulated_spectra_norm_all)
        ss_norm = pd.DataFrame(np.concatenate([simulated_spectra_norm_all, ss_derivateive], axis = 1))
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP().fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.1)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_{}_excitation_adjusted_normalized_umap_transform.pkl'.format(reference_folder, str(spc)))
    return

def load_training_data_simulate_normalized_select(reference_folder, learning_mode, spc, input_tab_filename):
    input_tab = pd.read_csv(input_tab_filename)
    mix_id = int(re.sub('mix_', '', re.search('mix_[0-9]*', input_tab_filename).group(0)))
    reference_list = input_tab.Barcodes.values
    files = glob.glob(reference_folder + '/'+ '*_avgint_norm.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        if enc in reference_list:
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
            csa_norm = spectra_mean/np.max(spectra_mean)
            simulated_spectra = np.random.multivariate_normal(csa_norm, spectra_cov, simulation_per_code)
            ss_norm = pd.DataFrame(simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None])
            ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP().fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.5)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_select_mix_{}_{}_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(mix_id), str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_select_mix_{}_{}_normalized_umap_transform.pkl'.format(reference_folder, str(mix_id), str(spc)))
    return

def load_training_data_simulate_normalized_select_excitation_adjusted(reference_folder, learning_mode, spc, input_tab_filename):
    input_tab = pd.read_csv(input_tab_filename)
    mix_id = int(re.sub('mix_', '', re.search('mix_[0-9]*', input_tab_filename).group(0)))
    reference_list = input_tab.Barcodes.values
    files = glob.glob(reference_folder + '/'+ '*_avgint_norm.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        if enc in reference_list:
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = 3*np.cov(pd.read_csv(files[i], header = None).transpose())
            csa_norm = spectra_mean/np.max(spectra_mean)
            simulated_spectra = np.random.multivariate_normal(csa_norm, spectra_cov, simulation_per_code)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,32,55,75,89,95]
            simulated_spectra_norm_adjusted_list = [simulated_spectra_norm]
            for k in range(0,5):
                simulated_spectra_adjusted = simulated_spectra_norm.copy()
                simulated_spectra_adjusted[:,indices[k]:indices[k+1]] = (0.7+0.3*np.random.random(simulated_spectra_adjusted.shape[0]))[:,None]*simulated_spectra_adjusted[:,indices[k]:indices[k+1]]
                simulated_spectra_adjusted_norm = simulated_spectra_adjusted/np.max(simulated_spectra_adjusted, axis = 1)[:,None]
                simulated_spectra_norm_adjusted_list.append(simulated_spectra_adjusted_norm)
            ss_norm = pd.DataFrame(np.vstack(simulated_spectra_norm_adjusted_list))
            ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
            training_data = training_data.append(ss_norm, ignore_index = True)
    umap_transform = umap.UMAP().fit(training_data.drop(columns = ['code'], axis = 1), y = training_data.code.values)
    clf = svm.SVC(C = 10, gamma = 0.5)
    clf.fit(umap_transform.embedding_, training_data.code.values)
    joblib.dump(clf, '{}/reference_simulate_select_mix_{}_{}_excitation_adjusted_normalized_umap_transformed_svc.pkl'.format(reference_folder, str(mix_id), str(spc)))
    joblib.dump(umap_transform, '{}/reference_simulate_select_mix_{}_{}_excitation_adjusted_normalized_umap_transform.pkl'.format(reference_folder, str(mix_id), str(spc)))
    return

def load_training_data_simulate_select(reference_folder, learning_mode, spc, input_tab_filename):
    input_tab = pd.read_csv(input_tab_filename)
    mix_id = int(re.sub('mix_', '', re.search('mix_[0-9]*', input_tab_filename).group(0)))
    reference_list = input_tab.Barcodes.values
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    nchannels = 95
    simulation_per_code = spc
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        if enc in reference_list:
            spectra_mean = np.average(pd.read_csv(files[i], header = None), axis = 0)
            spectra_cov = np.cov(pd.read_csv(files[i], header = None).transpose())
            csa_norm = spectra_mean/np.max(spectra_mean)
            simulated_spectra = np.random.multivariate_normal(csa_norm, spectra_cov, simulation_per_code)
            ss_norm = pd.DataFrame(simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None])
            ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
            training_data = training_data.append(ss_norm, ignore_index = True)
    if learning_mode == 'SVM':
        clf = svm.SVC()
    elif learning_mode == 'RFC':
        clf = RandomForestClassifier()
    clf.fit(training_data.drop(columns = ['code'], axis = 1), training_data['code'])
    joblib.dump(clf, '{}/reference_simulate_select_mix_{}_{}.pkl'.format(reference_folder, str(mix_id), str(spc)))
    return

def load_training_data(reference_folder, learning_mode):
    files = glob.glob(reference_folder + '/'+ '*_avgint.csv')
    nbit = 10
    training_data = pd.DataFrame()
    for i in range(0, len(files)):
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', files[i]).group(0)))
        spectra = pd.read_csv(files[i], header = None)
        ss_norm = pd.DataFrame(spectra.values/np.max(spectra.values, axis = 1)[:,None])
        ss_norm['code'] = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        training_data = training_data.append(ss_norm, ignore_index = True)
    if learning_mode == 'SVM':
        clf = svm.SVC(C = 10, gamma = 0.5)
    elif learning_mode == 'RFC':
        clf = RandomForestClassifier()
    clf.fit(training_data.drop(columns = ['code'], axis = 1), training_data['code'])
    joblib.dump(clf, '{}/reference_all.pkl'.format(reference_folder))
    return

def main():
    # parameters are hard coded within the function
    load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b_limited('','','',2000)

if __name__ == '__main__':
    main()
