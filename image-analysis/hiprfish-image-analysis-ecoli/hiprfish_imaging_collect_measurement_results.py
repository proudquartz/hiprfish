
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import argparse
import pandas as pd
import numpy as np

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

def collect_reference_measurement_results(data_dir, simulation_table, output_filename):
    print('Loading samples table: %s' % (simulation_table))
    sim_tab = pd.read_csv(simulation_table)
    sim_tab['NCells'] = 0
    sim_tab['BarcodeComplexity'] = 0
    sim_tab['Barcodes'] = 0
    sim_tab['MostCommonSingleErrorBit'] = 0
    print('Loading result files:')
    for i in range(0, sim_tab.shape[0]):
        print('Saving collected results to %s...' % (output_filename))
        image_folder = sim_tab.SAMPLE.values[i]
        image_name = sim_tab.IMAGES.values[i]
        enc = int(re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0)))
        sim_tab.loc[i, 'Barcodes'] = enc
        spectra_measurement_filename = '{}/{}/{}_avgint.csv'.format(data_dir, image_folder, image_name)
        spectra_identification_filename = '{}/{}/{}_cell_ids.txt'.format(data_dir, image_folder, image_name)
        sim_tab.loc[i, 'BarcodeComplexity'] = np.sum(list(map(int, list(re.sub('0b', '', format(enc, '#0' + str(10+2) + 'b'))))))
        if os.path.exists(spectra_measurement_filename):
            spectra_measurement = pd.read_csv(spectra_measurement_filename, header = None)
            sim_tab.loc[i, 'NCells'] = spectra_measurement.shape[0]
        else:
            print('Sample result file %s does not exist' % spectra_measurement_filename)
        if os.path.exists(spectra_identification_filename):
            cell_ids = pd.read_csv(spectra_identification_filename, header = None, dtype = str)
            cell_ids.columns = ['Barcodes']
            binary_barcode = re.sub('0b', '', format(enc, '#0' + str(10+2) + 'b'))
            error_rate = 1 - np.sum(cell_ids.Barcodes.values == binary_barcode)/cell_ids.shape[0]
            if error_rate == 0:
                sim_tab.loc[i, 'ErrorRate'] = 1/cell_ids.shape[0]
                sim_tab.loc[i, 'ErrorRateUpperLimit'] = 'T'
            else:
                sim_tab.loc[i, 'ErrorRate'] = error_rate
                sim_tab.loc[i, 'ErrorRateUpperLimit'] = 'F'
            cell_ids_wrong = cell_ids.loc[cell_ids.Barcodes.values != binary_barcode]
            one_bit_error = 0
            two_bit_error = 0
            multiple_bit_error = 0
            for j in range(cell_ids_wrong.shape[0]):
                measured_barcode_list = list(cell_ids_wrong.Barcodes.values[j])
                error_barcode = [int(measured_barcode_list[k]) - int(list(binary_barcode)[k]) for k in range(10)]
                error_code = np.sum([np.abs(x) for x in error_barcode])
                if error_code == 1:
                    one_bit_error += 1
                elif error_code == 2:
                    two_bit_error += 1
                else:
                    multiple_bit_error += 1
            sim_tab.loc[i, 'OneBitError'] = one_bit_error/cell_ids.shape[0]
            sim_tab.loc[i, 'TwoBitError'] = two_bit_error/cell_ids.shape[0]
            sim_tab.loc[i, 'MultipleBitError'] = multiple_bit_error/cell_ids.shape[0]
            sim_tab.to_csv(output_filename, index = False, header = True)
    return

def collect_mix_measurement_results(data_dir, simulation_table, output_filename):
    print('Loading samples table: %s' % (simulation_table))
    sim_tab = pd.read_csv(simulation_table)
    sim_tab['NCells'] = 0
    sim_tab['FOV'] = 0
    abundance_tab = pd.DataFrame(np.arange(1,1024), columns = ['Barcodes'])
    print('Loading result files:')
    for i in range(0, sim_tab.shape[0]):
        print('Saving collected results to %s...' % (output_filename))
        image_folder = sim_tab.SAMPLE.values[i]
        image_name = sim_tab.IMAGES.values[i]
        fov = int(re.sub('fov_', '', re.search('fov_[0-9]*', image_name).group(0)))
        sim_tab.loc[i, 'FOV'] = fov
        spectra_measurement_filename = '{}/{}/{}_avgint.csv'.format(data_dir, image_folder, image_name)
        spectra_identification_filename = '{}/{}/{}_cell_ids.txt'.format(data_dir, image_folder, image_name)
        if os.path.exists(spectra_measurement_filename):
            spectra_measurement = pd.read_csv(spectra_measurement_filename, header = None)
            sim_tab.loc[i, 'NCells'] = spectra_measurement.shape[0]
        else:
            print('Sample result file %s does not exist' % spectra_measurement_filename)
        if os.path.exists(spectra_identification_filename):
            cell_ids = pd.read_csv(spectra_identification_filename, header = None, dtype = str)
            cell_ids.columns = ['Barcodes']
            # abundance = cell_ids.groupby('Barcodes').count()
            fov_abundance = pd.DataFrame(cell_ids.Barcodes.value_counts())
            fov_abundance.columns = ['FOV{}'.format(str(i+1))]
            fov_abundance['Barcodes'] = [int(x, 2) for x in fov_abundance.index.tolist()]
            abundance_tab = abundance_tab.merge(fov_abundance, on = 'Barcodes', how = 'left').fillna(0)
        abundance_filename = re.sub('.csv', '_abundance.csv', output_filename)
        sim_tab.to_csv(output_filename, index = False, header = True)
        abundance_tab.to_csv(abundance_filename, index = False, header = True)
    return



###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Collect summary statistics of HiPRFISH probes for a complex microbial community')

    # data directory
    parser.add_argument('data_dir', type = str, help = 'Directory of the data files')

    # input simulation table
    parser.add_argument('simulation_table', type = str, help = 'Input csv table containing simulation information')

    # output simulation results table
    parser.add_argument('simulation_results', type = str, help = 'Output csv table containing simulation results')

    parser.add_argument('-t', '--type', dest = 'type', type = str, default = 'R')
    args = parser.parse_args()

    if args.type == 'R':
        collect_reference_measurement_results(args.data_dir, args.simulation_table, args.simulation_results)
    else:
        collect_mix_measurement_results(args.data_dir, args.simulation_table, args.simulation_results)

if __name__ == '__main__':
    main()
