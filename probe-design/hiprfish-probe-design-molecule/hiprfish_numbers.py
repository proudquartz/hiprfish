import argparse
import subprocess
import os
from Bio.Blast.Applications import NcbiblastnCommandline
import re
import glob
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster


###############################################################################################################
# HiPR-FISH-strain : design probes
###############################################################################################################

@dask.delayed
def get_file_line_number(f):
    blast_result = pd.read_csv(f)
    return(blast_result.shape[0])

def get_numbers(data_folder):
    data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_7/'
    design_id = 'DSGN0673'
    sample = '04_02_2017'
    target_rank = 'genus'
    genus_list = glob.glob('{}/{}/probes/genus/*'.format(data_folder, sample))
    probe_fasta_files = glob.glob('{}/{}/probes/genus/*/*.fasta'.format(data_folder, sample))
    probe_evaluation_files = glob.glob('{}/{}/evaluate/genus/*/*.csv.gz'.format(data_folder, sample))
    index_list = np.arange(0, len(probe_evaluation_files), 1000)
    index_list = np.append(index_list, len(probe_evaluation_files))
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    probe_evaluation_file_length_full = []
    for i in range(len(index_list) - 1):
        print(i)
        probe_evaluation_file_length = []
        for j in range(int(index_list[i]), int(index_list[i+1])):
            file_line_number = get_file_line_number(probe_evaluation_files[j])
            probe_evaluation_file_length.append(file_line_number)
        probe_evaluation_file_length_compute = dask.delayed(np.sum)(probe_evaluation_file_length).compute()
        probe_evaluation_file_length_full.append(probe_evaluation_file_length_compute)
    probe_evaluation_file_total_length = np.sum(probe_evaluation_file_length_full)

    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Blast FISH probes designed for a complex microbial community')

    parser.add_argument('data_folder', type = str, help = 'Input FASTA file containing full length 16S sequences of the complex microbial community')

    args = parser.parse_args()

    return

if __name__ == '__main__':
    main()
