import argparse
import subprocess
import os
from Bio.Blast.Applications import NcbiblastnCommandline
import re
import glob
import pandas as pd
import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster
import dask.bag as db
import numpy as np

###############################################################################################################
# HiPR-FISH : blast probes
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################



###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Blast FISH probes designed for a complex microbial community')
    parser.add_argument('taxon_evaluate_dir', type = str, help = 'Input file containing all probes designed by primer3')
    args = parser.parse_args()

    probe_evaluation_filenames = glob.glob('{}/*_probe_evaluation_complete.txt'.format(args.taxon_evaluate_dir))
    evaluate_dir, design_level = os.path.split(args.taxon_evaluate_dir)
    sample_dir = os.path.split(evaluate_dir)[0]
    log_dir = '{}/log'.format(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    probe_evaluation_list_filenames = '{}/{}_probe_evaluation_filenames.csv'.format(log_dir, design_level)
    probe_evaluation_filenames = pd.DataFrame(probe_evaluation_filenames)
    probe_evaluation_filenames.to_csv(probe_evaluation_list_filenames, index = None, header = None)
    return

if __name__ == '__main__':
    main()
