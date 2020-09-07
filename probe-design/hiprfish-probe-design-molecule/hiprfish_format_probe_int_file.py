import argparse
import subprocess
import os
from Bio.Blast.Applications import NcbiblastnCommandline
import re
import glob
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster


###############################################################################################################
# HiPR-FISH-strain : design probes
###############################################################################################################

def format_probe_int_file(primer3_int_filename):
    probe_csv_filename = re.sub('.int', '_probes.csv', primer3_int_filename)
    probes = pd.read_table(primer3_int_filename, skiprows = 3, header = None, delim_whitespace = True)
    probes.columns = ['probe_id', 'seq', 'start', 'length', 'N', 'GC', 'Tm', 'self_any_th', 'self_end_th', 'hair-pin', 'quality']
    probes['source'] = probe_csv_filename
    probes.to_csv(probe_csv_filename, index = None)
    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Blast FISH probes designed for a complex microbial community')

    parser.add_argument('primer3_int_filename', type = str, help = 'Input FASTA file containing full length 16S sequences of the complex microbial community')

    args = parser.parse_args()
    format_probe_int_file(args.primer3_int_filename)
    return

if __name__ == '__main__':
    main()
