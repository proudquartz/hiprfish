import argparse
import subprocess
import os
from Bio.Blast.Applications import NcbiblastnCommandline
import re
import glob
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client

###############################################################################################################
# HiPR-FISH-strain : design probes
###############################################################################################################

def design_probes(primer3_input_filename, primer3_settings_filename, probes_summary_dir):
    primer3_output_filename = re.sub('_primer3_input.txt', '_primer3_output.txt', primer3_input_filename)
    primer3_input_basename = re.sub('_primer3_input.txt', '', os.path.basename(primer3_input_filename))
    return_code = subprocess.check_call(['/programs/primer3-2.3.5/src/primer3_core', '-p3_settings_file', primer3_settings_filename, '-output', primer3_output_filename, '-format_output', primer3_input_filename])
    probe_int_filename = '{}/{}.int'.format(probes_summary_dir, primer3_input_basename)
    probe_csv_filename = '{}/{}_probes.csv'.format(probes_summary_dir, primer3_input_basename)
    probes = pd.read_table(probe_int_filename, skiprows = 3, header = None, delim_whitespace = True)
    probes.columns = ['probe_id', 'seq', 'start', 'length', 'N', 'GC', 'Tm', 'self_any_th', 'self_end_th', 'hair-pin', 'quality']
    probes['source'] = probe_csv_filename
    probes.to_csv(probe_csv_filename, index = None)
    return(0)

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Blast FISH probes designed for a complex microbial community')

    parser.add_argument('primer3_dir', type = str, help = 'Input FASTA file containing full length 16S sequences of the complex microbial community')
    parser.add_argument('-n_workers', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input FASTA file containing full length 16S sequences of the complex microbial community')

    args = parser.parse_args()
    sample_dir = os.path.split(args.primer3_dir)[0]
    probe_design_complete_filename = '{}/log/probe_design_complete.txt'.format(sample_dir)
    probes_summary_dir = '{}/probes_summary'.format(sample_dir)
    primer3_settings_filename = '{}/primer3_settings.txt'.format(args.primer3_dir)
    os.chdir(probes_summary_dir)
    primer3_input_filenames = pd.DataFrame(glob.glob('{}/*_primer3_input.txt'.format(args.primer3_dir)), columns = ['primer3_input_filename'])
    index_list = np.arange(0, primer3_input_filenames.shape[0], 10000)
    index_list = np.append(index_list, primer3_input_filenames.shape[0])
    cluster = LocalCluster(n_workers = args.n_workers, threads_per_worker = 1, memory_limit = '16GB')
    client = Client(cluster)
    for i in range(len(index_list) - 1):
        primer3_input_filenames_sub = dd.from_pandas(primer3_input_filenames.iloc[index_list[i]:index_list[i+1],:], npartitions = 100)
        probe_design = primer3_input_filenames_sub.primer3_input_filename.apply(design_probes, args = (primer3_settings_filename, probes_summary_dir,), meta = ('int'))
        probe_design.compute()
    file = open(probe_design_complete_filename, 'w')
    file.write('Probe design is complete.')
    file.close()
    client.close()
    cluster.close()
    return

if __name__ == '__main__':
    main()
