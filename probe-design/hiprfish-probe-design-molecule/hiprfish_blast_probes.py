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

@dask.delayed
def blast_taxon_individual_probe(filename, blast_database):
    blast_output_filename = '{}.blast.out'.format(filename)
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq'
    return_code = subprocess.check_call(['blastn', '-db', blast_database, '-query', filename, '-out', blast_output_filename, '-outfmt', out_format, '-task', 'blastn-short', '-max_hsps', '1', '-max_target_seqs', '100000', '-strand', 'minus', '-evalue', '100', '-num_threads', '1'])
    return(return_code)

def blast_design_level_probes(design_level_directory, blast_database, n_workers):
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1)
    client = Client(cluster)
    probe_filenames = glob.glob('{}/*/*.fasta'.format(design_level_directory))
    pd.DataFrame(probe_filenames).to_csv('{}/probe_fasta_list.csv'.format(design_level_directory), index = None, header = None)
    probes_fasta_list = pd.read_csv('{}/probe_fasta_list.csv'.format(design_level_directory), header = None)
    index_list = np.arange(0, probes_fasta_list.shape[0], 10000)
    index_list = np.append(index_list, probes_fasta_list.shape[0])
    for i in range(index_list.shape[0] - 1):
        print('Blasting probes from group {} out of {} groups'.format(i+1, index_list.shape[0]-1))
        start_index = index_list[i]
        end_index = index_list[i+1]
        return_code_list = []
        for idx in range(start_index, end_index):
            return_code = blast_taxon_individual_probe(probes_fasta_list.iloc[idx,0], blast_database)
            return_code_list.append(return_code)
        return_code_total = dask.delayed(sum)(return_code_list)
        result = return_code_total.compute()
    client.close()
    cluster.close()
    return(0)

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Blast FISH probes designed for a complex microbial community')
    parser.add_argument('design_level_dir', type = str, help = 'Input file containing all probes designed by primer3')
    parser.add_argument('oriented_pacbio_filename', type = str, help = 'Input FASTA file containing full length 16S sequences of the complex microbial community')
    parser.add_argument('-n', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input FASTA file containing 16S sequences')
    args = parser.parse_args()

    design_level = os.path.basename(args.design_level_dir)
    blast_complete_filename = '{}/{}_probes_blast_complete.txt'.format(args.design_level_dir, design_level)
    return_code_total = blast_design_level_probes(args.design_level_dir, args.oriented_pacbio_filename, args.n_workers)
    file = open(blast_complete_filename, 'w')
    file.write('Probe blast finished with total return code {}.'.format(return_code_total))
    file.close()
    return

if __name__ == '__main__':
    main()
