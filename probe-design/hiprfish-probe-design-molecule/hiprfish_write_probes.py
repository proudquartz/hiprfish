import os
import re
import glob
import argparse
import subprocess
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC, generic_dna
from matplotlib import pyplot as plt
from Bio.Blast.Applications import NcbiblastnCommandline

os.environ['OMP_NUM_THREADS'] = "16"
os.environ['MKL_NUM_THREADS'] = "16"
os.environ['DASK_NUM_THREADS'] = "16"

###############################################################################################################
# HiPR-FISH-strain : combine probes
###############################################################################################################

def write_design_level_target_probes(probes, probes_dir):
    design_level = probes.design_level.values[0]
    design_target = probes.design_target.values[0]
    design_level_target_dir = '{}/{}/{}'.format(probes_dir, design_level, design_target)
    if not os.path.exists(design_level_target_dir):
        os.makedirs(design_level_target_dir)
    for i in range(probes.shape[0]):
        probe_seq = SeqRecord(Seq(probes.seq.values[i]).reverse_complement(), id = '{}_{}_{}'.format(design_level, design_target, probes.probe_id.values[i]), description = '')
        probe_fasta_filename = '{}/{}/{}/{}_{}_{}.fasta'.format(probes_dir, design_level, design_target, design_level, design_target, probes.probe_id.values[i])
        SeqIO.write(probe_seq, probe_fasta_filename, 'fasta')
    return(0)

def write_probes(probes_summary_filename, probes_dir, n_workers):
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = "0")
    client = Client(cluster)
    design_level_list = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    for design_level in design_level_list:
        print('Writing probes at the {} level...'.format(design_level))
        probes = dd.read_hdf(probes_summary_filename, '{}/*'.format(design_level), mode = 'r')
        probes_write = probes.groupby(['design_level', 'design_target']).apply(write_design_level_target_probes, probes_dir, meta = ('int'))
        probes_write.compute()
    client.close()
    cluster.close()
    return(0)

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('probes_summary_filename', type = str, help = 'Input FASTA file containing 16S sequences')
    parser.add_argument('-n', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input FASTA file containing 16S sequences')
    args = parser.parse_args()
    primer3_dir = os.path.split(args.probes_summary_filename)[0]
    data_dir = os.path.split(primer3_dir)[0]
    probes_dir = '{}/probes'.format(data_dir)
    evaluate_dir = '{}/evaluate'.format(data_dir)
    probe_write_complete_filename = '{}/probes_write_complete.txt'.format(probes_dir)
    if not os.path.exists(probes_dir):
        os.makedirs(probes_dir)
    if not os.path.exists(evaluate_dir):
        os.makedirs(evaluate_dir)
    probe_write_status = write_probes(args.probes_summary_filename, probes_dir, args.n_workers)
    print('Writing probe write complete file...')
    file = open(probe_write_complete_filename, 'w')
    file.write('Probe write is done.')
    file.close()
    print('Finished with probe write')
    return

if __name__ == '__main__':
    main()
