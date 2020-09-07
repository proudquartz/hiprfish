import argparse
import pandas as pd
import subprocess
import os
import multiprocessing
import glob
import re
import itertools
import numpy as np
import random
from ete3 import NCBITaxa
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC, generic_dna
from Bio.Blast.Applications import NcbiblastnCommandline
from joblib import Parallel, delayed

###############################################################################################################
# HiPR-FISH-strain : initialize probe design
###############################################################################################################

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('probe_evaluation_filenames_list', type = str, help = 'Input FASTA file containing 16S sequences')
    args = parser.parse_args()
    log_dir, probe_evaluation_filenames_basename = os.path.split(args.probe_evaluation_filenames_list)
    design_level = re.sub('_probe_evaluation_filenames.csv', '', probe_evaluation_filenames_basename)
    sample_dir = os.path.split(log_dir)[0]
    selection_dir = '{}/selection/{}'.format(sample_dir, design_level)
    if not os.path.exists(selection_dir):
        os.makedirs(selection_dir)
    probe_evaluation_files = pd.read_csv(args.probe_evaluation_filenames_list, header = None)
    for f in probe_evaluation_files.loc[:,0].values:
        probe_evaluation_filenames_basename = os.path.basename(f)
        design_target = re.sub('_probe_evaluation_complete.txt', '', probe_evaluation_filenames_basename)
        probe_seletion_initiation_filename = '{}/{}_probe_selection_initiation.txt'.format(selection_dir, design_target)
        file = open(probe_seletion_initiation_filename, 'w')
        file.write('Probe evaluation for {} is complete'.format(f))
        file.close()
    return

if __name__ == '__main__':
    main()
