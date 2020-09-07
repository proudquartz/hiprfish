import argparse
import pandas as pd
import os
import glob
import re
import numpy as np
import random
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC, generic_dna
from Bio.Blast.Applications import NcbiblastnCommandline
from matplotlib import pyplot as plt

###############################################################################################################
# HiPR-FISH Probe Design Pipeline
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################

def add_spacer(taxon_best_probes, output_filename, consensus_directory):
    probes = pd.read_csv(taxon_best_probes)
    probes['seqrcsa'] = ''
    for i in range(0, probes.shape[0]):
        probe_seq = Seq(probes['seq'][i], generic_dna)
        rrna_file = consensus_directory + '/' + str(probes['target_taxon'][i]) + '.consensus.fasta'
        rrna_file_length = sum(1 for record in SeqIO.parse(rrna_file, 'fasta'))
        if rrna_file_length > 1:
            cluster = re.sub('.*_', '', probes['target_taxon_full'][i])
            rrna_seq = SeqIO.to_dict(SeqIO.parse(rrna_file, 'fasta'))[cluster].seq
        else:
            rrna_seq = SeqIO.read(rrna_file, 'fasta').seq
        sstart = probes['p_start'][i]
        probe_length = probes['ln'][i]
        send = sstart + probe_length - 1
        right_spacer = rrna_seq[sstart - 2] + rrna_seq[sstart - 3] + rrna_seq[sstart - 4]
        left_spacer = rrna_seq[send + 2] + rrna_seq[send + 1] + rrna_seq[send]
        probe_seq_rcsa = str(left_spacer).upper() + str(probe_seq.reverse_complement()) + str(right_spacer).upper()
        probes.ix[i, 'seqrcsa'] = probe_seq_rcsa
    probes.loc[:,'quadg'] = (probes['seqrcsa'].str.upper().str.count('GGGG')) + (probes['seqrcsa'].str.upper().str.count('GGGGG'))
    probes = probes.loc[probes['quadg'] == 0, :]
    probes.to_csv(output_filename, index = False)
    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    # input blast filename
    parser.add_argument('input_filename', type = str, help = 'Input file containing blast results')
    parser.add_argument('output_filename', type = str, help = 'Input file containing blast results')
    parser.add_argument('consensus_directory', type = str, help = 'Input file containing blast results')

    args = parser.parse_args()
    add_spacer(args.input_filename, args.output_filename, args.consensus_directory)
    return

if __name__ == '__main__':
    main()
