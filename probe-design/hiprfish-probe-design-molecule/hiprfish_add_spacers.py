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
from dask.distributed import LocalCluster, Client
import dask.dataframe as dd
from matplotlib import pyplot as plt

###############################################################################################################
# HiPR-FISH Probe Design Pipeline
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################

def get_target_taxon_rrna_seqs(oriented_fasta_list, probe_blast_filtered):
    molecule_id_list = probe_blast_filtered.molecule_id.values
    rrna_seqs = pd.DataFrame([r for r in oriented_fasta_list if r[0] in molecule_id_list])
    rrna_seqs.columns = ['molecule_id', 'rrna_seq']
    rrna_seqs['molecule_id'] = pd.Categorical(rrna_seqs['molecule_id'], molecule_id_list)
    rrna_seqs_sorted = rrna_seqs.sort_values(by = 'molecule_id').reset_index().drop(columns = ['index'])
    return(rrna_seqs_sorted)

def get_right_spacers(rrna_seq, sstart):
    if sstart >= 4:
        right_spacers = rrna_seq[sstart - 2] + rrna_seq[sstart - 3] + rrna_seq[sstart - 4]
    elif sstart == 3:
        right_spacers = rrna_seq[sstart - 2] + rrna_seq[sstart - 3]
    elif sstart == 2:
        right_spacers = rrna_seq[sstart - 2]
    else:
        right_spacers = ''
    return(right_spacers)

def get_left_spacers(rrna_seq, send):
    send_dif = len(rrna_seq) - send
    if send_dif >= 3:
        left_spacers = rrna_seq[send + 2] + rrna_seq[send + 1] + rrna_seq[send]
    elif send_dif == 2:
        left_spacers = rrna_seq[send + 1] + rrna_seq[send]
    elif send_dif == 1:
        left_spacers = rrna_seq[send]
    else:
        left_spacers = ''
    return(left_spacers)

def get_spacer_combinations(df):
    print(df)
    return('{}_{}'.format(df['left_spacer'], df['right_spacer']))

def add_spacer(taxon_best_probes, probe_evaluation_dir, oriented_fasta_list, blast_lineage, max_continuous_homology):
    output_filename = re.sub('.csv', '_sa.csv', taxon_best_probes)
    target_taxon = re.sub('_probe_selection.csv', '', os.path.basename(taxon_best_probes))
    probes = pd.read_csv(taxon_best_probes)
    probe_seq_rcsa_list = []
    for probe_idx in probes.probe_id.values:
        probe_blast = pd.read_csv('{}/{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, target_taxon, probe_idx))
        probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
        probe_seq = Seq(probes.loc[probes.probe_id.values == probe_idx,'seq'].values[0], generic_dna)
        sstart = probe_blast_filtered.molecule_end.values
        send = probe_blast_filtered.molecule_start.values
        # right_spacer = rrna_seq[sstart - 2] + rrna_seq[sstart - 3] + rrna_seq[sstart - 4]
        # left_spacer = rrna_seq[send + 2] + rrna_seq[send + 1] + rrna_seq[send]
        rrna_seqs = get_target_taxon_rrna_seqs(oriented_fasta_list, probe_blast_filtered)
        for j in range(rrna_seqs.shape[0]):
            rrna_seqs.loc[j,'left_spacers'] = get_left_spacers(rrna_seqs.loc[j, 'rrna_seq'], send[j])
            rrna_seqs.loc[j,'right_spacers'] = get_right_spacers(rrna_seqs.loc[j, 'rrna_seq'], sstart[j])
        rrna_seqs['combined_spacers'] = rrna_seqs['left_spacers'] + '_' + rrna_seqs['right_spacers']
        unique_spacer_combinations = rrna_seqs.combined_spacers.drop_duplicates()
        probe_seq_rcsa = pd.DataFrame(columns = ['probe_id', 'seqrcsa', 'spacer_combination'])
        for j in range(unique_spacer_combinations.shape[0]):
            spacer_combination = unique_spacer_combinations.values[j]
            left_spacer, right_spacer = spacer_combination.split('_')
            probe_seq_rcsa.loc[j, 'seqrcsa'] = str(left_spacer).upper() + str(probe_seq.reverse_complement()) + str(right_spacer).upper()
            probe_seq_rcsa.loc[j, 'spacer_combination'] = 'spacer_combination_{}'.format(j)
            probe_seq_rcsa.loc[j, 'spacer_combination_seq'] = spacer_combination
            probe_seq_rcsa.loc[j, 'probe_id'] = probe_idx
            probe_seq_rcsa.loc[j, 'probe_id_full'] = '{}_sc_{}'.format(probe_idx, j)
        probe_seq_rcsa_list.append(probe_seq_rcsa)
    probe_seq_rcsa_list = pd.concat(probe_seq_rcsa_list)
    probes = probes.merge(probe_seq_rcsa_list, on = 'probe_id', how = 'right')
    probes.loc[:,'quadg'] = (probes['seqrcsa'].str.upper().str.count('GGGG')) + (probes['seqrcsa'].str.upper().str.count('GGGGG'))
    probes = probes.loc[probes['quadg'] == 0, :]
    probes.to_csv(output_filename, index = False)
    return(0)

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    # input blast filename
    parser.add_argument('design_dir', type = str, help = 'Input file containing blast results')
    parser.add_argument('-n_workers', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input file containing blast results')
    parser.add_argument('-per', '--probe_evaluation_dir', dest = 'probe_evaluation_dir', type = str, help = 'Input file containing blast results')
    parser.add_argument('-off', '--oriented_fasta_filename', dest = 'oriented_fasta_filename', type = str, help = 'Input file containing blast results')
    parser.add_argument('-blf', '--blast_lineage_filename', dest = 'blast_lineage_filename', type = str, help = 'Input file containing blast results')
    parser.add_argument('-mch', '--max_continuous_homology', dest = 'max_continuous_homology', type = int, default = 14, help = 'Input file containing blast results')

    args = parser.parse_args()
    cluster = LocalCluster(n_workers = args.n_workers, threads_per_worker = 1, memory_limit = '16GB')
    client = Client(cluster)
    taxon_best_probes_filenames_list = pd.DataFrame(glob.glob('{}/*_probe_selection.csv'.format(args.design_dir)), columns = ['probe_selection_filename'])
    blast_lineage = pd.read_table(args.blast_lineage_filename, dtype = str)
    oriented_fasta_list = [[record.id, str(record.seq)] for record in SeqIO.parse(args.oriented_fasta_filename, 'fasta')]
    index_list = np.arange(0, taxon_best_probes_filenames_list.shape[0], 10000)
    index_list = np.append(index_list, taxon_best_probes_filenames_list.shape[0])
    for i in range(len(index_list) - 1):
        taxon_best_probes_filenames_list_sub = dd.from_pandas(taxon_best_probes_filenames_list.iloc[index_list[i]:index_list[i+1],:], npartitions = 100)
        spacer_addition = taxon_best_probes_filenames_list_sub.probe_selection_filename.apply(add_spacer, args = (args.probe_evaluation_dir, oriented_fasta_list, blast_lineage, args.max_continuous_homology,), meta = ('int'))
        spacer_addition.compute()
        # add_spacer(args.input_filename, args.output_filename, args.probe_evaluation_dir, args.oriented_fasta_filename, args.blast_lineage_filename, args.max_continuous_homology)
    probe_spacer_addition_complete_filename = '{}/{}_probe_spacer_addition_complete.txt'.format(args.design_dir, os.path.basename(args.design_dir))
    file = open(probe_spacer_addition_complete_filename, 'w')
    file.write('Probe spacer addition is complete.')
    file.close()
    client.close()
    cluster.close()
    return

if __name__ == '__main__':
    main()
