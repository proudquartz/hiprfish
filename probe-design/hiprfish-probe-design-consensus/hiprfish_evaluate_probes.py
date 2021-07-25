
import os
import re
import glob
import random
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from ete3 import NCBITaxa
from SetCoverPy import setcover
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Blast.Applications import NcbiblastnCommandline
import time
import smtplib
from joblib import Parallel, delayed
import tables
os.environ['OMP_NUM_THREADS'] = '1'
###############################################################################################################
# HiPR-FISH Probe Design Pipeline
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################

def send_confirmation():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('proudquartzprogramcheck@gmail.com', 'Shprogramcheck')
    server.sendmail('proudquartzprogramcheck@gmail.com', 'hs673@cornell.edu', 'Subject: {}\n\n{}'.format('Program finished', ''))

def calculate_max_continuous_homology(df):
    qseq = df['qseq']
    sseq = df['sseq']
    snp_indices = [i for i in range(len(list(qseq))) if qseq[i] != sseq[i]]
    seq_homology = [True if i not in snp_indices else False for i in range(0, len(list(qseq)))]
    mch = np.max([sum( 1 for _ in group ) for key, group in itertools.groupby(seq_homology ) if key])
    return(mch)

def mch_test(df):
    qseq = df['qseq']
    sseq = df['sseq']
    if qseq != sseq:
        snp_indices = np.where(np.array(list(qseq)) != np.array(list(sseq)))[0]
        snp_indices = np.insert(snp_indices, [0, snp_indices.size], [1, len(qseq)])
        mch = np.max(np.diff(snp_indices))
    else:
        mch = len(qseq)
    return(mch)

def mch_test_v4(df, blast_output_filename):
    qseq_array = df.qseq.values
    sseq_array = df.sseq.values
    mch_array = np.zeros(len(qseq_array))
    for i in range(len(qseq_array)):
        qseq = qseq_array[i]
        sseq = sseq_array[i]
        try:
            if qseq != sseq:
                snp_indices = np.where(np.array(list(qseq)) != np.array(list(sseq)))[0]
                diffs = np.diff(snp_indices)
                mch_array[i] = np.max(np.append(diffs,[snp_indices[0], len(qseq) - 1 - snp_indices[-1]]))
            else:
                mch_array[i] = len(qseq)
        except TypeError:
            num_line = subprocess.call(['wc', '-l', blast_output_filename])
            print(num_line)
            print(blast_output_filename)
            print(df.iloc[i,:])
    return(mch_array)

def sub_slash(str):
    return(re.sub('/', '_', str))

def get_blast_lineage_slim(blast_lineage_filename, otu):
    if otu == 'F':
        blast_lineage_df = pd.read_csv(blast_lineage_filename, dtype = {'staxids':str}, sep = '\t')
        lineage_columns = ['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain']
        blast_lineage_slim = blast_lineage_df[lineage_columns]
    else:
        blast_lineage_df = pd.read_csv(blast_lineage, header = None, dtype = {'staxids':str}, sep = '\t')
        blast_lineage_df.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
        blast_lineage_filtered = blast_lineage_df[blast_lineage_df['rec_type'] != 'C']
        blast_lineage_filtered['target_taxon'] = 'Cluster' + blast_lineage_filtered['cluster_num'].astype(str)
        blast_lineage_filtered['molecule_id'] = blast_lineage_filtered['query_label']
        blast_lineage_filtered[blast_lineage_filtered['rec_type'] == 'S']['molecule_id'] = blast_lineage_filtered['target_label']
        lineage_columns = ['molecule_id', 'target_taxon']
        blast_lineage_slim = blast_lineage_filtered[lineage_columns]
    return(blast_lineage_slim)

def evaluate_taxon_probes(taxon_probe_directory, blast_lineage_slim, store):
    probe_filenames = glob.glob(taxon_probe_directory + '/*.fasta')
    # Parallel(n_jobs = 4)(delayed(blast_taxon_probes)(filename, blast_database, filename + '.blast.out') for filename in probe_filenames)
    # for filename in probe_filenames:
    #     blast_output_filename = filename + '.blast.out'
    #     blast_taxon_probes(filename, blast_database, blast_output_filename)
    for filename in probe_filenames:
        blast_output_filename = filename + '.blast.out'
        try:
            chunk = pd.read_csv(blast_output_filename, header = None, sep = '\t')
            chunk.columns = ['probe_id', 'molecule_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'probe_start', 'probe_end', 'molecule_start', 'molecule_end', 'evalue', 'bitscore', 'staxids', 'qseq', 'sseq']
            chunk['mch'] = mch_test_v4(chunk, blast_output_filename)
            chunk['molecule_id'] = chunk.molecule_id.apply(sub_slash)
            chunk = chunk.merge(blast_lineage_slim, on = 'molecule_id', how = 'left')
            chunk_name = re.search('probe_[0-9]*', filename).group(0)
            store.append(chunk_name, chunk, index=False, data_columns = True, min_itemsize = {'sseq': 30, 'qseq': 30}, complib = 'lzo', complevel = 9)
        except pd.errors.EmptyDataError:
            pass

def evaluate_probes(similarity_directory, taxon, blast_lineage_filename):
    taxon_probe_directory = similarity_directory + '/primer3/' + taxon
    probe_blast_hdf5_filename = similarity_directory + '/blast/' + taxon + '.probe.evaluation.h5'
    blast_lineage_slim = get_blast_lineage_slim(blast_lineage_filename, 'F')
    if not os.path.exists(probe_blast_hdf5_filename):
        store = pd.HDFStore(probe_blast_hdf5_filename, complib = 'lzo', complevel = 9)
        evaluate_taxon_probes(taxon_probe_directory, blast_lineage_slim, store)
        store.close()
        # for blastfile in glob.glob(taxon_probe_directory + '/*.blast.out'):
        #     os.remove(blastfile)
    return

###############################################################################################################
# main function
###############################################################################################################


def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    # input blast filename
    parser.add_argument('design_probe_filename', type = str, help = 'Input file containing blast results')

    parser.add_argument('-cdir', '--consensus_dir', dest = 'cdir', type = str)

    # parser.add_argument('-bdir', '--blast_lineage_dir', dest = 'bdir', type = str)


    args = parser.parse_args()

    probe_directory, taxon_probe_filename = os.path.split(args.design_probe_filename)
    taxon = re.sub('_consensus.int', '', taxon_probe_filename)
    similarity_directory = os.path.split(probe_directory)[0]
    print('Evaluating probes for %s' % taxon)
    input_consensus_directory = similarity_directory + '/consensus'
    blast_lineage_filename = similarity_directory + '/consensus/blast_lineage.tab'
    blast_lineage_strain_filename = similarity_directory + '/consensus/blast_lineage_strain.tab'
    blast_lineage_strain_sub_slash_filename = similarity_directory + '/consensus/blast_lineage_strain_sub_slash.tab'
    # blast_lineage_filename = args.bdir + '/blast_lineage.tab'
    # blast_lineage_strain_filename = args.bdir + '/blast_lineage_strain.tab'
    # blast_lineage_strain_sub_slash_filename = args.bdir + '/blast_lineage_strain_sub_slash.tab'
    evaluation_complete_filename = similarity_directory + '/blast/' + taxon + '.probe.evaluation.complete.txt'
    # if not os.path.exists(blast_lineage_strain_filename):
    #     retrieve_cluster(blast_lineage_filename, input_consensus_directory, 'F')
    if not os.path.exists(similarity_directory + '/blast/' + taxon + '.probe.evaluation.h5'):
        evaluate_probes(similarity_directory, taxon, blast_lineage_strain_filename)
    file = open(evaluation_complete_filename, 'w')
    file.write('Taxon ' + taxon + ' probe evaluation is done.')
    file.close()
    return

if __name__ == '__main__':
    main()
