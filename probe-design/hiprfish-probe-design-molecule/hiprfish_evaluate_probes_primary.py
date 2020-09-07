import os
import re
import glob
import dask
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Alphabet import IUPAC, generic_dna
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['DASK_NUM_THREADS'] = '16'

###############################################################################################################
# HiPR-FISH Probe Design Pipeline
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################

def calculate_mch(df):
    qseq = df.qseq
    sseq = df.sseq
    if qseq != sseq:
        snp_indices = np.where(np.array(list(qseq)) != np.array(list(sseq)))[0]
        diffs = np.diff(snp_indices)
        mch = np.max(np.append(diffs,[snp_indices[0], len(qseq) - 1 - snp_indices[-1]]))
    else:
        mch = len(qseq)
    return(mch)

def sub_slash(str):
    return(re.sub('/', '_', str))

def get_design_info(probe_name):
    design_level, design_target, probe_nid = probe_name.split('_')
    return(pd.Series({'probe_id': probe_name, 'design_level': design_level, 'design_target': design_target, 'probe_nid': probe_nid}))

def get_blast_lineage_slim(blast_lineage_filename):
    blast_lineage = pd.read_table(blast_lineage_filename, dtype = {'staxids':str})
    lineage_columns = ['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    blast_lineage_slim = blast_lineage[lineage_columns]
    return(blast_lineage_slim)

def get_probe_blast_metadata(probe_blast_filename):
    design_target_dir, probe_blast_basename = os.path.split(probe_blast_filename)
    design_level_dir, design_target = os.path.split(design_target_dir)
    probes_dir, design_level = os.path.split(design_level_dir)
    sample_dir, sample = os.path.split(probes_dir)
    probe_nid = re.sub('.fasta.blast.out', '', probe_blast_basename)
    return(sample_dir, sample, design_level, design_target, probe_nid)

def read_probe_blast(probe_blast_filename):
    probes_blast = pd.read_table(probe_blast_filename, header = None)
    probes_blast.columns = ['probe_id', 'molecule_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'probe_start', 'probe_end', 'molecule_start', 'molecule_end', 'evalue', 'bitscore', 'qseq', 'sseq']
    return(probes_blast)

@dask.delayed
def blast_taxon_individual_probe(filename, blast_database):
    blast_output_filename = '{}.blast.out'.format(filename)
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq'
    return_code = subprocess.check_call(['blastn', '-db', blast_database, '-query', filename, '-out', blast_output_filename, '-outfmt', out_format, '-task', 'blastn-short', '-max_hsps', '1', '-max_target_seqs', '100000', '-strand', 'minus', '-evalue', '100', '-num_threads', '1', '-dust', 'no', '-soft_masking', 'false'])
    return(return_code)

@dask.delayed
def evaluate_individual_probes(probe_blast_filename, blast_lineage_slim):
    design_target_dir, probe_blast_basename = os.path.split(probe_blast_filename)
    design_level_dir, design_target = os.path.split(design_target_dir)
    probes_dir, design_level = os.path.split(design_level_dir)
    sample_dir, sample = os.path.split(probes_dir)
    probes_evaluation_filename = '{}/evaluate/{}/{}/{}_probe_evaluation.csv.gz'.format(sample_dir, design_level, design_target, re.sub('.fasta.blast.out', '', probe_blast_basename))
    probes_blast = read_probe_blast(probe_blast_filename)
    probes_design_info = probes_blast.probe_id.drop_duplicates().apply(get_design_info)
    probes_blast['mch'] = probes_blast.loc[:,['qseq', 'sseq']].apply(calculate_mch, axis = 1)
    probes_blast['molecule_id'] = probes_blast.molecule_id.apply(sub_slash).to_frame(name = 'molecule_id')
    probes_blast = probes_blast.merge(blast_lineage_slim, on = 'molecule_id', how = 'left')
    probes_blast = probes_blast.merge(probes_design_info, on = 'probe_id', how = 'left')
    probes_blast.to_csv(probes_evaluation_filename, index = None, compression = 'gzip')
    return(0)

def evaluate_probes(design_level_dir, blast_lineage_filename, blast_database, n_workers):
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    probes_dir, design_level = os.path.split(design_level_dir)
    sample_dir = os.path.split(probes_dir)[0]
    design_level_target_dir = glob.glob('{}/[0-9]*'.format(design_level_dir))
    design_target_list = [os.path.basename(dir) for dir in design_level_target_dir]
    design_target_evaluation_dir = ['{}/evaluate/{}/{}'.format(sample_dir, design_level, t) for t in design_target_list]
    blast_lineage_slim = get_blast_lineage_slim(blast_lineage_filename)
    for dir in design_target_evaluation_dir:
        if not os.path.exists(dir):
            os.makedirs(dir)
    for design_target in design_target_list:
        probe_filenames = pd.DataFrame(glob.glob('{}/{}/*.fasta'.format(design_level_dir, design_target)))
        probes_blast_filenames = pd.DataFrame(['{}.blast.out'.format(f) for f in probe_filenames.iloc[:,0].values])
        index_list = np.arange(0, probe_filenames.shape[0], 2000)
        index_list = np.append(index_list, probe_filenames.shape[0])
        for i in range(index_list.shape[0] - 1):
            print('Blasting probes from group {} out of {} groups in taxon {}'.format(i+1, index_list.shape[0]-1, design_target))
            start_index = index_list[i]
            end_index = index_list[i+1]
            return_code_list = []
            for idx in range(start_index, end_index):
                return_code = blast_taxon_individual_probe(probe_filenames.iloc[idx,0], blast_database)
                return_code_list.append(return_code)
            return_code_total = dask.delayed(sum)(return_code_list)
            rct = return_code_total.compute()
        print('Blasting done, return code {}'.format(rct))
        for i in range(index_list.shape[0] - 1):
            print('Primary evaluating probes from group {} out of {} groups in taxon {}'.format(i+1, index_list.shape[0]-1, design_target))
            start_index = index_list[i]
            end_index = index_list[i+1]
            return_code_list = []
            for idx in range(start_index, end_index):
                return_code = evaluate_individual_probes(probes_blast_filenames.iloc[idx,0], blast_lineage_slim)
                return_code_list.append(return_code)
            return_code_total = dask.delayed(sum)(return_code_list)
            rct = return_code_total.compute()
        blast_out_files = glob.glob('{}/{}/*.fasta.blast.out'.format(design_level_dir, design_target))
        for f in blast_out_files:
            os.remove(f)
        print('Primary evaluating done, return code {}'.format(rct))
        taxon_probe_evaluation_complete_filename = '{}/evaluate/{}/{}_probe_evaluation_complete.txt'.format(sample_dir, design_level, design_target, design_target)
        file = open(taxon_probe_evaluation_complete_filename, 'w')
        file.write('Probe evaluation for taxon {} is complete'.format(design_target))
        file.close()
    probe_evaluation_complete_filename = '{}/log/{}_probe_evaluation_primary_complete.txt'.format(sample_dir, design_level)
    file = open(probe_evaluation_complete_filename, 'w')
    file.write('Primary probe evaluation at the {} level is complete'.format(design_level))
    file.close()
    return

###############################################################################################################
# main function
###############################################################################################################


def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('probes_write_complete_filename', type = str, help = 'Input file containing blast results')
    parser.add_argument('oriented_pacbio_filename', type = str, help = 'Input FASTA file containing full length 16S sequences of the complex microbial community')
    parser.add_argument('-d', '--design_level_dir', dest = 'design_level_dir', type = str)
    parser.add_argument('-n', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input FASTA file containing 16S sequences')
    args = parser.parse_args()

    design_level = os.path.basename(args.design_level_dir)
    probes_dir = os.path.split(args.design_level_dir)[0]
    sample_dir = os.path.split(probes_dir)[0]
    evaluation_dir = '{}/evaluate/{}'.format(sample_dir, design_level)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    blast_lineage_filename = '{}/utilities/blast_lineage.tab'.format(sample_dir)
    evaluate_probes(args.design_level_dir, blast_lineage_filename, args.oriented_pacbio_filename, args.n_workers)
    return

if __name__ == '__main__':
    main()
