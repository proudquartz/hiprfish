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

def sub_slash(str):
    return(re.sub('/', '_', str))

def get_lineage_at_desired_ranks(taxid, desired_ranks):
    'Retrieve lineage information at desired taxonomic ranks'
    # initiate an instance of the ncbi taxonomy database
    ncbi = NCBITaxa()
    # retrieve lineage information for each full length 16S molecule
    lineage = ncbi.get_lineage(taxid)
    lineage2ranks = ncbi.get_rank(lineage)
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())
    ranki = [ranks2lineage.get(x) for x in desired_ranks]
    ranks = [x if x is not None else 0 for x in ranki]
    return(ranks)

def blast_pacbio(input_data_filename, blast_database_name, output_data_filename, max_hsps, max_target_seqs):
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids'
    blastn_cline = NcbiblastnCommandline(cmd = 'blastn', query = input_data_filename, db = blast_database_name, max_hsps = max_hsps, max_target_seqs = max_target_seqs, outfmt = '"' + out_format + '"', out = output_data_filename)
    blastn_cline()
    return

def write_oriented_pacbio_fasta(input_fasta_filename, input_blast_filename, oriented_fasta_filename):
    seq_dict = SeqIO.to_dict(SeqIO.parse(input_fasta_filename, 'fasta'))
    blast_result = pd.read_csv(input_blast_filename, header = None, delim_whitespace = True)
    blast_result.columns = ['molecule_id', 'reference_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'staxids']
    blast_result['oriented'] = (blast_result['sstart'] < blast_result['send'])
    oriented_seqs = [SeqRecord(seq_dict[mid].seq, id = sub_slash(mid), description = '') if blast_result[blast_result['molecule_id'] == mid]['oriented'].values else SeqRecord(seq_dict[mid].reverse_complement().seq, id = sub_slash(mid), description = '') for mid in blast_result['molecule_id']]
    SeqIO.write(oriented_seqs, oriented_fasta_filename, 'fasta')
    return

def make_blast_db(input_fasta_file):
    subprocess.call(['makeblastdb','-in', input_fasta_file, '-dbtype', 'nucl'])
    return

def generate_blast_lineage(input_blast_filename, util_dir):
    blast_result = pd.read_csv(input_blast_filename, header = None, dtype = {13: str}, delim_whitespace = True)
    blast_result.columns = ['molecule_id', 'reference_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'staxids']
    ncbi = NCBITaxa()
    desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    ranks = pd.DataFrame(columns = ['staxids'] + desired_ranks)
    blast_result_staxids = blast_result['staxids'].unique()
    ranks['staxids'] = blast_result_staxids
    for i in range(0, blast_result_staxids.shape[0]):
        taxid = blast_result_staxids[i]
        if not str(taxid).isdigit():
            taxid = taxid.split(';')[0]
        ranks.iloc[i,1:len(desired_ranks)+1] = get_lineage_at_desired_ranks(taxid, desired_ranks)
    blast_lineage = blast_result.merge(ranks, on = 'staxids', how = 'left')
    blast_lineage['molecule_id'] = blast_lineage['molecule_id'].apply(sub_slash)
    blast_lineage_filename = '{}/blast_lineage.tab'.format(util_dir)
    blast_lineage.to_csv(blast_lineage_filename, sep = '\t', index = False)
    return

def initialize_probe_design(input_fasta_filename, primer3_dir, include_start, include_end):

    # running the function
    settings_filename = primer3_dir + '/primer3_settings.txt'
    # write primer3 settings file
    primer3_settings = ['Primer3 File - http://primer3.sourceforge.net',
                        'P3_FILE_TYPE=settings',
                        '',
                        'P3_FILE_ID=FISH probe design',
                        'P3_FILE_FLAG=1',
                        'PRIMER_FIRST_BASE_INDEX=1',
                        'PRIMER_TASK=generic',
                        'PRIMER_EXPLAIN_FLAG=1',
                        'PRIMER_NUM_RETURN=10000',
                        'PRIMER_PICK_LEFT_PRIMER=0',
                        'PRIMER_PICK_INTERNAL_OLIGO=1',
                        'PRIMER_PICK_RIGHT_PRIMER=0',
                        'PRIMER_INTERNAL_OPT_SIZE=20',
                        'PRIMER_INTERNAL_MIN_SIZE=18',
                        'PRIMER_INTERNAL_MAX_SIZE=23',
                        'PRIMER_INTERNAL_MIN_TM=' + str(40),
                        'PRIMER_INTERNAL_MAX_SELF_ANY_TH=1000.00',
                        'PRIMER_INTERNAL_MAX_HAIRPIN_TH=1000.0',
                        'PRIMER_INTERNAL_MAX_NS_ACCEPTED=0',
                        'PRIMER_THERMODYNAMIC_OLIGO_ALIGNMENT=1',
                        'PRIMER_THERMODYNAMIC_TEMPLATE_ALIGNMENT=0',
                        'PRIMER_THERMODYNAMIC_PARAMETERS_PATH=/programs/primer3-2.3.5/src/primer3_config/',
                        'PRIMER_LOWERCASE_MASKING=0',
                        'PRIMER_PICK_ANYWAY=1',
                        '=']
    with open(settings_filename, 'w') as f:
        for l in primer3_settings:
            f.write('{}\n'.format(l))

    # write primer3 input file
    conseq = SeqIO.parse(input_fasta_filename, 'fasta')
    for record in conseq:
        primer3_input_filename = '{}/{}_primer3_input.txt'.format(primer3_dir, record.id)
        primer3_record = ['SEQUENCE_ID=' + str(record.id), 'SEQUENCE_TEMPLATE=' + str(record.seq).upper(), 'SEQUENCE_INCLUDED_REGION=' + str(include_start) + ',' + str(len(record.seq) - include_end - include_start), 'P3_FILE_FLAG=1', 'PRIMER_EXPLAIN_FLAG=1', '=']
        pd.DataFrame(primer3_record).to_csv(primer3_input_filename, header = None, index = False, mode = 'a', sep = ' ')
    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('input_filename', type = str, help = 'Input FASTA file containing 16S sequences')
    parser.add_argument('sample_dir', type = str, help = 'Directory to save all the outputs of the probe design pipeline')
    parser.add_argument('-db', '--blast_database', dest = 'blast_database', type = str, default = '', help = '16S database to identify the taxonomic lineage of each 16S sequence')
    args = parser.parse_args()
    util_dir = '{}/utilities/'.format(args.sample_dir)
    log_dir = '{}/log/'.format(args.sample_dir)
    primer3_dir = '{}/primer3'.format(args.sample_dir)
    probes_summary_dir = '{}/probes_summary'.format(args.sample_dir)
    blast_output_name = '{}.blast.out'.format(args.input_filename)
    oriented_fasta_filename = '{}.oriented.fasta'.format(re.sub('.fasta', '', args.input_filename))
    if not os.path.exists(util_dir):
        os.makedirs(util_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(primer3_dir):
        os.makedirs(primer3_dir)
    if not os.path.exists(probes_summary_dir):
        os.mkdir(probes_summary_dir)
    if not os.path.exists(blast_output_name):
        blast_pacbio(args.input_filename, args.blast_database, blast_output_name, 1, 1)
    if not os.path.exists(oriented_fasta_filename):
        write_oriented_pacbio_fasta(args.input_filename, blast_output_name, oriented_fasta_filename)
    make_blast_db(oriented_fasta_filename)
    generate_blast_lineage(blast_output_name, util_dir)
    initialize_probe_design(oriented_fasta_filename, primer3_dir, 10, 10)
    probe_design_initialization_complete_filename = '{}/probe_design_initialization_complete.txt'.format(log_dir)
    file = open(probe_design_initialization_complete_filename, 'w')
    file.write('Probe design is complete.')
    file.close()
    return

if __name__ == '__main__':
    main()
