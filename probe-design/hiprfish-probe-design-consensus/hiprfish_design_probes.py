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
from SetCoverPy import setcover
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC, generic_dna
from Bio.Blast.Applications import NcbiblastnCommandline
from joblib import Parallel, delayed


###############################################################################################################
# HiPR-FISH : design probes
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################

# get lineage at a desired taxonomic rank

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

# blast pacbio 16s sequences against 16S Microbial database
def blast_pacbio(input_data_filename, blast_database_name, output_data_filename, max_hsps, max_target_seqs):
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids'
    blastn_cline = NcbiblastnCommandline(cmd = 'blastn', query = input_data_filename, db = blast_database_name, max_hsps = max_hsps, max_target_seqs = max_target_seqs, outfmt = '"' + out_format + '"', out = output_data_filename)
    blastn_cline()
    return

# generate consensus sequence based on OTU
def generate_consensus_otu(input_blast_filename, input_fasta_filename, similarity, outdir):
    # read in blast result

    blast_result = pd.read_table(input_blast_filename, header = None, dtype = {13:str})
    blast_result.columns = ['molecule_id', 'reference_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'staxids']

    # initiate an instance of ncbi taxonomy database
    ncbi = NCBITaxa()

    # retrieve lineage information for each full length 16S molecule
    desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    ranks = pd.DataFrame(columns = ['staxids'] + desired_ranks)
    blast_result_staxids = blast_result['staxids'].unique()
    ranks['staxids'] = blast_result_staxids
    for i in range(0, blast_result_staxids.shape[0]):
        taxid = blast_result_staxids[i]
        ranks.ix[i,1:len(desired_ranks)+1] = get_lineage_at_desired_ranks(taxid, desired_ranks)

    # merge lineage information with PacBio 16S blast results
    blast_lineage = blast_result.merge(ranks, on = 'staxids', how = 'left')

    seq_dict = SeqIO.to_dict(SeqIO.parse(input_fasta_filename, 'fasta'))
    blast_lineage_filename = outdir + '/blast_lineage.tab'
    blast_lineage.to_csv(blast_lineage_filename, sep = '\t', index = False)
    taxon = os.path.basename(input_fasta_filename).replace('.fasta', '')
    taxon_consensus_filename = outdir + '/' + taxon + '.consensus.fasta'
    taxon_sequence_cluster_info_filename = outdir + '/' + taxon + '.uc'
    subprocess.check_call(['/home/hs673/Tools/usearch', '-cluster_fast', input_fasta_filename, '-id', str(similarity), '-consout', taxon_consensus_filename, '-uc', taxon_sequence_cluster_info_filename])
    return

# group fasta sequences by taxon and write to file
def write_taxon_fasta(df, taxon, seq_dict, similarity, outdir):
    # specify file names
    if taxon == 'strain':
        taxon = 'species'
    taxon_name = str(df[taxon].unique()[0])
    taxon_fasta_filename = outdir + '/' + taxon_name + '.fasta'
    taxon_alignment_filename = outdir + '/' + taxon_name + '.alignment.fasta'
    taxon_consensus_filename = outdir + '/' + taxon_name + '.consensus.fasta'
    taxon_sequence_cluster_info_filename = outdir + '/' + taxon_name + '.uc'
    # check for 16s sequence orientation
    df['oriented'] = (df['sstart'] < df['send'])
    # get sequences belonging to a particular taxon
    taxon_seqs = [seq_dict[mid] if df[df['molecule_id'] == mid]['oriented'].values else SeqRecord(seq_dict[mid].reverse_complement().seq, id = mid, description = '') for mid in df['molecule_id']]
    # write taxon specific sequences to file
    SeqIO.write(taxon_seqs, taxon_fasta_filename, 'fasta')
    if len(taxon_seqs) > 1:
        subprocess.check_call(['/home/hs673/Tools/usearch', '-cluster_fast', taxon_fasta_filename, '-id', str(similarity), '-consout', taxon_consensus_filename, '-uc', taxon_sequence_cluster_info_filename])
    else:
        consensus_seq = taxon_seqs[0].seq
        seq_rec = SeqRecord(consensus_seq, id = taxon_name, description = '')
        SeqIO.write(seq_rec, taxon_consensus_filename, 'fasta')
    return

# write oriented pacbio fasta file
def write_oriented_pacbio_fasta(input_fasta_filename, input_blast_filename, oriented_fasta_filename):
    seq_dict = SeqIO.to_dict(SeqIO.parse(input_fasta_filename, 'fasta'))
    blast_result = pd.read_table(input_blast_filename, header = None)
    blast_result.columns = ['molecule_id', 'reference_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'staxids']
    blast_result['oriented'] = (blast_result['sstart'] < blast_result['send'])
    oriented_seqs = [seq_dict[mid] if blast_result[blast_result['molecule_id'] == mid]['oriented'].values else SeqRecord(seq_dict[mid].reverse_complement().seq, id = mid, description = '') for mid in blast_result['molecule_id']]
    SeqIO.write(oriented_seqs, oriented_fasta_filename, 'fasta')
    return

# make blast database
def make_blast_db(input_fasta_file):
    subprocess.call(['makeblastdb','-in', input_fasta_file, '-parse_seqids', '-dbtype', 'nucl'])
    return

def generate_blast_lineage(input_blast_filename, similarity, outdir, target_rank):
    blast_result = pd.read_table(input_blast_filename, header = None, dtype = {13: str})
    blast_result.columns = ['molecule_id', 'reference_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'staxids']
    ncbi = NCBITaxa()

    # retrieve lineage information for each full length 16S molecule
    desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    ranks = pd.DataFrame(columns = ['staxids'] + desired_ranks)
    blast_result_staxids = blast_result['staxids'].unique()
    ranks['staxids'] = blast_result_staxids
    for i in range(0, blast_result_staxids.shape[0]):
        taxid = blast_result_staxids[i]
        if not str(taxid).isdigit():
            taxid = taxid.split(';')[0]
        ranks.ix[i,1:len(desired_ranks)+1] = get_lineage_at_desired_ranks(taxid, desired_ranks)
    blast_lineage = blast_result.merge(ranks, on = 'staxids', how = 'left')
    blast_lineage['molecule_id'] = blast_lineage['molecule_id'].apply(sub_slash)
    blast_lineage_filename = outdir + '/blast_lineage.tab'
    blast_lineage.to_csv(blast_lineage_filename, sep = '\t', index = False)
    if target_rank == 'strain':
        blast_lineage_strain = retrieve_cluster(blast_lineage_filename, outdir, 'F')
        taxon_abundance = blast_lineage_strain['strain'].value_counts().reset_index()
        taxon_abundance.columns = ['taxid', 'counts']
        taxon_abundance.to_csv(outdir + '/taxon_abundance.csv', index = False)
    else:
        blast_lineage_strain = retrieve_cluster(blast_lineage_filename, outdir, 'F')
        taxon_abundance = blast_lineage[target_rank].value_counts().reset_index()
        taxon_abundance.columns = ['taxid', 'counts']
        taxon_abundance.to_csv(outdir + '/taxon_abundance.csv', index = False)
    return

# generate consensus sequence
def generate_consensus(input_fasta_filename, input_blast_filename, similarity, outdir, target_rank):
    # read in blast result

    blast_result = pd.read_table(input_blast_filename, header = None, dtype = {13: str})
    blast_result.columns = ['molecule_id', 'reference_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'staxids']

    # initiate an instance of ncbi taxonomy database
    ncbi = NCBITaxa()

    # retrieve lineage information for each full length 16S molecule
    desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    ranks = pd.DataFrame(columns = ['staxids'] + desired_ranks)
    blast_result_staxids = blast_result['staxids'].unique()
    ranks['staxids'] = blast_result_staxids
    for i in range(0, blast_result_staxids.shape[0]):
        taxid = blast_result_staxids[i]
        if not str(taxid).isdigit():
            taxid = taxid.split(';')[0]
        ranks.ix[i,1:len(desired_ranks)+1] = get_lineage_at_desired_ranks(taxid, desired_ranks)

    # merge lineage information with PacBio 16S blast results
    blast_lineage = blast_result.merge(ranks, on = 'staxids', how = 'left')
    seq_dict = SeqIO.to_dict(SeqIO.parse(input_fasta_filename, 'fasta'))
    if target_rank == 'strain':
        blast_lineage.groupby(['species']).apply(write_taxon_fasta, taxon = target_rank, seq_dict = seq_dict, similarity = similarity, outdir = outdir)
    else:
        blast_lineage.groupby([target_rank]).apply(write_taxon_fasta, taxon = target_rank, seq_dict = seq_dict, similarity = similarity, outdir = outdir)
    return(blast_lineage)

# combine taxon consensus fasta files
def combine_fasta_files(input_file_directory, file_ext, output_file_name):
    # remove previously concatenated file if it exist
    try:
        os.path.exists(output_file_name)
    except:
        pass

    # running the function
    os.chdir(input_file_directory)
    fasta_files = glob.glob('*' + file_ext)
    # write primer3 input file
    for file in fasta_files:
        conseq = SeqIO.parse(file, 'fasta')
        num_rec = sum(1 for record in conseq)
        conseq = SeqIO.parse(file, 'fasta')
        for record in conseq:
            taxon_name = file.replace('.consensus.fasta', '')
            if num_rec > 1:
                record.id = taxon_name + '_' + record.id + '_consensus'
            else:
                record.id = taxon_name + '_consensus'
            record.description = ''
            with open(output_file_name, 'a') as output_handle:
                SeqIO.write(record, output_handle, 'fasta')

    return

# probe design
def probe_design(input_fasta_filename, outdir, target_rank, min_tm, include_start, include_end):

    # running the function
    settings_file_path = outdir + '/primer3_settings.txt'
    primer3_input_file_path = outdir + '/consensus_primer3_input.txt'
    output_file_path = outdir + '/consensus_probes.fasta'

    try:
        os.path.exists(primer3_input_file_path)
    except:
        pass

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
    pd.DataFrame(primer3_settings).to_csv(settings_file_path, header = None, index = False)

    # write primer3 input file
    conseq = SeqIO.parse(input_fasta_filename, 'fasta')
    for record in conseq:
        primer3_record = ['SEQUENCE_ID=' + str(record.id), 'SEQUENCE_TEMPLATE=' + str(record.seq).upper(), 'SEQUENCE_INCLUDED_REGION=' + str(include_start) + ',' + str(len(record.seq) - include_end - include_start), 'P3_FILE_FLAG=1', 'PRIMER_EXPLAIN_FLAG=1', '=']
        pd.DataFrame(primer3_record).to_csv(primer3_input_file_path, header = None, index = False, mode = 'a', sep = ' ')

    os.chdir(outdir)
    if os.path.exists(output_file_path):
        print('No probe design was rerun')
    else:
        print('Probe design was run')
        subprocess.check_call(['/programs/primer3-2.3.5/src/primer3_core', '-p3_settings_file', settings_file_path, '-output', output_file_path, '-format_output', primer3_input_file_path])

    return

def split_taxon_probe_file(probe_filename):
    probe_dir, taxon_consensus_filename = os.path.split(probe_filename)
    taxon = re.sub('_consensus.int', '', taxon_consensus_filename)
    taxon_probe_directory = probe_dir + '/' + taxon
    if not os.path.exists(taxon_probe_directory):
        os.makedirs(taxon_probe_directory)
    probes = pd.read_table(probe_filename, skiprows = 3, header = None, delim_whitespace = True)
    probes.columns = ['probe_num', 'seq', 'start', 'length', 'N', 'GC', 'Tm', 'self_any_th', 'self_end_th', 'hair-pin', 'quality']
    probes_list = [SeqRecord(Seq(probes['seq'][i]).reverse_complement(), id = str(probes['probe_num'][i]), description = '') for i in range (0, probes.shape[0])]
    probes_fasta_filenames = [probe_dir + '/' + taxon + '/' + taxon_consensus_filename + '.probe_' + str(probe_num) + '.fasta' for probe_num in range(probes.shape[0])]
    try:
        os.path.exists(probe_dir + '/' + taxon)
    except:
        os.makedirs(probe_dir + '/' + taxon)
    for i in range(probes.shape[0]):
        SeqIO.write(probes_list[i], probes_fasta_filenames[i], 'fasta')

def retrieve_cluster(input_blast_lineage, input_consensus_directory, otu):
    if otu == 'F':
        blast_lineage_df = pd.read_table(input_blast_lineage, dtype = {'staxids':str})
        lineage_columns = ['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        blast_lineage_slim = blast_lineage_df[lineage_columns]
    else:
        blast_lineage_df = pd.read_table(input_blast_lineage, dtype = {'staxids':str}, header = None)
        blast_lineage_df.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
        blast_lineage_filtered = blast_lineage_df[blast_lineage_df['rec_type'] != 'C']
        blast_lineage_filtered['target_taxon'] = 'Cluster' + blast_lineage_filtered['cluster_num'].astype(str)
        blast_lineage_filtered['molecule_id'] = blast_lineage_filtered['query_label']
        blast_lineage_filtered[blast_lineage_filtered['rec_type'] == 'S']['molecule_id'] = blast_lineage_filtered['target_label']
        lineage_columns = ['molecule_id', 'target_taxon']
        blast_lineage_slim = blast_lineage_filtered[lineage_columns]
    os.chdir(input_consensus_directory)
    taxon_fasta_list = glob.glob('*.consensus.fasta')
    cluster_lookup = pd.DataFrame(columns = ['molecule_id', 'strain'])
    for i in range(0, len(taxon_fasta_list)):
        taxon = re.sub('.consensus.fasta', '', taxon_fasta_list[i])
        taxon_fasta_filename = str(taxon) + '.fasta'
        taxon_consensus_filename = str(taxon) + '.consensus.fasta'
        fasta_num_seq = sum(1 for record in SeqIO.parse(taxon_fasta_filename, 'fasta'))
        consensus_num_seq = sum(1 for record in SeqIO.parse(taxon_consensus_filename, 'fasta'))
        if fasta_num_seq == 1:
            fasta_record = SeqIO.read(taxon_fasta_filename, 'fasta')
            molecule_cluster = pd.DataFrame([fasta_record.id, str(taxon)]).transpose()
            molecule_cluster.columns = ['molecule_id', 'strain']
            cluster_lookup = pd.concat([cluster_lookup, molecule_cluster], ignore_index = True)
        elif consensus_num_seq == 1:
            molecule_cluster = pd.DataFrame([[record.id, taxon] for record in SeqIO.parse(taxon_fasta_filename, 'fasta')])
            molecule_cluster.columns = ['molecule_id', 'strain']
            cluster_lookup = pd.concat([cluster_lookup, molecule_cluster], ignore_index = True)
        else:
            taxon_uc_file = input_consensus_directory + '/' + str(taxon) + '.uc'
            taxon_uc = pd.read_table(taxon_uc_file, header = None)
            taxon_uc.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
            taxon_uc_clusters = taxon_uc[(taxon_uc['rec_type'] == 'S') | (taxon_uc['rec_type'] == 'H')]
            molecule_cluster = pd.DataFrame([[taxon_uc_clusters.loc[i, 'query_label'], str(taxon) + '_Cluster' + str(taxon_uc_clusters.loc[i, 'cluster_num'])] for i in range(0, taxon_uc_clusters.shape[0])])
            molecule_cluster.columns = ['molecule_id', 'strain']
            cluster_lookup = pd.concat([cluster_lookup, molecule_cluster], ignore_index = True)
    cluster_lookup_df = pd.DataFrame(cluster_lookup)
    cluster_lookup_df['molecule_id'] = cluster_lookup_df['molecule_id'].apply(sub_slash)
    cluster_lookup_filename = input_consensus_directory + '/cluster_lookup.tab'
    cluster_lookup_df.to_csv(cluster_lookup_filename, sep = '\t', index = False)
    blast_lineage_strain = blast_lineage_df.merge(cluster_lookup, on = 'molecule_id')
    blast_lineage_strain_filename = input_consensus_directory + '/blast_lineage_strain.tab'
    blast_lineage_strain.to_csv(blast_lineage_strain_filename, sep = '\t', index = False)
    blast_lineage_strain_sub_slash_filename = input_consensus_directory + '/blast_lineage_strain_sub_slash.tab'
    blast_lineage_strain_sub_slash = blast_lineage_strain
    blast_lineage_strain_sub_slash['molecule_id'] = blast_lineage_strain.molecule_id.apply(sub_slash)
    blast_lineage_strain_sub_slash.to_csv(blast_lineage_strain_sub_slash_filename, sep = '\t', index = False)
    return(blast_lineage_strain)

###############################################################################################################
# main function
###############################################################################################################


def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    # input 16S sequence
    parser.add_argument('input_file_name', type = str, help = 'Input FASTA file containing 16S sequences')

    # output directory
    parser.add_argument('output_directory', type = str, help = 'Directory to save all the outputs of the probe design pipeline')

    # 16S database
    parser.add_argument('-db', '--blast_database', dest = 'blast_database', type = str, default = '', help = '16S database to identify the taxonomic lineage of each 16S sequence')

    # blast database for full pacbio dataset
    parser.add_argument('-sdb', '--special_blast_database', dest = 'special_blast_database', type = str, default = '', help = 'Pacbio database to identify the taxonomic lineage of each 16S sequence')

    # target taxon
    parser.add_argument('-t', '--target_rank', dest = 'target_rank', type = str, default = '.', help = 'The taxonomic level at which to design FISH probes')

    # OTU
    parser.add_argument('-o', '--otu_clustering', dest = 'otu', type = str, default = 'F', help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    # sequence similarity threshold
    parser.add_argument('-s', '--similarity', dest = 'similarity', type = float, default = 90, help = 'Similarity threshold for grouping pacbio 16S sequences to produce consensus sequences')

    # minimum homology length
    parser.add_argument('-mch', '--max_continous_homology', dest = 'max_continuous_homology', type = int, default = 15, help = 'Maximum length of any continous homology of a blast hit between any 16S FISH probe sequence and any target sequence for that hit to be significant')

     # minimum melting temperature
    parser.add_argument('-tm', '--minimum_melting_temp', dest = 'min_tm', type = float, default = 55.00, help = 'Minimum probe melting temperature')

    # Region to include in choosing probes - start
    parser.add_argument('-b', '--include_start', dest = 'include_start', type = int, default = 50, help = 'Starting position of included region for probe design, measured from the beginning of 16S rRNA molecules')

    # Region to include in choosing probes - end
    parser.add_argument('-e', '--include_end', dest = 'include_end', type = int, default = 50, help = 'Ending position of included region for probe design, measured from the end of 16S rRNA molecules')

    args = parser.parse_args()
    rank_dir = args.output_directory + '/' + args.target_rank + '/'
    if args.otu == 'F':
        sim_dir = args.output_directory + '/' + args.target_rank + '/' + 's_' + str(args.similarity)
    else:
        sim_dir = args.output_directory + '/' + args.target_rank + '/' + 's_' + str(args.similarity) + '_OTU'

    pacbio_blast_output_directory = args.output_directory + '/utilities/'
    replacements = (u'.fasta', u''), (u'.fa', u''), (u'.fna', u'')
    output_basename = os.path.basename(args.input_file_name)
    blast_output_name = output_basename + '.blast.out'

    if not os.path.exists(pacbio_blast_output_directory):
        os.makedirs(pacbio_blast_output_directory)

    pacbio_blast_output_file_name = pacbio_blast_output_directory + '/' + blast_output_name
    if not os.path.exists(pacbio_blast_output_file_name):
        blast_pacbio(args.input_file_name, args.blast_database, pacbio_blast_output_file_name, 1, 1)

    # generate oriented fasta file
    oriented_fasta_filename = re.sub('.fasta', '', args.input_file_name) + '.oriented.fasta'
    if not os.path.exists(oriented_fasta_filename):
        write_oriented_pacbio_fasta(args.input_file_name, pacbio_blast_output_file_name, oriented_fasta_filename)

    make_blast_db(oriented_fasta_filename)

    taxon_consensus_output_directory = sim_dir + '/consensus'
    if args.otu == 'F':
        # generate consensus 16S sequence for each taxon
        taxon_consensus_output_directory = sim_dir + '/consensus'
        if not os.path.exists(taxon_consensus_output_directory):
            os.makedirs(taxon_consensus_output_directory)
        generate_consensus(args.input_file_name, pacbio_blast_output_file_name, args.similarity, taxon_consensus_output_directory, args.target_rank)
    else:
        # generate consensus 16S sequence for each OTU
        taxon_consensus_output_directory = sim_dir + '/consensus/'
        if not os.path.exists(taxon_consensus_output_directory):
            os.makedirs(taxon_consensus_output_directory)
        generate_consensus_otu(args.special_blast_database, pacbio_blast_output_file_name, args.input_file_name, args.similarity, taxon_consensus_output_directory)

    blast_lineage_output_dir = args.output_directory + '/{}/s_{}/consensus/'.format(args.target_rank, args.similarity)
    if not os.path.exists(blast_lineage_output_dir):
        os.makedirs(blast_lineage_output_dir)
    generate_blast_lineage(pacbio_blast_output_file_name, args.similarity, blast_lineage_output_dir, args.target_rank)
    # design probes for each taxon
    taxon_consensus_sequences_filename = taxon_consensus_output_directory + '/taxon_consensus.fasta'
    consensus_fasta_ext = '.consensus.fasta'
    taxon_probes_output_directory = sim_dir + '/primer3/'
    try:
        os.path.exists(taxon_consensus_sequences_filename)
    except:
        pass
    if args.otu == 'F':
        combine_fasta_files(taxon_consensus_output_directory, consensus_fasta_ext, taxon_consensus_sequences_filename)
        probe_design(taxon_consensus_sequences_filename, taxon_probes_output_directory, args.target_rank, args.min_tm, args.include_start, args.include_end)
    else:
        taxon = os.path.basename(args.input_file_name).replace('.fasta', '')
        taxon_consensus_sequences_filename = sim_dir + '/consensus/' + taxon + '.consensus.fasta'
        probe_design(taxon_consensus_sequences_filename, taxon_probes_output_directory, args.target_rank, args.min_tm, args.include_start, args.include_end)
    taxon_probes_filename = glob.glob(taxon_probes_output_directory + '*_consensus.int')
    for filename in taxon_probes_filename:
        split_taxon_probe_file(filename)

    if not os.path.exists(sim_dir + '/blast'):
        os.makedirs(sim_dir + '/blast')
    return

if __name__ == '__main__':
    main()
