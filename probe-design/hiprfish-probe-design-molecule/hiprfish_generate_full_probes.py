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

def cm_to_inches(x):
    return(x/2.54)

def convert_numeric_barcode(num, nbit):
    code = re.sub('0b', '', format(num, '#0' + str(nbit+2) + 'b'))
    return(code)

def count_number_of_bits(binary_barcode):
    bin_list = list(binary_barcode)
    bin_int_list = [int(index) for index in bin_list]
    return(np.sum(bin_int_list))

def add_blocking_seqs(probe_blast_filtered, oriented_fasta_list, probes):
    probe_id = '{}_{}_{}'.format(probe_blast_filtered.design_level, probe_blast_filtered.design_target, probe_blast_filtered.probe_nid)
    if probe_id in probes.probe_id.drop_duplicates().values:
        probe_length = probes.loc[probes.probe_id.values == probe_id, 'probe_length'].values[0]
        molecule_id = probe_blast_filtered.molecule_id
        rrna_seq = oriented_fasta_list.loc[oriented_fasta_list.molecule_id.values == molecule_id, 'rrna_seq'].values[0]
        bp_start = probe_blast_filtered.molecule_start + probe_blast_filtered.probe_start - 1
        bp_end = probe_blast_filtered.molecule_end - probe_length + probe_blast_filtered.probe_end
        bp_seq = str(Seq(rrna_seq[bp_end:bp_start]).reverse_complement())
    else:
        bp_seq = ''
    return(pd.Series({'blocking_sequence':bp_seq}))

def generate_full_probes(design_dir, bot, plf = 'T', primer = 'T', primerset = 'B', barcode_selection = 'MostSimple', selection_method = 'AllSpecificPStartGroup'):
    design_id = os.path.basename(design_dir)
    RP = ['GATGATGTAGTAGTAAGGGT',
          'AGGTTAGGTTGAGAATAGGA',
          'AGGGTGTGTTTGTAAAGGGT',
          'TTGGAGGTGTAGGGAGTAAA',
          'AGAGTGAGTAGTAGTGGAGT',
          'ATAGGAAATGGTGGTAGTGT',
          'TGTGGAGGGATTGAAGGATA']
    nbit = len(RP)
    if plf == 'T':
        # probe_length_filler = 'GGAATCGATGGTGCACTGCT'
        probe_length_filler = 'TCTATTTTCTTATCCGACGT'
    else:
        probe_length_filler = ''
    if primer == 'T':
        if primerset == 'A':
            forward_primer = 'CGCGGGCTATATGCGAACCG'
            reverse_primer = 'GCGTTGTATGCCCTCCACGC'
            # TAATACGACTCACTATAGGGCGTGGAGGGCATACAACGC
        elif primerset == 'B':
            forward_primer = 'CGATGCGCCAATTCCGGTTC'
            reverse_primer = 'CAACCCGCGAGCGATGATCA'
            # TAATACGACTCACTATAGGGTGATCATCGCTCGCGGGTTG
        elif primerset == 'C':
            forward_primer = 'GTTGGTCGGCACTTGGGTGC'
            reverse_primer = 'CCACCGGATGAACCGGCTTT'
            # TAATACGACTCACTATAGGGAAAGCCGGTTCATCCGGTGG
    else:
        forward_primer = ''
        reverse_primer = ''
    taxon_best_probes_sa_filenames = glob.glob('{}/*_probe_selection_sa.csv'.format(design_dir))
    if '{}/0_probe_selection_sa.csv'.format(design_dir) in taxon_best_probes_sa_filenames:
        taxon_best_probes_sa_filenames.remove('{}/0_probe_selection_sa.csv'.format(design_dir))
    taxon_best_probes_sa_filenames.sort()
    taxon_best_probes_list = [pd.read_csv(filename) for filename in taxon_best_probes_sa_filenames]
    taxon_best_probes_filtered_list = []
    for i in range(len(taxon_best_probes_list)):
        if not taxon_best_probes_list[i].empty:
            if taxon_best_probes_list[i].selection_method.drop_duplicates().values[0] == selection_method:
                taxon_best_probes_filtered_list.append(taxon_best_probes_list[i])
    oligo_list = []
    blocking_probe_list = []
    barcodes = pd.DataFrame(np.arange(1,2**nbit))
    barcodes.columns = ['NumericBarcode']
    barcodes.loc[:,'BinaryBarcode'] = barcodes.NumericBarcode.apply(convert_numeric_barcode, args = (7,))
    barcodes.loc[:,'NumberBits'] = barcodes.BinaryBarcode.apply(count_number_of_bits)
    if barcode_selection == 'MostSimple':
        barcodes_sorted = barcodes.sort_values(by = ['NumberBits', 'NumericBarcode'])
    elif barcode_selection == 'MostComplex':
        barcodes_sorted = barcodes.sort_values(by = ['NumberBits', 'NumericBarcode'], ascending = [False, False])
    elif barcode_selection == 'Random':
        barcodes_sorted = barcodes.reindex(np.random.permutation(barcodes.index))
    elif barcode_selection == 'Sequential':
        barcodes_sorted = barcodes.copy()
    for i in range(min(127,len(taxon_best_probes_filtered_list))):
        probes = taxon_best_probes_filtered_list[i]
        probes = probes.loc[(probes.blast_on_target_rate.values > bot), :]
        assigned = barcodes_sorted.NumericBarcode.values[i]
        if barcodes_sorted.NumberBits.values[i] > 2:
            barcode_repeat = np.round(15/barcodes_sorted.NumberBits.values[i]).astype(int)
        else:
            barcode_repeat = 15
        if probes.shape[0] > 0:
            for k in range(barcode_repeat):
                probes = probes.sort_values(by = 'quality', ascending = True).reset_index().drop(columns = ['index'])
                for i in range(0, probes.shape[0]):
                    tarseq = probes.loc[i, 'seqrcsa']
                    if 'N' not in list(tarseq):
                        code = np.asarray(list(re.sub('0b', '', format(assigned, '#0' + str(7+2) + 'b'))), dtype = np.int64)
                        indices = np.where(code == 1)[0]
                        if len(indices) > 2:
                            indices = np.append(indices, indices[0])
                            subcode = np.zeros((len(indices)-1, nbit), dtype = np.int64)
                            for j in range(0, len(indices) - 1):
                                subcode[j, indices[j]] = 1
                                subcode[j, indices[j+1]] = 1
                            oligo = [[str(probes.loc[i,'target_taxon']), probes.loc[i,'mean_probe_start'], probes.loc[i, 'length'], probes.loc[i,'taxon_abundance'], tarseq, probes.loc[i,'Tm'], probes.loc[i,'GC'], re.sub('\[|\]| ','',str(code)), assigned, probes.loc[i,'probe_id'], str(probes.loc[i,'target_taxon']) + '_' + str(assigned) + '_' + re.sub('\[|\]| ','',str(subcode[k])) + '_' + str(probes.loc[i,'probe_id_full']), forward_primer + RP[np.where(subcode[k] == 1)[0][0]] + tarseq + RP[np.where(subcode[k] == 1)[0][1]] + reverse_primer] for k in range(0, len(subcode))]
                        elif len(indices) == 2:
                            oligo = [[str(probes.loc[i,'target_taxon']), probes.loc[i,'mean_probe_start'], probes.loc[i, 'length'], probes.loc[i,'taxon_abundance'], tarseq, probes.loc[i,'Tm'], probes.loc[i,'GC'], re.sub('\[|\]| ','',str(code)), assigned, probes.loc[i,'probe_id'], str(probes.loc[i,'target_taxon']) + '_' + str(assigned) + '_' + re.sub('\[|\]| ','', str(code)) + '_' + str(probes.loc[i,'probe_id_full']), forward_primer + RP[np.where(code == 1)[0][0]] + tarseq + RP[np.where(code == 1)[0][1]] + reverse_primer]]
                        else:
                            oligo = [[str(probes.loc[i,'target_taxon']), probes.loc[i,'mean_probe_start'], probes.loc[i, 'length'], probes.loc[i,'taxon_abundance'], tarseq, probes.loc[i,'Tm'], probes.loc[i,'GC'], re.sub('\[|\]| ','',str(code)), assigned, probes.loc[i,'probe_id'], str(probes.loc[i,'target_taxon']) + '_' + str(assigned) + '_' + re.sub('\[|\]| ','',str(code)) + '_' + str(probes.loc[i,'probe_id_full']), forward_primer + RP[np.where(code == 1)[0][0]] + tarseq + probe_length_filler + reverse_primer]]
                        oligo_list = oligo_list + oligo
    oligo_df = pd.DataFrame(oligo_list)
    print(oligo_df.shape)
    oligo_df.columns = ['target_taxon', 'mean_probe_start', 'length', 'abundance', 'rna_seq', 'Tm', 'GC', 'code', 'numeric_code', 'probe_id', 'probe_id_full', 'probe_full_seq']
    return(oligo_df)

def blocking_probes_blast_summarize(off_target_summary, probes, evaluation_directory, target_rank):
    blocked_taxa = pd.DataFrame(off_target_summary.loc[:,target_rank].drop_duplicates())
    blocked_taxa.loc[:,'average_encoding_interference_fraction_0bp'] = 0.0
    blocked_taxa.loc[:,'average_encoding_interference_fraction_5bp'] = 0.0
    blocked_taxa.loc[:,'average_encoding_interference_fraction_10bp'] = 0.0
    for tt in blocked_taxa.loc[:,target_rank].values:
        off_target_summary_blocked_taxon = off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, :]
        off_target_summary_p_start = off_target_summary_blocked_taxon.loc[:, ['molecule_id', 'molecule_start', 'molecule_end']].drop_duplicates()
        off_target_summary_p_start.columns = ['molecule_id', 'blocking_molecule_start', 'blocking_molecule_end']
        taxon_probes = probes.loc[probes.target_taxon.values == tt, :]
        encoding_p_start = taxon_probes.loc[:,['probe_id', 'mean_probe_start', 'length']].drop_duplicates()
        if not encoding_p_start.empty:
            significant_overlap_fraction = np.zeros((encoding_p_start.shape[0], 3))
            for i in range(encoding_p_start.shape[0]):
                encoding_probe_blast = pd.read_csv('{}/{}/{}/{}_probe_evaluation.csv.gz'.format(evaluation_directory, target_rank, tt, encoding_p_start.probe_id.values[i]))
                encoding_probe_blast_primary = encoding_probe_blast.loc[(encoding_probe_blast['qcovhsp'] >= 99.9) & (encoding_probe_blast['mch'] == encoding_probe_blast['length']), ['molecule_id', 'molecule_start', 'molecule_end']]
                encoding_probe_blast_primary.columns = ['molecule_id', 'encoding_molecule_start', 'encoding_molecule_end']
                blocking_encoding_merge = off_target_summary_p_start.merge(encoding_probe_blast_primary, on = 'molecule_id', how = 'left')
                blocking_encoding_merge = blocking_encoding_merge.loc[~np.isnan(blocking_encoding_merge.encoding_molecule_start),:]
                if not blocking_encoding_merge.empty:
                    min_end = blocking_encoding_merge.loc[:, ['blocking_molecule_start', 'encoding_molecule_start']].min(axis = 1)
                    max_start = blocking_encoding_merge.loc[:, ['blocking_molecule_end', 'encoding_molecule_end']].max(axis = 1)
                    overlap = min_end - max_start
                    significant_overlap_fraction[i, 0] = np.sum(overlap > 0)/overlap.shape[0]
                    significant_overlap_fraction[i, 1] = np.sum(overlap > 5)/overlap.shape[0]
                    significant_overlap_fraction[i, 2] = np.sum(overlap > 10)/overlap.shape[0]
                else:
                    significant_overlap_fraction[i, 0] = 0
                    significant_overlap_fraction[i, 1] = 0
                    significant_overlap_fraction[i, 2] = 0
            off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, 'average_encoding_interference_fraction_0bp'] = np.average(significant_overlap_fraction[:,0])
            off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, 'average_encoding_interference_fraction_5bp'] = np.average(significant_overlap_fraction[:,1])
            off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, 'average_encoding_interference_fraction_10bp'] = np.average(significant_overlap_fraction[:,2])
        else:
            off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, 'average_encoding_interference_fraction_0bp'] = 0.0
            off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, 'average_encoding_interference_fraction_5bp'] = 0.0
            off_target_summary.loc[off_target_summary.loc[:,target_rank].values == tt, 'average_encoding_interference_fraction_10bp'] = 0.0
    max_average_encoding_interference_fraction_0bp = off_target_summary.average_encoding_interference_fraction_0bp.max()
    max_average_encoding_interference_fraction_5bp = off_target_summary.average_encoding_interference_fraction_5bp.max()
    max_average_encoding_interference_fraction_10bp = off_target_summary.average_encoding_interference_fraction_10bp.max()
    return(pd.Series({'max_average_encoding_interference_fraction_0bp': max_average_encoding_interference_fraction_0bp,
                      'max_average_encoding_interference_fraction_5bp': max_average_encoding_interference_fraction_5bp,
                      'max_average_encoding_interference_fraction_10bp': max_average_encoding_interference_fraction_10bp}))

def generate_blocking_probes(design_dir, bplc, oriented_fasta_filename, evaluation_directory, target_rank, plf = 'T', primer = 'T', primerset = 'B', barcode_selection = 'MostSimple'):
    design_dir_folder = os.path.split(design_dir)[1]
    blocking_probes_filenames = glob.glob('{}/*_off_target_summary_info.csv'.format(design_dir))
    probes_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    probes = pd.read_csv(probes_filename)
    oriented_fasta_list = pd.DataFrame([[record.id, str(record.seq)] for record in SeqIO.parse(oriented_fasta_filename, 'fasta')], columns = ['molecule_id', 'rrna_seq'])
    if '{}/0_off_target_summary_info.csv'.format(design_dir) in blocking_probes_filenames:
        blocking_probes_filenames.remove('{}/0_off_target_summary_info.csv'.format(design_dir))
    blocking_probes_list = []
    for f in blocking_probes_filenames:
        target_taxon = re.sub('_off_target_summary_info.csv', '', os.path.basename(f))
        print('Generating blocking probes for taxon {}...'.format(target_taxon))
        if not probes.loc[probes.target_taxon.values.astype(str) == target_taxon, :].empty:
            off_target_summary_full = pd.read_csv(f)
            off_target_interference = pd.DataFrame()
            off_target_summary_full['blocking_sequence'] = ''
            if not off_target_summary_full.empty:
                off_target_summary_full_probes = off_target_summary_full.probe_id.drop_duplicates()
                index_list = np.arange(0, off_target_summary_full_probes.shape[0], 1000)
                index_list = np.append(index_list, off_target_summary_full_probes.shape[0])
                otsf_index = off_target_summary_full_probes.index.append(pd.Index([off_target_summary_full.shape[0]]))
                for i in range(len(index_list) - 1):
                    off_target_summary_dd = dd.from_pandas(off_target_summary_full.loc[otsf_index[index_list[i]]:otsf_index[index_list[i+1]],:], npartitions = 100)
                    meta_dtype = {'blocking_sequence': 'str'}
                    off_target_summary_bpseq = off_target_summary_dd.apply(add_blocking_seqs, oriented_fasta_list = oriented_fasta_list, probes = probes, meta = meta_dtype, axis = 1)
                    off_target_summary_bpseq_compute = off_target_summary_bpseq.compute()
                    off_target_summary_full.loc[otsf_index[index_list[i]]:otsf_index[index_list[i+1]],'blocking_sequence'] = off_target_summary_bpseq_compute
                off_target_summary_full = off_target_summary_full[off_target_summary_full.blocking_sequence.values != '']
                if not off_target_summary_full.empty:
                    off_target_summary_full_probes = off_target_summary_full.probe_id.drop_duplicates()
                    index_list = np.arange(0, off_target_summary_full_probes.shape[0], 1000)
                    index_list = np.append(index_list, off_target_summary_full_probes.shape[0])
                    otsf_index = off_target_summary_full_probes.index.append(pd.Index([off_target_summary_full.index[-1] + 1]))
                    for i in range(len(index_list) - 1):
                        off_target_summary_dd = dd.from_pandas(off_target_summary_full.loc[otsf_index[index_list[i]]:otsf_index[index_list[i+1]],:], npartitions = 100)
                        meta_dtype = {'max_average_encoding_interference_fraction_0bp': 'float',
                                      'max_average_encoding_interference_fraction_5bp': 'float',
                                      'max_average_encoding_interference_fraction_10bp': 'float'}
                        off_target_summary_temp = off_target_summary_dd.groupby(['probe_id']).apply(blocking_probes_blast_summarize, probes = probes, evaluation_directory = evaluation_directory, target_rank = target_rank, meta = meta_dtype)
                        off_target_summary_compute = off_target_summary_temp.compute()
                        off_target_interference = off_target_interference.append(off_target_summary_compute)
                    off_target_summary_full = off_target_summary_full.merge(off_target_interference.reset_index(), on = 'probe_id')
            blocking_probes_list.append(off_target_summary_full)
    blocking_probes = pd.concat(blocking_probes_list, sort = False)
    blocking_probes = blocking_probes.loc[blocking_probes.length.values >= bplc, :]
    blocking_probes.to_csv('{}/{}_primerset_{}_barcode_selection_{}_full_length_blocking_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), index = False)
    blocking_probes_order_format = blocking_probes.loc[:,['sseq', 'length']]
    blocking_probes_order_format.columns = ['Sequence', 'Length']
    blocking_probes_order_format = blocking_probes_order_format.drop_duplicates()
    blocking_probes_order_format = blocking_probes_order_format.sort_values(by = 'Length', ascending = False)
    # blocking_probes_order_format = blocking_probes_order_format.loc[blocking_probes_order_format.length.values >= bplc]
    blocking_probes_order_format.insert(0, 'Blocking_Probe_Name', ['BP_{}'.format(i) for i in range(blocking_probes_order_format.shape[0])])
    blocking_probes_order_format = blocking_probes_order_format.assign(Amount = '25nm', Purification = 'STD')
    blocking_probes_order_format.loc[blocking_probes_order_format.Length.values < 15, 'Amount'] = '100nm'
    blocking_probes_order_format = blocking_probes_order_format.loc[:,['Blocking_Probe_Name', 'Sequence', 'Amount', 'Purification', 'Length']]
    blocking_probes_order_format.to_excel('{}/{}_primerset_{}_barcode_selection_{}_full_length_blocking_probes_order_format.xlsx'.format(design_dir, design_dir_folder, primerset, barcode_selection))
    probe_length_filler = 'TCTATTTTCTTATCCGACGT'
    # 'GTCTATTTTCTTATCCGACGTGTTG'
    if primerset == 'A':
        forward_primer = 'CGCGGGCTATATGCGAACCG'
        reverse_primer = 'GCGTTGTATGCCCTCCACGC'
        # TAATACGACTCACTATAGGGCGTGGAGGGCATACAACGC
    elif primerset == 'B':
        forward_primer = 'CGATGCGCCAATTCCGGTTC'
        reverse_primer = 'CAACCCGCGAGCGATGATCA'
        # TAATACGACTCACTATAGGGTGATCATCGCTCGCGGGTTG
    elif primerset == 'C':
        forward_primer = 'GTTGGTCGGCACTTGGGTGC'
        reverse_primer = 'CCACCGGATGAACCGGCTTT'
        # TAATACGACTCACTATAGGGAAAGCCGGTTCATCCGGTGG
    else:
        forward_primer = ''
        reverse_primer = ''
    blocking_probe_seq_list = []
    for i in range(blocking_probes_order_format.shape[0]):
        bp_seq = blocking_probes_order_format.Sequence.values[i]
        if len(bp_seq) == 15:
            probe_length_filler = 'TCTATTTTCTTATCCGACGTGTT'
        elif len(bp_seq) == 16:
            probe_length_filler = 'TCTATTTTCTTATCCGACGTGT'
        elif len(bp_seq) == 17:
            probe_length_filler = 'TCTATTTTCTTATCCGACGT'
        elif len(bp_seq) == 18:
            probe_length_filler = 'TCTATTTTCTTATCCGACGT'
        elif len(bp_seq) == 19:
            probe_length_filler = 'TCTATTTTCTTATCCGACGT'
        elif len(bp_seq) >= 20:
            probe_length_filler = 'TCTATTTTCTTATCCGACGT'
        blocking_probe = [[forward_primer + probe_length_filler + bp_seq + probe_length_filler + reverse_primer]]
        blocking_probe_seq_list = blocking_probe_seq_list + blocking_probe
    blocking_probe_seq_list = pd.DataFrame(blocking_probe_seq_list)
    if not blocking_probe_seq_list.empty:
        blocking_probe_seq_list.columns = ['blocking_sequences']
        print('There are {} blocking probes for {}.'.format(blocking_probe_seq_list.shape[0], design_dir_folder))
        blocking_probe_seq_list.blocking_sequences.repeat(15).str.upper().to_csv('{}/{}_full_length_blocking_probes_sequences.txt'.format(design_dir, design_dir_folder), header = False, index = False)
    else:
        blocking_probe_seq_list.to_csv('{}/{}_full_length_blocking_probes_sequences.txt'.format(design_dir, design_dir_folder), header = False, index = False)
    return(blocking_probes)

def helper_probe_blast_summarize(hp_idx, probe_evaluation_dir, evaluation_directory, probes, max_continuous_homology, target_rank):
    probe_blast = pd.read_csv('{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, hp_idx))
    probe_blast_filtered = probe_blast.loc[(probe_blast['qcovhsp'] >= 99.9) & (probe_blast['mch'] == probe_blast['length']), :]
    if probe_blast_filtered.shape[0] > 0:
        interfered_taxa = pd.DataFrame(probe_blast_filtered.loc[:,target_rank].drop_duplicates())
        interfered_taxa['average_encoding_interference_fraction_0bp'] = 0.0
        for tt in interfered_taxa.loc[:,target_rank].values:
            helper_probes_interfered_taxon = probe_blast_filtered.loc[probe_blast_filtered[target_rank].values == tt, ['molecule_id', 'molecule_start', 'molecule_end']]
            helper_probes_interfered_taxon.columns = ['molecule_id', 'helper_molecule_start', 'helper_molecule_end']
            taxon_probes = probes.loc[probes.target_taxon.values == tt, :]
            encoding_p_start = taxon_probes.loc[:,['probe_id', 'mean_probe_start', 'length']].drop_duplicates()
            if not encoding_p_start.empty:
                significant_overlap_fraction = np.zeros(encoding_p_start.shape[0])
                for i in range(encoding_p_start.shape[0]):
                    encoding_probe_blast = pd.read_csv('{}/{}/{}/{}_probe_evaluation.csv.gz'.format(evaluation_directory, target_rank, tt, encoding_p_start.probe_id.values[i]))
                    encoding_probe_blast_primary = encoding_probe_blast.loc[(encoding_probe_blast['qcovhsp'] >= 99.9) & (encoding_probe_blast['mch'] == encoding_probe_blast['length']), ['molecule_id', 'molecule_start', 'molecule_end']]
                    encoding_probe_blast_primary.columns = ['molecule_id', 'encoding_molecule_start', 'encoding_molecule_end']
                    helper_encoding_merge = helper_probes_interfered_taxon.merge(encoding_probe_blast_primary, on = 'molecule_id', how = 'left')
                    helper_encoding_merge = helper_encoding_merge.loc[~np.isnan(helper_encoding_merge.encoding_molecule_start),:]
                    if not helper_encoding_merge.empty:
                        min_end = helper_encoding_merge.loc[:, ['helper_molecule_start', 'encoding_molecule_start']].min(axis = 1)
                        max_start = helper_encoding_merge.loc[:, ['helper_molecule_end', 'encoding_molecule_end']].max(axis = 1)
                        overlap = min_end - max_start
                        significant_overlap_fraction[i] = np.sum(overlap > 0)/overlap.shape[0]
                    else:
                        significant_overlap_fraction[i] = 0
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_0bp'] = np.average(significant_overlap_fraction)
            max_average_encoding_interference_fraction_0bp = interfered_taxa.average_encoding_interference_fraction_0bp.max()
    else:
        max_average_encoding_interference_fraction_0bp = 0.0
    return(pd.Series({'probe_id': probe_blast.probe_id.values[0],
                      'max_average_encoding_interference_fraction_0bp': max_average_encoding_interference_fraction_0bp}))

def select_top_taxon_group_helper_probe(helper_probes):
    helper_probes = helper_probes.reset_index().drop(columns = ['index'])
    return(helper_probes.iloc[0,:])

def get_seq_rc(seq):
    seqrc = str(Seq(seq, generic_dna).reverse_complement())
    return(pd.Series({'seqrc': seqrc}))

def get_quadg(seqrc):
    quadg = seqrc.upper().count('GGGG') + seqrc.upper().count('GGGGG')
    return(pd.Series({'quadg':quadg}))

def generate_helper_probes(design_dir, evaluation_directory, target_rank, max_continuous_homology, plf = 'T', primer = 'T', primerset = 'B', barcode_selection = 'MostSimple', helper_probe_repeat = 15):
    design_dir_folder = os.path.split(design_dir)[1]
    helper_probes_filenames = glob.glob('{}/[0-9]*_helper_probes.csv'.format(design_dir))
    print(helper_probes_filenames)
    probes_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    probes = pd.read_csv(probes_filename)
    if '{}/0_helper_probes.csv'.format(design_dir) in helper_probes_filenames:
        helper_probes_filenames.remove('{}/0_helper_probes.csv'.format(design_dir))
    helper_probes_filenames.sort()
    helper_probes_list = []
    for f in helper_probes_filenames:
        target_taxon = re.sub('_helper_probes.csv', '', os.path.basename(f))
        probe_evaluation_dir = '{}/{}/{}'.format(evaluation_directory, target_rank, target_taxon)
        print('Generating helper probes for taxon {}...'.format(target_taxon))
        if not probes.loc[probes.target_taxon.values.astype(str) == target_taxon, :].empty:
            try:
                helper_probes = pd.read_csv(f)
                helper_probes_summary = pd.DataFrame()
                helper_probes_unique = helper_probes.loc[:,['probe_id', 'seq']].drop_duplicates()
                index_list = np.arange(0, helper_probes_unique.shape[0], 1000)
                index_list = np.append(index_list, helper_probes_unique.shape[0])
                for i in range(len(index_list) - 1):
                    helper_probes_dd = dd.from_pandas(helper_probes_unique.loc[index_list[i]:index_list[i+1],:], npartitions = 100)
                    meta_dtype = {'probe_id': 'str',
                                  'max_average_encoding_interference_fraction_0bp': 'float'}
                    helper_probes_temp = helper_probes_dd.probe_id.apply(helper_probe_blast_summarize, args = (probe_evaluation_dir, evaluation_directory, probes, max_continuous_homology, target_rank), meta = meta_dtype)
                    helper_probes_compute = helper_probes_temp.compute()
                    helper_probes_summary = helper_probes_summary.append(helper_probes_compute)
                print('Finished evaluating helper probes...')
                helper_probes_summary_dd = dd.from_pandas(helper_probes_summary, npartitions = 100)
                helper_probes_dd = dd.from_pandas(helper_probes, npartitions = 100)
                helper_probes_summary_dd = helper_probes_summary_dd.drop_duplicates()
                helper_probes_dd = helper_probes_dd.merge(helper_probes_summary_dd, on = 'probe_id')
                meta_dtype = {'seqrc': 'str'}
                helper_probes_seqrc = helper_probes_dd.seq.apply(get_seq_rc, meta = meta_dtype).compute()
                helper_probes_compute = helper_probes_dd.compute()
                helper_probes_compute['seqrc'] = helper_probes_seqrc
                helper_probes_compute_dd = dd.from_pandas(helper_probes_compute, npartitions = 100)
                meta_dtype = {'quadg': 'int'}
                helper_probes_quadg = helper_probes_compute_dd.seqrc.apply(get_quadg, meta = meta_dtype).compute()
                helper_probes_compute['quadg'] = helper_probes_quadg.values
                print('Finished calcuating reverse complement and quad G...')
                helper_probes = helper_probes_compute.loc[(helper_probes_compute.quadg.values == 0) & (helper_probes_compute.max_average_encoding_interference_fraction_0bp.values < 1e-3),:]
                helper_probes = helper_probes.sort_values(by = 'helper_source_probe').reset_index().drop(columns = 'index')
                helper_probes_source_probes = helper_probes.helper_source_probe.drop_duplicates()
                index_list = np.arange(0, helper_probes_source_probes.shape[0], 1000)
                index_list = np.append(index_list, helper_probes_source_probes.shape[0])
                hsp_index = helper_probes_source_probes.index.append(pd.Index([helper_probes.shape[0]]))
                helper_probes_selected_taxon = pd.DataFrame()
                for i in range(len(index_list) - 1):
                    helper_probes_dd = dd.from_pandas(helper_probes.loc[hsp_index[index_list[i]]:hsp_index[index_list[i+1]],:], npartitions = 100)
                    helper_probes_selected = helper_probes_dd.groupby(['helper_source_probe', 'helper_group']).apply(select_top_taxon_group_helper_probe, meta = helper_probes.head())
                    helper_probes_selected_compute = helper_probes_selected.compute()
                    helper_probes_selected_taxon = helper_probes_selected_taxon.append(helper_probes_selected_compute)
                helper_probes_list.append(helper_probes_selected_taxon)
            except:
                print('No helper probes for taxon {}...'.format(target_taxon))
        else:
            print('No helper probes for taxon {} because there is no probes...'.format(target_taxon))
            pass
    helper_probes_selection = pd.concat(helper_probes_list, sort = False)
    helper_probes_selection.to_csv('{}/{}_primerset_{}_barcode_selection_{}_full_length_helper_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), index = False)
    helper_probes_order_format = helper_probes_selection.loc[:,['seqrc', 'length']]
    helper_probes_order_format.columns = ['Sequence', 'Length']
    helper_probes_order_format = helper_probes_order_format.drop_duplicates()
    helper_probes_order_format = helper_probes_order_format.sort_values(by = 'Length', ascending = False)
    helper_probes_order_format.insert(0, 'Helper_Probe_Name', ['HP_{}'.format(i) for i in range(helper_probes_order_format.shape[0])])
    helper_probes_order_format = helper_probes_order_format.assign(Amount = '25nm', Purification = 'STD')
    helper_probes_order_format.loc[helper_probes_order_format.Length.values < 15, 'Amount'] = '100nm'
    helper_probes_order_format = helper_probes_order_format[['Helper_Probe_Name', 'Sequence', 'Amount', 'Purification', 'Length']]
    helper_probes_order_format.to_excel('{}/{}_primerset_{}_barcode_selection_{}_full_length_helper_probes_order_format.xlsx'.format(design_dir, design_dir_folder, primerset, barcode_selection))
    probe_length_filler = 'TCTATTTTCTTATCCGACGT'
    # 'GTCTATTTTCTTATCCGACGTGTTG'
    if primerset == 'A':
        forward_primer = 'CGCGGGCTATATGCGAACCG'
        reverse_primer = 'GCGTTGTATGCCCTCCACGC'
        # TAATACGACTCACTATAGGGCGTGGAGGGCATACAACGC
    elif primerset == 'B':
        forward_primer = 'CGATGCGCCAATTCCGGTTC'
        reverse_primer = 'CAACCCGCGAGCGATGATCA'
        # TAATACGACTCACTATAGGGTGATCATCGCTCGCGGGTTG
    elif primerset == 'C':
        forward_primer = 'GTTGGTCGGCACTTGGGTGC'
        reverse_primer = 'CCACCGGATGAACCGGCTTT'
        # TAATACGACTCACTATAGGGAAAGCCGGTTCATCCGGTGG
    else:
        forward_primer = ''
        reverse_primer = ''
    helper_probe_seq_list = []
    for i in range(helper_probes_order_format.shape[0]):
        hp_seq = helper_probes_order_format.Sequence.values[i]
        helper_probe = [[forward_primer + probe_length_filler + hp_seq + probe_length_filler + reverse_primer]]
        helper_probe_seq_list = helper_probe_seq_list + helper_probe
    helper_probe_seq_list = pd.DataFrame(helper_probe_seq_list)
    helper_probe_seq_list.columns = ['helper_sequences']
    print('There are {} helper probes for {}.'.format(helper_probe_seq_list.shape[0], design_dir_folder))
    helper_probe_seq_list.helper_sequences.repeat(helper_probe_repeat).str.upper().to_csv('{}/{}_full_length_helper_probes_sequences.txt'.format(design_dir, design_dir_folder, primerset, barcode_selection), header = False, index = False)
    return

def write_final_probes_fasta(probes, design_dir, primerset, barcode_selection):
    design_dir_folder = os.path.split(design_dir)[1]
    probes_fasta_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.fasta'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    probes_list = [SeqRecord(Seq(probes['probe_full_seq'][i]), id = str(probes['probe_id'][i]), description = '') for i in range (0, probes.shape[0])]
    SeqIO.write(probes_list, probes_fasta_filename, 'fasta')
    return

def write_final_unique_probes_fasta(probes, design_dir, primerset, barcode_selection):
    design_dir_folder = os.path.split(design_dir)[1]
    probes = probes.drop_duplicates().reset_index().drop(['index'], axis = 1)
    probes_fasta_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_unique.fasta'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    if not os.path.exists(probes_fasta_filename):
        probes_list = [SeqRecord(Seq(probes['probe_full_seq'][i]), id = str(probes['probe_id_full'][i]), description = '') for i in range (0, probes.shape[0])]
        SeqIO.write(probes_list, probes_fasta_filename, 'fasta')
    else:
        pass
    return

def blast_final_probes(design_dir, primerset, barcode_selection, blast_database):
    # read in probe information
    # print('    Blasting ' + os.path.basename(infile))
    design_dir_folder = os.path.split(design_dir)[1]
    infile = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_unique.fasta'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    blast_output_filename = '{}/{}_full_length_probes_unique.blast.out'.format(design_dir, design_dir_folder)
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids qseq sseq'
    if not os.path.exists(blast_output_filename):
        return_code = subprocess.call(['blastn', '-db', blast_database, '-query', infile, '-out', blast_output_filename, '-outfmt', out_format, '-task', 'blastn-short', '-max_hsps', '1', '-max_target_seqs', '100000', '-strand', 'minus', '-evalue', '100', '-num_threads', '1'])
    else:
        return_code = 0
    return(return_code)

def calculate_mch(df):
    qseq_array = df.qseq.values
    sseq_array = df.sseq.values
    mch_array = np.zeros(len(qseq_array))
    for i in range(len(qseq_array)):
        qseq = qseq_array[i]
        sseq = sseq_array[i]
        if qseq != sseq:
            snp_indices = np.where(np.array(list(qseq)) != np.array(list(sseq)))[0]
            diffs = np.diff(snp_indices)
            mch_array[i] = np.max(np.append(diffs,[snp_indices[0], len(qseq) - 1 - snp_indices[-1]]))
        else:
            mch_array[i] = len(qseq)
    return(mch_array)

def summarize_final_probe_blast(df, mch, target_rank = None):
    df['mch'] = calculate_mch(df)
    df_filtered = df.loc[df.mch.values >= mch, :]
    if df_filtered.shape[0] > 0:
        probe_id = np.unique(df_filtered.loc[:,'probe_id'])[0]
        probe_length = np.unique(df_filtered.loc[:,'probe_length'])[0]
        check = np.sum(((df_filtered.loc[:,'probe_start'] >= 44) & (df_filtered.loc[:,'probe_end'] <= 44 + probe_length - 1) & (df_filtered.loc[:,'mch'] <= probe_length)) | (df_filtered.loc[:,'target_taxon'] == df_filtered.loc[:,target_rank]))/df_filtered.shape[0]
    else:
        print('{} has no blast hits...'.format(df.probe_id.values[0]))
        check = 0
    return(check)

def check_final_probes_blast(design_dir, probes, mch, bot, target_rank, blast_lineage_filename, primerset, barcode_selection):
    # probes_filename = '{}/full_length_probes.csv'.format(design_dir)
    design_dir_folder = os.path.split(design_dir)[1]
    probes_blast_filename = '{}/{}_full_length_probes_unique.blast.out'.format(design_dir, design_dir_folder)
    probes_blast = dd.read_table(probes_blast_filename, delim_whitespace = True, header = None)
    probes_blast.columns = ['probe_id', 'molecule_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'probe_start', 'probe_end', 'molecule_start', 'molecule_end', 'evalue', 'bitscore', 'staxids', 'qseq', 'sseq']
    # probes = pd.read_csv(probes_filename, dtype = {'code': str})
    probes['probe_length'] = probes.loc[:,'rna_seq'].apply(len) - 6
    probes_length_df = probes.loc[:,['probe_id', 'target_taxon', 'probe_length']].drop_duplicates()
    probes_blast = probes_blast.merge(probes_length_df, on = 'probe_id', how = 'left')
    blast_lineage = pd.read_table(blast_lineage_filename, dtype = {'staxids':str})
    blast_lineage_slim = blast_lineage.loc[:,['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']]
    probes_blast = probes_blast.merge(blast_lineage_slim, on = 'molecule_id', how = 'left')
    probes_blast_summary = probes_blast.groupby(['probe_id']).apply(summarize_final_probe_blast, mch = mch, target_rank = target_rank, meta = ('int'))
    probes_blast_summary_compute = probes_blast_summary.compute()
    probes_blast_summary_final = probes_blast_summary_compute.reset_index()
    probes_blast_summary_final.columns = ['probe_id', 'final_check']
    probes_final_check = probes.merge(probes_blast_summary_final, on = 'probe_id', how = 'left')
    probe_number_prefiltering = probes_final_check.shape[0]
    problematic_probes = probes_final_check.loc[(probes_final_check.final_check.values < bot), ['target_taxon', 'probe_id', 'probe_id_full', 'final_check']].drop_duplicates()
    probes_final_filter = probes_final_check.loc[~(probes_final_check.target_taxon.isin(problematic_probes.target_taxon.values) & probes_final_check.probe_id.isin(problematic_probes.probe_id.values)), :]
    probe_filtering_statistics_filename = '{}/{}_probe_filtering_statistics.csv'.format(design_dir, design_dir_folder)
    probe_filtering_statistics = pd.DataFrame([[probe_number_prefiltering, problematic_probes.shape[0], probes_final_filter.shape[0]]])
    probe_filtering_statistics.columns = ['PreFiltering', 'Filtered', 'PostFiltering']
    probe_filtering_statistics.to_csv(probe_filtering_statistics_filename, header = True, index = False)
    probes_final_filter.loc[:,'probe_full_seq_length'] = probes_final_filter.probe_full_seq.apply(len)
    probes_final_filter.to_csv('{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), header = True, index = False)
    probes_final_filter_summary = probes_final_filter.target_taxon.value_counts()
    probes_final_filter_summary.columns = ['Taxon', 'Probe_Richness']
    probes_final_filter_summary.to_csv('{}/{}_primerset_{}_barcode_selection_{}_probe_richness_summary.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection))
    probes_final_filter.loc[:,'probe_full_seq'].str.upper().to_csv('{}/{}_full_length_probes_sequences.txt'.format(design_dir, design_dir_folder), header = False, index = False)
    probes_order_format = probes_final_filter[['abundance', 'probe_id', 'probe_full_seq']]
    probes_order_format = probes_order_format.assign(Amount = '25nm', Purification = 'STD')
    probes_order_format.loc[probes_order_format.loc[:,'probe_full_seq'].str.len() > 60, 'Amount'] = '100nm'
    probes_order_format.loc[probes_order_format.loc[:,'probe_full_seq'].str.len() > 60, 'Purification'] = 'PAGE'
    probes_order_format = probes_order_format.sort_values(by = ['abundance', 'probe_id'], ascending = [False,True]).reset_index().drop(columns = ['index'])
    probes_order_format.to_excel('{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_order_format.xlsx'.format(design_dir, design_dir_folder, primerset, barcode_selection))
    return(probes_final_filter)

def generate_probe_statistics_plots(design_dir, probes_final_filter, primerset, barcode_selection, theme_color):
    design_dir_folder = os.path.split(design_dir)[1]
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    plt.hist(probes_final_filter.Tm.values, bins = 20, color = (0,0.5,1))
    plt.xlabel(r'Melting Temperature [$^\circ$C]', fontsize = 8, color = theme_color)
    plt.ylabel('Frequency', fontsize = 8, color = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.25, right = 0.98, top = 0.98)
    probes_tm_histogram_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_tm_histogram.pdf'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    fig.savefig(probes_tm_histogram_filename, dpi = 300, transparent = True)
    plt.close()
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    plt.hist(probes_final_filter.GC.values, bins = 20, color = (0,0.5,1))
    plt.xlabel('GC Content [-]', fontsize = 8, color = theme_color)
    plt.ylabel('Frequency', fontsize = 8, color = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.25, right = 0.98, top = 0.98)
    probes_gc_histogram_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_gc_histogram.pdf'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    fig.savefig(probes_gc_histogram_filename, dpi = 300, transparent = True)
    plt.close()
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    abundance_probe_multiplexity = probes_final_filter.groupby('target_taxon').agg({'abundance': np.unique, 'probe_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
    abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_id': 'probe_multiplexity'})
    abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
    plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = (0,0.5,1), alpha = 0.5)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative Abundance', fontsize = 8, color = theme_color)
    plt.ylabel('Probe Plurality', fontsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.25, right = 0.98, top = 0.98)
    abundance_plurality_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_abundance_plurality.pdf'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    fig.savefig(abundance_plurality_filename, dpi = 300, transparent = True)
    plt.close()
    abundance_plurality_table_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_abundance_plurality.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    abundance_probe_multiplexity.to_csv(abundance_plurality_table_filename)
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    plt.hist(probes_final_filter.probe_full_seq_length.values, bins = np.arange(102, 112, 1), color = (0, 0.5, 1))
    plt.xlabel('Probe Length [bp]', fontsize = 8, color = theme_color)
    plt.ylabel('Frequency', fontsize = 8, color = theme_color)
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.25, right = 0.98, top = 0.98)
    probe_length_histogram_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_length_histogram.pdf'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    fig.savefig(probe_length_histogram_filename, dpi = 300, transparent = True)
    plt.close()
    return

def generate_probe_summary_file(design_dir, probes_final_filter, primerset, barcode_selection):
    design_dir_folder = os.path.split(design_dir)[1]
    probes_summary_filename = '{}/{}_full_length_probes_summary.txt'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    with open(probes_summary_filename, 'w') as tf:
        tf.write('Probe design complete.')
    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')

    parser.add_argument('design_dir', type = str, help = 'Input file containing blast results')

    parser.add_argument('utilities_directory', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('evaluation_directory', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('blast_database', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('bot', type = float, help = 'Input file containing blast lineage')

    parser.add_argument('mch', type = int, help = 'Input file containing blast lineage')

    parser.add_argument('bplc', type = int, help = 'Input file containing blast lineage')


    parser.add_argument('-n_workers', '--n_workers', dest = 'n_workers', default = 20, type = int, help = 'Input file containing blast lineage')
    parser.add_argument('-ps', '--primerset', dest = 'primerset', default = 'B', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-plf', '--plf', dest = 'plf', default = 'T', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-p', '--primer', dest = 'primer', default = 'T', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-t', '--target_rank', dest = 'target_rank', default = '', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-sm', '--selection_method', dest = 'selection_method', default = 'AllSpecificPStartGroup', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-bs', '--barcode_selection', dest = 'barcode_selection', default = 'MostSimple', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-hr', '--helper_probe_repeat', dest = 'helper_probe_repeat', default = 15, type = int, help = 'Input file containing blast lineage')
    parser.add_argument('-tc', '--theme_color', dest = 'theme_color', default = 'white', type = str, help = 'Input file containing blast lineage')
    args = parser.parse_args()

    print('Generating full probes for design {}...'.format(os.path.basename(args.design_dir)))
    taxon_best_probes = glob.glob('{}/*_probe_selection.csv'.format(args.design_dir))
    blast_lineage_filename = '{}/blast_lineage.tab'.format(args.utilities_directory)
    design_dir_folder = os.path.split(args.design_dir)[1]
    oligo_df = generate_full_probes(args.design_dir, args.bot, plf = args.plf, primer = args.primer, primerset = args.primerset, barcode_selection = args.barcode_selection, selection_method = args.selection_method)
    # write_final_probes_fasta(oligo_df, args.design_dir, args.primerset, args.barcode_selection)
    print('Writing final probes fasta...')
    write_final_unique_probes_fasta(oligo_df, args.design_dir, args.primerset, args.barcode_selection)
    print('Blasting final probes...')
    blast_final_probes(args.design_dir, args.primerset, args.barcode_selection, args.blast_database)
    probes_final_filter = check_final_probes_blast(args.design_dir, oligo_df, args.mch, args.bot, args.target_rank, blast_lineage_filename, args.primerset, args.barcode_selection)
    cluster = LocalCluster(n_workers = args.n_workers, threads_per_worker = 1, memory_limit = '0')
    client = Client(cluster)
    blocking_probes_sequences_filename = '{}/{}_full_length_blocking_probes_sequences.txt'.format(args.design_dir, design_dir_folder)
    if not os.path.exists(blocking_probes_sequences_filename):
        generate_blocking_probes(args.design_dir, args.bplc, args.blast_database, args.evaluation_directory, args.target_rank, plf = args.plf, primer = args.primer, primerset = args.primerset, barcode_selection = args.barcode_selection)
    generate_helper_probes(args.design_dir, args.evaluation_directory, args.target_rank, args.mch, plf = args.plf, primer = args.primer, primerset = args.primerset, barcode_selection = args.barcode_selection, helper_probe_repeat = args.helper_probe_repeat)
    generate_probe_statistics_plots(args.design_dir, probes_final_filter, args.primerset, args.barcode_selection, args.theme_color)
    generate_probe_summary_file(args.design_dir, probes_final_filter, args.primerset, args.barcode_selection)
    client.close()
    cluster.close()
    return

if __name__ == '__main__':
    main()
