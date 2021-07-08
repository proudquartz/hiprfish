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
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.SeqUtils import GC
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

def generate_full_probes(design_dir, bot, plf = 'T', primer = 'T', primerset = 'B', barcode_selection = 'MostSimple'):
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
    taxon_best_probes_filtered_list = [x for x in taxon_best_probes_list if x.blast_on_target_rate.max() > bot]
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
        probes = probes.loc[(probes['blast_on_target_rate'] > bot), :]
        assigned = barcodes_sorted.NumericBarcode.values[i]
        if barcodes_sorted.NumberBits.values[i] > 2:
            barcode_repeat = np.round(15/barcodes_sorted.NumberBits.values[i]).astype(int)
        else:
            barcode_repeat = 15
        if probes.shape[0] > 0:
            for k in range(barcode_repeat):
                probes = probes.sort_values(by = 'quality', ascending = True).reset_index().drop(columns = ['index'])
                for i in range(0, probes.shape[0]):
                    tarseq = probes['seqrcsa'][i]
                    if 'N' not in list(tarseq):
                        code = np.asarray(list(re.sub('0b', '', format(assigned, '#0' + str(7+2) + 'b'))), dtype = np.int64)
                        indices = np.where(code == 1)[0]
                        if len(indices) > 2:
                            indices = np.append(indices, indices[0])
                            subcode = np.zeros((len(indices)-1, nbit), dtype = np.int64)
                            for j in range(0, len(indices) - 1):
                                subcode[j, indices[j]] = 1
                                subcode[j, indices[j+1]] = 1
                            oligo = [[str(probes['target_taxon_full'][i]), probes['p_start'][i], probes['ln'][i], probes['taxon_abundance'][i], tarseq, probes['Tm'][i], probes['GC'][i], re.sub('\[|\]| ','',str(code)), assigned, probes['probe_id'][i], str(probes['target_taxon_full'][i]) + '_' + str(assigned) + '_' + re.sub('\[|\]| ','',str(subcode[k])) + '_' + str(probes['probe_id'][i]), forward_primer + RP[np.where(subcode[k] == 1)[0][0]] + tarseq + RP[np.where(subcode[k] == 1)[0][1]] + reverse_primer] for k in range(0, len(subcode))]
                        elif len(indices) == 2:
                            oligo = [[str(probes['target_taxon_full'][i]), probes['p_start'][i], probes['ln'][i], probes['taxon_abundance'][i], tarseq, probes['Tm'][i], probes['GC'][i], re.sub('\[|\]| ','',str(code)), assigned, probes['probe_id'][i], str(probes['target_taxon_full'][i]) + '_' + str(assigned) + '_' + re.sub('\[|\]| ','', str(code)) + '_' + str(probes['probe_id'][i]), forward_primer + RP[np.where(code == 1)[0][0]] + tarseq + RP[np.where(code == 1)[0][1]] + reverse_primer]]
                        else:
                            oligo = [[str(probes['target_taxon_full'][i]), probes['p_start'][i], probes['ln'][i], probes['taxon_abundance'][i], tarseq, probes['Tm'][i], probes['GC'][i], re.sub('\[|\]| ','',str(code)), assigned, probes['probe_id'][i], str(probes['target_taxon_full'][i]) + '_' + str(assigned) + '_' + re.sub('\[|\]| ','',str(code)) + '_' + str(probes['probe_id'][i]), forward_primer + RP[np.where(code == 1)[0][0]] + tarseq + probe_length_filler + reverse_primer]]
                        oligo_list = oligo_list + oligo
    oligo_df = pd.DataFrame(oligo_list)
    oligo_df.columns = ['target_taxon', 'p_start', 'ln', 'abundance', 'rna_seq', 'Tm', 'GC', 'code', 'numeric_code', 'probe_numeric_id', 'probe_id', 'probe_full_seq']
    return(oligo_df)

def generate_blocking_probes(design_dir, bplc, target_rank, plf = 'T', primer = 'T', primerset = 'B', barcode_selection = 'MostSimple'):
    design_dir_folder = os.path.split(design_dir)[1]
    blocking_probes_filenames = glob.glob('{}/*_off_target_summary_info.csv'.format(design_dir))
    probes_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    probes = pd.read_csv(probes_filename)
    if '{}/0_off_target_summary_info.csv'.format(design_dir) in blocking_probes_filenames:
        blocking_probes_filenames.remove('{}/0_off_target_summary_info.csv'.format(design_dir))
    blocking_probes_list = []
    for f in blocking_probes_filenames:
        target_taxon = re.sub('_off_target_summary_info.csv', '', os.path.basename(f))
        if not probes.loc[probes.target_taxon.values.astype(str) == target_taxon, :].empty:
            off_target_summary_full = pd.read_csv(f)
            if not off_target_summary_full.empty:
                off_target_summary_full.loc[:, 'max_average_encoding_interference_fraction_0bp'] = 0.0
                off_target_summary_full.loc[:, 'max_average_encoding_interference_fraction_5bp'] = 0.0
                off_target_summary_full.loc[:, 'max_average_encoding_interference_fraction_10bp'] = 0.0
                for bp in off_target_summary_full.probe_id.drop_duplicates().values:
                    off_target_summary = off_target_summary_full.loc[off_target_summary_full.probe_id.values == bp, :]
                    off_target_summary.loc[:, 'average_encoding_interference_fraction_0bp'] = 0.0
                    off_target_summary.loc[:, 'average_encoding_interference_fraction_5bp'] = 0.0
                    off_target_summary.loc[:, 'average_encoding_interference_fraction_10bp'] = 0.0
                    blocked_taxa = off_target_summary[target_rank].drop_duplicates()
                    for tt in blocked_taxa.values:
                        off_target_summary_blocked_taxon = off_target_summary.loc[off_target_summary[target_rank].values == tt, :]
                        off_target_summary_p_start = off_target_summary_blocked_taxon.loc[:, ['molecule_start', 'molecule_end']].drop_duplicates()
                        taxon_probes = probes.loc[probes.target_taxon.values == tt, :]
                        encoding_p_start = taxon_probes.loc[:,['p_start', 'ln']].drop_duplicates()
                        if not encoding_p_start.empty:
                            significant_overlap_fraction = np.zeros((encoding_p_start.shape[0], 3))
                            for i in range(encoding_p_start.shape[0]):
                                min_end = off_target_summary_p_start.molecule_start.apply(min, args = (encoding_p_start.p_start.values[i] + encoding_p_start.ln.values[i], ))
                                max_start = off_target_summary_p_start.molecule_end.apply(max, args = (encoding_p_start.p_start.values[i], ))
                                overlap = min_end - max_start
                                significant_overlap_fraction[i, 0] = np.sum(overlap > 0)/overlap.shape[0]
                                significant_overlap_fraction[i, 1] = np.sum(overlap > 5)/overlap.shape[0]
                                significant_overlap_fraction[i, 2] = np.sum(overlap > 10)/overlap.shape[0]
                            off_target_summary.loc[off_target_summary[target_rank].values == tt, 'average_encoding_interference_fraction_0bp'] = np.average(significant_overlap_fraction[:,0])
                            off_target_summary.loc[off_target_summary[target_rank].values == tt, 'average_encoding_interference_fraction_5bp'] = np.average(significant_overlap_fraction[:,1])
                            off_target_summary.loc[off_target_summary[target_rank].values == tt, 'average_encoding_interference_fraction_10bp'] = np.average(significant_overlap_fraction[:,2])
                    off_target_summary_full.loc[off_target_summary_full.probe_id.values == bp, 'max_average_encoding_interference_fraction_0bp'] = off_target_summary.average_encoding_interference_fraction_0bp.max()
                    off_target_summary_full.loc[off_target_summary_full.probe_id.values == bp, 'max_average_encoding_interference_fraction_5bp'] = off_target_summary.average_encoding_interference_fraction_5bp.max()
                    off_target_summary_full.loc[off_target_summary_full.probe_id.values == bp, 'max_average_encoding_interference_fraction_10bp'] = off_target_summary.average_encoding_interference_fraction_10bp.max()
            blocking_probes_list.append(off_target_summary_full)
    blocking_probes = pd.concat(blocking_probes_list, sort = False)
    blocking_probes = blocking_probes.loc[blocking_probes.length.values >= bplc, :]
    blocking_probes.to_csv('{}/{}_primerset_{}_barcode_selection_{}_full_length_blocking_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), index = False)
    blocking_probes_order_format = blocking_probes[['sseq', 'length']]
    blocking_probes_order_format.columns = ['Sequence', 'Length']
    blocking_probes_order_format = blocking_probes_order_format.drop_duplicates()
    blocking_probes_order_format = blocking_probes_order_format.sort_values(by = 'Length', ascending = False)
    # blocking_probes_order_format = blocking_probes_order_format.loc[blocking_probes_order_format.length.values >= bplc]
    blocking_probes_order_format.insert(0, 'Blocking_Probe_Name', ['BP_{}'.format(i) for i in range(blocking_probes_order_format.shape[0])])
    blocking_probes_order_format = blocking_probes_order_format.assign(Amount = '25nm', Purification = 'STD')
    blocking_probes_order_format.loc[blocking_probes_order_format.Length.values < 15, 'Amount'] = '100nm'
    blocking_probes_order_format = blocking_probes_order_format[['Blocking_Probe_Name', 'Sequence', 'Amount', 'Purification', 'Length']]
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

def helper_probe_blast_summarize(probe_blast, probes, max_continuous_homology, target_rank):
    probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
    if probe_blast_filtered.shape[0] > 0:
        interfered_taxa = pd.DataFrame(probe_blast_filtered.loc[:,target_rank].drop_duplicates())
        interfered_taxa['average_encoding_interference_fraction_0bp'] = 0.0
        interfered_taxa['average_encoding_interference_fraction_5bp'] = 0.0
        interfered_taxa['average_encoding_interference_fraction_10bp'] = 0.0
        for tt in interfered_taxa[target_rank].values:
            helper_probes_interfered_taxon = probe_blast_filtered.loc[probe_blast_filtered[target_rank].values == tt, :]
            helper_probes_p_start = helper_probes_interfered_taxon.loc[:, ['molecule_start', 'molecule_end']].drop_duplicates()
            taxon_probes = probes.loc[probes.target_taxon.values == tt, :]
            encoding_p_start = taxon_probes.loc[:,['p_start', 'ln']].drop_duplicates()
            if not encoding_p_start.empty:
                significant_overlap_fraction = np.zeros((encoding_p_start.shape[0], 3))
                for i in range(encoding_p_start.shape[0]):
                    min_end = helper_probes_p_start.molecule_start.apply(min, args = (encoding_p_start.p_start.values[i] + encoding_p_start.ln.values[i], ))
                    max_start = helper_probes_p_start.molecule_end.apply(max, args = (encoding_p_start.p_start.values[i], ))
                    overlap = min_end - max_start
                    significant_overlap_fraction[i, 0] = np.sum(overlap > 0)/overlap.shape[0]
                    significant_overlap_fraction[i, 1] = np.sum(overlap > 5)/overlap.shape[0]
                    significant_overlap_fraction[i, 2] = np.sum(overlap > 10)/overlap.shape[0]
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_0bp'] = np.average(significant_overlap_fraction[:,0])
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_5bp'] = np.average(significant_overlap_fraction[:,1])
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_10bp'] = np.average(significant_overlap_fraction[:,2])
            max_average_encoding_interference_fraction_0bp = interfered_taxa.average_encoding_interference_fraction_0bp.max()
            max_average_encoding_interference_fraction_5bp = interfered_taxa.average_encoding_interference_fraction_5bp.max()
            max_average_encoding_interference_fraction_10bp = interfered_taxa.average_encoding_interference_fraction_10bp.max()
    else:
        max_average_encoding_interference_fraction_0bp = 0.0
        max_average_encoding_interference_fraction_5bp = 0.0
        max_average_encoding_interference_fraction_10bp = 0.0
    return(pd.Series({'probe_id': probe_blast.probe_id.values[0],
                      'max_average_encoding_interference_fraction_0bp': max_average_encoding_interference_fraction_0bp,
                      'max_average_encoding_interference_fraction_5bp': max_average_encoding_interference_fraction_5bp,
                      'max_average_encoding_interference_fraction_10bp': max_average_encoding_interference_fraction_10bp}))

def helper_probe_blast_summarize_v2(probe_blast, probes, max_continuous_homology, target_rank):
    probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
    if probe_blast_filtered.shape[0] > 0:
        interfered_taxa = pd.DataFrame(probe_blast_filtered.loc[:,target_rank].drop_duplicates())
        interfered_taxa.loc[:, 'average_encoding_interference_fraction_0bp'] = 0.0
        interfered_taxa.loc[:, 'average_encoding_interference_fraction_5bp'] = 0.0
        interfered_taxa.loc[:, 'average_encoding_interference_fraction_10bp'] = 0.0
        for tt in interfered_taxa[target_rank].values:
            helper_probes_interfered_taxon = probe_blast_filtered.loc[probe_blast_filtered[target_rank].values == tt, :]
            helper_probes_p_start = helper_probes_interfered_taxon.loc[:, ['molecule_start', 'molecule_end']].drop_duplicates()
            taxon_probes = probes.loc[probes.target_taxon.values == tt, :]
            encoding_p_start = taxon_probes.loc[:,['p_start', 'ln']].drop_duplicates()
            encoding_p_start['p_end'] = encoding_p_start.p_start.values + encoding_p_start.ln.values - 1
            if not encoding_p_start.empty:
                significant_overlap_fraction = np.zeros((encoding_p_start.shape[0], 3))
                print(helper_probes_p_start.molecule_start.drop_duplicates())
                min_end = encoding_p_start.p_end.apply(min, args = (helper_probes_p_start.molecule_start.values[0], ))
                max_start = encoding_p_start.p_start.apply(max, args = (helper_probes_p_start.molecule_end.values.values[0], ))
                overlap = min_end - max_start
                significant_overlap_fraction[:, 0] = np.sum(overlap > 0)/overlap.shape[0]
                significant_overlap_fraction[:, 1] = np.sum(overlap > 5)/overlap.shape[0]
                significant_overlap_fraction[:, 2] = np.sum(overlap > 10)/overlap.shape[0]
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_0bp'] = np.average(significant_overlap_fraction[:,0])
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_5bp'] = np.average(significant_overlap_fraction[:,1])
                interfered_taxa.loc[tt, 'average_encoding_interference_fraction_10bp'] = np.average(significant_overlap_fraction[:,2])
            max_average_encoding_interference_fraction_0bp = interfered_taxa.average_encoding_interference_fraction_0bp.max()
            max_average_encoding_interference_fraction_5bp = interfered_taxa.average_encoding_interference_fraction_5bp.max()
            max_average_encoding_interference_fraction_10bp = interfered_taxa.average_encoding_interference_fraction_10bp.max()
    else:
        max_average_encoding_interference_fraction_0bp = 0.0
        max_average_encoding_interference_fraction_5bp = 0.0
        max_average_encoding_interference_fraction_10bp = 0.0
    return(pd.Series({'probe_id': int(probe_blast.probe_id.values[0]),
                      'max_average_encoding_interference_fraction_0bp': max_average_encoding_interference_fraction_0bp,
                      'max_average_encoding_interference_fraction_5bp': max_average_encoding_interference_fraction_5bp,
                      'max_average_encoding_interference_fraction_10bp': max_average_encoding_interference_fraction_10bp}))

def generate_helper_probes(design_dir, blast_directory, target_rank, max_continuous_homology, plf = 'T', primer = 'T', primerset = 'B', barcode_selection = 'MostSimple', helper_probe_repeat = 15):
    design_dir_folder = os.path.split(design_dir)[1]
    helper_probes_filenames = glob.glob('{}/*_helper_probes.csv'.format(design_dir))
    helper_probes = pd.concat([pd.read_csv(f) for f in helper_probes_filenames], sort = False)
    probes_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    probes = pd.read_csv(probes_filename)
    if '{}/0_helper_probes.csv'.format(design_dir) in helper_probes_filenames:
        helper_probes_filenames.remove('{}/0_helper_probes.csv'.format(design_dir))
    helper_probes_filenames.sort()
    helper_probes_list = []
    for f in helper_probes_filenames:
        target_taxon = re.sub('_helper_probes.csv', '', os.path.basename(f))
        probe_evaluation_filename = '{}/{}.probe.evaluation.h5'.format(blast_directory, target_taxon)
        if not probes.loc[probes.target_taxon.values.astype(str) == target_taxon, :].empty:
            helper_probes = pd.read_csv(f)
            helper_probes_summary = pd.DataFrame()
            for hp_idx in helper_probes.probe_id:
                probe_name = 'probe_' + str(hp_idx)
                try:
                    probe_blast = pd.read_hdf(probe_evaluation_filename, probe_name)
                    probe_blast = probe_blast[probe_blast['mch'] >= max_continuous_homology]
                    p_start_group = np.floor(helper_probes.loc[helper_probes.probe_id == hp_idx, 'p_start'].values[0]/25).astype(int)
                    probe_blast['p_start_group'] = p_start_group
                    if probe_blast.shape[0] > 0:
                        helper_probes_summary = helper_probes_summary.append(helper_probe_blast_summarize(probe_blast, probes, max_continuous_homology = max_continuous_homology, target_rank = target_rank), ignore_index = True, sort = False)
                except KeyError:
                    pass
            helper_probes = helper_probes.merge(helper_probes_summary, on = 'probe_id')
            helper_probes['seqrc'] = ''
            for i in range(helper_probes.shape[0]):
                helper_probes.loc[i, 'seqrc'] = str(Seq(helper_probes.loc[i, 'seq']).reverse_complement())
            helper_probes.loc[:,'quadg'] = (helper_probes['seqrc'].str.upper().str.count('GGGG')) + (helper_probes['seqrc'].str.upper().str.count('GGGGG'))
            helper_probes = helper_probes.loc[(helper_probes.quadg.values == 0) & (helper_probes.max_average_encoding_interference_fraction_0bp.values < 1e-3),:]
            for hsp in helper_probes.helper_source_probe.drop_duplicates():
                helper_probes_taxon_hsp = helper_probes.loc[helper_probes.helper_source_probe.values == hsp, :]
                for group in helper_probes_taxon_hsp.helper_group.drop_duplicates():
                    helper_probes_group = helper_probes_taxon_hsp.loc[helper_probes_taxon_hsp.helper_group.values == group, :]
                    helper_probes_list.append(helper_probes_group.iloc[[0],:])
        else:
            print(f)
    helper_probes_selection = pd.concat(helper_probes_list, sort = False)
    helper_probes_selection.to_csv('{}/{}_primerset_{}_barcode_selection_{}_full_length_helper_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), index = False)
    helper_probes_order_format = helper_probes_selection[['seqrc', 'ln']]
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
    print(helper_probe_seq_list.shape[0])
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
    probes_list = [SeqRecord(Seq(probes['probe_full_seq'][i]), id = str(probes['probe_id'][i]), description = '') for i in range (0, probes.shape[0])]
    SeqIO.write(probes_list, probes_fasta_filename, 'fasta')
    return

def blast_final_probes(design_dir, primerset, barcode_selection, blast_database):
    # read in probe information
    # print('    Blasting ' + os.path.basename(infile))
    design_dir_folder = os.path.split(design_dir)[1]
    infile = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_unique.fasta'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    blast_output_filename = '{}/{}_full_length_probes_unique.blast.out'.format(design_dir, design_dir_folder)
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids qseq sseq'
    return_code = subprocess.call(['blastn', '-db', blast_database, '-query', infile, '-out', blast_output_filename, '-outfmt', out_format, '-task', 'blastn-short', '-max_hsps', '1', '-max_target_seqs', '100000', '-strand', 'minus', '-evalue', '100', '-num_threads', '1'])
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
    df_filtered = df.loc[df.mch >= mch]
    if df_filtered.shape[0] > 0:
        probe_id = np.unique(df_filtered['probe_id'])[0]
        probe_length = np.unique(df_filtered['probe_length'])[0]
        check = np.sum(((df_filtered['probe_start'] >= 44) & (df_filtered['probe_end'] <= 44 + probe_length - 1) & (df_filtered['mch'] <= probe_length)) | (df_filtered['target_taxon'] == df_filtered[target_rank]))/df_filtered.shape[0]
    else:
        print('{} has no blast hits...'.format(df.probe_id.values[0]))
        check = 0
    return(check)

def check_final_probes_blast(design_dir, probes, mch, bot, target_rank, blast_lineage_filename, primerset, barcode_selection):
    # probes_filename = '{}/full_length_probes.csv'.format(design_dir)
    design_dir_folder = os.path.split(design_dir)[1]
    probes_blast_filename = '{}/{}_full_length_probes_unique.blast.out'.format(design_dir, design_dir_folder)
    probes_blast = pd.read_csv(probes_blast_filename, header = None, sep = '\t')
    probes_blast.columns = ['probe_id', 'molecule_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'probe_start', 'probe_end', 'molecule_start', 'molecule_end', 'evalue', 'bitscore', 'staxids', 'qseq', 'sseq']
    # probes = pd.read_csv(probes_filename, dtype = {'code': str})
    probes.loc[:,'probe_length'] = probes.loc[:,'rna_seq'].apply(len) - 6
    probes_length_df = probes.loc[:,['probe_id', 'target_taxon', 'probe_length']]
    probes_blast = probes_blast.merge(probes_length_df, on = 'probe_id', how = 'left')
    blast_lineage = pd.read_csv(blast_lineage_filename, dtype = {'staxids':str}, sep = '\t')
    blast_lineage_slim = blast_lineage.loc[:,['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain']]
    probes_blast = probes_blast.merge(blast_lineage_slim, on = 'molecule_id', how = 'left')
    probes_blast_summary = probes_blast.groupby('probe_id', axis = 0).apply(summarize_final_probe_blast, mch = mch, target_rank = target_rank)
    probes_blast_summary = probes_blast_summary.reset_index()
    probes_blast_summary.columns = ['probe_id', 'final_check']
    probes_final_check = probes.merge(probes_blast_summary, on = 'probe_id', how = 'left')
    probe_number_prefiltering = probes_final_check.shape[0]
    problematic_probes = probes_final_check.loc[(probes_final_check.final_check < bot)][['target_taxon', 'probe_numeric_id', 'probe_id', 'final_check']].drop_duplicates()
    probes_final_filter = probes_final_check.loc[~(probes_final_check.target_taxon.isin(problematic_probes.target_taxon.values) & probes_final_check.probe_numeric_id.isin(problematic_probes.probe_numeric_id.values))]
    probe_filtering_statistics_filename = '{}/{}_probe_filtering_statistics.csv'.format(design_dir, design_dir_folder)
    probe_filtering_statistics = pd.DataFrame([[probe_number_prefiltering, problematic_probes.shape[0], probes_final_filter.shape[0]]])
    probe_filtering_statistics.columns = ['PreFiltering', 'Filtered', 'PostFiltering']
    probe_filtering_statistics.to_csv(probe_filtering_statistics_filename, header = True, index = False)
    probes_final_filter.loc[:,'probe_full_seq_length'] = probes_final_filter.probe_full_seq.apply(len)
    probes_final_filter.to_csv('{}/{}_primerset_{}_barcode_selection_{}_full_length_probes.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), header = True, index = False)
    probes_final_filter_summary = probes_final_filter.target_taxon.value_counts()
    probes_final_filter_summary.columns = ['Taxon', 'Probe_Richness']
    probes_final_filter_summary.to_csv('{}/{}_primerset_{}_barcode_selection_{}_probe_richness_summary.csv'.format(design_dir, design_dir_folder, primerset, barcode_selection), header = True, index = False)
    probes_final_filter.loc[:,'probe_full_seq'].str.upper().to_csv('{}/{}_full_length_probes_sequences.txt'.format(design_dir, design_dir_folder), header = False, index = False)
    probes_order_format = probes_final_filter.loc[:,['abundance', 'probe_id', 'probe_full_seq']].copy()
    probes_order_format = probes_order_format.assign(Amount = '25nm', Purification = 'STD')
    probes_order_format.loc[probes_order_format.loc[:,'probe_full_seq'].str.len() > 60, 'Amount'] = '100nm'
    probes_order_format.loc[probes_order_format.loc[:,'probe_full_seq'].str.len() > 60, 'Purification'] = 'PAGE'
    probes_order_format = probes_order_format.sort_values(by = ['abundance', 'probe_id'], ascending = [False,True]).reset_index().drop(columns = ['index'])
    probes_order_format.to_excel('{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_order_format.xlsx'.format(design_dir, design_dir_folder, primerset, barcode_selection))
    return(probes_final_filter)

def generate_probe_statistics_plots(design_dir, probes_final_filter, primerset, barcode_selection, theme_color, target_rank):
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
    abundance_probe_multiplexity = probes_final_filter.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
    abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
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
    off_target_filenames = glob.glob('{}/*_off_target_summary_info.csv'.format(design_dir))
    off_target_filenames.sort()
    taxid_list = probes_final_filter.target_taxon.drop_duplicates().astype(str).values
    taxid_list.sort()
    ot_gc = pd.DataFrame(np.zeros((taxid_list.shape[0], taxid_list.shape[0])))
    ot_gc.columns = taxid_list
    ot_gc.index = taxid_list
    for i in range(taxid_list.shape[0]):
        ot = pd.read_csv(off_target_filenames[i], dtype = {'species':str})
        ot['OTGC'] = ot.qseq.apply(GC)
        ot['OTGCINT'] = ot.OTGC.values*ot.length.values
        ot_gc_temp = ot.groupby(target_rank).agg({'OTGCINT': 'max'}).reset_index().sort_values(by = target_rank, ascending = True)
        ot_gc.loc[ot_gc_temp.loc[:,target_rank].values,taxid_list[i]] = ot_gc_temp.OTGCINT.values/100
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(7))
    plt.imshow(ot_gc.values, cmap = 'inferno')
    plt.xticks(np.arange(taxid_list.shape[0]), ot_gc.columns, rotation = -90, fontsize = 8, color = theme_color)
    plt.yticks(np.arange(taxid_list.shape[0]), ot_gc.columns, fontsize = 8, color = theme_color)
    plt.axes().xaxis.tick_top()
    plt.axes().spines['left'].set_color(theme_color)
    plt.axes().spines['bottom'].set_color(theme_color)
    plt.axes().spines['right'].set_color(theme_color)
    plt.axes().spines['top'].set_color(theme_color)
    plt.tick_params(direction = 'in', length = 0, labelsize = 8, color = theme_color)
    plt.subplots_adjust(left = 0.25, bottom = 0.02, right = 0.98, top = 0.75)
    cbar = plt.colorbar(fraction = 0.1)
    cbar.ax.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    cbar.ax.spines['left'].set_color(theme_color)
    cbar.ax.spines['bottom'].set_color(theme_color)
    cbar.ax.spines['right'].set_color(theme_color)
    cbar.ax.spines['top'].set_color(theme_color)
    probe_ot_gc_filename = '{}/{}_primerset_{}_barcode_selection_{}_full_length_probes_ot_gc_heatmap.pdf'.format(design_dir, design_dir_folder, primerset, barcode_selection)
    fig.savefig(probe_ot_gc_filename, dpi = 300, transparent = True)
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

    parser.add_argument('consensus_directory', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('utilities_directory', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('blast_directory', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('blast_database', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('bot', type = float, help = 'Input file containing blast lineage')

    parser.add_argument('mch', type = int, help = 'Input file containing blast lineage')

    parser.add_argument('bplc', type = int, help = 'Input file containing blast lineage')

    parser.add_argument('-ps', '--primerset', dest = 'primerset', default = 'B', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-plf', '--plf', dest = 'plf', default = 'T', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-p', '--primer', dest = 'primer', default = 'T', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-t', '--target_rank', dest = 'target_rank', default = '', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-bs', '--barcode_selection', dest = 'barcode_selection', default = 'MostSimple', type = str, help = 'Input file containing blast lineage')
    parser.add_argument('-hr', '--helper_probe_repeat', dest = 'helper_probe_repeat', default = 15, type = int, help = 'Input file containing blast lineage')
    parser.add_argument('-tc', '--theme_color', dest = 'theme_color', default = 'white', type = str, help = 'Input file containing blast lineage')
    args = parser.parse_args()

    print('Generating full probes for design {}...'.format(os.path.basename(args.design_dir)))
    taxon_best_probes = glob.glob('{}/*_probe_selection.csv'.format(args.design_dir))
    blast_lineage_filename = '{}/blast_lineage_strain.tab'.format(args.consensus_directory)
    oligo_df = generate_full_probes(args.design_dir, args.bot, plf = args.plf, primer = args.primer, primerset = args.primerset, barcode_selection = args.barcode_selection)
    write_final_probes_fasta(oligo_df, args.design_dir, args.primerset, args.barcode_selection)
    write_final_unique_probes_fasta(oligo_df, args.design_dir, args.primerset, args.barcode_selection)
    blast_final_probes(args.design_dir, args.primerset, args.barcode_selection, args.blast_database)
    probes_final_filter = check_final_probes_blast(args.design_dir, oligo_df, args.mch, args.bot, args.target_rank, blast_lineage_filename, args.primerset, args.barcode_selection)
    generate_blocking_probes(args.design_dir, args.bplc, args.target_rank, plf = args.plf, primer = args.primer, primerset = args.primerset, barcode_selection = args.barcode_selection)
    generate_helper_probes(args.design_dir, args.blast_directory, args.target_rank, args.mch, plf = args.plf, primer = args.primer, primerset = args.primerset, barcode_selection = args.barcode_selection, helper_probe_repeat = args.helper_probe_repeat)
    generate_probe_statistics_plots(args.design_dir, probes_final_filter, args.primerset, args.barcode_selection, args.theme_color, args.target_rank)
    generate_probe_summary_file(args.design_dir, probes_final_filter, args.primerset, args.barcode_selection)
    return

if __name__ == '__main__':
    main()
