import os
import re
import glob
import random
import argparse
import itertools
import threading
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from ete3 import NCBITaxa
from SetCoverPy import setcover
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Blast.Applications import NcbiblastnCommandline
import tables

pd.options.mode.chained_assignment = None

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH Probe Design Pipeline
###############################################################################################################

###############################################################################################################
# Workflow functions
###############################################################################################################

def select_all_specific_probes(probe_summary_info, bot):
    if np.max(probe_summary_info['blast_on_target_rate']) > bot:
        best_probes = probe_summary_info[probe_summary_info['blast_on_target_rate'] > bot]
        best_probes.loc[:,'selection_method'] = 'AllSpecific'
    else:
        best_probes = probe_summary_info.iloc[[0],:]
        best_probes.loc[:,'selection_method'] = 'AllSpecificSingleBest'
    return(best_probes)

def select_all_specific_p_start_group_probes(probe_summary_info, group_distance, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff):
    best_probes_group = pd.DataFrame()
    best_probes = probe_summary_info.loc[(probe_summary_info['blast_on_target_rate'] > bot) & (probe_summary_info['off_target_max_bitscore'] < bitscore_thresh) & (probe_summary_info['off_target_max_tm'] < mt_cutoff) & (probe_summary_info['off_target_max_gc'] < ot_gc_cutoff)]
    if not best_probes.empty:
        for group in range(int(np.floor(1500/group_distance))):
            best_probes_temp = best_probes.loc[best_probes.p_start_group.values == group]
            best_probes_temp.sort_values(['taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'on_target_full_match', 'quality'], ascending = [False, True, True, True, True, True, False, True], inplace = True)
            if not best_probes_temp.empty:
                best_probes_group = best_probes_group.append(best_probes_temp.iloc[[0],:], sort = False)
                best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroup'
    else:
        probe_summary_info.sort_values(['blast_on_target_rate', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'on_target_full_match', 'quality'], ascending = [False, False, True, True, True, True, True, False, True], inplace = True)
        best_probes_group = probe_summary_info.iloc[[0],:]
        best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroupSingleBest'
    return(best_probes_group)

def select_min_overlap_probes(probe_summary_info, taxon_fasta_filename, probe_evaluation_filename, max_continuous_homology, bot):
    probe_summary_info.sort_values(['blast_on_target_rate', 'taxon_coverage', 'quality'], ascending = [False, False, True], inplace = True)
    if (probe_summary_info['blast_on_target_rate'][0] > bot and probe_summary_info['taxon_coverage'][0] > bot):
        best_probes = probe_summary_info.iloc[[0], :]
        best_probes.loc[:,'selection_method'] = 'MinOverlapSingleBest'
    elif probe_summary_info['blast_on_target_rate'][0] > bot:
        taxon_molecules = [record.id for record in SeqIO.parse(taxon_fasta_filename, 'fasta')]
        taxon_molecules_set = [sub_slash(mol) for mol in set(taxon_molecules)]
        probe_summary_filtered = probe_summary_info[probe_summary_info['blast_on_target_rate'] > bot]
        probe_ids = probe_summary_filtered['probe_id'].unique()
        cover_matrix = np.zeros((len(taxon_molecules), len(probe_ids)), dtype = int)
        cost = np.ones(len(probe_ids), dtype = int)
        for i in range(probe_summary_filtered.shape[0]):
            probe_idx = probe_summary_filtered.probe_id.values[i]
            probe_info = probe_summary_filtered[probe_summary_filtered.probe_id == probe_idx]
            probe_name = 'probe_' + str(probe_idx)
            probe_blast = pd.read_hdf(probe_evaluation_filename, probe_name)
            probe_blast = probe_blast[probe_blast['mch'] >= max_continuous_homology]
            blasted_molecules = list(probe_blast['molecule_id'])
            indices = [i for i, e in enumerate(blasted_molecules) if e in taxon_molecules_set]
            cover_matrix[indices,i] = 1
        cover_matrix_filt = np.array(np.delete(cover_matrix, np.where(np.sum(cover_matrix, axis = 1) == 0), axis = 0), dtype = bool)
        g = setcover.SetCover(cover_matrix_filt, cost)
        g.SolveSCP()
        set_cover_indices = np.flatnonzero(g.s*1 == 1)
        best_probes = probe_summary_filtered.iloc[set_cover_indices,:]
        if set_cover_indices.shape[0] > 1:
            best_probes.loc[:,'selection_method'] = 'MinOverlap'
        else:
            best_probes.loc[:,'selection_method'] = 'MinOverlapSingleBest'
    else:
        best_probes = probe_summary_info.iloc[[0],:]
        best_probes.loc[:,'selection_method'] = 'MinOverlapSingleBest'
    return(best_probes)

def select_top_probes(probe_summary_info, tpn, bot):
    if np.max(probe_summary_info['blast_on_target_rate']) > bot:
        best_probes_all = probe_summary_info[probe_summary_info['blast_on_target_rate'] > bot]
        if best_probes_all.shape[0] > nprobes:
            best_probes = best_probes_all.iloc[0:nprobes,:]
            best_probes.loc[:,'selection_method'] = 'Top' + str(nprobes)
        else:
            best_probes = best_probes_all
            best_probes.loc[:,'selection_method'] = 'AllTop'
    else:
        best_probes = probe_summary_info.iloc[[0],:]
        best_probes.loc[:,'selection_method'] = 'AllTopSingleBest'
    return(best_probes)

def get_off_target_last_common_taxon_rank(df, target_rank, target_taxon):
    ncbi = NCBITaxa()
    if (target_taxon != 0) & (df.loc[target_rank] != 0):
        if not pd.isnull(df.loc[target_rank]):
            last_common_taxon = ncbi.get_topology([df[target_rank], target_taxon])
            last_common_taxon_rank = last_common_taxon.rank
            if last_common_taxon_rank != 'no rank':
                lineage = ncbi.get_lineage(last_common_taxon.taxid)
                last_common_taxon_rank = ncbi.get_rank([lineage[-1]])[lineage[-1]]
            else:
                last_common_taxon_rank = 'no rank'
        else:
            last_common_taxon_rank = 'no rank'
    else:
        last_common_taxon_rank = 'no rank'
    return(last_common_taxon_rank)

def probe_blast_summarize(probe_blast, max_continuous_homology, taxon_abundance, target_rank):
    if target_rank == 'strain':
        probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
        blast_on_target_rate = probe_blast_filtered.loc[:,'target_taxon_full_hit'].sum()/probe_blast_filtered.shape[0]
        blast_on_target_full_rate = blast_on_target_rate
        taxon_coverage = probe_blast_filtered.loc[:,'target_taxon_full_hit'].sum()/taxon_abundance
        on_target_full_match = probe_blast_filtered.target_taxon_hit_full_match.sum()/taxon_abundance
    else:
        probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
        if probe_blast_filtered.shape[0] > 0:
            blast_on_target_rate = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/probe_blast_filtered.shape[0]
            blast_on_target_full_rate = probe_blast_filtered.loc[:,'target_taxon_full_hit'].sum()/probe_blast_filtered.shape[0]
            taxon_coverage = probe_blast_filtered.loc[:,'target_taxon_full_hit'].sum()/taxon_abundance
            on_target_full_match = probe_blast.target_taxon_hit_full_match.sum()/taxon_abundance
        else:
            blast_on_target_rate = 0
            blast_on_target_full_rate = 0
            taxon_coverage = 0
            on_target_full_match = 0
    probe_blast_below_mch = probe_blast[probe_blast['mch'] < max_continuous_homology]
    if probe_blast_below_mch.shape[0] > 0:
        off_target_min_evalue = probe_blast_below_mch.evalue.min()
        off_target_max_bitscore = probe_blast_below_mch.bitscore.max()
        off_target_max_mch = probe_blast_below_mch.mch.max()
        probe_blast_below_mch.sort_values(['mch', 'qcovhsp'], ascending = [False, False], inplace = True)
        off_target_max_mch_sort = probe_blast_below_mch.mch.values[0]
        off_target_max_mch_qcovhsp = probe_blast_below_mch.qcovhsp.values[0]
        off_target_full_qcovhsp_fraction = np.sum(probe_blast_below_mch.qcovhsp.values > 99.9)/probe_blast_below_mch.shape[0]
        off_target_max_tm = probe_blast_below_mch.melting_temp.max()
        off_target_max_gc = probe_blast_below_mch.GC_count.max()
    else:
        off_target_min_evalue = 100
        off_target_max_bitscore = 0
        off_target_max_mch = 0
        off_target_max_mch_sort = 0
        off_target_max_mch_qcovhsp = 0
        off_target_full_qcovhsp_fraction = 0
        off_target_max_tm = 0
        off_target_max_gc = 0
    return(pd.Series({'probe_id': probe_blast.probe_id.values[0],
                      'blast_on_target_rate': blast_on_target_rate,
                      'blast_on_target_full_rate': blast_on_target_full_rate,
                      'taxon_coverage': taxon_coverage,
                      'on_target_full_match': on_target_full_match,
                      'off_target_min_evalue': off_target_min_evalue,
                      'off_target_max_bitscore': off_target_max_bitscore,
                      'off_target_max_mch': off_target_max_mch,
                      'off_target_max_mch_sort': off_target_max_mch_sort,
                      'off_target_max_mch_qcovhsp': off_target_max_mch_qcovhsp,
                      'off_target_full_qcovhsp_fraction': off_target_full_qcovhsp_fraction,
                      'off_target_max_tm': off_target_max_tm,
                      'off_target_max_gc': off_target_max_gc,
                      'taxon_abundance': taxon_abundance}))

def get_probe_blast_off_target_ranks(probe_blast, probe_info, target_rank, target_taxon):
    ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'no rank']
    if probe_info.blast_on_target_rate.values[0] < 1-1e-8:
        probe_blast_filtered = probe_blast[probe_blast['target_taxon_hit'] == False]
        off_target_last_common_taxon_rank = probe_blast_filtered.apply(get_off_target_last_common_taxon_rank, target_rank = target_rank, target_taxon = target_taxon, axis = 1)
        n_total = off_target_last_common_taxon_rank.shape[0]
        if n_total > 0:
            rank_fractions = [np.sum(off_target_last_common_taxon_rank == rank)/n_total for rank in ranks]
        else:
            rank_fractions = [np.nan for rank in ranks]
    else:
        rank_fractions = [0 for rank in ranks]
    probe_off_target_summary = pd.DataFrame([int(probe_blast.probe_id.values[0])] + rank_fractions).transpose()
    probe_off_target_summary.columns = ['probe_id'] + ['off_target_' + s for s in ranks]
    return(probe_off_target_summary)

def sub_slash(str):
    return(re.sub('/', '_', str))

def calculate_tm(df, Na = 390, dnac1_oligo = 5):
    qseq_array = df.qseq.values
    sseq_array = df.sseq.values
    tm_array = np.zeros(len(qseq_array))
    for i in range(len(qseq_array)):
        qseq = qseq_array[i]
        cseq = Seq(sseq_array[i]).complement()
        tm_array[i] = mt.Tm_NN(qseq, Na = Na, saltcorr = 7, dnac1 = dnac1_oligo*15, dnac2 = 1)
    return(tm_array)

def calculate_gc_count(df):
    qseq_array = df.qseq.values
    sseq_array = df.sseq.values
    gc_count_array = np.zeros(len(qseq_array), dtype = int)
    for i in range(len(qseq_array)):
        gc_count_array[i] = int(GC(qseq_array[i])*len(qseq_array[i])/100)
    return(gc_count_array)

def get_blast_lineage_slim(blast_lineage_filename, otu):
    if otu == 'F':
        blast_lineage_df = pd.read_csv(blast_lineage_filename, dtype = {'staxids':str}, sep = '\t')
        lineage_columns = ['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        blast_lineage_slim = blast_lineage_df[lineage_columns]
        blast_lineage_slim.loc[:,'molecule_id'] = blast_lineage_slim.molecule_id.apply(sub_slash)
    else:
        blast_lineage_df = pd.read_csv(blast_lineage, header = None, dtype = {'staxids':str}, sep = '\t')
        blast_lineage_df.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
        blast_lineage_filtered = blast_lineage_df[blast_lineage_df['rec_type'] != 'C']
        blast_lineage_filtered.loc[:,'target_taxon'] = 'Cluster' + blast_lineage_filtered['cluster_num'].astype(str)
        blast_lineage_filtered.loc[:,'molecule_id'] = blast_lineage_filtered['query_label']
        blast_lineage_filtered[blast_lineage_filtered['rec_type'] == 'S']['molecule_id'] = blast_lineage_filtered['target_label']
        lineage_columns = ['molecule_id', 'target_taxon']
        blast_lineage_slim = blast_lineage_filtered[lineage_columns]
        blast_lineage_slim.loc[:,'molecule_id'] = blast_lineage_slim.molecule_id.apply(sub_slash)
    return(blast_lineage_slim)

def get_blast_lineage_strain_slim(blast_lineage_filename, otu):
    if otu == 'F':
        blast_lineage_df = pd.read_csv(blast_lineage_filename, dtype = {'staxids':str}, sep = '\t')
        lineage_columns = ['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain']
        blast_lineage_slim = blast_lineage_df[lineage_columns]
        blast_lineage_slim.loc[:,'molecule_id'] = blast_lineage_slim.molecule_id.apply(sub_slash)
    else:
        blast_lineage_df = pd.read_csv(blast_lineage, header = None, dtype = {'staxids':str}, sep = '\t')
        blast_lineage_df.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
        blast_lineage_filtered = blast_lineage_df[blast_lineage_df['rec_type'] != 'C']
        blast_lineage_filtered.loc[:,'target_taxon'] = 'Cluster' + blast_lineage_filtered['cluster_num'].astype(str)
        blast_lineage_filtered.loc[:,'molecule_id'] = blast_lineage_filtered['query_label']
        blast_lineage_filtered[blast_lineage_filtered['rec_type'] == 'S']['molecule_id'] = blast_lineage_filtered['target_label']
        lineage_columns = ['molecule_id', 'target_taxon']
        blast_lineage_slim = blast_lineage_filtered[lineage_columns]
        blast_lineage_slim.loc[:,'molecule_id'] = blast_lineage_slim.molecule_id.apply(sub_slash)
    return(blast_lineage_slim)

def get_taxon_abundance(input_consensus_directory, target_taxon, target_taxon_full, otu, blast_lineage_slim):
    if otu == 'F':
        taxon_fasta_filename = input_consensus_directory + '/' + str(target_taxon) + '.fasta'
        taxon_consensus_filename = input_consensus_directory + '/' + str(target_taxon) + '.consensus.fasta'
        fasta_num_seq = sum(1 for record in SeqIO.parse(taxon_fasta_filename, 'fasta'))
        consensus_num_seq = sum(1 for record in SeqIO.parse(taxon_consensus_filename, 'fasta'))
        if fasta_num_seq == 1:
            taxon_abundance = 1
        elif consensus_num_seq == 1:
            cluster = 0
            taxon_uc_file = input_consensus_directory + '/' + str(target_taxon) + '.uc'
            taxon_uc = pd.read_csv(taxon_uc_file, header = None, sep = '\t')
            taxon_uc.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
            taxon_uc_cluster_counts = taxon_uc[taxon_uc['rec_type'] == 'C']
            taxon_abundance = taxon_uc_cluster_counts[taxon_uc_cluster_counts['cluster_num'] == cluster]['seq_length'].values[0]
        else:
            cluster = int(re.sub('.*_Cluster', '', target_taxon_full))
            taxon_uc_file = input_consensus_directory + '/' + str(target_taxon) + '.uc'
            taxon_uc = pd.read_csv(taxon_uc_file, header = None, sep = '\t')
            taxon_uc.columns = ['rec_type', 'cluster_num', 'seq_length', 'pid', 'strand', 'notused', 'notused', 'comp_alignment', 'query_label', 'target_label']
            taxon_uc_cluster_counts = taxon_uc[taxon_uc['rec_type'] == 'C']
            taxon_abundance = taxon_uc_cluster_counts[taxon_uc_cluster_counts['cluster_num'] == cluster]['seq_length'].values[0]
    else:
        taxon_abundance = blast_lineage_slim[blast_lineage_slim['target_taxon'] == target_taxon].shape[0]
    return(taxon_abundance)

def get_probes(probe_evaluation_filename, input_probe_directory, otu):
    if otu == 'F':
        target_taxon_full = re.sub('.probe.evaluation.h5', '', os.path.basename(probe_evaluation_filename))
        target_taxon = int(re.sub('_Cluster[0-9]+', '', target_taxon_full))
        probes = pd.read_csv(input_probe_directory + '/' + str(target_taxon_full) + '_consensus.int', skiprows = 3, header = None, delim_whitespace = True)
    else:
        target_taxon_full = re.sub('.probe_evaluation.h5', '', os.path.basename(probe_evaluation_filename))
        target_taxon = os.path.split(os.path.split(probe_blast_directory)[0])[0]
        probes = pd.read_csv(input_probe_directory + '/' + str(target_taxon) + '.int', skiprows = 3, header = None, delim_whitespace = True)
    probes.columns = ['probe_id', 'seq', 'p_start', 'ln', 'N', 'GC', 'Tm', 'self_any_th', 'self_end_th', 'hairpin', 'quality']
    probes.loc[:,'target_taxon'] = target_taxon
    probes.loc[:,'target_taxon_full'] = target_taxon_full
    return(target_taxon, target_taxon_full, probes)

def summarize_probes(probe_evaluation_filename, blast_lineage_strain_filename, probe_summary_info_filename, sample_directory, similarity, input_consensus_directory, input_probe_directory, target_rank, min_tm, max_tm, gc_cutoff, max_continuous_homology, otu, probe_selection_method, tpn, freqll, bot, bitscore_thresh, Na, dnac1oligo, mt_cutoff, ot_gc_cutoff):
    blast_dir, taxon_evaluaton_filename = os.path.split(probe_evaluation_filename)
    community_taxon_abundance = pd.read_csv('{}/{}/s_{}/consensus/taxon_abundance.csv'.format(sample_directory, target_rank, similarity))
    community_taxon_abundance['relative_freq'] = community_taxon_abundance.counts.values/community_taxon_abundance.counts.sum()
    community_taxon_abundance = community_taxon_abundance[community_taxon_abundance.relative_freq.values > freqll]
    community_taxon_abundance = community_taxon_abundance.rename(columns = {'taxid':target_rank})
    similiarity_dir = os.path.split(blast_dir)[0]
    blast_lineage_slim = get_blast_lineage_strain_slim(blast_lineage_strain_filename, otu)
    target_taxon, target_taxon_full, probes = get_probes(probe_evaluation_filename, input_probe_directory, otu)
    taxon_fasta_filename = input_consensus_directory + '/' + str(target_taxon) + '.fasta'
    probe_summary = pd.DataFrame()
    for probe_id_idx in probes.probe_id:
        probe_name = 'probe_' + str(probe_id_idx)
        try:
            probe_blast = pd.read_hdf(probe_evaluation_filename, probe_name)
            probe_blast['melting_temp'] = calculate_tm(probe_blast, Na, dnac1oligo)
            probe_blast['GC_count'] = calculate_gc_count(probe_blast)
            if otu == 'F':
                if target_rank == 'strain':
                    probe_blast.loc[:,'target_taxon_hit'] = probe_blast['species'].values.astype(str) == str(target_taxon)
                    probe_blast.loc[:,'target_taxon_full_hit'] = probe_blast['species'].values.astype(str) == str(target_taxon_full)
                    probe_blast.loc[:,'target_taxon_hit_full_match'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))*(probe_blast.pid.values >= 99.9)*(probe_blast.qcovhsp.values >= 99.9)
                else:
                    probe_blast.loc[:,'target_taxon_hit'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))
                    probe_blast.loc[:,'target_taxon_full_hit'] = (probe_blast['strain'].values.astype(str) == str(target_taxon_full))
                    probe_blast.loc[:,'target_taxon_hit_full_match'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))*(probe_blast.pid.values >= 99.9)*(probe_blast.qcovhsp.values >= 99.9)
            else:
                probe_blast.loc[:,'target_taxon_hit'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))
                probe_blast.loc[:,'target_taxon_full_hit'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon_full))
                probe_blast.loc[:,'target_taxon_hit_full_match'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))*(probe_blast.pid.values >= 99.9)*(probe_blast.qcovhsp.values >= 99.9)
            taxon_abundance = get_taxon_abundance(input_consensus_directory, target_taxon, target_taxon_full, otu, blast_lineage_slim)
            if probe_blast.shape[0] > 0:
                probe_summary = probe_summary.append(probe_blast_summarize(probe_blast, max_continuous_homology = max_continuous_homology, taxon_abundance = taxon_abundance, target_rank = target_rank), ignore_index = True, sort = False)
        except KeyError:
            pass
    if not probe_summary.empty:
        probe_summary.loc[:, 'probe_id'] = probe_summary.probe_id.astype(int)
    probes['p_start_group'] = np.floor(probes.p_start.values/20).astype(int)
    probes_merge = probes.merge(probe_summary, on = 'probe_id', how = 'left', sort = False)
    probes_merge = probes_merge[(probes_merge['Tm'] >= min_tm) & (probes_merge['Tm'] <= max_tm) & (probes_merge['GC'] >= gc_cutoff)]
    probes_merge['p_end'] = probes_merge.p_start.values + probes_merge.ln.values - 1
    if probes.shape[0] > 0:
        if probe_selection_method == 'SingleBestProbe':
            best_probes = pd.DataFrame(probes.iloc[[0],:])
            best_probes.loc[:,'selection_method'] = 'SingleBest'
        elif probe_selection_method == 'AllSpecific':
            best_probes = select_all_specific_probes(probes, bot)
        elif probe_selection_method == 'AllSpecificPStartGroup':
            best_probes = pd.DataFrame()
            group_distance = 120
            while (best_probes.shape[0] <= 15) & (group_distance > 20):
                group_distance -= 20
                probes['p_start_group'] = np.floor(probes.p_start.values/group_distance).astype(int)
                probes_merge = probes.merge(probe_summary, on = 'probe_id', how = 'left', sort = False)
                probes_merge = probes_merge[(probes_merge['Tm'] >= min_tm) & (probes_merge['Tm'] <= max_tm) & (probes_merge['GC'] >= gc_cutoff)]
                if probes_merge.empty:
                    print('{}, {}'.format(probe_evaluation_filename, probe_summary.off_target_max_gc.min()))
                probes_merge['p_end'] = probes_merge.p_start.values + probes_merge.ln.values - 1
                best_probes = select_all_specific_p_start_group_probes(probes_merge, group_distance, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
            if best_probes.shape[0] > 15:
                group_distance += 20
                probes['p_start_group'] = np.floor(probes.p_start.values/group_distance).astype(int)
                probes_merge = probes.merge(probe_summary, on = 'probe_id', how = 'left', sort = False)
                probes_merge = probes_merge[(probes_merge['Tm'] >= min_tm) & (probes_merge['Tm'] <= max_tm) & (probes_merge['GC'] >= gc_cutoff)]
                probes_merge['p_end'] = probes_merge.p_start.values + probes_merge.ln.values - 1
                best_probes = select_all_specific_p_start_group_probes(probes_merge, group_distance, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
            elif best_probes.shape[0] == 0:
                probes_sorted = probes_merge.sort_values(by = 'blast_on_target_rate', ascending = False)
                best_probes = pd.DataFrame(probes_sorted.iloc[[0],:])
                best_probes.loc[:,'selection_method'] = 'SingleBest'
                best_probes['p_end'] = best_probes.p_start.values + best_probes.ln.values - 1
        elif probe_selection_method == 'MinOverlap':
            best_probes = select_min_overlap_probes(probes, taxon_fasta_filename, probe_evaluation_filename, max_continuous_homology)
        elif probe_selection_method == 'TopN':
            best_probes = select_top_probes(probes, tpn)
        probes['p_start_group'] = np.floor(probes.p_start.values/25).astype(int)
        probes_merge = probes.merge(probe_summary, on = 'probe_id', how = 'left', sort = False)
        probes_merge = probes_merge[(probes_merge['Tm'] >= min_tm) & (probes_merge['Tm'] <= max_tm) & (probes_merge['GC'] >= gc_cutoff)]
        probes_merge['p_end'] = probes_merge.p_start.values + probes_merge.ln.values - 1
        helper_probes_all = pd.DataFrame()
        for probe_idx in best_probes.probe_id:
            p_start = best_probes.loc[best_probes.probe_id == probe_idx, 'p_start'].values[0]
            p_end = best_probes.loc[best_probes.probe_id == probe_idx, 'p_end'].values[0]
            five_prime_helpers = probes_merge.loc[(probes_merge.p_start.values > p_start - 103) & (probes_merge.p_end.values < p_start - 3)]
            three_prime_helpers = probes_merge.loc[(probes_merge.p_start.values > p_end + 3) & (probes_merge.p_end.values < p_end + 103)]
            helper_probes = pd.concat([five_prime_helpers,three_prime_helpers])
            helper_probes['helper_group'] = ((helper_probes.p_start.values - p_start)/20).astype(int)
            helper_probes['helper_source_probe'] = probe_idx
            helper_probes_all = helper_probes_all.append(helper_probes)
            helper_probes_blast = pd.DataFrame()
        probe_off_target_summary = pd.DataFrame()
        probe_blast_off_target = pd.DataFrame()
        probe_blast_cumulative = pd.DataFrame(columns = ['molecule_id'])
        for probe_idx in best_probes.probe_id:
            probe_info = best_probes.loc[best_probes.probe_id == probe_idx]
            probe_name = 'probe_' + str(probe_idx)
            probe_blast = pd.read_hdf(probe_evaluation_filename, probe_name)
            probe_blast_off_target = probe_blast_off_target.append(probe_blast.loc[(probe_blast.mch.values < max_continuous_homology) & (probe_blast.gapopen.values == 0)])
            probe_blast = probe_blast[probe_blast['mch'] >= max_continuous_homology]
            if probe_blast.empty:
                ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'no rank']
                probe_off_target_ranks = pd.DataFrame([int(probe_idx), 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']).transpose()
                probe_off_target_ranks.columns = ['probe_id'] + ['off_target_' + s for s in ranks]
                probe_off_target_summary = probe_off_target_summary.append(probe_off_target_ranks, ignore_index = True, sort = False)
            else:
                if otu == 'F':
                    probe_blast.loc[:,'target_taxon_hit'] = probe_blast[target_rank].values.astype(str) == str(target_taxon)
                    probe_blast.loc[:,'target_taxon_full_hit'] = probe_blast['strain'].values.astype(str) == str(target_taxon_full)
                else:
                    probe_blast.loc[:,'hit'] = probe_blast['target_taxon'] == target_taxon
                if probe_info.selection_method.values[0] != 'SingleBest':
                    probe_blast_cumulative = probe_blast_cumulative.merge(probe_blast.loc[probe_blast.target_taxon_full_hit == True].loc[:,['molecule_id']], on = 'molecule_id', how = 'outer', sort = False)
                if target_rank == 'strain':
                    probe_off_target_summary = probe_off_target_summary.append(get_probe_blast_off_target_ranks(probe_blast, probe_info, 'species', target_taxon), ignore_index = True, sort = False)
                else:
                    probe_off_target_summary = probe_off_target_summary.append(get_probe_blast_off_target_ranks(probe_blast, probe_info, target_rank, target_taxon), ignore_index = True, sort = False)
        if not probe_off_target_summary.empty:
            probe_off_target_summary['probe_id'] = probe_off_target_summary.probe_id.astype(int)
            best_probes = best_probes.merge(probe_off_target_summary, on = 'probe_id', how = 'left', sort = False)
        if not best_probes.empty:
            if probe_info.selection_method.values[0] != 'SingleBest':
                best_probes.loc[:,'taxon_coverage'] = probe_blast_cumulative.shape[0]/taxon_abundance
            probe_off_target_summary_filename = re.sub('_probe_selection.csv', '_off_target_summary_info.csv', probe_summary_info_filename)
            helper_probes_filename = re.sub('_probe_selection.csv', '_helper_probes.csv', probe_summary_info_filename)
            probe_blast_off_target.to_csv(probe_off_target_summary_filename, index = False)
            helper_probes_all.to_csv(helper_probes_filename, index = False)
            best_probes.to_csv(probe_summary_info_filename, index = False)
        else:
            best_probes = pd.DataFrame(columns = ['GC', 'N', 'Tm', 'blast_on_target_rate', 'hairpin', 'ln', 'off_target_full_qcovhsp_fraction',
                                   'off_target_max_mch', 'off_target_max_mch_qcovhsp', 'off_target_max_mch_sort', 'off_target_min_evalue',
                                   'on_target_full_match', 'p_start', 'p_start_group', 'probe_id', 'quality', 'self_any_th', 'self_end_th',
                                   'seq', 'target_taxon', 'target_taxon_full', 'taxon_abundance', 'taxon_coverage', 'selection_method',
                                   'off_target_class', 'off_target_family', 'off_target_genus', 'off_target_norank', 'off_target_order',
                                   'off_target_phylum', 'off_target_species', 'off_target_superkingdom'])
            best_probes.to_csv(probe_summary_info_filename, index = False)
            probe_blast_off_target = pd.DataFrame(columns = ['probe_id', 'molecule_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'probe_start', 'probe_end', 'molecule_start',
            	                              'molecule_end', 'evalue', 'bitscore', 'staxids', 'qseq', 'sseq', 'mch', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain'])
            probe_off_target_summary_filename = re.sub('_probe_selection.csv', '_off_target_summary_info.csv', probe_summary_info_filename)
            probe_blast_off_target.to_csv(probe_off_target_summary_filename, index = False)
    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')

    parser.add_argument('probe_evaluation_complete_filename', type = str, help = 'Input file containing blast results')

    parser.add_argument('design_id', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('probe_summary_info_filename', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('-t', '--target_rank', dest = 'target_rank', type = str, default = 'phylum', help = 'Input file containing blast lineage')

    parser.add_argument('-s', '--similarity', dest = 'similarity', type = str, default = '', help = 'Input file containing blast lineage')

    parser.add_argument('-o', '--otu_clustering', dest = 'otu', type = str, default = 'F', help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-tmin', '--min_tm', dest = 'min_tm', type = float, default = 55.0, help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-tmax', '--max_tm', dest = 'max_tm', type = float, default = 55.0, help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-m', '--mch', dest = 'mch', type = int, default = 14, help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-tpn', '--top_n_probes', dest = 'tpn', type = int, default = 1, help = 'Number of top probes to keep')

    parser.add_argument('-freqll', '--freq_ll', dest = 'freqll', type = float, default = 0.0, help = 'Number of top probes to keep')

    parser.add_argument('-gc', '--gc', dest = 'gc', type = float, default = 40.0, help = 'Number of top probes to keep')

    parser.add_argument('-bot', '--bot', dest = 'bot', type = float, default = 0.0, help = 'Number of top probes to keep')

    parser.add_argument('-bt', '--bitscore_thresh', dest = 'bitscore_thresh', type = float, default = 0.001, help = 'Number of top probes to keep')

    parser.add_argument('-sod', '--sod', dest = 'sod', type = float, default = 390, help = 'sodium concentration in nM')

    parser.add_argument('-dnaconc', '--dnaconc', dest = 'dnaconc', type = float, default = 5, help = 'oligo concentration in nM')

    parser.add_argument('-mt', '--mt_cutoff', dest = 'mt_cutoff', type = float, default = 60, help = 'oligo concentration in nM')

    parser.add_argument('-otgc', '--ot_gc_cutoff', dest = 'ot_gc_cutoff', type = float, default = 7, help = 'oligo concentration in nM')

    parser.add_argument('-c', '--probe_selection_method', dest = 'probe_selection_method', type = str, default = 'AllSpecific', help = 'Probe selection method. AllSpecific (default) | Single Best | MinOverlap | Top N')

    args = parser.parse_args()
    similarity_directory = os.path.split(os.path.split(args.probe_evaluation_complete_filename)[0])[0]
    probe_evaluation_filename = re.sub('.complete.txt', '.h5', args.probe_evaluation_complete_filename)
    input_consensus_directory = '{}/consensus'.format(similarity_directory)
    input_probe_directory = '{}/primer3'.format(similarity_directory)
    sample_directory = os.path.split(os.path.split(similarity_directory)[0])[0]
    blast_lineage_strain_filename = '{}/{}/s_{}/consensus/blast_lineage_strain.tab'.format(sample_directory, args.target_rank, args.similarity)
    summarize_probes(probe_evaluation_filename, blast_lineage_strain_filename, args.probe_summary_info_filename, sample_directory, args.similarity, input_consensus_directory, input_probe_directory, args.target_rank, args.min_tm, args.max_tm, args.gc, args.mch, args.otu, args.probe_selection_method, args.tpn, args.freqll, args.bot, args.bitscore_thresh, args.sod, args.dnaconc, args.mt_cutoff, args.ot_gc_cutoff)
    return

if __name__ == '__main__':
    main()
