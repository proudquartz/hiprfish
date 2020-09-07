import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from ete3 import NCBITaxa
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import dask.dataframe as dd
from Bio.Alphabet import IUPAC, generic_dna
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.SeqUtils import MeltingTemp as mt
from dask.distributed import LocalCluster, Client
from Bio.SeqUtils import GC
import tables
from SetCoverPy import setcover

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

def select_all_specific_probes(probe_summary_info, blast_lineage, bitscore_thresh, mt_cutoff, ot_gc_cutoff):
    best_probes = pd.DataFrame()
    bot = 1 - 1/blast_lineage.shape[0]
    best_probes = probe_summary_info.loc[(probe_summary_info['blast_on_target_rate'] > bot) & (probe_summary_info['off_target_max_bitscore'] < bitscore_thresh) & (probe_summary_info['off_target_max_tm'] < mt_cutoff) & (probe_summary_info['off_target_max_gc'] < ot_gc_cutoff)]
    if not best_probes.empty:
        best_probes.loc[:,'selection_method'] = 'AllSpecific'
    else:
        probe_summary_info.sort_values(['blast_on_target_rate', 'on_target_full_match', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'quality'], ascending = [False, False, False, True, True, True, True, True, True], inplace = True)
        best_probes = probe_summary_info.iloc[[0],:]
        best_probes.loc[:,'selection_method'] = 'AllSpecificSingleBest'
    return(best_probes)

def select_all_specific_p_start_group_probes(probe_summary_info, blast_lineage, group_distance, bitscore_thresh, mt_cutoff, ot_gc_cutoff):
    bot = 1 - 1/blast_lineage.shape[0]
    best_probes_group = pd.DataFrame()
    best_probes = probe_summary_info.loc[(probe_summary_info['blast_on_target_rate'] > bot) & (probe_summary_info['off_target_max_bitscore'] < bitscore_thresh) & (probe_summary_info['off_target_max_tm'] < mt_cutoff) & (probe_summary_info['off_target_max_gc'] < ot_gc_cutoff)]
    if not best_probes.empty:
        for group in range(int(np.floor(1500/group_distance))):
            best_probes_temp = best_probes.loc[best_probes.mean_probe_start_group.values == group]
            best_probes_temp_sorted = best_probes_temp.sort_values(['on_target_full_match', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'quality'], ascending = [False, False, True, True, True, True, True, True])
            if not best_probes_temp_sorted.empty:
                best_probes_group = best_probes_group.append(best_probes_temp_sorted.iloc[[0],:], sort = False)
                best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroup'
    else:
        probe_summary_info_sorted = probe_summary_info.sort_values(['blast_on_target_rate', 'on_target_full_match', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'quality'], ascending = [False, False, False, True, True, True, True, True, True])
        best_probes_group = probe_summary_info_sorted.iloc[[0],:]
        best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroupSingleBest'
    return(best_probes_group)

def select_all_specific_p_start_group_min_overlap_probes(probe_summary_info, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, group_distance, max_continuous_homology, bitscore_thresh, mt_cutoff, ot_gc_cutoff):
    best_probes_group = pd.DataFrame()
    bot = 1 - 1/blast_lineage.shape[0]
    best_probes = probe_summary_info.loc[(probe_summary_info['blast_on_target_rate'] > bot) & (probe_summary_info['off_target_max_bitscore'] < bitscore_thresh) & (probe_summary_info['off_target_max_tm'] < mt_cutoff) & (probe_summary_info['off_target_max_gc'] < ot_gc_cutoff)]
    if not best_probes.empty:
        for group in range(int(np.floor(1500/group_distance))):
            best_probes_temp = best_probes.loc[best_probes.mean_probe_start_group.values == group]
            best_probes_temp_sorted = best_probes_temp.sort_values(['on_target_full_match', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'quality'], ascending = [False, False, True, True, True, True, True, True])
            target_taxon_molecule_ids = blast_lineage.loc[blast_lineage[target_rank].values == target_taxon, 'molecule_id']
            probe_ids = best_probes_temp.probe_id.unique()
            cover_matrix = pd.DataFrame(np.zeros((len(target_taxon_molecule_ids), len(probe_ids)), dtype = int), columns = probe_ids, index = target_taxon_molecule_ids)
            cost = np.ones(len(probe_ids), dtype = int)
            for i in range(best_probes_temp_sorted.shape[0]):
                probe_idx = best_probes_temp_sorted.probe_id.values[i]
                probe_info = best_probes_temp_sorted.loc[best_probes_temp_sorted.probe_id.values == probe_idx,:]
                probe_blast = pd.read_csv('{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, probe_idx))
                probe_blast = probe_blast.loc[probe_blast.mch.values >= max_continuous_homology,:]
                blasted_molecules = list(probe_blast.molecule_id.values)
                cover_matrix.loc[blasted_molecules, probe_idx] = 1
            g = setcover.SetCover(cover_matrix.values, cost)
            g.SolveSCP()
            set_cover_indices = np.flatnonzero(g.s*1 == 1)
            best_probes_minoverlap = best_probes_temp.iloc[set_cover_indices,:]
            if not best_probes_minoverlap.empty:
                best_probes_group = best_probes_group.append(best_probes_minoverlap, sort = False)
                best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroupMinOverlap'
    else:
        probe_summary_info.sort_values(['blast_on_target_rate', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'on_target_full_match', 'quality'], ascending = [False, False, True, True, True, True, True, False, True], inplace = True)
        best_probes_group = probe_summary_info.iloc[[0],:]
        best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroupMinOverlapSingleBest'
    return(best_probes_group)

def select_all_specific_min_overlap_probes(probe_summary_info, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, group_distance, max_continuous_homology, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff):
    best_probes_group = pd.DataFrame()
    best_probes = probe_summary_info.loc[(probe_summary_info['blast_on_target_rate'] > bot) & (probe_summary_info['off_target_max_bitscore'] < bitscore_thresh) & (probe_summary_info['off_target_max_tm'] < mt_cutoff) & (probe_summary_info['off_target_max_gc'] < ot_gc_cutoff)]
    if not best_probes.empty:
        # best_probes_temp = best_probes.loc[best_probes.mean_probe_start_group.values == group]
        # best_probes_sorted = best_probes.sort_values(['on_target_full_match', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'quality'], ascending = [False, False, True, True, True, True, True, True])
        target_taxon_molecule_ids = blast_lineage.loc[blast_lineage[target_rank].values == target_taxon, 'molecule_id']
        probe_ids = best_probes.probe_id.unique()
        cover_matrix = pd.DataFrame(np.zeros((len(target_taxon_molecule_ids), len(probe_ids)), dtype = int), columns = probe_ids, index = target_taxon_molecule_ids)
        qcovhsp_matrix = pd.DataFrame(np.zeros((len(target_taxon_molecule_ids), len(probe_ids)), dtype = int), columns = probe_ids, index = target_taxon_molecule_ids)
        cost = np.ones(len(probe_ids), dtype = int)
        for i in range(best_probes.shape[0]):
            probe_idx = best_probes.probe_id.values[i]
            probe_info = best_probes.loc[best_probes.probe_id.values == probe_idx,:]
            probe_blast = pd.read_csv('{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, probe_idx))
            probe_blast = probe_blast.loc[probe_blast.mch.values >= max_continuous_homology,:]
            blasted_molecules = list(probe_blast.molecule_id.values)
            cover_matrix.loc[blasted_molecules, probe_idx] = 1
            qcovhsp_matrix.loc[blasted_molecules, probe_idx] = probe_blast.loc[probe_blast.molecule_id.values == blasted_molecules, 'qcovhsp'].values
        g = setcover.SetCover(cover_matrix.values, cost)
        g.SolveSCP()
        set_cover_indices = np.flatnonzero(g.s*1 == 1)
        best_probes_minoverlap = best_probes.iloc[set_cover_indices,:]
        if not best_probes_minoverlap.empty:
            best_probes_group = best_probes_group.append(best_probes_minoverlap, sort = False)
            best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroupMinOverlap'
    else:
        probe_summary_info.sort_values(['blast_on_target_rate', 'taxon_coverage', 'off_target_full_qcovhsp_fraction', 'off_target_max_mch', 'off_target_max_bitscore', 'off_target_max_tm', 'off_target_max_gc', 'on_target_full_match', 'quality'], ascending = [False, False, True, True, True, True, True, False, True], inplace = True)
        best_probes_group = probe_summary_info.iloc[[0],:]
        best_probes_group.loc[:,'selection_method'] = 'AllSpecificPStartGroupMinOverlapSingleBest'
        qcovhsp_matrix = pd.DataFrame()
        cover_matrix = pd.DataFrame()
    return(best_probes_group, cover_matrix, qcovhsp_matrix)

def calculate_cover_matrix(best_probes, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, max_continuous_homology, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff):
    if not best_probes.empty:
        target_taxon_molecule_ids = blast_lineage.loc[blast_lineage[target_rank].values == target_taxon, 'molecule_id']
        probe_ids = best_probes.probe_id.unique()
        cover_matrix = pd.DataFrame(np.zeros((len(target_taxon_molecule_ids), len(probe_ids)), dtype = int), columns = probe_ids, index = target_taxon_molecule_ids)
        qcovhsp_matrix = pd.DataFrame(np.zeros((len(target_taxon_molecule_ids), len(probe_ids)), dtype = int), columns = probe_ids, index = target_taxon_molecule_ids)
        for i in range(best_probes.shape[0]):
            probe_idx = best_probes.probe_id.values[i]
            probe_info = best_probes.loc[best_probes.probe_id.values == probe_idx,:]
            probe_blast = pd.read_csv('{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, probe_idx))
            probe_blast = probe_blast.loc[(probe_blast.mch.values >= max_continuous_homology),:]
            blasted_molecules = list(probe_blast.molecule_id.values)
            qcovhsp_matrix.loc[blasted_molecules, probe_idx] = probe_blast.loc[probe_blast.molecule_id.values == blasted_molecules, 'qcovhsp'].values
            probe_blast = probe_blast.loc[(probe_blast.qcovhsp.values == 100),:]
            blasted_molecules = list(probe_blast.molecule_id.values)
            cover_matrix.loc[blasted_molecules, probe_idx] = 1
    return(cover_matrix, qcovhsp_matrix)

def plot_qcovhsp_matrix(qcovhsp_matrix, theme_color, filename):
    fig = plt.figure()
    heatmap = sns.clustermap(qcovhsp_matrix.values, cmap = 'inferno', figsize = (cm_to_inches(8), cm_to_inches(8)), tree_kws = {'colors':theme_color})
    plt.axes(heatmap.ax_heatmap)
    plt.xlabel('Fusobacterium Probe ID', fontsize = 8, color = theme_color)
    plt.ylabel('Fusobacterium Target Molecule ID', fontsize = 8, color = theme_color)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(direction = 'in', length = 0, colors = theme_color)
    plt.axes(heatmap.ax_cbar)
    plt.tick_params(direction = 'in', colors = theme_color, labelsize = 8)
    plt.savefig(filename, dpi = 300, transparent = True)
    return

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
    probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
    if probe_blast_filtered.shape[0] > 0:
        blast_on_target_rate = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/probe_blast_filtered.shape[0]
        taxon_coverage = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/taxon_abundance
        on_target_full_match = probe_blast.target_taxon_hit_full_match.sum()/taxon_abundance
        min_probe_start = probe_blast_filtered.molecule_end.min()
        max_probe_start = probe_blast_filtered.molecule_end.max()
        mean_probe_start = probe_blast_filtered.molecule_end.mean()
    else:
        blast_on_target_rate = 0
        taxon_coverage = 0
        on_target_full_match = 0
        min_probe_start = 0
        max_probe_start = 0
        mean_probe_start = 0
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
                      'min_probe_start': min_probe_start,
                      'max_probe_start': max_probe_start,
                      'mean_probe_start': mean_probe_start,
                      'blast_on_target_rate': blast_on_target_rate,
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
    if probe_info.blast_on_target_rate.values < 1-1e-8:
        probe_blast_filtered = probe_blast.loc[probe_blast.target_taxon_hit.values == False, :]
        off_target_last_common_taxon_rank = probe_blast_filtered.apply(get_off_target_last_common_taxon_rank, target_rank = target_rank, target_taxon = target_taxon, axis = 1)
        n_total = off_target_last_common_taxon_rank.shape[0]
        if n_total > 0:
            rank_fractions = [np.sum(off_target_last_common_taxon_rank == rank)/n_total for rank in ranks]
        else:
            rank_fractions = [np.nan for rank in ranks]
    else:
        rank_fractions = [0 for rank in ranks]
    probe_off_target_summary = pd.DataFrame([probe_blast.probe_id.values[0]] + rank_fractions).transpose()
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

def get_blast_lineage(blast_lineage_filename):
    blast_lineage_df = pd.read_table(blast_lineage_filename, dtype = str)
    lineage_columns = ['molecule_id', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    blast_lineage_slim = blast_lineage_df.loc[:,lineage_columns]
    blast_lineage_slim.loc[:,'molecule_id'] = blast_lineage_slim.molecule_id.apply(sub_slash)
    return(blast_lineage_slim)

def get_taxon_abundance(blast_lineage, target_taxon):
    taxon_abundance = blast_lineage
    return(taxon_abundance)

def get_probes(probes_summary_filename, target_rank, target_taxon):
    probes = pd.read_hdf(probes_summary_filename, key = '{}/{}_{}'.format(target_rank, target_rank, target_taxon), mode = 'r')
    probes.loc[:,'target_taxon'] = target_taxon
    return(probes)

def get_full_probe_id(pid, target_rank, target_taxon):
    return('{}_{}_{}'.format(target_rank, target_taxon, pid))

def summarize_probe_properties(probe_blast, probe_evaluation_dir, target_rank, target_taxon, taxon_abundance, max_continuous_homology):
    # probe_blast = pd.read_csv('{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, probe_idx))
    probe_blast.loc[:,'melting_temp'] = calculate_tm(probe_blast, Na, dnac1oligo)
    probe_blast.loc[:,'GC_count'] = calculate_gc_count(probe_blast)
    probe_blast.loc[:,'target_taxon_hit'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))
    probe_blast.loc[:,'target_taxon_hit_full_match'] = (probe_blast[target_rank].values.astype(str) == str(target_taxon))*(probe_blast.pid.values >= 99.9)*(probe_blast.qcovhsp.values >= 99.9)
    probe_blast_filtered = probe_blast[(probe_blast['mch'] >= max_continuous_homology) | (probe_blast['length'] >= max_continuous_homology)]
    if probe_blast_filtered.shape[0] > 0:
        blast_on_target_rate = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/probe_blast_filtered.shape[0]
        taxon_coverage = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/taxon_abundance
        on_target_full_match = probe_blast.target_taxon_hit_full_match.sum()/taxon_abundance
        min_probe_start = probe_blast_filtered.molecule_end.min()
        max_probe_start = probe_blast_filtered.molecule_end.max()
        mean_probe_start = probe_blast_filtered.molecule_end.mean()
    else:
        blast_on_target_rate = 0
        taxon_coverage = 0
        on_target_full_match = 0
        min_probe_start = 0
        max_probe_start = 0
        mean_probe_start = 0
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
                      'min_probe_start': min_probe_start,
                      'max_probe_start': max_probe_start,
                      'mean_probe_start': mean_probe_start,
                      'blast_on_target_rate': blast_on_target_rate,
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

def summarize_probe_properties_v2(probe_idx, probe_evaluation_dir, target_rank, target_taxon, target_taxon_abundance, max_continuous_homology, Na, dnac1oligo):
    probe_blast = pd.read_csv('{}/{}_{}_{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, target_rank, target_taxon, probe_idx))
    probe_blast.loc[:,'melting_temp'] = calculate_tm(probe_blast, Na, dnac1oligo)
    probe_blast.loc[:,'GC_count'] = calculate_gc_count(probe_blast)
    probe_blast.loc[:,'target_taxon_hit'] = (probe_blast.loc[:,target_rank].values.astype(str) == str(target_taxon))
    probe_blast.loc[:,'target_taxon_hit_full_match'] = (probe_blast.loc[:,target_rank].values.astype(str) == str(target_taxon))*(probe_blast.loc[:,'pid'].values >= 99.9)*(probe_blast.loc[:,'qcovhsp'].values >= 99.9)
    probe_blast_filtered = probe_blast.loc[(probe_blast.loc[:,'mch'].values >= max_continuous_homology) | (probe_blast.loc[:,'length'].values >= max_continuous_homology),:]
    if probe_blast_filtered.shape[0] > 0:
        blast_on_target_rate = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/probe_blast_filtered.shape[0]
        taxon_coverage = probe_blast_filtered.loc[:,'target_taxon_hit'].sum()/target_taxon_abundance
        on_target_full_match = probe_blast.target_taxon_hit_full_match.sum()/target_taxon_abundance
        min_probe_start = probe_blast_filtered.molecule_end.min()
        max_probe_start = probe_blast_filtered.molecule_end.max()
        mean_probe_start = probe_blast_filtered.molecule_end.mean()
    else:
        blast_on_target_rate = 0
        taxon_coverage = 0
        on_target_full_match = 0
        min_probe_start = 0
        max_probe_start = 0
        mean_probe_start = 0
    probe_blast_below_mch = probe_blast.loc[probe_blast.loc[:,'mch'].values < max_continuous_homology, :]
    if probe_blast_below_mch.shape[0] > 0:
        off_target_min_evalue = probe_blast_below_mch.evalue.min()
        off_target_max_bitscore = probe_blast_below_mch.bitscore.max()
        off_target_max_mch = probe_blast_below_mch.mch.max()
        probe_blast_below_mch_sorted = probe_blast_below_mch.sort_values(['mch', 'qcovhsp'], ascending = [False, False])
        off_target_max_mch_sort = probe_blast_below_mch_sorted.mch.values[0]
        off_target_max_mch_qcovhsp = probe_blast_below_mch_sorted.qcovhsp.values[0]
        off_target_full_qcovhsp_fraction = np.sum(probe_blast_below_mch_sorted.qcovhsp.values > 99.9)/probe_blast_below_mch_sorted.shape[0]
        off_target_max_tm = probe_blast_below_mch_sorted.melting_temp.max()
        off_target_max_gc = probe_blast_below_mch_sorted.GC_count.max()
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
                      'probe_seq': probe_blast.qseq.values[0],
                      'min_probe_start': min_probe_start,
                      'max_probe_start': max_probe_start,
                      'mean_probe_start': mean_probe_start,
                      'blast_on_target_rate': blast_on_target_rate,
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
                      'taxon_abundance': target_taxon_abundance}))

# def summarize_probe_seq(probe_idx, probe_evaluation_dir, target_rank, target_taxon):
#     probe_blast = pd.read_csv('{}/{}_{}_{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, target_rank, target_taxon, probe_idx))
#     probe_blast_filtered = probe_blast.loc[(probe_blast.loc[:,'mch'].values >= max_continuous_homology) | (probe_blast.loc[:,'length'].values >= max_continuous_homology),:]
#     if probe_blast_filtered.shape[0] > 0:
#         min_probe_start = probe_blast_filtered.molecule_end.min()
#         max_probe_start = probe_blast_filtered.molecule_end.max()
#         mean_probe_start = probe_blast_filtered.molecule_end.mean()
#     else:
#         min_probe_start = 2000
#         max_probe_start = 2000
#         mean_probe_start = 2000
#     return(pd.Series({'probe_id': probe_blast.probe_id.values[0],
#                      'probe_seq': probe_blast.qseq.values[0],
#                      'min_probe_start': min_probe_start,
#                      'max_probe_start': max_probe_start,
#                      'mean_probe_start': mean_probe_start}))

def summarize_probes(probe_evaluation_dir, blast_lineage_filename, design_dir, sample_directory, probes_summary_filename, target_rank, min_tm, max_tm, gc_cutoff, max_continuous_homology, bot, bitscore_thresh, Na, dnac1oligo, mt_cutoff, ot_gc_cutoff):
    design_level_evaluation_dir, target_taxon = os.path.split(probe_evaluation_dir)
    probes_summary_info_filename = '{}/{}_probes_summary_information.csv'.format(design_dir, target_taxon)
    blast_lineage = get_blast_lineage(blast_lineage_filename)
    taxon_abundance = blast_lineage.groupby(target_rank).molecule_id.count().reset_index()
    taxon_abundance.loc[:, target_rank] = taxon_abundance.loc[:, target_rank].astype(str)
    probes = get_probes(probes_summary_filename, target_rank, target_taxon)
    target_taxon_abundance = taxon_abundance.loc[taxon_abundance.loc[:,target_rank].values == target_taxon, 'molecule_id'].values[0]
    index_list = np.arange(0, probes.shape[0], 10000)
    index_list = np.append(index_list, probes.shape[0])
    probes_summary_full = pd.DataFrame()
    for i in range(len(index_list) - 1):
        probes_sub = dd.from_pandas(probes.iloc[index_list[i]:index_list[i+1],:], npartitions = 100)
        meta_dtype = {'probe_id': 'int',
                      'probe_seq': 'str',
                      'min_probe_start': 'float',
                      'max_probe_start': 'float',
                      'mean_probe_start': 'float',
                      'blast_on_target_rate': 'float',
                      'taxon_coverage': 'float',
                      'on_target_full_match': 'float',
                      'off_target_min_evalue': 'float',
                      'off_target_max_bitscore': 'float',
                      'off_target_max_mch': 'int',
                      'off_target_max_mch_sort': 'int',
                      'off_target_max_mch_qcovhsp': 'float',
                      'off_target_full_qcovhsp_fraction': 'float',
                      'off_target_max_tm': 'float',
                      'off_target_max_gc': 'float',
                      'taxon_abundance': 'float'}
        probe_summary = probes_sub.probe_id.apply(summarize_probe_properties_v2, args = (probe_evaluation_dir, target_rank, target_taxon, target_taxon_abundance, max_continuous_homology, Na, dnac1oligo,), meta = meta_dtype)
        probe_summary_compute = probe_summary.compute()
        probes_summary_full = probes_summary_full.append(probe_summary_compute)
    probes.loc[:,'probe_id'] = probes.probe_id.apply(get_full_probe_id, args = (target_rank, target_taxon))
    probes_merge = probes.merge(probes_summary_full, on = 'probe_id', how = 'left', sort = False)
    probes_merge = probes_merge.loc[(probes_merge.loc[:,'Tm'].values >= min_tm) & (probes_merge.loc[:,'Tm'].values <= max_tm) & (probes_merge.loc[:,'GC'].values >= gc_cutoff),:]
    probes_merge['mean_probe_end'] = probes_merge.mean_probe_start.values + probes_merge.length.values - 1
    probes_merge['mean_probe_start_group'] = np.floor(probes_merge.mean_probe_start.values/20).astype(int)
    probes_merge.to_csv(probes_summary_info_filename, index = False)
    # all_specific_probes = select_all_specific_probes(probes_merge, blast_lineage, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    # cover_matrix_full, qcovhsp_matrix_full = calculate_cover_matrix(all_specific_probes, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, max_continuous_homology, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    # qcovhsp_matrix_full_filename = re.sub('_probe_selection.csv', '_full_qcovhsp_matrix.csv', probe_summary_info_filename)
    # cover_matrix_full_filename = re.sub('_probe_selection.csv', '_full_cover_matrix.csv', probe_summary_info_filename)
    # qcovhsp_matrix_full.to_csv(qcovhsp_matrix_full_filename)
    # cover_matrix_full.to_csv(cover_matrix_full_filename)
    # if probes.shape[0] > 0:
    #     best_probes = pd.DataFrame()
    #     group_distance = 120
    #     while (best_probes.shape[0] <= 15) & (group_distance > 20):
    #         group_distance -= 20
    #         probes_merge.loc[:,'mean_probe_start_group'] = np.floor(probes_merge.mean_probe_start.values/group_distance).astype(int)
    #         # if probes_merge.empty:
    #         #     print('{}, {}'.format(probe_evaluation_filename, probe_summary.off_target_max_gc.min()))
    #         best_probes = select_all_specific_p_start_group_probes(probes_merge, blast_lineage, group_distance, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    #         # best_probes = select_all_specific_p_start_group_min_overlap_probes(probes_merge, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, group_distance, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    #     if best_probes.shape[0] > 15:
    #         group_distance += 20
    #         probes_merge['mean_probe_start_group'] = np.floor(probes_merge.mean_probe_start.values/group_distance).astype(int)
    #         best_probes = select_all_specific_p_start_group_probes(probes_merge, blast_lineage, group_distance, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    #         # best_probes = select_all_specific_p_start_group_min_overlap_probes(probes_merge, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, group_distance, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    #         cover_matrix, qcovhsp_matrix = calculate_cover_matrix(best_probes, probe_evaluation_dir, blast_lineage, target_rank, target_taxon, max_continuous_homology, bot, bitscore_thresh, mt_cutoff, ot_gc_cutoff)
    #         qcovhsp_matrix_best_filename = re.sub('_probe_selection.csv', '_best_qcovhsp_matrix.csv', probe_summary_info_filename)
    #         cover_matrix_best_filename = re.sub('_probe_selection.csv', '_best_cover_matrix.csv', probe_summary_info_filename)
    #         qcovhsp_matrix.to_csv(qcovhsp_matrix_best_filename)
    #         cover_matrix.to_csv(cover_matrix_best_filename)
    #     elif best_probes.shape[0] == 0:
    #         probes_sorted = probes_merge.sort_values(by = 'blast_on_target_rate', ascending = False)
    #         best_probes = pd.DataFrame(probes_sorted.iloc[[0],:])
    #         best_probes.loc[:,'selection_method'] = 'SingleBest'
    #     helper_probes_all = pd.DataFrame()
    #     for probe_idx in best_probes.probe_id:
    #         mean_probe_start = best_probes.loc[best_probes.probe_id == probe_idx, 'mean_probe_start'].values[0]
    #         mean_probe_end = best_probes.loc[best_probes.probe_id == probe_idx, 'mean_probe_end'].values[0]
    #         five_prime_helpers = probes_merge.loc[(probes_merge.mean_probe_start.values > mean_probe_start - 103) & (probes_merge.mean_probe_end.values < mean_probe_start - 3)]
    #         three_prime_helpers = probes_merge.loc[(probes_merge.mean_probe_start.values > mean_probe_end + 3) & (probes_merge.mean_probe_end.values < mean_probe_end + 103)]
    #         helper_probes = pd.concat([five_prime_helpers,three_prime_helpers])
    #         if not helper_probes.empty:
    #             helper_probes.loc[:,'helper_group'] = ((helper_probes.mean_probe_start.values - mean_probe_start)/20).astype(int)
    #             helper_probes.loc[:,'helper_source_probe'] = probe_idx
    #             helper_probes_all = helper_probes_all.append(helper_probes)
    #     probe_off_target_summary = pd.DataFrame()
    #     probe_blast_off_target = pd.DataFrame()
    #     probe_blast_cumulative = pd.DataFrame(columns = ['molecule_id'])
    #     for probe_idx in best_probes.probe_id.values:
    #         probe_info = best_probes.loc[best_probes.probe_id.values == probe_idx,:]
    #         probe_blast = pd.read_csv('{}/{}_probe_evaluation.csv.gz'.format(probe_evaluation_dir, probe_idx))
    #         probe_blast_off_target = probe_blast_off_target.append(probe_blast.loc[(probe_blast.mch.values < max_continuous_homology) & (probe_blast.gapopen.values == 0)])
    #         probe_blast = probe_blast[probe_blast['mch'] >= max_continuous_homology]
    #         probe_blast.loc[:,'target_taxon_hit'] = probe_blast[target_rank].values.astype(str) == str(target_taxon)
    #         if probe_info.selection_method.values[0] != 'SingleBest':
    #             probe_blast_cumulative = probe_blast_cumulative.merge(probe_blast.loc[probe_blast.target_taxon_hit.values == True,:].loc[:,['molecule_id']], on = 'molecule_id', how = 'outer', sort = False)
    #         probe_off_target_summary = probe_off_target_summary.append(get_probe_blast_off_target_ranks(probe_blast, probe_info, target_rank, target_taxon), ignore_index = True, sort = False)
    #     if not probe_off_target_summary.empty:
    #         best_probes = best_probes.merge(probe_off_target_summary, on = 'probe_id', how = 'left', sort = False)
    #     probes_merge_filename = re.sub('_probe_selection.csv', '_probes_summary_infomration.csv', probe_summary_info_filename)
    #     probes_merge.to_csv(probes_merge_filename, index = False)
    #     if not best_probes.empty:
    #         probe_off_target_summary_filename = re.sub('_probe_selection.csv', '_off_target_summary_info.csv', probe_summary_info_filename)
    #         helper_probes_filename = re.sub('_probe_selection.csv', '_helper_probes.csv', probe_summary_info_filename)
    #         all_specific_probe_filename = re.sub('_probe_selection.csv', '_probe_selection_all_specific.csv', probe_summary_info_filename)
    #         prove_coverage_filename = re.sub('_probe_selection.csv', '_probe_coverage.csv', probe_summary_info_filename)
    #         probe_blast_off_target.to_csv(probe_off_target_summary_filename, index = False)
    #         helper_probes_all.to_csv(helper_probes_filename, index = False)
    #         best_probes.to_csv(probe_summary_info_filename, index = False)
    #         all_specific_probes.to_csv(all_specific_probe_filename, index = False)
    #         probe_blast_cumulative.to_csv(prove_coverage_filename, index = False)
    #     else:
    #         best_probes = pd.DataFrame(columns = ['GC', 'N', 'Tm', 'blast_on_target_rate', 'hairpin', 'length', 'off_target_full_qcovhsp_fraction',
    #                                'off_target_max_mch', 'off_target_max_mch_qcovhsp', 'off_target_max_mch_sort', 'off_target_min_evalue',
    #                                'on_target_full_match', 'mean_probe_start', 'mean_probe_start_group', 'probe_id', 'quality', 'self_any_th', 'self_end_th',
    #                                'seq', 'target_taxon', 'taxon_abundance', 'taxon_coverage', 'selection_method',
    #                                'off_target_class', 'off_target_family', 'off_target_genus', 'off_target_norank', 'off_target_order',
    #                                'off_target_phylum', 'off_target_species', 'off_target_superkingdom'])
    #         best_probes.to_csv(probe_summary_info_filename, index = False)
    #         probe_blast_off_target = pd.DataFrame(columns = ['probe_id', 'molecule_id', 'pid', 'qcovhsp', 'length', 'mismatch', 'gapopen', 'probe_start', 'probe_end', 'molecule_start',
    #         	                              'molecule_end', 'evalue', 'bitscore', 'staxids', 'qseq', 'sseq', 'mch', 'superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'])
    #         probe_off_target_summary_filename = re.sub('_probe_selection.csv', '_off_target_summary_info.csv', probe_summary_info_filename)
    #         probe_blast_off_target.to_csv(probe_off_target_summary_filename, index = False)
    #         probe_blast_cumulative = pd.DataFrame(columns = ['molecule_id'])
    #         prove_coverage_filename = re.sub('_probe_selection.csv', '_probe_coverage.csv', probe_summary_info_filename)
    #         probe_blast_cumulative.to_csv(prove_coverage_filename)
    return

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')

    parser.add_argument('design_level_evaluation_dir', type = str, help = 'Input file containing blast results')

    parser.add_argument('design_id', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('design_dir', type = str, help = 'Input file containing blast lineage')

    parser.add_argument('-n_workers', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input file containing blast lineage')

    parser.add_argument('-t', '--target_rank', dest = 'target_rank', type = str, default = 'phylum', help = 'Input file containing blast lineage')

    parser.add_argument('-tmin', '--min_tm', dest = 'min_tm', type = float, default = 55.0, help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-tmax', '--max_tm', dest = 'max_tm', type = float, default = 55.0, help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-m', '--mch', dest = 'mch', type = int, default = 14, help = 'Boolean to indicate whether to group sequences by similarity instead of taxonomic info')

    parser.add_argument('-gc', '--gc', dest = 'gc', type = float, default = 40.0, help = 'Number of top probes to keep')

    parser.add_argument('-bot', '--bot', dest = 'bot', type = float, default = 0.0, help = 'Number of top probes to keep')

    parser.add_argument('-bt', '--bitscore_thresh', dest = 'bitscore_thresh', type = float, default = 27, help = 'Number of top probes to keep')

    parser.add_argument('-sod', '--sod', dest = 'sod', type = float, default = 390, help = 'sodium concentration in nM')

    parser.add_argument('-dnaconc', '--dnaconc', dest = 'dnaconc', type = float, default = 5, help = 'oligo concentration in nM')

    parser.add_argument('-mt', '--mt_cutoff', dest = 'mt_cutoff', type = float, default = 60, help = 'oligo concentration in nM')

    parser.add_argument('-otgc', '--ot_gc_cutoff', dest = 'ot_gc_cutoff', type = float, default = 7, help = 'oligo concentration in nM')

    args = parser.parse_args()
    evaluation_dir = os.path.split(args.design_level_evaluation_dir)[0]
    sample_dir = os.path.split(evaluation_dir)[0]
    probes_summary_filename = '{}/probes_summary/probes_summary.h5'.format(sample_dir)
    blast_lineage_filename = '{}/utilities/blast_lineage.tab'.format(sample_dir)
    cluster = LocalCluster(n_workers = args.n_workers, threads_per_worker = 1, memory_limit = '16GB')
    client = Client(cluster)
    probe_evaluation_complete_filenames = glob.glob('{}/*_probe_evaluation_complete.txt'.format(args.design_level_evaluation_dir))
    for i in range(len(probe_evaluation_complete_filenames)):
        f = probe_evaluation_complete_filenames[i]
        probe_evaluation_dir = re.sub('_probe_evaluation_complete.txt', '', f)
        target_taxon = os.path.basename(probe_evaluation_dir)
        print('Secondary evaluation of probes for {}, taxon {} out of {}...'.format(target_taxon, i, len(probe_evaluation_complete_filenames)))
        summarize_probes(probe_evaluation_dir, blast_lineage_filename, args.design_dir, sample_dir, probes_summary_filename, args.target_rank, args.min_tm, args.max_tm, args.gc, args.mch, args.bot, args.bitscore_thresh, args.sod, args.dnaconc, args.mt_cutoff, args.ot_gc_cutoff)
        taxon_probe_evaluation_secondary_complete_filename = '{}/{}_probe_evaluation_secondary_complete.txt'.format(args.design_dir, target_taxon)
        file = open(taxon_probe_evaluation_secondary_complete_filename, 'w')
        file.write('Secondary probe evaluation for taxon {} is complete.'.format(target_taxon))
        file.close()
    client.close()
    cluster.close()
    probe_evaluation_secondary_complete_filename = '{}/probe_evaluation_secondary_complete.txt'.format(args.design_dir)
    file = open(probe_evaluation_secondary_complete_filename, 'w')
    file.write('Secondary probe evaluation is complete.')
    file.close()
    return

if __name__ == '__main__':
    main()
