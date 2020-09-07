
"""
Collect HiPRFISH probe design results
Hao Shi 2017
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import glob
from ete3 import NCBITaxa
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib import cm


###############################################################################################################
# HiPR-FISH : collect probe design results
###############################################################################################################

def cm_to_inches(x):
    return(x/2.54)

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def get_cumulative_coverage(blast_lineage, target_rank, taxon_best_probes_filtered):
    n_total = blast_lineage.shape[0]
    taxon_best_probes_filtered['taxon_coverage_absolute'] = taxon_best_probes_filtered.taxon_coverage.values*taxon_best_probes_filtered.taxon_abundance.values
    taxon_best_probes_filtered = taxon_best_probes_filtered.sort_values(['taxon_coverage_absolute'], ascending = False)
    taxon_cumsum = taxon_best_probes_filtered[['target_taxon', 'taxon_coverage_absolute']].drop_duplicates().taxon_coverage_absolute.cumsum()
    cumcov = taxon_cumsum.values/n_total
    return(cumcov)

def get_lineage_sciname_at_desired_ranks(taxid, desired_ranks):
    'Retrieve lineage information at desired taxonomic ranks'
    # initiate an instance of the ncbi taxonomy database
    ncbi = NCBITaxa()
    # retrieve lineage information for each full length 16S molecule
    lineage = ncbi.get_lineage(taxid)
    lineage2ranks = ncbi.get_rank(lineage)
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())
    ranki = [ranks2lineage.get(x) for x in desired_ranks]
    ranks = [x if x is not None else 0 for x in ranki]
    ranks_translation = ncbi.get_taxid_translator(ranks)
    ranks_sciname = [ranks_translation[x] if x != 0 else 'NA' for x in ranks]
    return(ranks, ranks_sciname)

def analyze_taxon_probe_selections(data_dir, simulation_table, pipeline_version, theme_color):
    sim_tab = pd.read_csv(simulation_table)
    for i in range(sim_tab.shape[0]):
        sample = sim_tab.SAMPLE[i]
        target_rank = sim_tab.TARGET_RANK[i]
        design_id = sim_tab.DESIGN_ID[i]
        taxon_probe_selection_files = glob.glob('{}/simulation/{}/*_probe_selection.csv'.format(data_dir, design_id))
        taxon_list = [re.sub('_probe_selection.csv', '', os.path.basename(f)) for f in taxon_probe_selection_files]
        taxon_selection_summary = pd.DataFrame(columns = ['taxid','max_on_target_full_coverage','detectability'])
        taxon_selection_summary['taxid'] = taxon_list
        desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        desired_ranks_taxid = ['superkingdom_taxid', 'phylum_taxid', 'class_taxid', 'order_taxid', 'family_taxid', 'genus_taxid', 'species_taxid']
        desired_ranks_sciname = ['superkingdom_sciname', 'phylum_sciname', 'class_sciname', 'order_sciname', 'family_sciname', 'genus_sciname', 'species_sciname']
        ranks = pd.DataFrame(columns = ['taxid'] + desired_ranks_taxid + desired_ranks_sciname)
        taxon_selection_summary_taxid_list = taxon_selection_summary.taxid.unique()
        ranks['taxid'] = taxon_selection_summary_taxid_list
        for i in range(0, taxon_selection_summary_taxid_list.shape[0]):
            taxid = taxon_selection_summary_taxid_list[i]
            if taxid == '0':
                ranks.iloc[i, 1:len(desired_ranks)+1] = 'NA'
            else:
                if not str(taxid).isdigit():
                    taxid = taxid.split(';')[0]
                ranks_taxid, ranks_sciname = get_lineage_sciname_at_desired_ranks(taxid, desired_ranks)
                ranks.loc[i,desired_ranks_taxid] = ranks_taxid
                ranks.loc[i,desired_ranks_sciname] = ranks_sciname
        taxon_selection_summary = taxon_selection_summary.merge(ranks, on = 'taxid', how = 'left')
        for i in range(len(taxon_list)):
            taxon = taxon_list[i]
            print(taxon)
            taxon_probe_selection_filename = '{}/simulation/{}/{}_probe_selection.csv'.format(data_dir, design_id, taxon)
            taxon_coverage_filename = '{}/simulation/{}/{}_full_cover_matrix.csv'.format(data_dir, design_id, taxon)
            taxon_qcovhsp_filename = '{}/simulation/{}/{}_full_qcovhsp_matrix.csv'.format(data_dir, design_id, taxon)
            taxon_probe_selection = pd.read_csv(taxon_probe_selection_filename)
            selection_method = taxon_probe_selection.selection_method.drop_duplicates().values
            if ('AllSpecificPStartGroup' in selection_method) or ('AllSpecificPStartGroupMinOverlap' in selection_method):
                taxon_coverage = pd.read_csv(taxon_coverage_filename, index_col = ['molecule_id'])
                taxon_qcovhsp = pd.read_csv(taxon_qcovhsp_filename, index_col = ['molecule_id'])
                taxon_best_probe_coverage = taxon_coverage.loc[:,taxon_probe_selection.probe_id.values]
                taxon_best_probe_qcovhsp = taxon_qcovhsp.loc[:,taxon_probe_selection.probe_id.values]
                taxon_target_molecule_max_qcovhsp_fraction = taxon_best_probe_qcovhsp.max(axis = 1)/100
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'relative_abundance'] = taxon_probe_selection.taxon_abundance.values[0]/18899
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'detectability'] = 1
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'detectability_average'] = taxon_target_molecule_max_qcovhsp_fraction.mean()
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'cumulative_molecule_coverage'] = np.sum(taxon_best_probe_coverage.values.sum(axis = 1) > 0)/taxon_best_probe_coverage.shape[0]
                # taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'max_on_target_full_coverage'] = taxon_probe_selection.on_target_full_match.max()
            else:
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'relative_abundance'] = taxon_probe_selection.taxon_abundance.values[0]/18899
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'detectability'] = 0
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'detectability_average'] = 0
                taxon_selection_summary.loc[taxon_selection_summary.taxid.values == taxon, 'max_on_target_full_coverage'] = 0
        taxon_selection_summary_sorted = taxon_selection_summary.sort_values(by = 'relative_abundance', ascending = False).reset_index().drop(columns = ['index'])
        division = int(np.ceil(taxon_selection_summary.shape[0]/30))
        indices = np.arange(0, taxon_selection_summary.shape[0], 30)
        indices = np.append(indices, taxon_selection_summary.shape[0])
        fig = plt.figure()
        fig.set_size_inches(cm_to_inches(2 + 5*(division+1)), cm_to_inches(12))
        gs = GridSpec(1, division)
        for d in range(division):
            ax = plt.subplot(gs[0,d])
            taxon_selection_summary_sub = taxon_selection_summary_sorted.iloc[indices[d]:indices[d+1],:].reset_index().drop(columns = 'index')
            relative_abundance_max = taxon_selection_summary_sorted.relative_abundance.max()
            relative_abundance_max_log = np.log10(relative_abundance_max) - np.log10(0.5/18899)
            detectability_color_map = cm.get_cmap('Greens')
            plt.yticks(np.arange(0, -taxon_selection_summary_sub.shape[0], -1), taxon_selection_summary_sub.genus_sciname.values, fontsize = 8)
            plt.xlim(-0.5, 1.5)
            plt.ylim(-30+0.5, 0.5)
            for i in range(taxon_selection_summary_sub.shape[0]):
                relative_abundance = taxon_selection_summary_sub.loc[i, 'relative_abundance']
                relative_abundance_log = np.log10(relative_abundance) - np.log10(0.5/18899)
                rra = relative_abundance_log/relative_abundance_max_log
                max_on_target_full_coverage = taxon_selection_summary_sub.loc[i, 'max_on_target_full_coverage']
                plt.hlines(-i, 0, rra*0.8*max_on_target_full_coverage, color = (0,0.5,1), linewidth = 5)
                plt.hlines(-i, rra*0.8*max_on_target_full_coverage, rra*0.8, color = (1,0.5,0), linewidth = 5)
                if taxon_selection_summary_sub.loc[i, 'detectability'] == 1:
                    plt.plot(1, -i, 'o', color = (0.5,1,0), markersize = 4)
                else:
                    plt.plot(1, -i, 'o', color = (1,0.5,0), markersize = 4)
                ad = taxon_selection_summary_sub.loc[i, 'detectability_average']
                plt.plot(1.4, -i, 'o', color = (0, 0.5*ad, 1*ad), markersize = 4, markeredgecolor = (0.5,0.5,0.5), markeredgewidth = 0.5)
            # plt.xticks([0, 1, 1.4], ['Rel. Abundance', 'Detectability'], rotation = 45, ha = 'right', fontsize = 8)
            plt.xticks([0, 1, 1.4], ['Rel. Abundance', 'Detectability', 'AD'], rotation = 45, ha = 'right', fontsize = 8)
            plt.tick_params(direction = 'in', length = 0, colors = theme_color)
            ax.spines['left'].set_color(None)
            ax.spines['bottom'].set_color(None)
            ax.spines['right'].set_color(None)
            ax.spines['top'].set_color(None)

        plt.subplots_adjust(left = 0.2, bottom = 0.2, top = 0.98, right = 0.98, wspace = 1.7)
        plt.savefig('{}/simulation/{}/design_coverage.pdf'.format(data_dir, design_id), transparent = True, dpi = 300)
        plt.close()
    total_coverage = np.sum(taxon_selection_summary.max_on_target_full_coverage.values*taxon_selection_summary.relative_abundance.values)
    return(total_coverage)

def generate_color_map(top_color, n):
    vals = np.ones((n, 4))
    vals[:, 0] = np.linspace(0, top_color[0], n)
    vals[:, 1] = np.linspace(0, top_color[1], n)
    vals[:, 2] = np.linspace(0, top_color[2], n)
    new_cmap = ListedColormap(vals)
    return(new_cmap)

def plot_taxon_coverage_heatmap(taxon_best_probe_qcovhsp, theme_color):
    fig = plt.figure()
    fig.set_size_inches(cm_to_inches(8), cm_to_inches(6))
    cmap = generate_color_map((0.5,1,0), 256)
    heatmap = plt.imshow(taxon_best_probe_qcovhsp.values.astype(float), cmap = cmap, vmin = 0, vmax = 100)
    fig.colorbar(heatmap, anchor = (0,0))
    plt.xticks(np.arange(taxon_best_probe_qcovhsp.shape[1]), taxon_best_probe_qcovhsp.columns, rotation = 90)
    plt.yticks(np.arange(taxon_best_probe_qcovhsp.shape[0]), taxon_best_probe_qcovhsp.index)
    plt.tick_params(direction = 'in', length = 2, colors = theme_color, labelsize = 8)
    plt.xlabel('Probe ID', fontsize = 8)
    plt.ylabel('Target ID', fontsize = 8)
    plt.subplots_adjust(left = 0.3, bottom = 0.6, right = 0.98, top = 0.98)
    return

def collect_probe_coverage_results(data_dir, simulation_table, pipeline_version, output_filename):
    print('Loading samples table: {}'.format(simulation_table))
    sim_tab = pd.read_csv(simulation_table)
    sim_tab['Gini_Index'] = np.nan
    sim_tab['Inverse_Simpson_Index'] = np.nan
    sim_tab['Shannon_Index'] = np.nan
    sim_tab['COVERED_TAXA_RICHNESS'] = np.nan
    sim_tab['COVERED_TAXA_RICHNESS_FRACTION'] = np.nan
    sim_tab['CUMCOV_all'] = np.nan
    sim_tab['CUMCOV_TOP10'] = np.nan
    sim_tab['CUMCOV_TOP20'] = np.nan
    sim_tab['OFF_TARGET_PHYLUM'] = np.nan
    sim_tab['OFF_TARGET_CLASS'] = np.nan
    sim_tab['OFF_TARGET_ORDER'] = np.nan
    sim_tab['OFF_TARGET_FAMILY'] = np.nan
    sim_tab['OFF_TARGET_GENUS'] = np.nan
    sim_tab['OFF_TARGET_SPECIES'] = np.nan
    sim_tab['PreFiltering'] = np.nan
    sim_tab['Filtered'] = np.nan
    sim_tab['PostFiltering'] = np.nan
    print('Loading result files:')
    for i in range(0, sim_tab.shape[0]):
        sample = sim_tab.SAMPLE[i]
        target_rank = sim_tab.TARGET_RANK[i]
        taxon_best_probes_filtered_filename = '{}/simulation/{}/taxon_best_probes_filtered.csv'.format(data_dir, sim_tab.DESIGN_ID[i])
        blast_lineage_filename = '{}/{}/utilities/blast_lineage.tab'.format(data_dir, sample)
        blast_lineage = pd.read_table(blast_lineage_filename)
        if os.path.exists(taxon_best_probes_filtered_filename):
            taxon_best_probes_filtered = pd.read_csv(taxon_best_probes_filtered_filename)
            sim_tab.COVERED_TAXA_RICHNESS.loc[i] = taxon_best_probes_filtered.target_taxon.drop_duplicates().shape[0]
            sim_tab.COVERED_TAXA_RICHNESS_FRACTION.loc[i] = taxon_best_probes_filtered.target_taxon.drop_duplicates().shape[0]/blast_lineage.loc[:,sim_tab.loc[0,'TARGET_RANK']].drop_duplicates().shape[0]
            cumcov = get_cumulative_coverage(blast_lineage, target_rank, taxon_best_probes_filtered)
            sim_tab.CUMCOV_all.loc[i] = cumcov[-1]
            if cumcov.shape[0] >= 10:
                sim_tab.CUMCOV_TOP10.loc[i] = cumcov[9]
            if cumcov.shape[0] >= 20:
                sim_tab.CUMCOV_TOP20.loc[i] = cumcov[19]
            print('Saving collected results to {}...'.format(output_filename))
        else:
            print('Sample result file {} does not exist'.format(taxon_best_probes_filtered_filename))
        taxon_abundance = blast_lineage[target_rank].value_counts().reset_index()
        taxon_abundance.columns = ['target_taxon', 'counts']
        taxon_abundance['rel_freq'] = taxon_abundance.counts.values/taxon_abundance.counts.sum()
        sim_tab.loc[i, 'Gini_Index'] = gini(taxon_abundance.rel_freq.sort_values(ascending = True).values)
        sim_tab.loc[i, 'Inverse_Simpson_Index'] = 1/(np.sum(taxon_abundance.rel_freq**2))
        sim_tab.loc[i, 'Shannon_Index'] = -np.sum(taxon_abundance.rel_freq*np.log(taxon_abundance.rel_freq))
        probe_filtering_statistics = pd.read_csv('{}/simulation/{}/{}_probe_filtering_statistics.csv'.format(data_dir, sim_tab.DESIGN_ID[i], sim_tab.DESIGN_ID[i]))
        sim_tab.loc[i, 'PreFiltering'] = probe_filtering_statistics.loc[0,'PreFiltering']
        sim_tab.loc[i, 'Filtered'] = probe_filtering_statistics.loc[0,'Filtered']
        sim_tab.loc[i, 'PostFiltering'] = probe_filtering_statistics.loc[0,'PostFiltering']
        full_length_probes_filename = '{}/simulation/{}/{}_full_length_probes_sequences.txt'.format(data_dir, sim_tab.DESIGN_ID[i], sim_tab.DESIGN_ID[i])
        full_length_blocking_probes_filename = '{}/simulation/{}/{}_full_length_blocking_probes_sequences.txt'.format(data_dir, sim_tab.DESIGN_ID[i], sim_tab.DESIGN_ID[i])
        full_length_helper_probes_filename = '{}/simulation/{}/{}_full_length_helper_probes_sequences.txt'.format(data_dir, sim_tab.DESIGN_ID[i], sim_tab.DESIGN_ID[i])
        encoding_probes = pd.read_csv(full_length_probes_filename, header = None)
        sim_tab.loc[i, 'ENCODING_PROBES_COUNT'] = encoding_probes.shape[0]
        helper_probes = pd.read_csv(full_length_helper_probes_filename, header = None)
        sim_tab.loc[i, 'HELPER_PROBES_COUNT'] = helper_probes.shape[0]
        try:
            blocking_probes = pd.read_csv(full_length_blocking_probes_filename, header = None)
            sim_tab.loc[i, 'BLOCKING_PROBES_COUNT'] = blocking_probes.shape[0]
        except:
            sim_tab.loc[i, 'BLOCKING_PROBES_COUNT'] = 0
            pass
        sim_tab.loc[i, 'PIPELINE_VERSION'] = pipeline_version
    sim_tab['TOTAL_COUNT'] = sim_tab.ENCODING_PROBES_COUNT.values + sim_tab.BLOCKING_PROBES_COUNT.values + sim_tab.HELPER_PROBES_COUNT.values
    sim_tab['ENCODING_PROBES_FRACTION'] = sim_tab.ENCODING_PROBES_COUNT.values/sim_tab.TOTAL_COUNT.values
    sim_tab['BLOCKING_PROBES_FRACTION'] = sim_tab.BLOCKING_PROBES_COUNT.values/sim_tab.TOTAL_COUNT.values
    sim_tab['HELPER_PROBES_FRACTION'] = sim_tab.HELPER_PROBES_COUNT.values/sim_tab.TOTAL_COUNT.values
    sim_tab.to_csv(output_filename, index = False, header = True)
    return(sim_tab)

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Collect summary statistics of HiPRFISH probes for a complex microbial community')

    # data directory
    parser.add_argument('data_dir', type = str, help = 'Directory of the data files')

    # input simulation table
    parser.add_argument('simulation_table', type = str, help = 'Input csv table containing simulation information')

    parser.add_argument('pipeline_version', type = str, help = 'Input csv table containing simulation information')
    parser.add_argument('theme_color', type = str, help = 'Input csv table containing simulation information')
    # output simulation results table
    parser.add_argument('simulation_results', type = str, help = 'Output csv table containing simulation results')

    args = parser.parse_args()

    sim_tab = collect_probe_coverage_results(args.data_dir, args.simulation_table, args.pipeline_version, args.simulation_results)
    analyze_taxon_probe_selections(args.data_dir, args.simulation_table, args.pipeline_version, args.theme_color)

if __name__ == '__main__':
    main()
