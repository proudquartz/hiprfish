# HiPR-FISH Probe Design and Image Analysis 

## Acknowledgement
This suite of code makes use of open source packages, including `numpy`, `pandas`, `biopython`, `bioformats`, `javabridge`, `scikit-image`, `scikit-learn`, and `scipy`.

## HiPR-FISH Image Analysis
Image analysis pipelines and scripts for HiPR-FISH experiments

### Overview

This pipeline enables automated image analysis for highly multiplexed FISH experiments on microbial communities. In most cases, the main pipeline is a snakemake workflow. There are also standalone scripts used for specific analyses presented in our paper.

### Before running the pipeline
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html),
2. Install the environment by running the following command\
     `conda env create hiprfish python=3.5`\
     `conda install pandas`\
     `conda install -c anaconda biopython`\
     `conda install javabridge`\
     `conda install -c bioconda python-bioformats`\

## HiPR-FISH Probe Design
Probe design pipeline for HiPR-FISH experiments

### Acknowledgements
We would like to thank Jakob Wirbel for their help with testing the probe design pipeline. 

### Overview

This pipeline enables design of complex oligo probe sets used for highly multiplexed FISH experiments on microbial communities. The main pipeline is a snakemake workflow. There are two versions of the pipeline. The `hiprfish-probe-design-consensus` version uses the consensus approach by designing probes from the taxon consensus sequence for each taxon. The `hiprfish-probe-design-molecule` version designs probes from each individual 16S molecule from PacBio sequencing datasets and pool all unique probes for subsequent evaluation. The probe evaluation and selection is identifical in either version.  

### Required resources

The pipeline requires a local copy of the 16SMicrobial database from NCBI.

### Before running the pipeline
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html),
2. Install the environment by either
   - Running `conda env create -f hiprfish.yml` in a Terminal window,\

  OR

   - Running the following command\
     `conda env create hiprfish python=3.5`\
     `conda install pandas=0.25`\
     `conda install -c anaconda biopython`\
     `conda install -c etetoolkit ete3`\
     `conda install snakemake`\
     `conda install blast`\
     `source activate hiprfish`\
     `pip install SetCoverPy`
     `pip install tables`
     `pip install openpyxl`
     `pip install matplotlib`
     

3. Activate the environment by running `source activate hiprfish`,
4. Edit the `hiprfish_config.json file` to point the pipeline to the correct directories.
   - `__default__`
      * `SCRIPTS_PATH`: path to the folder that contains all the scripts
      * `DATA_DIR`: path to the folder that contains input folders and files
   - `blast`
      * `16s_db`: path to the local copy of NCBI 16SMicrobial database
   -  `usearch`
      * `path`: path to the usearch program
   - `simulations`
      * `simulation_table`: path to the simulation summary file

### Input
1. Simulation summary file (simulation_table_test.csv)
   - A csv file containing all the designs to be run.
      * `DESIGN_ID`: identifier for each design
      * `SAMPLE`: name of the input FASTA file without file extension
      * `TARGET_RANK`: desired taxonomic rank for the probe design. Availabel options: phylum, class, order, family, genus, and species
      * `SIMILARITY`: similarity cut off for grouping 16S sequences. A low cut off (e.g. 0.1) essentially means 16S sequences will be grouped by their lineage information in the NCBI 16SMicrobial database. A higher cut off can be used to subdivide sequences within a given taxon. Higher cut off values generally leads to longer run time.
      * `MAX_CONTINUOUS_HOMOLOGY`: maximum continuous homology (measured in bp) for a probe-target hit to be considered significant. Lower values leads to more stringent designs. Default is 14 bp.
      * `MIN_TM`: minimum melting temperature threhold
      * `MAX_TM`: maximum melting temperature threhold
      * `GC`: minimum probe GC content threhold
      * `INCLUDE_START`: number of nucleotides to exclude at the beginning of the 16S sequences
      * `INCLUDE_END`: number of nucleotides to exclude at the end of the 16S sequences
      * `PROBE_SELECTION_METHOD`: method for selecting probes. Available options are
         1. `SingleBestProbe`: select the top probe for each taxa, if available
         2. `AllSpecific`: select all probes that are specific and only specific to its target taxon
         3. `AllSpecificPStartGroup`: select all probes that are specific and only specific to its target taxon within each segment of the 16S sequences. By default the 16S sequences are dividied into block resolutions of 100bp regions. If there are less than 15 probes available (average one probe per block), the block resolution is modified in 20bp decrements until there are 15 probes or the block resolution is zero, whichever happens first.
         4. `MinOverlap`: select all probes that are specific and only specific to its target taxon with minimum overlap in their target coverage
         5. `TopN`: select the top *n* probes for each taxa
       * `PRIMERSET`: primer sets to include in the final probes. There are three sets (A, B, and C) availble in the current version. User specific primer sets can also be added if necessary.
       * `OTU`: boolean to indicate whether to group 16S sequences only by their similarity. Generally set to `F` for ease of taxonomic interpretation of the probe designs, but could be useful if very high taxonomic resolution is desired.
       * `TPN`: number of top probes to select for each taxon, if the probe selection method is set to `TopN`
       * `FREQLL`: minimum abundance threshold. Default is zero, and is generally left at zero. Can be increased in situations where the in silico taxonomic coverage is not as good as desired. A higher value means increasing the probe design space for the more abundance sequences at the risk of those probes mishybridizing to the lower abundance taxa in the experiment.
       * `BOT`: minimum blast on target rate threshold. Probes with blast on target values lower than this value is considered *promiscuous*, and is not included in the final probe pool.
       * `BARCODESELECTION`: method for barcode assignment to taxa. Available options are:
         1. MostSimple: assign barcodes by barcode complexity, starting with the simplest ones. Barcodes with more bits are considered more complex.
         2. Random: randomly assign barcodes to taxa
         3. MostComplex: assign barcodes by barcode complexity, starting with the most complex ones. Barcodes with more bits are considered more complex.
       * `BPLC`: minimum blocking probe length threhold. Blocking probes with length lower than this threshold is considered likely to be washed off and do not need to be included in the final probe pool. Default is 15 bp.
2. FASTA file
   - A FASTA file containing full length 16S sequences of the community to be probed. This file can be curated from public databases, or it can come from your own long read sequencing datasets, such as those from PacBio. The input file should be placed in `DATA_DIR/[SAMPLE]/input/[SAMPLE].fasta`.


### Output

1. Simulation results file
   - A csv file containing all the parameters for all the designs, as well as some summary statistics for each design
2. Probe folder
   - A folder containing selected probe summary files for each taxa, a concatenated file containing all selected probes, a file containing information for all the blocking probes, as well as text files that can be sent as is to array synthesis vendors for complex oligo pool synthesis.

### Running the pipeline
Run `snakemake --configfile hiprfish_config.json -j n`, where `n` is the number of cores to be used. If the pipeline excuted without errors, you should see a file called `simulation_table_test_results.csv` in the same directory where you put the `simulation_table_test.csv` file. It can be useful to run a design at a high taxonomic rank (phylum, for example) to make sure that the pipeline runs correctly with the input files.
