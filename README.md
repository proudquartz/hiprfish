# HiPR-FISH Probe Design and Image Analysis 

## Acknowledgement
This suite of code makes use of open source packages, including `numpy`, `pandas`, `biopython`, `bioformats`, `javabridge`, `scikit-image`, `scikit-learn`, and `scipy`.

## HiPR-FISH Image Analysis
Image analysis pipelines and scripts for HiPR-FISH experiments

### Overview

This pipeline enables automated image analysis for highly multiplexed FISH experiments on microbial communities. In most cases, the main pipeline is a snakemake workflow. There are also standalone scripts used for specific analyses presented in our paper.

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
2. Install the environment by running the following commands:
     `conda create -n hiprfish python=3.6`\
     `conda activate hiprfish`\
     `conda install pandas`\
     `conda install -c bioconda primer3 -y`\
     `conda install -c anaconda joblib -y`\
     `conda install -c anaconda biopython -y`\
     `conda install -c etetoolkit ete3 ete_toolchain -y`\
     `conda install -n hiprfish -c conda-forge mamba -y`\
     `mamba install -c conda-forge -c bioconda snakemake -y`\
     `pip install SetCoverPy`\
     `pip install tables`\
     `pip install openpyxl`\
     `pip install matplotlib`\
     `python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose`
     
3. Install usearch. [Download](https://www.drive5.com/usearch/download.html) usearch executable. Move the downloaded file to a directory of your choosing. 
     
4. Install blast. [Download](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) the latest version and install the downloaded package. 

6. Edit the `hiprfish_config.json file` to point the pipeline to the correct directories.
   - `__default__`
      * `SCRIPTS_PATH`: path to the folder that contains all the scripts
      * `DATA_DIR`: path to the folder that contains input folders and files
   - `blast`
      * `16s_db`: path to the local copy of NCBI 16SMicrobial database. If you put the database files in `/[PATH_TO_16S_DB]/16S_ribosoma_RNA` (i.e. the path name to the full database files look like `/[PATH_TO_16S_DB]/16S_ribosoma_RNA/16S_ribosoma_RNA.n*`), you should set the value of this variable to `/[PATH_TO_16S_DB]/16S_ribosoma_RNA/16S_ribosoma_RNA`.
   -  `primer3`
      * `primer3_exec_dir`: path to the primer3 executable. If you installed primer3 via conda, you can likely just put "primer3_core" here. If that alias somehow does not work, you can put the full path to the primer3_core executable instead.
      * `primer3_config_dir`: configuration files for primer3. You can [download](https://github.com/primer3-org/primer3) the source repository and copy the primer3_config folder to a location your choosing on your local system.
   -  `usearch`
      * `usearch_dir`: path to the usearch executable
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
       * `BITSCORE_THRESH`: blast bitscore cutoff. Any blast hits (between probe and target sequence) with a score higher than this number will be considered significant and used for evaluation of probe specificity.
       * `BARCODESELECTION`: method for barcode assignment to taxa. Available options are:
         1. MostSimple: assign barcodes by barcode complexity, starting with the simplest ones. Barcodes with more bits are considered more complex.
         2. Random: randomly assign barcodes to taxa
         3. MostComplex: assign barcodes by barcode complexity, starting with the most complex ones. Barcodes with more bits are considered more complex.
       * `BPLC`: minimum blocking probe length threhold. Blocking probes with length lower than this threshold is considered likely to be washed off and do not need to be included in the final probe pool. Default is 15 bp.
       * `HELPER_PROBE_REPEAT`: number of times to repeat helper sequences in the final complex oligo pool. Default is 14 so that the any inidividual helper sequence is roughly at the same concentration as any individual encoding probe.
       * `SOD`: assumed sodium concentration for caluclation of melting temperatures. Default is 390.
       * `DNACONC`: assumed probe concentration for calculation of melting temperatures. Default is 5.
       * `MT_CUTOFF`: off target melting temperature cutoff. Probes would need to have a maximum off target melting temp smaller than this number to be considered specific. Default is 60. Note that this seems high because there seem to be a constant offset between melting temperatures calculated by primer3 and biopython built-in melting temperature calculation. This parameter refers to the calculation from the biopython implementation, which generally are higher than the primer3 calculations.
       * `OT_GC_CUTOFF`: off target maximum GC count. A probe would only be considered specific if any of its off-target binding sites have less than this many bases of G or C. Default is 7.
       * `THEME_COLOR`: overall theme color for axes and labels of the generated plots. Available options are:
          1. black: plots will have black axes and labels - works well against light background slides
          2. white: plots will have white axes and labels - works well against dark background slides
2. FASTA file
   - A FASTA file containing full length 16S sequences of the community to be probed. This file can be curated from public databases, or it can come from your own long read sequencing datasets, such as those from PacBio. The input file should be placed in `DATA_DIR/[SAMPLE]/input/[SAMPLE].fasta`.


### Output

1. Simulation results file
   - A csv file containing all the parameters for all the designs, as well as some summary statistics for each design
2. Probe folder
   - A folder containing selected probe summary files for each taxa, a concatenated file containing all selected probes, a file containing information for all the blocking probes, as well as text files that can be sent as is to array synthesis vendors for complex oligo pool synthesis.

### Running the pipeline
Run `snakemake --configfile hiprfish_config.json -j n`, where `n` is the number of cores to be used. If the pipeline excuted without errors, you should see a file called `simulation_table_test_results.csv` in the same directory where you put the `simulation_table_test.csv` file. It can be useful to run a design at a high taxonomic rank (phylum, for example) to make sure that the pipeline runs correctly with the input files.
