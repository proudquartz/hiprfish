import os
import re
import glob
import argparse
import subprocess
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC, generic_dna
from matplotlib import pyplot as plt
from Bio.Blast.Applications import NcbiblastnCommandline

os.environ['OMP_NUM_THREADS'] = "16"
os.environ['MKL_NUM_THREADS'] = "16"
os.environ['DASK_NUM_THREADS'] = "16"

###############################################################################################################
# HiPR-FISH-strain : combine probes
###############################################################################################################

def sub_slash(x):
    return(re.sub('/', '_', x))

@dask.delayed
def write_blast_pacbio_sequence(sequence, temp_dir):
    seq = SeqRecord(Seq(sequence.SEQUENCE), id = sequence.SEQID, description = '')
    seq_fasta_filename = '{}/{}.fasta'.format(temp_dir, sub_slash(sequence.SEQID))
    seq_blast_out_filename = '{}/{}.fasta.blast.out'.format(temp_dir, sub_slash(sequence.SEQID))
    SeqIO.write(seq, seq_fasta_filename, 'fasta')
    out_format = '6 qseqid sseqid pident qcovhsp length mismatch gapopen qstart qend sstart send evalue bitscore staxids'
    return_code = subprocess.check_call(['blastn', '-db', blast_database, '-query', seq_fasta_filename, '-out', seq_blast_out_filename, '-outfmt', out_format, '-task', 'blastn-short', '-max_hsps', '1', '-max_target_seqs', '1'])
    return(return_code)

def write_probes(probes_summary_filename, probes_dir, n_workers):
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1, memory_limit = "16GB")
    client = Client(cluster)
    num_record = len([s for s in SeqIO.parse(input_fasta_filename, 'fasta')])
    pacbio_sequence_list = pd.DataFrame(index = np.arange(num_record), columns = ['SEQID', 'SEQUENCE'])
    pacbio_sequence_list['SEQID'] = [s.id for s in SeqIO.parse(input_fasta_filename, 'fasta')]
    pacbio_sequence_list['SEQUENCE'] = [s.seq for s in SeqIO.parse(input_fasta_filename, 'fasta')]
    return_code_list = []
    for i in range(num_record):
        return_code = write_blast_pacbio_sequence(pacbio_sequence_list.iloc[i, :], temp_dir)
        return_code_list.append(return_code)
    return_code_total = dask.delayed(sum)(return_code_list)
    result = return_code_total.compute()
    probes_blast_results = dd.read_table('{}/*.blast.out'.format(temp_dir), delim_whitespace = True, header = None, dtype = {13:str})
    probes_blast = probes_blast_results.compute()
    probes_blast_results_filename = os.path.basename(input_fasta_filename)
    probes_blast_results.to_csv('{}/{}.blast.out'.format(temp_dir, probes_blast_results_filename), index = None, header = None, sep = ' ')
    client.close()
    cluster.close()
    return(0)

###############################################################################################################
# main function
###############################################################################################################

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('probes_summary_filename', type = str, help = 'Input FASTA file containing 16S sequences')
    parser.add_argument('-n', '--n_workers', dest = 'n_workers', type = int, default = 20, help = 'Input FASTA file containing 16S sequences')
    args = parser.parse_args()
    primer3_dir = os.path.split(args.probes_summary_filename)[0]
    data_dir = os.path.split(primer3_dir)[0]
    probes_dir = '{}/probes'.format(data_dir)
    evaluate_dir = '{}/evaluate'.format(data_dir)
    probe_write_complete_filename = '{}/probes_write_complete.txt'.format(probes_dir)
    if not os.path.exists(probes_dir):
        os.makedirs(probes_dir)
    if not os.path.exists(evaluate_dir):
        os.makedirs(evaluate_dir)
    probe_write_status = write_probes(args.probes_summary_filename, probes_dir, args.n_workers)
    print('Writing probe write complete file...')
    file = open(probe_write_complete_filename, 'w')
    file.write('Probe write is done.')
    file.close()
    print('Finished with probe write')
    return

if __name__ == '__main__':
    main()
