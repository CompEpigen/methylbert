from .bam import process_bismark_read, process_dorado_read
import argparse, os, uuid, sys

import multiprocessing as mp

import pandas as pd
import numpy as np

from functools import partial
from Bio import SeqIO

import pickle, pysam, os, glob, random, re, time
import warnings

# https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f
def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def kmers(seq: str, k: int=3):
    '''
        Covert sequences including both reference DNA and methylation patterns into 3-mer DNA and methylation seqs
    '''
    converted_seq = list()
    methyl = list()

    if k % 2 == 0:
        raise ValueError(f"k must be an odd number because of CpG methylation being at the middle of the token. The given k is {k}")

    mid = int(k/2)
    for seq_idx in range(len(seq)-k):
        token = seq[seq_idx:seq_idx+k]
        if token[mid] =='z':
            m = 0
        elif token[mid] =='Z':
            m = 1
        else:
            m = 2

        # six alphabets indicating cytosine methylation in bismark processed files
        token = re.sub("[h|H|z|Z|x|X]", "C", token) 
        converted_seq.append(token)
        methyl.append(str(m))

    return converted_seq, methyl


def read_extract(bam_file_path: str, dict_ref: dict, k: int, dmrs: pd.DataFrame, ncores: int=1, methyl_caller: str = "bismark"):
    '''
        Extract reads including methylation patterns overlapping with DMRs
        and convert those into 3-mer sequences

        bam_file_path: (str)
            single-cell bam file path 
        dict_ref: (dict)
            Reference genome given in a dictionary whose key is chromosome and value is DNA sequences
        k: (int)
            k value for k-mers
        dmrs: (pd.Dataframe)
            dataframe where DMR information is stored (chromo, start, end are required)
        ncores: (int)
            Number of cores to be used for parallel processing, default: 1
    '''
    
    def _reads_overlapping(aln, chromo, start, end, methyl_caller="bismark"):
        '''
        seq_list = list()
        dna_seq = list()
        xm_tags = list()
        '''

        if methyl_caller not in ["bismark", "dorado"]:
            raise ValueError(f"Methylation caller must be one of [bismark, dorado]")

        # get all reads overlapping with chromo:start-end
        fetched_reads = aln.fetch(chromo, start, end, until_eof=True)
        processed_reads = list()

        for read in fetched_reads:
            # Only select fully overlapping reads
            if (read.pos  < start) or ((read.pos+read.query_alignment_length) > end):
                continue
            
            # Remove case-specific mode occured by the quality
            ref_seq = dict_ref[chromo][read.pos:(read.pos+read.query_alignment_length)].upper() 

            if methyl_caller == "bismark":
                ref_seq = process_bismark_read(ref_seq, read)
            elif methyl_caller == "dorado":
                ref_seq = process_dorado_read(ref_seq, read)

            if ref_seq is None:
                continue

            # When there is no CpG methylation patterns after processing
            if "z" not in ref_seq.lower():
                continue

            # K-mers
            s, m = kmers(ref_seq, k=3)

            if len(s) != len(m):
                raise ValueError("DNA and methylation sequences have different length (%d and %d)"%(len(s), len(m)))

            # Add processed results as a tag
            read.setTag("RF", value=" ".join(s), replace=True) # reference sequence
            read.setTag("ME", value="".join(m), replace=True) # methylation pattern sequence 

            # Process back to a dictionary
            read_tags = {t:v for t, v in read.get_tags()}
            read = read.to_dict()
            read.update(read_tags)
            processed_reads.append(read)

        processed_reads = pd.DataFrame(processed_reads)
        if "tags" in processed_reads.keys():
            processed_reads = processed_reads.drop(columns=["tags"])

        return processed_reads
        
    @globalize
    def _get_methylseq(dmr, bam_file_path: str, k: int, methyl_caller: str):
        '''
            Return a dictionary of DNA seq, cell type and methylation seq processed in a 3-mer seq
        '''
        aln = pysam.AlignmentFile(bam_file_path, "rb")

        processed_reads = _reads_overlapping(aln, 
                                             dmr["chr"], int(dmr["start"]), int(dmr["end"]),
                                             methyl_caller)
        if processed_reads.shape[0] > 0:
            processed_reads = processed_reads.assign(dmr_ctype = dmr["ctype"],
                                                     dmr_label = dmr["dmr_id"])
            return processed_reads

    if ncores > 1:
        with mp.Pool(ncores) as pool:
            # Convert read sequences to k-mer sequences 
            seqs = pool.map(partial(_get_methylseq, 
                                    bam_file_path = bam_file_path, k=k, methyl_caller=methyl_caller),
                            dmrs.to_dict("records"))
    else:
        seqs = [_get_methylseq(dmr, bam_file_path = bam_file_path, k=k, methyl_caller = methyl_caller) for dmr in dmrs.to_dict("records")]
        
    # Filter None values that means no overlapping read with the given DMR
    seqs = list(filter(lambda i: i is not None, seqs))
    if len(seqs) > 0:
        return pd.concat(seqs, ignore_index=True)
    else:
        warnings.warn(f"Zero reads were extracted from {bam_file_path}")

def finetune_data_generate(
        f_dmr: str,
        output_dir: str,
        f_ref: str,
        sc_dataset: str = None,
        input_file: str = None, 
        n_mers: int = 3,
        split_ratio: float = 1.0,
        n_dmrs: int = -1,
        n_cores: int = 1,
        seed: int = 950410,
        ignore_sex_chromo: bool = True,
        methyl_caller: str = "bismark"
    ):

    # Setup random seed 
    random.seed(seed)
    np.random.seed(seed)

    # Setup output files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fp_dmr = os.path.join(output_dir, "dmrs.csv") # File to save selected DMRs

    # Reference genome
    record_iter = SeqIO.parse(f_ref, "fasta")
    
    # Save the reference genome into a dictionary with chr as a key value
    dict_ref = dict()
    for r in record_iter:
        dict_ref[r.id] = str(r.seq.upper())
    del record_iter
    
    # Load DMRs into a data frame
    dmrs = pd.read_csv(f_dmr, sep="\t", index_col=None)
    if ("chr" not in dmrs.keys()) or ("start" not in dmrs.keys()) or ("end" not in dmrs.keys()):
        ValueError("The .csv file for DMRs must contain chr, start and end in the header.")

    # Remove chrX, chrY, chrM and so on in DMRs
    regex_expr = "chr\d+" if ignore_sex_chromo else "chr[\d+|X|Y]"
    dmrs = dmrs[dmrs["chr"].str.contains(regex_expr, regex=True)]
    if dmrs.shape[0] == 0:
        ValueError("Could not find any DMRs. Please make sure chromosomes have \'chr\' at the beginning.")

    # Sort by statistics if available
    if "areaStat" in dmrs.keys():
        print("DMRs sorted by areaStat")
        dmrs = dmrs.sort_values(by="areaStat", ascending=False)
    elif "diff.Methy" in dmrs.keys():
        print("DMRs sorted by diff.Methy")
        dmrs = dmrs.sort_values(by="diff.Methy", ascending=False)
    else:
        print("Could not find any statistics to sort DMRs")

    # Select top n dmrs based on 
    if n_dmrs > 0:
        print(f"{n_dmrs} are selected based on the statistics")
        list_dmrs = list()
        for c in dmrs["ctype"].unique(): #  For the case when multiple cell types are given
            ctype_dmrs = dmrs[dmrs["ctype"]==c]
            if ctype_dmrs.shape[0] > n_dmrs:
                list_dmrs.append(ctype_dmrs[:n_dmrs])
            else:
                list_dmrs.append(ctype_dmrs)
        dmrs = pd.concat(list_dmrs)
        del list_dmrs

    # Newly assign dmr label from 0
    if "dmr_id" not in dmrs.keys():
      dmrs["dmr_id"] = range(len(dmrs))

    # Save DMRs in a new file 
    dmrs.to_csv(fp_dmr, sep="\t", index=False)
    print(dmrs.head())

    print(f"Number of DMRs to extract sequence reads: {len(dmrs)}")

    # check whether the input is a file or a file list
    if ( not sc_dataset ) and ( not input_file ):
        ValueError("Please provide either a list of input files or a file path. Both are given.")
    elif ( not sc_dataset ):
        # one input file in the list
        sc_files = [input_file]
    elif ( not input_file ):
        # Collect train data (single-cell samples)
        train_sc_samples = []
        with open(sc_dataset, "r") as fp_sc_dataset:
            sc_files = fp_sc_dataset.readlines()
    else:
        sys.exit("Either a list of input files or a file path must be given.")
    
    # Collect reads from the .bam files
    df_reads = list()
    for f_sc in sc_files:
        f_sc = f_sc.strip().split("\t")
        f_sc_bam = f_sc[0]

        

        extracted_reads = read_extract(f_sc_bam, dict_ref, k=3, dmrs=dmrs, ncores=n_cores, methyl_caller=methyl_caller)
        if extracted_reads is None:
            continue

        # cell type for the single-cell data
        '''
        if "RG" in extracted_reads.columns:
            extracted_reads = extracted_reads.rename(columns={"RG":"read_ctype"})
            #extracted_reads["ctype"] = [c.split("_")[1][:3]+"-"+c.split("_")[1][3] for c in extracted_reads["read_ctype"]] # mouse single-cell
            extracted_reads["ctype"] = [c.split("_")[1] for c in extracted_reads["read_ctype"]] # tumour pseudo bulk
        else:
        '''
        if len(f_sc) > 1:
            print(f"{f_sc_bam} processing ({f_sc[1]})...")
            extracted_reads["ctype"] = f_sc[1]
        else:
            extracted_reads["ctype"] = "NA"
        extracted_reads = extracted_reads.rename(columns={"RF":"dna_seq", "ME":"methyl_seq"})

        if(extracted_reads.shape[0] > 0):
            df_reads.append(extracted_reads)

    # Integrate all reads and shuffle
    if len(df_reads) > 0:
        df_reads = pd.concat(df_reads, ignore_index=True).sample(frac=1).reset_index(drop=True) # sample is for shuffling
        print("Fine-tuning data generated:", df_reads.head())
    else:
        ValueError("Could not find any reads overlapping with the given DMRs. Please try different regions.")

    # Split the data into train and valid/test
    if (split_ratio < 1.0) and (split_ratio > 0.0):
        fp_train_seq = os.path.join(output_dir, "train_seq.csv")
        fp_test_seq = os.path.join(output_dir, "test_seq.csv")
        n_train = int(df_reads.shape[0]*split_ratio)
        print("Size - train %d seqs , valid %d seqs "%(n_train, df_reads.shape[0] - n_train))

        # Write train & test files
        df_reads.loc[:n_train,:].to_csv(fp_train_seq, sep="\t", header=True, index=None)
        df_reads.loc[n_train:,:].to_csv(fp_test_seq, sep="\t", header=True, index=None)
    else:
        fp_data_seq = os.path.join(output_dir, "data.csv")
        df_reads.to_csv(fp_data_seq, sep="\t", header=True, index=None)

