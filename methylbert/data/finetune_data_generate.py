import argparse, os, uuid, sys

import multiprocessing as mp

import pandas as pd
import numpy as np

from functools import partial

from Bio import SeqIO

import pickle, pysam, os, glob, random, re, time

non_assign_chars = ["h","H", "x", "X"]


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--sc_dataset", required=True, type=str, help=".txt file all single-cell bam files are listed up")
    parser.add_argument("-d", "--f_dmr", required=True, type=str, help=".bed or .csv file DMRs information is contained")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="a directory where all generated results will be saved")
    parser.add_argument("-r", "--f_ref", required=True, type=str, help=".fasta file for the reference genome")
    parser.add_argument("-nm", "--n_mers", type=int, default=3, help="n-mers")
    parser.add_argument("-p", "--split_ratio", type=float, default=0.8, help="the ratio between train and test dataset")
    parser.add_argument("-nd", "--n_dmrs", type=int, default=-1, help="Number of DMRs to take from the dmr file. If the value is not given, all DMRs will be used")
    parser.add_argument("-c", "--n_cores", type=int, default=1, help="number of cores for the multiprocessing")
    parser.add_argument("--seed", type=int, default=950410, help="seed number")

    return parser.parse_args()

# https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f
def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def kmers(seq, k=3):
    converted_seq = list()
    methyl = list()
    for seq_idx in range(len(seq)-k):
        token = seq[seq_idx:seq_idx+k]
        if token[1] =='z':
            m = 0
        elif token[1] =='Z':
            m = 1
        else:
            m = 2

        token = re.sub("[h|H|z|Z|x|X]", "C", token) 
        converted_seq.append(token)
        methyl.append(str(m))


    return converted_seq, methyl


def parse_cigar(cigar):
    num = 0
    cigar_list = list()
    cigar = list(cigar)
    for c in cigar:
        if c.isdigit() : 
            num = num*10 + int(c)
        else:
            cigar_list += [num, c]
            num = 0
    return cigar_list

def handling_cigar(ref_seq, xm_tag, cigar):
    cigar_list = parse_cigar(cigar)
    cur_idx = 0
    
    ref_seq = list(ref_seq)
    xm_tag = list(xm_tag)
    
    for cigar_idx in range(int(len(cigar_list)/2)):
        cigar_num = cigar_list[cigar_idx*2]
        cigar_char = cigar_list[(cigar_idx*2)+1]
        
        if cigar_char in ["I", "S"]: # insertion, soft clip
            ref_seq = ref_seq[:cur_idx] + [cigar_char for i in range(cigar_num)] + ref_seq[cur_idx:]
        elif cigar_char in ["D" , "N"]: # deletion, skip
            xm_tag = xm_tag[:cur_idx] + [cigar_char for i in range(cigar_num)] + xm_tag[cur_idx:]
        cur_idx += cigar_num
        
    return "".join(ref_seq), "".join(xm_tag)

def read_extract(bam_file_path: str, dict_ref: dict, k: int, dmrs: pd.DataFrame, ncores=1, single_end=None):
    '''
        Extract reads including methylation patterns overlapping with DMRs
        and convert those into 3-mers sequences

        bam_file_path (str)
            single-cell bam file path 
        dict_ref (dict)
            dictionary whose key is chromosome and value is corresponding DNA sequence
        k (int)
            k value for k-mers
        dmrs (pd.Dataframe)
            dataframe where DMR information is stored (chromo, start, end are required)
        ncores (int)
            Number of cores to be used for parallel processing

    '''
    
    def _reads_overlapping(aln, chromo, start, end, single_end):
        seq_list = list()
        dna_seq = list()
        xm_tags = list()

        # get all reads overlapping with chromo:start-end
        fetched_reads = aln.fetch(chromo, start, end, until_eof=True)
        processed_reads = list()
        for reads in fetched_reads:
            ref_seq = dict_ref[chromo][reads.pos:(reads.pos+reads.query_alignment_end)].upper() # Remove case-specific mode occured by the quality
            xm_tag = reads.get_tag("XM")
            cigarstring = reads.cigarstring
            
            if xm_tag.count(".") == len(xm_tag):
                continue
            
            ref_seq, xm_tag = handling_cigar(ref_seq, xm_tag, cigarstring)
            
            # Extract all cytosines
            methylatable_sites = [idx for idx, r in enumerate(ref_seq) if ref_seq[idx:idx+2] == "CG"]

            if len(methylatable_sites) == 0:
                continue 
            
            # Disregard CHH context (h and H)
            for idx in methylatable_sites:
                methyl_state = None
                methyl_idx = -1
                # Taking the complemented cytosine's methylation for the reversed reads
                if idx >= len(xm_tag):
                    methyl_state = "."
                    methyl_idx=idx
                elif (not reads.is_reverse and single_end) or (reads.is_reverse != reads.is_read1 and not single_end):
                    methyl_state = xm_tag[idx]
                    methyl_idx = idx 
                elif idx+1 < len(xm_tag): 
                    methyl_state = xm_tag[idx+1]
                    methyl_idx = idx+1
                else:
                    methyl_state = "."
                    methyl_idx = idx+1
                    
                if methyl_state is not None:
                    if methyl_state in (".", "D"): # Missing or occured by deletion
                        methyl_state = "C"
                        
                    elif (methyl_state in ["x", "h", "X", "H"]):
                        if (xm_tag[idx] in ["D"]) or (xm_tag[idx+1] in ["D"]):
                            methyl_state="C"
                        else:
                            raise ValueError("Error in the conversion: %s %s %s %s %s"%(xm_tag[idx],   
                                             methyl_state, "Reverse" if reads.is_reverse else "Forward",
                                             ref_seq, xm_tag))
                    ref_seq = ref_seq[:idx] + methyl_state + ref_seq[idx+1:]

            # Remove inserted and soft clip bases 
            ref_seq = ref_seq.replace("I", "")
            ref_seq = ref_seq.replace("S", "")


            # K-mers
            s, m = kmers(ref_seq, k=3)

            if len(s) != len(m):
                raise ValueError("DNA and methylation sequences have different length (%d and %d)"%(len(s), len(m)))

            # Add processed results as a tag
            reads.setTag("RF", value=" ".join(s), replace=True)
            reads.setTag("ME", value="".join(m), replace=True)

            read_tags = {t:v for t, v in reads.get_tags()}
            reads = reads.to_dict()
            reads.update(read_tags)
            processed_reads.append(reads)
        
        return pd.DataFrame(processed_reads).drop(columns=["tags"])
        
    @globalize
    def _get_methylseq(dmr, bam_file_path, k, single_end):
        '''
            Return a dictionary of DNA seq, cell type and methylation seq
        '''
        aln = pysam.AlignmentFile(bam_file_path, "rb")

        # Decide whether the file is aligned in single-end mode or paired-end mode
        if not single_end:
            single_end=True
            for program_line in aln.header.as_dict()["PG"]:
                if "bismark" in program_line["CL"]:
                    parsed_line = program_line["CL"].split(" ")
                    for p in parsed_line:
                        if "-2" in p:
                            single_end = False

        processed_reads = _reads_overlapping(aln, dmr["chr"], int(dmr["start"]), int(dmr["end"]), single_end)

        processed_reads["ctype"] = dmr["ctype"]
        processed_reads["dmr_id"] = dmr["dmr_id"]

        return processed_reads
    
    with mp.Pool(ncores) as pool:
        # Convert sequences to K-mers
        seqs = pool.map(partial(_get_methylseq, 
                                bam_file_path = bam_file_path, 
                                k=k, single_end=single_end), 
        dmrs.to_dict("records"))
    
    return pd.concat(seqs, ignore_index=True)



def finetune_data_generate(args):
    # Setup random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup output files
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    fp_train_seq = os.path.join(args.output_dir, "train_seq.txt")
    fp_test_seq = os.path.join(args.output_dir, "test_seq.txt")
    fp_dmr = os.path.join(args.output_dir, "dmrs.csv")

    # Reference genome
    record_iter = SeqIO.parse(args.f_ref, "fasta")
    
    # Save the reference genome into a dictionary with chr as a key value
    #global dict_ref # NEED TO FIX IT AT SOME POINT 
    dict_ref = dict()
    for r in record_iter:
        dict_ref[r.id] = str(r.seq.upper())
    del record_iter
    
    # Load DMRs into a data frame
    dmrs = pd.read_csv(args.f_dmr, sep="\t")
    print(dmrs.head())

    # Remove chrX, chrY, chrM in DMRs
    regex_expr = "chr\d+"
    dmrs = dmrs[dmrs["chr"].str.contains(regex_expr, regex=True)]

    # Select top-dmrs based on diff.Methy
    if args.n_dmrs > 0:
        #dmrs = dmrs[:args.n_dmrs]
        dmrs = [dmrs[dmrs["ctype"]==c][:args.n_dmrs] for c in dmrs["ctype"].unique()]
        dmrs = pd.concat(dmrs)

    
    # Newly assign dmr label from 0
    dmrs["dmr_id"] = range(len(dmrs))
    dmrs.to_csv(fp_dmr, sep="\t", index=False)
    print(dmrs)

    print(f"Number of DMRs : {len(dmrs)}")

    
    # Collect train data (single-cell samples)
    train_sc_samples = []
    with open(args.sc_dataset, "r") as fp_sc_dataset:
        sc_files = fp_sc_dataset.readlines()
    
    # Collect reads from the .bam files
    print("Number of cpu : ", mp.cpu_count())
    df_reads = list()
    random.shuffle(sc_files)

    for f_sc in sc_files:
        f_sc = f_sc.strip().split("\t")
        f_sc_ctype = f_sc[1]
        f_sc = f_sc[0]
        print("%s processing (%s)..."%(f_sc, f_sc_ctype))

        extracted_reads = read_extract(f_sc, dict_ref, k=3, dmrs=dmrs, ncores=args.n_cores)
        
        # cell type for the single-cell data
        extracted_reads["read_ctype"] = f_sc_ctype
        df_reads.append(extracted_reads)

    # Integrate all reads and shuffle
    df_reads = pd.concat(df_reads, ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Split the data into train and valid/test
    n_train = int(df_reads.shape[0]*args.split_ratio)
    print("Size - train %d seqs , valid %d seqs "%(n_train, df_reads.shape[0] - n_train))

    # Write train & test files
    df_reads.loc[:n_train,:].to_csv(fp_train_seq, sep="\t", header=None, index=None)
    df_reads.loc[n_train:,:].to_csv(fp_test_seq, sep="\t", header=None, index=None)


if __name__ == "__main__":
    args = arg_parser()
    finetune_data_generate(args)
