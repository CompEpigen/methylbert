import numpy as np
from scipy.stats import binom
from tqdm import tqdm
import os, argparse
from Bio import SeqIO
import random 
import pandas as pd

def arg_parser():
    parser = argparse.ArgumentParser()
    # Required hyperparameters 
    parser.add_argument("-d", "--f_dmr", required=True, type=str, help="tab-separated .csv file, DMRs should be given with mean methylation level of each cell type, chr, start and end")
    #parser.add_argument("-i", "--f_input", required=False, type=str, help=".csv file, DMRs should be given with mean methylation level of each cell type, chr, start and end")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="a directory where all generated results will be saved")
    parser.add_argument("-r", "--f_ref", required=True, type=str, help=".fasta file for the reference genome")

    #Hyperparameters for read-level methylome simulation 
    parser.add_argument("-nm", "--n_mers", type=int, default=3, help="n-mers")
    parser.add_argument("-nr", "--n_reads", type=int, default=120, help="Read coverage to simulate in each DMR")
    parser.add_argument("-l", "--len_read", type=int, default=150, help="Read length to simulate")
    parser.add_argument("--seed", type=int, default=950410, help="seed number")

    # Hyperparameters for pseudo-bulks 
    parser.add_argument("--bulk", type=bool, default=True, help="Whether you want to generate pseudo-bulks or the entire dataset")
    parser.add_argument("-nb", "--n_bulks", type=int, default=None, help="Number of bulks, Applicable only when --bulk is True")
    parser.add_argument("-s", "--std", type=float, default=0.0, help="Standard deviation to sample local proportions. The larger value is given, the more varied local proportions are sampled from a Gaussian distribution centred at the global proportion")
    parser.add_argument("-t", "--f_gt", type=str, help="tab-separated .csv file, each column should be a proportion for each cell-type. bulk name can be included as dataframe index/row name, only used when n_bulks is not given")
    

    return parser.parse_args()


def simulation(args):
    # Setup random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup output files
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Save hyperparameters 
    pd.DataFrame({"param":vars(args).keys(),
                  "values":vars(args).values()}).to_csv(args.output_dir+"/parameters.txt", sep="\t")

    # Reference genome
    record_iter = SeqIO.parse(args.f_ref, "fasta")

    dict_ref = dict()
    for r in record_iter:
        dict_ref[r.id] = str(r.seq.upper())
    del record_iter

    # Load DMRs
    df_dmr = pd.read_csv(args.f_dmr, sep="\t")
    if "dmr_id" in df_dmr.keys():
        df_dmr = df_dmr.set_index("dmr_id")
    print(df_dmr)


    reads = list()
    for i in tqdm(range(df_dmr.shape[0])):
        dmr = df_dmr.loc[i]
        refseq = dict_ref[dmr["chr"]][dmr["start"]:dmr["end"]]

        # Sample start positions
        r_starts = np.random.randint(len(refseq)-args.len_read, size=args.n_reads).tolist()

        for r_idx, start in enumerate(r_starts):
            # simulate same number of reads for all cell types 
            r_ctype = "N" if r_idx < int(args.n_reads/2) else "T"

            # read position
            r_start = dmr["start"] + start
            r_end = r_start + args.len_read + 2
            r_seq = dict_ref[dmr["chr"]][r_start:r_end]
            
            # Set probability based on mean methylation level for each cell type
            p=dmr["meanMethy1"] if r_ctype=="N" else dmr["meanMethy2"]
            p = p>0.5

            # Methylation pattern
            cpg_loci = [j for j in range(len(r_seq)-2) if r_seq[j:j+2] == "CG"]

            methyl_patterns = binom.rvs(n=1, p=p, 
                                        size=len(cpg_loci)).tolist()
            r_methyl = list(np.ones(len(r_seq), dtype=int)*2)
            for ri, rm in zip(cpg_loci, methyl_patterns):
                r_methyl[ri] = rm
            r_methyl = "".join([str(r) for r in r_methyl[1:-1]])

            # 3-mers
            r_seq = [r_seq[i:i+3] for i in range(len(r_seq)-2)]
            r_seq = " ".join(r_seq)
            reads.append(pd.DataFrame({"dna_seq": [r_seq], 
                                 "methyl_seq": [r_methyl],
                                 "dmr_ctype":[ dmr["ctype"]],
                                 "ctype": [r_ctype],
                                 "dmr_label":[i],
                                 "ref_name":[dmr["chr"]],
                                 "ref_pos":[r_start],
                                 "XM": ["z"]}))

    reads = pd.concat(reads)
    if not args.bulk:
        # Entire dataset simulation
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        reads.to_csv(f"{args.output_dir}/data.txt", sep="\t", index=False)
    else:
        # pseudo_bulk simulation
        ctypes = reads["ctype"].unique()
        
        if not args.n_bulks:
            df_gt = pd.read_csv(args.f_gt, sep="\t")
            print(df_gt)
            pri_ratios = df_gt["T"].tolist()
            bulk_names = df_gt["bulk"] if "bulk" in df_gt.keys() else df_gt.index
            bulk_names = bulk_names.tolist()
            n_bulks = df_gt.shape[0]
        else:
            pri_ratios = np.random.uniform(size=args.n_bulks)
            n_bulks = args.n_bulks
            bulk_names = ["bulk_%d"%i for i in range(n_bulks)]

        post_ratios = list()

        for bulk_idx, bulk_name in tqdm(enumerate(bulk_names)):
            # proportion for each dmr
            local_proportions = np.random.normal(loc=pri_ratios[bulk_idx], scale=args.std, size=df_dmr.shape[0])
            bulk_reads = list()
            
            for dmr_idx in range(df_dmr.shape[0]):
                dmr_reads = reads.loc[reads["dmr_label"]==dmr_idx,].copy().reset_index().drop(columns=["index"])
                local_prop = local_proportions[dmr_idx]
                for ctype in ctypes:
                    ctype_ratio = local_prop if ctype=="T" else 1-local_prop
                    sampled_idces = random.sample(dmr_reads.loc[dmr_reads["ctype"]==ctype].index.tolist(), int((args.n_reads/len(ctypes)) * ctype_ratio))
                    bulk_reads.append(dmr_reads.loc[sampled_idces, ])
            bulk_reads = pd.concat(bulk_reads)
            bulk_reads.to_csv(f"{args.output_dir}/{bulk_name}.txt", sep="\t", index=False)
            post_ratios.append((bulk_reads["ctype"]=="T").sum()/bulk_reads.shape[0])

        gt_ratios = pd.DataFrame({"bulk":bulk_names,
                                "T": post_ratios,
                                "N": [1-r for r in post_ratios]})
        gt_ratios.to_csv(f"{args.output_dir}/gt_ratio.csv", sep="\t", index=False)


if __name__=="__main__":
    args = arg_parser()
    simulation(args)