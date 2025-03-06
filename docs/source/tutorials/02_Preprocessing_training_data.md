# Preprocessing for _MethylBERT_ fine-tuning training data

_MethylBERT_ fine-tuning needs DNA methylation data from tumour (T) and normal (N) samples as training data. You can give a list of sample files with annotations in a tab-deliminated file. 


```python
cat ../test/data/bam_list.txt
```

    ../test/data/T_sample.bam	T
    ../test/data/N_sample.bam	N



As described in the [data preparation](https://github.com/hanyangii/methylbert/blob/main/tutorials/01_Data_Preparation.md) tutorial, DMRs and the reference genome should be prepared in the required format. 

_MethylBERT_ provides `finetune_data_generate` function to preprocess the given tumour and normal data.


```python
from methylbert.data import finetune_data_generate as fdg

f_bam_file_list = "../test/data/bam_list.txt"
f_dmr = "../test/data/dmrs.csv"
f_ref = "../../../genome/hg19.fa"
out_dir = "tmp/"

fdg.finetune_data_generate(
    sc_dataset = f_bam_file_list,
    f_dmr = f_dmr,
    f_ref = f_ref,
    output_dir=out_dir,
    split_ratio = 0.8, # Split ratio to make training and validation data
    n_mers=3, # 3-mer DNA sequences 
    n_cores=20
)
```

    DMRs sorted by areaStat
         chr      start          end  length  nCG  meanMethy1  meanMethy2  \
    1  chr10  134597480  134602875.0    5396  670    0.861029    0.140400   
    0   chr7    1268957    1277884.0    8928  753    0.793278    0.129747   
    2   chr4    1395812    1402597.0    6786  663    0.831162    0.185272   
    5  chr16   54962053   54967980.0    5928  546    0.783631    0.096095   
    9  chr18   76736906   76741580.0    4675  510    0.829475    0.104403   
    
       diff.Methy     areaStat  abs_areaStat  abs_diff.Methy ctype  dmr_id  
    1    0.720629  6144.089331   6144.089331        0.720629     T       0  
    0    0.663531  5722.091790   5722.091790        0.663531     T       1  
    2    0.645891  4941.410089   4941.410089        0.645891     T       2  
    5    0.687536  4714.551799   4714.551799        0.687536     T       3  
    9    0.725072  4684.608381   4684.608381        0.725072     T       4  
    Number of DMRs to extract sequence reads: 20
    ../test/data/T_sample.bam processing (T)...
    ../test/data/N_sample.bam processing (N)...
    Fine-tuning data generated:                                        name flag ref_name    ref_pos  \
    0  SRR10165994.69237235_69237235_length=151  163     chr7  156797584   
    1  SRR10165464.24148712_24148712_length=151   99    chr10  131770809   
    2  SRR10165994.26664131_26664131_length=151  163    chr10  131766813   
    3  SRR10165464.61126854_61126854_length=150  147    chr10  131769430   
    4  SRR10165994.14375046_14375046_length=150   83     chr5    1884762   
    
      map_quality cigar next_ref_name next_ref_pos length  \
    0          24  151M             =    156797848    344   
    1          42  151M             =    131770846    188   
    2          42  149M             =    131767027    365   
    3          24  151M             =    131769291   -290   
    4          40  149M             =      1884665   -246   
    
                                                     seq  ...              PG  XG  \
    0  GGGGAAGAAAAAAAACTAAATAATAATTTAACATACATACGTAAAC...  ...  MarkDuplicates  GA   
    1  GGTTTGTCGGGAAGGTTGTGAGTAGAGGCCAACGGAGGTCTCCCAG...  ...  MarkDuplicates  CT   
    2  GGGGGCCTCTAAAAACGCTCCAAATTCGTCTTACGCCACGAAATCA...  ...  MarkDuplicates  GA   
    3  GTTGGGTGGTAAGGTGGTTTAGGGTATAGTTAGGGGTTATGTAGAA...  ...  MarkDuplicates  CT   
    4  AATAATTATTTCTAAATTCTATATTAATTTCGCGACAAACCGCGTT...  ...  MarkDuplicates  GA   
    
       NM                                                 XM  XR  \
    0  23  HHH.z..hhh.h..h...............h...h.....Z..h.....  GA   
    1  11  ..hxz.xZ.......xz.z...x.....HH..Z.....hH.HHX.....  CT   
    2  38  .Z.ZX.....x.h.h.Z......h...Z....h.Z....Zxhh......  GA   
    3  31  .x....z..h....z..hhx....h....h......hh...........  GA   
    4  18  .......h.....x......x....h.....Z.Zx..xh..Z.Z.....  CT   
    
                                                 dna_seq  \
    0  GGG GGC GCG CGA GAT ATG TGG GGG GGA GAG AGA GA...   
    1  GGC GCC CCC CCG CGC GCC CCG CGG GGG GGA GAA AA...   
    2  CGC GCG CGG GGC GCC CCT CTC TCT CTG TGA GAG AG...   
    3  GCT CTG TGG GGG GGC GCG CGG GGC GCA CAA AAG AG...   
    4  AAT ATA TAA AAT ATT TTG TGT GTT TTT TTC TCT CT...   
    
                                              methyl_seq dmr_ctype dmr_label ctype  
    0  2202222222222222222222222222222222222212222212...         T        17     T  
    1  2220221222222220202222222222222122222222222222...         T         5     N  
    2  2122222222222212222222222122222212222122222221...         T         5     T  
    3  2222202222222022222222222222222222222222222222...         T         5     N  
    4  2222222222222222222222222222212122222221212222...         T         8     T  
    
    [5 rows x 22 columns]
    Size - train 3051 seqs , valid 763 seqs 


After the preprocessing, you get three different files:
1. dmrs.csv : Selected DMRs (when the number of DMRs is given) with `dmr_label` column
2. train_seq.csv : Preprocessed training data
3. test_seq.csv : Preprocessed evaluation data (20% of given data, due to the split_ratio=0.8)


```python
ls tmp/
```

    dmrs.csv  test_seq.csv  train_seq.csv


Each preprocessed data is a tab-deliminated .csv file where each column contains the individual field of given BAM/SAM file. Additionally `dmr_ctype`, `dmr_label` and `ctype` are given:
1. `dmr_ctype`: The specific cell type for each DMR
2. `dmr_label`: DMR label. This is used for the read classifier fully-connected network in _MethylBERT_
3. `ctype` : Cell-type of the read (indicated in the input file)


```python
import pandas as pd
pd.read_csv("tmp/test_seq.csv", sep='\t').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>flag</th>
      <th>ref_name</th>
      <th>ref_pos</th>
      <th>map_quality</th>
      <th>cigar</th>
      <th>next_ref_name</th>
      <th>next_ref_pos</th>
      <th>length</th>
      <th>seq</th>
      <th>...</th>
      <th>PG</th>
      <th>XG</th>
      <th>NM</th>
      <th>XM</th>
      <th>XR</th>
      <th>dna_seq</th>
      <th>methyl_seq</th>
      <th>dmr_ctype</th>
      <th>dmr_label</th>
      <th>ctype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SRR10165464.6790597_6790597_length=151</td>
      <td>83</td>
      <td>chr2</td>
      <td>176943541</td>
      <td>40</td>
      <td>151M</td>
      <td>=</td>
      <td>176943475</td>
      <td>-217</td>
      <td>AATTAACAATTTTCATCATAATCTACACATTATTAACATCAAACTT...</td>
      <td>...</td>
      <td>MarkDuplicates</td>
      <td>GA</td>
      <td>37</td>
      <td>h...hh........z.........x..........h.............</td>
      <td>CT</td>
      <td>GAT ATT TTG TGG GGC GCA CAA AAT ATT TTT TTT TT...</td>
      <td>2222222222220222222222222222222222222222222222...</td>
      <td>T</td>
      <td>12</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SRR10165994.18752987_18752987_length=149</td>
      <td>163</td>
      <td>chr7</td>
      <td>157486616</td>
      <td>40</td>
      <td>149M</td>
      <td>=</td>
      <td>157486650</td>
      <td>183</td>
      <td>AGGCACGCGACCACCCTAAACCTCGAACAAAACTAAAAAAACGCAA...</td>
      <td>...</td>
      <td>MarkDuplicates</td>
      <td>GA</td>
      <td>51</td>
      <td>..Z...Z.Zx.......xhh....Zx...xhh...hhhhh..Z..x...</td>
      <td>GA</td>
      <td>CCG CGC GCA CAC ACG CGC GCG CGG GGC GCC CCA CA...</td>
      <td>1222121222222222222222122222222222222222122222...</td>
      <td>T</td>
      <td>11</td>
      <td>T</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SRR10165994.2935274_2935274_length=150</td>
      <td>83</td>
      <td>chr7</td>
      <td>1270222</td>
      <td>42</td>
      <td>150M</td>
      <td>=</td>
      <td>1269981</td>
      <td>-391</td>
      <td>ACGAACATTAAAACGCACGGAACCGCCGCGACGCGGACTCGCTCTT...</td>
      <td>...</td>
      <td>MarkDuplicates</td>
      <td>GA</td>
      <td>27</td>
      <td>h.Z.h....hhh..Z...ZX.h..Z..Z.Zx.Z.ZX....Z........</td>
      <td>CT</td>
      <td>GCG CGA GAG AGC GCA CAT ATT TTG TGG GGG GGA GA...</td>
      <td>1222222222221222122222122121221212222212222222...</td>
      <td>T</td>
      <td>1</td>
      <td>T</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SRR10165464.56090327_56090327_length=151</td>
      <td>163</td>
      <td>chr2</td>
      <td>176949511</td>
      <td>42</td>
      <td>149M</td>
      <td>=</td>
      <td>176949602</td>
      <td>242</td>
      <td>AGGATTTCTTACTACATAACCACAAAAATACATTAAACCCACACCT...</td>
      <td>...</td>
      <td>MarkDuplicates</td>
      <td>GA</td>
      <td>36</td>
      <td>h.Z.......h....z.hh..z.zx.hh.h....hhh...z.z......</td>
      <td>GA</td>
      <td>GCG CGC GCT CTT TTT TTC TCT CTT TTG TGC GCT CT...</td>
      <td>1222222222222022222020222222222222222202022222...</td>
      <td>T</td>
      <td>12</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SRR10165464.47924911_47924911_length=150</td>
      <td>147</td>
      <td>chr7</td>
      <td>1272480</td>
      <td>42</td>
      <td>151M</td>
      <td>=</td>
      <td>1272378</td>
      <td>-253</td>
      <td>AATTATTGGGAGTTTGATGTTGATAAGTAAAGTGTTGGAGTGTGGG...</td>
      <td>...</td>
      <td>MarkDuplicates</td>
      <td>CT</td>
      <td>31</td>
      <td>......z.....h...................z.xz......z......</td>
      <td>GA</td>
      <td>AAT ATT TTA TAT ATC TCG CGG GGG GGA GAG AGC GC...</td>
      <td>2222202222222222222222222222222022022222202220...</td>
      <td>T</td>
      <td>1</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>


