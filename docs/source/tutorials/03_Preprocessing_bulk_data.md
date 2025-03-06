# Preprocessing for bulk data 

The bulk sample you want to deconvolute using _MethylBERT_ also needs to be preprocessed using `finetune_data_generate` function. 


```python
from methylbert.data import finetune_data_generate as fdg

f_bam = "../test/data/bulk.bam"
f_dmr = "../test/data/dmrs.csv"
f_ref = "../../../genome/hg19.fa"
out_dir = "tmp/"

fdg.finetune_data_generate(
    input_file = f_bam,
    f_dmr = f_dmr,
    f_ref = f_ref,
    output_dir=out_dir,
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
    Fine-tuning data generated:                                        name flag ref_name    ref_pos  \
    0    SRR10166000.9089788_9089788_length=151  147    chr10  131767360   
    1  SRR10165998.65829390_65829390_length=150  163     chr4   20254248   
    2  SRR10165467.85837758_85837758_length=151   99     chr4    1401206   
    3  SRR10165995.16747267_16747267_length=149   83     chr2  176945656   
    4  SRR10165995.46034072_46034072_length=151   99     chr4   20253524   
    
      map_quality cigar next_ref_name next_ref_pos length  \
    0          42  151M             =    131767187   -324   
    1          23  151M             =     20254343    244   
    2          40  151M             =      1401285    227   
    3          40  149M             =    176945572   -233   
    4          40  151M             =     20253771    398   
    
                                                     seq  ...  NM  \
    0  GTGGAGTGTCGTTGCGTAGTCGGGAGTCGGGAGTAGAATAGTTTGG...  ...  49   
    1  GGGGATTCTACCTTTACCATCAAATATCTACCGCGAAACTACGACT...  ...  35   
    2  AAAATGAGAGATTGTTTGTTTTTTTTAATTTGTTTTTAAAAGGGGG...  ...  40   
    3  AAATAACTTAATCTACTTCTCTCCGACCAAACCCAACCCCAAATAC...  ...  35   
    4  TCGGATTTGGTGTTATTTATTTGGGAAGCGTCCGGACGGCGGAGCT...  ...   2   
    
                                                      XM  XR  \
    0  ........xZ.x..Z.x..xZ.....xZ.....x....x..hx......  GA   
    1  H..............h......xh.h...x..Z.Zx.h..x.Zx.....  GA   
    2  ...........x..h....hhh.h....hxz.hhhhh............  CT   
    3  x...hh...hh.............Z.....h.........z.h......  CT   
    4  .Z...h......................Z.hXZ...Z..Z....H....  CT   
    
                            PG                                    RG  \
    0  MarkDuplicates-287B47C6  diffuse_large_B_cell_lymphoma_test_8   
    1  MarkDuplicates-3DAAB091  diffuse_large_B_cell_lymphoma_test_8   
    2  MarkDuplicates-36E4BA78                Bcell_noncancer_test_8   
    3  MarkDuplicates-74536757  diffuse_large_B_cell_lymphoma_test_8   
    4  MarkDuplicates-74536757  diffuse_large_B_cell_lymphoma_test_8   
    
                                                 dna_seq  \
    0  GTG TGG GGA GAG AGT GTG TGC GCC CCG CGC GCT CT...   
    1  GTT TTT TTC TCT CTT TTC TCT CTA TAC ACC CCT CT...   
    2  AAA AAA AAT ATG TGA GAG AGA GAG AGA GAC ACT CT...   
    3  GAA AAT ATG TGG GGC GCT CTT TTG TGG GGT GTC TC...   
    4  TCG CGG GGA GAC ACT CTT TTG TGG GGT GTG TGT GT...   
    
                                              methyl_seq dmr_ctype dmr_label ctype  
    0  2222222212222122222122222212222222222222222222...         T         5    NA  
    1  2222222222222222222222222222221212222222122222...         T        19    NA  
    2  2222222222222222222222222222202222222222222222...         T         2    NA  
    3  2222222222222222222222122222222222222202222222...         T        12    NA  
    4  1222222222222222222222222221222122212212222222...         T        19    NA  
    
    [5 rows x 23 columns]


This process generates a new file `data.csv` where the preprocessed bulk data is contained. 


```python
ls tmp/
```

    data.csv  dmrs.csv  test_seq.csv  train_seq.csv


Since the cell-type information is not given with the bulk sample, `ctype` column only contains `NaN` value. 


```python
import pandas as pd
pd.read_csv("tmp/data.csv", sep="\t").head()
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
      <th>NM</th>
      <th>XM</th>
      <th>XR</th>
      <th>PG</th>
      <th>RG</th>
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
      <td>SRR10166000.9089788_9089788_length=151</td>
      <td>147</td>
      <td>chr10</td>
      <td>131767360</td>
      <td>42</td>
      <td>151M</td>
      <td>=</td>
      <td>131767187</td>
      <td>-324</td>
      <td>GTGGAGTGTCGTTGCGTAGTCGGGAGTCGGGAGTAGAATAGTTTGG...</td>
      <td>...</td>
      <td>49</td>
      <td>........xZ.x..Z.x..xZ.....xZ.....x....x..hx......</td>
      <td>GA</td>
      <td>MarkDuplicates-287B47C6</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>GTG TGG GGA GAG AGT GTG TGC GCC CCG CGC GCT CT...</td>
      <td>2222222212222122222122222212222222222222222222...</td>
      <td>T</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SRR10165998.65829390_65829390_length=150</td>
      <td>163</td>
      <td>chr4</td>
      <td>20254248</td>
      <td>23</td>
      <td>151M</td>
      <td>=</td>
      <td>20254343</td>
      <td>244</td>
      <td>GGGGATTCTACCTTTACCATCAAATATCTACCGCGAAACTACGACT...</td>
      <td>...</td>
      <td>35</td>
      <td>H..............h......xh.h...x..Z.Zx.h..x.Zx.....</td>
      <td>GA</td>
      <td>MarkDuplicates-3DAAB091</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>GTT TTT TTC TCT CTT TTC TCT CTA TAC ACC CCT CT...</td>
      <td>2222222222222222222222222222221212222222122222...</td>
      <td>T</td>
      <td>19</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SRR10165467.85837758_85837758_length=151</td>
      <td>99</td>
      <td>chr4</td>
      <td>1401206</td>
      <td>40</td>
      <td>151M</td>
      <td>=</td>
      <td>1401285</td>
      <td>227</td>
      <td>AAAATGAGAGATTGTTTGTTTTTTTTAATTTGTTTTTAAAAGGGGG...</td>
      <td>...</td>
      <td>40</td>
      <td>...........x..h....hhh.h....hxz.hhhhh............</td>
      <td>CT</td>
      <td>MarkDuplicates-36E4BA78</td>
      <td>Bcell_noncancer_test_8</td>
      <td>AAA AAA AAT ATG TGA GAG AGA GAG AGA GAC ACT CT...</td>
      <td>2222222222222222222222222222202222222222222222...</td>
      <td>T</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SRR10165995.16747267_16747267_length=149</td>
      <td>83</td>
      <td>chr2</td>
      <td>176945656</td>
      <td>40</td>
      <td>149M</td>
      <td>=</td>
      <td>176945572</td>
      <td>-233</td>
      <td>AAATAACTTAATCTACTTCTCTCCGACCAAACCCAACCCCAAATAC...</td>
      <td>...</td>
      <td>35</td>
      <td>x...hh...hh.............Z.....h.........z.h......</td>
      <td>CT</td>
      <td>MarkDuplicates-74536757</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>GAA AAT ATG TGG GGC GCT CTT TTG TGG GGT GTC TC...</td>
      <td>2222222222222222222222122222222222222202222222...</td>
      <td>T</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SRR10165995.46034072_46034072_length=151</td>
      <td>99</td>
      <td>chr4</td>
      <td>20253524</td>
      <td>40</td>
      <td>151M</td>
      <td>=</td>
      <td>20253771</td>
      <td>398</td>
      <td>TCGGATTTGGTGTTATTTATTTGGGAAGCGTCCGGACGGCGGAGCT...</td>
      <td>...</td>
      <td>2</td>
      <td>.Z...h......................Z.hXZ...Z..Z....H....</td>
      <td>CT</td>
      <td>MarkDuplicates-74536757</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>TCG CGG GGA GAC ACT CTT TTG TGG GGT GTG TGT GT...</td>
      <td>1222222222222222222222222221222122212212222222...</td>
      <td>T</td>
      <td>19</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>


