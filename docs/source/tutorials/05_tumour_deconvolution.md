# Tumour deconvolution with the fine-tuned `MethylBERT` model 

### Load the bulk data and the fine-tuned model

Please load your preprocessed bulk data following the [tutorial](https://github.com/hanyangii/methylbert/blob/main/tutorials/03_Preprocessing_bulk_data.ipynb) into the `MethylBertFinetuneDataset` and `DataLoader`.


```python
from methylbert.utils import set_seed

set_seed(42)
seq_len=150
n_mers=3
batch_size=5
num_workers=20
output_path="tmp/deconvolution/"
```


```python
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from torch.utils.data import DataLoader

tokenizer = MethylVocab(k=n_mers)
dataset = MethylBertFinetuneDataset("tmp/data.csv", 
                                    tokenizer, 
                                    seq_len=seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

```

    Building Vocab
    Total number of sequences :  3070
    # of reads in each label:  [3070.]


### Load the fine-tuned `MethylBERT` model

`load` function in `MethylBertFinetuneTrainer` automatically detects `config.json`, `pytorch_model.bin`, `dmr_encoder.pickle` and `read_classification_model.pickle` files in the given directory and load the fine-tuned model. 


```python
from methylbert.trainer import MethylBertFinetuneTrainer

restore_dir = "tmp/fine_tune/"
trainer = MethylBertFinetuneTrainer(len(tokenizer), 
                                    train_dataloader=data_loader, 
                                    test_dataloader=data_loader,
                                    )
trainer.load(restore_dir) # Load the fine-tuned MethylBERT model
```

    No detected GPU device. Load the model on CPU
    The model is loaded on CPU
    Restore the pretrained model tmp/fine_tune/
    Restore DMR encoder from tmp/fine_tune/dmr_encoder.pickle
    Restore read classification FCN model from tmp/fine_tune/read_classification_model.pickle
    Total Parameters: 32754130


### Deconvolution
For the deconvolution, the training data as a `pandas.DataFrame` object is required for the maginal probability of cell types. 


```python
import pandas as pd
from methylbert.deconvolute import deconvolute

deconvolute(trainer = trainer,
            tokenizer = tokenizer,                                    
            data_loader = data_loader,
            output_path = output_path,
            df_train = pd.read_csv("tmp/train_seq.csv", sep="\t"))
```

      0%|          | 0/614 [00:00<?, ?it/s]/omics/groups/OE0219/internal/Yunhee/anaconda3/envs/dnabert/lib/python3.6/site-packages/torch/autocast_mode.py:156: UserWarning: In CPU autocast, but the target dtype is not supported. Disabling autocast.
    CPU Autocast only supports dtype of torch.bfloat16 currently.
      warnings.warn(error_message)
    100%|██████████| 614/614 [01:03<00:00,  9.73it/s]


    Margins :  [0.5114678899082569, 0.48853211009174313]
                                           name flag ref_name    ref_pos  \
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
    
                                                     seq  ...  \
    0  GTGGAGTGTCGTTGCGTAGTCGGGAGTCGGGAGTAGAATAGTTTGG...  ...   
    1  GGGGATTCTACCTTTACCATCAAATATCTACCGCGAAACTACGACT...  ...   
    2  AAAATGAGAGATTGTTTGTTTTTTTTAATTTGTTTTTAAAAGGGGG...  ...   
    3  AAATAACTTAATCTACTTCTCTCCGACCAAACCCAACCCCAAATAC...  ...   
    4  TCGGATTTGGTGTTATTTATTTGGGAAGCGTCCGGACGGCGGAGCT...  ...   
    
                                         RG  \
    0  diffuse_large_B_cell_lymphoma_test_8   
    1  diffuse_large_B_cell_lymphoma_test_8   
    2                Bcell_noncancer_test_8   
    3  diffuse_large_B_cell_lymphoma_test_8   
    4  diffuse_large_B_cell_lymphoma_test_8   
    
                                                 dna_seq  \
    0  GTGGAGTGCCGCTGCGCAGCCGGGAGCCGGGAGCAGAACAGCCTGG...   
    1  GTTTCTTCTACCTTTGCCATCAGGTGTCTGCCGCGGAGCTGCGGCT...   
    2  AAAATGAGAGACTGCTTGTCCCTCTTAACCCGCCCCCAAAAGGGGG...   
    3  GAATGGCTTGGTCTACTTCTCTCCGACCAAGCCCAACCCCGAGTAC...   
    4  TCGGACTTGGTGTTATTTATTTGGGAAGCGCCCGGACGGCGGAGCT...   
    
                                              methyl_seq dmr_ctype dmr_label  \
    0  2222222221222212222212222221222222222222222222...         T         5   
    1  2222222222222222222222222222222121222222212222...         T        19   
    2  2222222222222222222222222222220222222222222222...         T         2   
    3  2222222222222222222222212222222222222220222222...         T        12   
    4  2122222222222222222222222222122212221221222222...         T        19   
    
      ctype pred n_cpg       P_N       P_T  
    0    NA    0    14  0.502180  0.497820  
    1    NA    0    10  0.515485  0.484515  
    2    NA    0     5  0.512614  0.487386  
    3    NA    0    10  0.505069  0.494931  
    4    NA    0    19  0.513832  0.486168  
    
    [5 rows x 27 columns]


    100%|██████████| 10000/10000 [00:02<00:00, 3849.83it/s]


`deconvolute` function creates three files in the given directiory:
1. `decconvolution.csv` : tumour deconvolution result
2. `FI.csv` : the Fisher information value
3. `res.csv` : read classification result (the classification result for each read is given in `pred` column)


```python
import os
pd.read_csv(os.path.join(output_path, "deconvolution.csv"), sep="\t").head()
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
      <th>cell_type</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>T</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv(os.path.join(output_path, "FI.csv"), sep="\t").head()
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
      <th>fi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.243251</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv(os.path.join(output_path, "res.csv"), sep="\t").head()
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
      <th>XM</th>
      <th>XR</th>
      <th>PG</th>
      <th>RG</th>
      <th>dna_seq</th>
      <th>methyl_seq</th>
      <th>dmr_ctype</th>
      <th>dmr_label</th>
      <th>ctype</th>
      <th>pred</th>
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
      <td>........xZ.x..Z.x..xZ.....xZ.....x....x..hx......</td>
      <td>GA</td>
      <td>MarkDuplicates-287B47C6</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>GTGGAGTGCCGCTGCGCAGCCGGGAGCCGGGAGCAGAACAGCCTGG...</td>
      <td>2222222221222212222212222221222222222222222222...</td>
      <td>T</td>
      <td>5</td>
      <td>NaN</td>
      <td>0</td>
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
      <td>H..............h......xh.h...x..Z.Zx.h..x.Zx.....</td>
      <td>GA</td>
      <td>MarkDuplicates-3DAAB091</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>GTTTCTTCTACCTTTGCCATCAGGTGTCTGCCGCGGAGCTGCGGCT...</td>
      <td>2222222222222222222222222222222121222222212222...</td>
      <td>T</td>
      <td>19</td>
      <td>NaN</td>
      <td>0</td>
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
      <td>...........x..h....hhh.h....hxz.hhhhh............</td>
      <td>CT</td>
      <td>MarkDuplicates-36E4BA78</td>
      <td>Bcell_noncancer_test_8</td>
      <td>AAAATGAGAGACTGCTTGTCCCTCTTAACCCGCCCCCAAAAGGGGG...</td>
      <td>2222222222222222222222222222220222222222222222...</td>
      <td>T</td>
      <td>2</td>
      <td>NaN</td>
      <td>0</td>
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
      <td>x...hh...hh.............Z.....h.........z.h......</td>
      <td>CT</td>
      <td>MarkDuplicates-74536757</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>GAATGGCTTGGTCTACTTCTCTCCGACCAAGCCCAACCCCGAGTAC...</td>
      <td>2222222222222222222222212222222222222220222222...</td>
      <td>T</td>
      <td>12</td>
      <td>NaN</td>
      <td>0</td>
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
      <td>.Z...h......................Z.hXZ...Z..Z....H....</td>
      <td>CT</td>
      <td>MarkDuplicates-74536757</td>
      <td>diffuse_large_B_cell_lymphoma_test_8</td>
      <td>TCGGACTTGGTGTTATTTATTTGGGAAGCGCCCGGACGGCGGAGCT...</td>
      <td>2122222222222222222222222222122212221221222222...</td>
      <td>T</td>
      <td>19</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Deconvolution with the estimate adjustment

_MethylBERT_ supports the tumour purity estimation adjustment considering the different distribution of tumour-derived reads in DMRs. 

You can turn `adjustment` option on for this. 


```python
from methylbert.deconvolute import deconvolute
import pandas as pd

# Read classification
deconvolute(trainer = trainer,
            tokenizer = tokenizer,                                    
            data_loader = data_loader,
            output_path = output_path,
            df_train = pd.read_csv("tmp/train_seq.csv", sep="\t"),
           adjustment = True) # tumour purity estimation adjustment
```

      0%|          | 0/614 [00:00<?, ?it/s]/omics/groups/OE0219/internal/Yunhee/anaconda3/envs/dnabert/lib/python3.6/site-packages/torch/autocast_mode.py:156: UserWarning: In CPU autocast, but the target dtype is not supported. Disabling autocast.
    CPU Autocast only supports dtype of torch.bfloat16 currently.
      warnings.warn(error_message)
    100%|██████████| 614/614 [00:56<00:00, 10.86it/s]


    Margins :  [0.5114678899082569, 0.48853211009174313]
                                           name flag ref_name    ref_pos  \
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
    
                                                     seq  ...  \
    0  GTGGAGTGTCGTTGCGTAGTCGGGAGTCGGGAGTAGAATAGTTTGG...  ...   
    1  GGGGATTCTACCTTTACCATCAAATATCTACCGCGAAACTACGACT...  ...   
    2  AAAATGAGAGATTGTTTGTTTTTTTTAATTTGTTTTTAAAAGGGGG...  ...   
    3  AAATAACTTAATCTACTTCTCTCCGACCAAACCCAACCCCAAATAC...  ...   
    4  TCGGATTTGGTGTTATTTATTTGGGAAGCGTCCGGACGGCGGAGCT...  ...   
    
                                         RG  \
    0  diffuse_large_B_cell_lymphoma_test_8   
    1  diffuse_large_B_cell_lymphoma_test_8   
    2                Bcell_noncancer_test_8   
    3  diffuse_large_B_cell_lymphoma_test_8   
    4  diffuse_large_B_cell_lymphoma_test_8   
    
                                                 dna_seq  \
    0  GTGGAGTGCCGCTGCGCAGCCGGGAGCCGGGAGCAGAACAGCCTGG...   
    1  GTTTCTTCTACCTTTGCCATCAGGTGTCTGCCGCGGAGCTGCGGCT...   
    2  AAAATGAGAGACTGCTTGTCCCTCTTAACCCGCCCCCAAAAGGGGG...   
    3  GAATGGCTTGGTCTACTTCTCTCCGACCAAGCCCAACCCCGAGTAC...   
    4  TCGGACTTGGTGTTATTTATTTGGGAAGCGCCCGGACGGCGGAGCT...   
    
                                              methyl_seq dmr_ctype dmr_label  \
    0  2222222221222212222212222221222222222222222222...         T         5   
    1  2222222222222222222222222222222121222222212222...         T        19   
    2  2222222222222222222222222222220222222222222222...         T         2   
    3  2222222222222222222222212222222222222220222222...         T        12   
    4  2122222222222222222222222222122212221221222222...         T        19   
    
      ctype pred n_cpg       P_N       P_T  
    0    NA    0    14  0.502180  0.497820  
    1    NA    0    10  0.515485  0.484515  
    2    NA    0     5  0.512614  0.487386  
    3    NA    0    10  0.505069  0.494931  
    4    NA    0    19  0.513832  0.486168  
    
    [5 rows x 27 columns]


     10%|█         | 1/10 [00:00<00:00, 11.06it/s]


When the adjustment is applied, the Fisher information is calculated in each DMR. 


```python
pd.read_csv(os.path.join(output_path, "FI.csv"), sep="\t").head()
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
      <th>dmr_label</th>
      <th>fi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


