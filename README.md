# MethylBERT: A Transformer-based model for read-level DNA methylation pattern identification and tumour deconvolution
[![DOI](https://zenodo.org/badge/559284606.svg)](https://doi.org/10.5281/zenodo.14025051)

![methylbert_scheme](https://github.com/CompEpigen/methylbert/blob/main/img/introduction_methylbert.png)
_The figure was generated using [biorender](https://www.biorender.com/)_

BERT model to classify read-level DNA methylation data into tumour/normal and perform tumour deconvolution.
_MethylBERT_ is implemented using [pytorch](https://pytorch.org/) and [transformers](https://huggingface.co/docs/transformers/index) 🤗.

## Paper
_MethylBERT_ paper is now online on [__Nature Communications__](https://www.nature.com/articles/s41467-025-55920-z#article-info)!!

__MethylBERT enables read-level DNA methylation pattern identification and tumour deconvolution using a Transformer-based model__

Yunhee Jeong, Clarissa Gerhäuser, Guido Sauter, Thorsten Schlomm, Karl Rohr and Pavlo Lutsik 

## Installation
_MethylBERT_ runs most stably with __Python=3.11__

### Pip Installation
_MethylBERT_ is available as a [python package](https://pypi.org/project/methylbert/).
```
conda create -n methylbert -c conda-forge python=3.11 cudatoolkit==11.8 pip freetype-py
conda activate methylbert
pip install methylbert
```

### Manual Installation
You can set up your conda environment with the `environment.yml` file by
 running `conda env create --file environment.yml` or instead:
```
conda create -n methylbert -c conda-forge python=3.11 cudatoolkit==11.8 pip freetype-py
conda activate methylbert
git clone https://github.com/hanyangii/methylbert.git
cd methylbert
pip3 install .
```

## Quick start
### Python library
If you want to use _MethylBERT_ as a python library, please follow the [tutorials](https://methylbert.readthedocs.io/en/latest/).

### Command line
MethylBERT supports a command line tool. Before using the command line tool, please check [the input file requirements](https://github.com/hanyangii/methylbert/blob/main/tutorials/01_Data_Preparation.md)
```
> methylbert
MethylBERT v2.0.1
One option must be given from ['preprocess_finetune', 'finetune', 'deconvolute']
```
`-h` or `--help` provides available arguments for each function. (e.g., `methylbert preprocess_finetune --help`)

#### 1. Data Preprocessing to fine-tune MethylBERT
**e.g.)** `methylbert preprocess_finetune -f bulk.bam -d dmrs.csv -r genome.fa -p 0.8 -c 50 -o data/`
```
  -s SC_DATASET, --sc_dataset SC_DATASET
                        a file all single-cell bam files are listed up. The first and second columns must indicate file names and cell types if cell types are given. Otherwise, each line must have one file path.
  -f INPUT_FILE, --input_file INPUT_FILE
                        .bam file to be processed
  -d F_DMR, --f_dmr F_DMR
                        .bed or .csv file DMRs information is contained
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        a directory where all generated results will be saved
  -r F_REF, --f_ref F_REF
                        .fasta file containing reference genome
  -nm N_MERS, --n_mers N_MERS
                        K for K-mer sequences (default: 3)
  -m METHYLCALLER, --methylcaller METHYLCALLER
                        Used methylation caller. It must be either bismark or dorado (default: bismark)
  -p SPLIT_RATIO, --split_ratio SPLIT_RATIO
                        the ratio between train and test dataset (default: 0.8)
  -nd N_DMRS, --n_dmrs N_DMRS
                        Number of DMRs to take from the dmr file. If the value is not given, all DMRs will be used
  -c N_CORES, --n_cores N_CORES
                        number of cores for the multiprocessing (default: 1)
  --seed SEED           random seed number (default: 950410)
  --ignore_sex_chromo IGNORE_SEX_CHROMO
                        Whether DMRs at sex chromosomes (chrX and chrY) will be ignored (default: True)
```
#### 2. MethylBERT fine-tuning
**e.g.)** `methylbert finetune -c data/train_seq.csv -t data/test_seq.csv -o model/ -l 12 -s 150 -b 256 --gradient_accumulation_steps 4 -e 600 -w 8 --log_freq 1 --eval_freq 1 --warm_up 1 --lr 1e-4 --decrease_steps 200`
```
  -c TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        train dataset for train bert
  -t TEST_DATASET, --test_dataset TEST_DATASET
                        test set for evaluate train set
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        ex)output/bert.model
  -p PRETRAIN, --pretrain PRETRAIN
                        path to the saved pretrained model to restore
  -l N_ENCODER, --n_encoder N_ENCODER
                        number of encoder blocks. One of [12, 8, 6] need to be given. A pre-trained MethylBERT model is downloaded accordingly. Ignored when -p (--pretrain) is given.
  -nm N_MERS, --n_mers N_MERS
                        n-mers (default: 3)
  -s SEQ_LEN, --seq_len SEQ_LEN
                        maximum sequence len (default: 150)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of batch_size (default: 50)
  --valid_batch VALID_BATCH
                        number of batch_size in valid set. If it's not given, valid_set batch size is set same as the train_set batch size
  --corpus_lines CORPUS_LINES
                        total number of lines in corpus
  --loss LOSS           Loss function for fine-tuning. It can be either 'bce' or 'focal_bce' (default: bce)
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm (default: 1.0)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass. (default: 1)
  -e STEPS, --steps STEPS
                        number of training steps (default: 600)
  --save_freq SAVE_FREQ
                        Steps to save the interim model
  -w NUM_WORKERS, --num_workers NUM_WORKERS
                        dataloader worker size (default: 20)
  --with_cuda WITH_CUDA
                        training with CUDA: true, or false (default: True)
  --log_freq LOG_FREQ   Frequency (steps) to print the loss values (default: 100)
  --eval_freq EVAL_FREQ
                        Evaluate the model every n iter (default: 10)
  --lr LR               learning rate of adamW (default: 4e-4)
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        weight_decay of adamW (default: 0.01)
  --adam_beta1 ADAM_BETA1
                        adamW first beta value (default: 0.9)
  --adam_beta2 ADAM_BETA2
                        adamW second beta value (default: 0.98)
  --warm_up WARM_UP     steps for warm-up (default: 100)
  --decrease_steps DECREASE_STEPS
                        step to decrease the learning rate (default: 200)
  --seed SEED           seed number (default: 950410)
```
#### 3. MethylBERT tumour deconvolution
**e.g.)** `methylbert deconvolute -i data/data.txt -m model/ -o res/ -b 128 --adjustment`
```
  -i INPUT_DATA, --input_data INPUT_DATA
                        Bulk data to deconvolute
  -m MODEL_DIR, --model_dir MODEL_DIR
                        Trained methylbert model
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Directory to save deconvolution results. (default: ./)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size. Please decrease the number if you do not have enough memory to run the software (default: 64)
  --save_logit          Save logits from the model (default: False)
  --adjustment          Adjust the estimated tumour purity (default: False)
```
## Citation
```
@article{jeong2025methylbert,
  title={MethylBERT enables read-level DNA methylation pattern identification and tumour deconvolution using a Transformer-based model},
  author={Jeong, Yunhee and Gerh{\"a}user, Clarissa and Sauter, Guido and Schlomm, Thorsten and Rohr, Karl and Lutsik, Pavlo},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={788},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
