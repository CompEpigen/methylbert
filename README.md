# MethylBERT: A Transformer-based model for read-level DNA methylation pattern identification and tumour deconvolution

![methylbert_scheme](https://github.com/CompEpigen/methylbert/blob/main/img/introduction_methylbert.png)
_The figure was generated using [biorender](https://www.biorender.com/)_

BERT model to classify read-level DNA methylation data into tumour/normal and perform tumour deconvolution.
_MethylBERT_ is implemented using [pytorch](https://pytorch.org/) and [transformers](https://huggingface.co/docs/transformers/index) 🤗.

## Paper
_MethylBERT_ paper is now online on [__bioRxiv__](https://www.biorxiv.org/content/10.1101/2023.10.29.564590v1)!!

MethylBERT: A Transformer-based model for read-level DNA methylation pattern identification and tumour deconvolution,
Yunhee Jeong, Karl Rohr, Pavlo Lutsik,
bioRxiv 2023.10.29.564590; doi: https://doi.org/10.1101/2023.10.29.564590

## Installation
_MethylBERT_ runs most stably with __Python=3.7__

### Pip Installation 
_MethylBERT_ is available as a [python package](https://pypi.org/project/methylbert/).
```
conda create -n methylbert -c conda-forge python=3.7 cudatoolkit==11.1.1 pip
conda activate methylbert
pip install methylbert
```

### Manual Installation
You can set up your conda environment with the `setup.py` file. 
```
conda create -n methylbert -c conda-forge python=3.7 cudatoolkit==11.1.1 pip
conda activate methylbert
git clone https://github.com/hanyangii/methylbert.git
cd methylbert
pip3 install .
```

## Quick start
### Python library
If you want to use _MethylBERT_ as a python library, please follow the [tutorials](https://github.com/hanyangii/methylbert/tree/main/tutorials).

### Command line
MethylBERT supports a command line tool. Before using the command line tool, please check [the input file requirements](https://github.com/hanyangii/methylbert/blob/main/tutorials/01_Data_Preparation.md)
```
> methylbert 
MethylBERT v0.0.2
One option must be given from ['preprocess_finetune', 'finetune', 'deconvolute']
```
#### 1. Data Preprocessing to fine-tune MethylBERT
```
> methylbert preprocess_finetune --help
MethylBERT v0.0.2
usage: methylbert preprocess_finetune [-h] [-s SC_DATASET] [-f INPUT_FILE] -d
                                      F_DMR -o OUTPUT_DIR -r F_REF
                                      [-nm N_MERS] [-p SPLIT_RATIO]
                                      [-nd N_DMRS] [-c N_CORES] [--seed SEED]
                                      [--ignore_sex_chromo IGNORE_SEX_CHROMO]

optional arguments:
  -h, --help            show this help message and exit
  -s SC_DATASET, --sc_dataset SC_DATASET
                        a file all single-cell bam files are listed up. The
                        first and second columns must indicate file names and
                        cell types if cell types are given. Otherwise, each
                        line must have one file path.
  -f INPUT_FILE, --input_file INPUT_FILE
                        .bam file to be processed
  -d F_DMR, --f_dmr F_DMR
                        .bed or .csv file DMRs information is contained
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        a directory where all generated results will be saved
  -r F_REF, --f_ref F_REF
                        .fasta file containing reference genome
  -nm N_MERS, --n_mers N_MERS
                        K for K-mer sequences (default: 3)
  -p SPLIT_RATIO, --split_ratio SPLIT_RATIO
                        the ratio between train and test dataset (default:
                        0.8)
  -nd N_DMRS, --n_dmrs N_DMRS
                        Number of DMRs to take from the dmr file. If the value
                        is not given, all DMRs will be used
  -c N_CORES, --n_cores N_CORES
                        number of cores for the multiprocessing (default: 1)
  --seed SEED           random seed number (default: 950410)
  --ignore_sex_chromo IGNORE_SEX_CHROMO
                        Whether DMRs at sex chromosomes (chrX and chrY) will
                        be ignored (default: True)

```
#### 2. MethylBERT fine-tuning
```
> methylbert finetune --help
MethylBERT v0.0.2
usage: methylbert finetune [-h] -c TRAIN_DATASET [-t TEST_DATASET] -o
                           OUTPUT_PATH [-p PRETRAIN] [-l N_ENCODER]
                           [-nm N_MERS] [-s SEQ_LEN] [-b BATCH_SIZE]
                           [--valid_batch VALID_BATCH]
                           [--corpus_lines CORPUS_LINES]
                           [--max_grad_norm MAX_GRAD_NORM]
                           [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                           [-e STEPS] [--save_freq SAVE_FREQ] [-w NUM_WORKERS]
                           [--with_cuda WITH_CUDA] [--log_freq LOG_FREQ]
                           [--eval_freq EVAL_FREQ] [--lr LR]
                           [--adam_weight_decay ADAM_WEIGHT_DECAY]
                           [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                           [--warm_up WARM_UP]
                           [--decrease_steps DECREASE_STEPS] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -c TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        train dataset for train bert
  -t TEST_DATASET, --test_dataset TEST_DATASET
                        test set for evaluate train set
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        ex)output/bert.model
  -p PRETRAIN, --pretrain PRETRAIN
                        path to the saved pretrained model to restore
  -l N_ENCODER, --n_encoder N_ENCODER
                        number of encoder blocks. One of [12, 8, 6] need to be
                        given. A pre-trained MethylBERT model is downloaded
                        accordingly. Ignored when -p (--pretrain) is given.
  -nm N_MERS, --n_mers N_MERS
                        n-mers (default: 3)
  -s SEQ_LEN, --seq_len SEQ_LEN
                        maximum sequence len (default: 150)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of batch_size (default: 50)
  --valid_batch VALID_BATCH
                        number of batch_size in valid set. If it's not given,
                        valid_set batch size is set same as the train_set
                        batch size
  --corpus_lines CORPUS_LINES
                        total number of lines in corpus
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm (default: 1.0)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass. (default: 1)
  -e STEPS, --steps STEPS
                        number of training steps (default: 600)
  --save_freq SAVE_FREQ
                        Steps to save the interim model
  -w NUM_WORKERS, --num_workers NUM_WORKERS
                        dataloader worker size (default: 20)
  --with_cuda WITH_CUDA
                        training with CUDA: true, or false (default: True)
  --log_freq LOG_FREQ   Frequency (steps) to print the loss values (default:
                        100)
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
```
> methylbert deconvolute --help
MethylBERT v0.0.2
usage: methylbert deconvolute [-h] -i INPUT_DATA -m MODEL_DIR [-o OUTPUT_PATH]
                              [-b BATCH_SIZE] [--save_logit] [--adjustment]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DATA, --input_data INPUT_DATA
                        Bulk data to deconvolute
  -m MODEL_DIR, --model_dir MODEL_DIR
                        Trained methylbert model
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Directory to save deconvolution results. (default: ./)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size. Please decrease the number if you do not
                        have enough memory to run the software (default: 64)
  --save_logit          Save logits from the model (default: False)
  --adjustment          Adjust the estimated tumour purity (default: False)
```
## Citation
```
@article{jeong2023methylbert,
  title={MethylBERT: A Transformer-based model for read-level DNA methylation pattern identification and tumour deconvolution},
  author={Jeong, Yunhee and Rohr, Karl and Lutsik, Pavlo},
  journal={bioRxiv},
  pages={2023--10},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
