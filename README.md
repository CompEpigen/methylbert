# MethylBERT
Pytorch BERT model to train methylation data 


## Setup
You can set up your own conda environment with the __requirements.txt__ file. 

```
conda create -n methylbert python=3.6
conda activate methylbert
conda install pytorch torchvision -c pytorch

git clone https://github.com/hanyangii/MethylBERT.git
cd MethylBERT
python3 -m pip install -r requirements.txt
```

## Create n-mers input files from DMRs

MethylBERT requires an input text file containing DNA sequenence + CpG and CHG context methylation pattern. 
We provide a tool to generate the required input from bam files. 

#### 1. BAM File

The methylation pattern must be extracted by aligning the samples using `bismark`.
`bismark` includes `XM` tag for methylation patterns at each read as follows:

- `x` : Unmethylated cytosine at CHH
- `X` : Methylated cytosine at CHH
- `h` : Unmethylated cytosine at CHG context
- `H` : Methylated cytosine at CHG context 
- `z` : Unmethylated cytosine at CpG context
- `Z` : Methylated cytosine at CpG context

For example, 
```
SRR5390326.sra.2060072_2060072_length=150       16      chr1    3000485 42      118M    *       0       0 
AATTTCAACTCTAAATTTAATTATTTCCTACTATCTACTCATCTTAAATAAATTTACTTCCTTTTATTCTAAAACTTCTAAATTTACTATCAAACTACTAATATATACTCTAATTTCC  
JA-FFJJJFJJJJJJJJJJJJJJFJJJJJJJJJFJJJJFJJFJJJJJJJJJJJJJJJJFJJJJJJJJJJJJJJJFJJFJFJFJJJFJJJJJJJFJJAJJ<JJFFJAAJJJJFF<JJJJ  
MD:Z:0G6G4G1G3G10G2G12G0G0G1G5G9G5G1G6G4G2G3G0G2G3G1G9G5        XG:Z:GA NM:i:24 
XM:Z:h......x....x.h...h..........x..x............hhh.h.....h.........h.....h.h......h....h..x...xh..x...h.h.........h.....
XR:Z:CT PG:Z:MarkDuplicates-6C1DF036
```

However, we only use methylation patterns at CHG and CpG context.

#### 2. Reference file

Since _MethylBERT_ uses a reference genome to find corresponding cytosines, the reference genome used for the alignment must be provided as a __fasta__ file. 

#### 3. DMRs

_MethylBert_ deals with only reads overlapping with given genomic regions. For cell-type deconvolution, we strongly recommend to use cell-type differentially methylated regions (DMRs). 

Regions must be given as __BED__ file or __csv__ style file including three columns (chr, start, end). 

```
chr     start   end     length  nCG     meanMethy1      meanMethy2      diff.Methy      areaStat        ctype
chr2    107469896       107470095       200     5       0.0376047481667001      0.925271144570319       -0.887666396403618      -97.6899490136342       mPv
chr19   46474351        46474642        292     13      0.0905692796563347      0.945569253529007       -0.854999973872673      -187.403587988962       mPv
chr4    107989023       107989302       280     8       0.05951346726543        0.908318149867157       -0.848804682601727      -162.881526978275       mL6-2
chr17   65373109        65373244        136     5       0.0858110427730819      0.918285474669721       -0.832474431896639      -85.6599363826153       mL6-2
chr2    83028176        83028389        214     5       0.117763559475859       0.944228847738697       -0.826465288262838      -51.3069729996581       mPv
```
We used [DSS](https://bioconductor.org/packages/release/bioc/html/DSS.html) R package to generate DMRs from single-cell data.

#### 2. Input data generation
The required input file can be generated from the bismark-aligned bam file. 

```
python3 src/pretrain_data_generate.py -s bam_file_list.txt -d cell_type_dmrs.bed 
-o ./methylbert_input/ -r mm10.fa -nm 3 -p 0.8 -nd 10 -c 3

usage: pretrain_data_generate.py [-h] -s SC_DATASET -d F_DMR -o OUTPUT_DIR -r
                                 F_REF [-nm N_MERS] [-p SPLIT_RATIO]
                                 [-nd N_DMRS] [-c N_CORES]

optional arguments:
  -h, --help            show this help message and exit
  -s SC_DATASET, --sc_dataset SC_DATASET
                        .txt file all single-cell bam files are listed up
  -d F_DMR, --f_dmr F_DMR
                        .bed or .csv file DMRs information is contained
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        a directory where all generated results will be saved
  -r F_REF, --f_ref F_REF
                        .fasta file for the reference genome
  -nm N_MERS, --n_mers N_MERS
                        n-mers
  -p SPLIT_RATIO, --split_ratio SPLIT_RATIO
                        the ratio between train and test dataset
  -nd N_DMRS, --n_dmrs N_DMRS
                        Number of DMRs to take from the dmr file. If the value
                        is not given, all DMRs will be used
  -c N_CORES, --n_cores N_CORES
                        number of cores for the multiprocessing
```

If you give a split ratio for train and test data, this will output two files `train_seq.txt` and `test_seq.txt` as follows:

```
GCT CTT TTG TGA GAT ATG TGG GGG GGA GAG AGG GGA GAG AGA GAA AAC ACT CTG TGA GAG AGC GCA CAA AAG AGC GCC CCA CAT ATG TGT GTG TGC GCA CAT ATC TCA CAG AGA GAT ATG TGC GCT CTG TGT GTA TAA AAC ACA CAG AGT GTT TTA TAA AAG AGC GCC CCG CGT GTG TGT GTT TTT TTT TTC TCA CAG AGG GGC GCA CAG AGC GCG CGA GAA AAT ATT TTT TTC TCA CAT ATT TTG TGG GGA GAA AAG AGC GCC CCA CAC ACT CTT TTT TTA TAA AAA AAT ATA TAA AAT ATG TGA GAG AGA GAT ATG TGC GCA CAA AAC ACA CAC ACA CAT
CGT GTA TAA AAA AAC ACG CGT GTC TCA CAA AAC ACT CTG TGT GTT TTG TGA GAA AAT ATC TCT CTG TGC GCC CCG CGG GGT GTG TGC GCT CTT TTT TTC TCA CAA AAC ACT CTG TGT GTA TAA AAG AGT GTC TCT CTT TTT TTT TTT TTC TCT CTG TGG GGG GGC GCA CAT ATC TCC CCT CTT TTC TCT CTA TAA AAC ACA CAC ACA CAC ACC CCA CAG AGA GAT ATG TGA GAG AGA GAG AGA GAT ATG TGT GTT TTA TAC ACT CTC TCA
TGA GAC ACA CAA AAC ACA CAG AGA GAC ACT CTT TTT TTA TAT ATC TCT CTA TAA AAC ACA CAC ACC CCT CTC TCA CAC ACT CTC TCA CAC ACT CTC TCT CTC TCC CCT CTC TCA CAT ATT TTT TTG TGC GCC CCA CAA AAA AAT ATG TGT GTG TGC GCT CTT TTC TCA CAT ATG TGA GAA AAA AAG AGG GGA GAA AAG AGT GTC TCT CTC TCA CAA AAA AAT ATG TGT GTT TTA TAA AAC ACT CTC TCA CAT ATG TGG GGT GTG TGT GTT TTT TTT TTG TGC GCC CCT CTA TAA AAT ATG TGG GGA GAA AAT ATG TGG GGA GAG AGT GTT TTG TGT GTG TGT GTT TTC
```

## Run Pre-train 

We provide 10,000 sequences of 3-mers DNA + Methylation patterns for both train and test data in the data/3mers/ directory.
After generating the required data, you can run _MethylBert_ as follows:

```
python src/torch_pretrain.py -c data/3mers/train_seq.txt  -t data/3mers/test_seq.txt  -o ./methylbert_pretrain/ -s 120 -e 120000 -b 1000 -w 32 -a 16 -l 10 -hs 512 -nm 3 --log_freq 1000

usage: torch_pretrain.py [-h] -c TRAIN_DATASET [-t TEST_DATASET] -o
                         OUTPUT_PATH [-nm N_MERS] [-hs HIDDEN] [-l LAYERS]
                         [-a ATTN_HEADS] [-s SEQ_LEN]
                         [--max_grad_norm MAX_GRAD_NORM]
                         [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                         [-b BATCH_SIZE] [-e STEPS] [-w NUM_WORKERS]
                         [--with_cuda WITH_CUDA] [--log_freq LOG_FREQ]
                         [--eval_freq EVAL_FREQ] [--corpus_lines CORPUS_LINES]
                         [--cuda_devices CUDA_DEVICES [CUDA_DEVICES ...]]
                         [--on_memory ON_MEMORY] [--lr LR]
                         [--adam_weight_decay ADAM_WEIGHT_DECAY]
                         [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                         [--warm_up WARM_UP] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -c TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        train dataset for train bert
  -t TEST_DATASET, --test_dataset TEST_DATASET
                        test set for evaluate train set
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        ex)output/bert.model
  -nm N_MERS, --n_mers N_MERS
                        n-mers
  -hs HIDDEN, --hidden HIDDEN
                        hidden size of transformer model
  -l LAYERS, --layers LAYERS
                        number of layers
  -a ATTN_HEADS, --attn_heads ATTN_HEADS
                        number of attention heads
  -s SEQ_LEN, --seq_len SEQ_LEN
                        maximum sequence len
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of batch_size
  -e STEPS, --steps STEPS
                        number of steps
  -w NUM_WORKERS, --num_workers NUM_WORKERS
                        dataloader worker size
  --with_cuda WITH_CUDA
                        training with CUDA: true, or false
  --log_freq LOG_FREQ   printing loss every n iter: setting n
  --eval_freq EVAL_FREQ
                        Evaluate the model every n iter
  --corpus_lines CORPUS_LINES
                        total number of lines in corpus
  --cuda_devices CUDA_DEVICES [CUDA_DEVICES ...]
                        CUDA device ids
  --on_memory ON_MEMORY
                        Loading on memory: true or false
  --lr LR               learning rate of adam
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        weight_decay of adam
  --adam_beta1 ADAM_BETA1
                        adam first beta value
  --adam_beta2 ADAM_BETA2
                        adam second beta value
  --warm_up WARM_UP     steps for warm-up
  --seed SEED           seed number
```

It records loss and accuracy values from train and test data as well as the best performing model at the given output directory. 
