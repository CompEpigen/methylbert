# Data Preparation for your own BAM/SAM file to run _MethylBERT_

## Input requirements

In order to run _MethylBERT_, these files are required:
1. Input bulk sample as a BAM/SAM file
2. Reference genome as a FASTA file
3. DMRs as a tab-separated .csv file
4. (Optional, in case you want to fine-tune the MethylBERT model with your data) Pure tumour and normal samples as BAM/SAM files 

#### 1. BAM/SAM File format
_MethylBERT_ currently supports only [bismark](https://www.bioinformatics.babraham.ac.uk/projects/bismark/)-aligned samples where read-level methylation calls are given with `XM` tag. `XM` tage stores methylation calls as follows: 
- `x` : Unmethylated cytosine at CHH
- `X` : Methylated cytosine at CHH
- `h` : Unmethylated cytosine at CHG context
- `H` : Methylated cytosine at CHG context 
- `z` : Unmethylated cytosine at CpG context
- `Z` : Methylated cytosine at CpG context

Each sequence read has its methylation call with `XM` tag like: 
```
SRR5390326.sra.2060072_2060072_length=150       16      chr1    3000485 42      118M    *       0       0 
AATTTCAACTCTAAATTTAATTATTTCCTACTATCTACTCATCTTAAATAAATTTACTTCCTTTTATTCTAAAACTTCTAAATTTACTATCAAACTACTAATATATACTCTAATTTCC  
JA-FFJJJFJJJJJJJJJJJJJJFJJJJJJJJJFJJJJFJJFJJJJJJJJJJJJJJJJFJJJJJJJJJJJJJJJFJJFJFJFJJJFJJJJJJJFJJAJJ<JJFFJAAJJJJFF<JJJJ  
MD:Z:0G6G4G1G3G10G2G12G0G0G1G5G9G5G1G6G4G2G3G0G2G3G1G9G5        XG:Z:GA NM:i:24 
XM:Z:h......x....x.h...h..........x..x............hhh.h.....h.........h.....h.h......h....h..x...xh..x...h.h.........h.....
XR:Z:CT PG:Z:MarkDuplicates-6C1DF036
```
If you have any suggestions for a read aligner/mapper or methylation caller to be supported by _MethylBERT_, please leave it in the [issue](https://github.com/hanyangii/methylbert/issues) tab. 

#### 2. Reference file

_MethylBERT_ uses a reference genome to acquire consistent DNA sequences (disregarding SNVs, mutations etc.) for the given sequence reads. The reference genome FASTA file used for aligning the BAM/SAM files should be provided. You can download various reference genomes on the [UCSC](https://hgdownload.soe.ucsc.edu/downloads.html#hg38sequence) web-site. 

#### 3. DMRs

_MethylBERT_ uses only reads overlapping with given differentially methylated regions (DMRs). The DMRs must be given as a tab-separated .csv style file including three columns (chr, start, end):

```
chr     start   end     length  nCG     meanMethy1      meanMethy2      diff.Methy      areaStat        ctype
chr2    107469896       107470095       200     5       0.0376047481667001      0.925271144570319       -0.887666396403618      -97.6899490136342       mPv
chr19   46474351        46474642        292     13      0.0905692796563347      0.945569253529007       -0.854999973872673      -187.403587988962       mPv
chr4    107989023       107989302       280     8       0.05951346726543        0.908318149867157       -0.848804682601727      -162.881526978275       mL6-2
chr17   65373109        65373244        136     5       0.0858110427730819      0.918285474669721       -0.832474431896639      -85.6599363826153       mL6-2
chr2    83028176        83028389        214     5       0.117763559475859       0.944228847738697       -0.826465288262838      -51.3069729996581       mPv
```
We used [DSS](https://bioconductor.org/packages/release/bioc/html/DSS.html) R package to call DMRs from pure tumour and normal methylomes.

## Simulated read-level methylomes
If you want to test MethylBERT or practice the usage of MethylBERT, the simulated read-level methylomes from our [methylseq_simulation](https://github.com/CompEpigen/methylseq_simulation) tool can be used. 
