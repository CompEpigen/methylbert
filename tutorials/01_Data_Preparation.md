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