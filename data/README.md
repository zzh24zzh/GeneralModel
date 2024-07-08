## ATAC-seq
### Dependencies
* samtools (1.11)
* deeptools (3.5.1)

```
bamCoverage --bam [BAM_FILE] -o [OUTPUT_BIGWIG_FILE] --outFileFormat bigwig --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --Offset 1 --binSize 1 --numberOfProcessors 12 --blackListFileName black_list.bed
```

## RNA-seq, CAGE-seq, Bru-seq, BruUV-seq, BruChase-seq, GRO-seq, GRO-cap, TT-seq, PRO-seq, PRO-cap, STARR-seq, and NET-CAGE

```
cl='[CELL TYPE]'
modal='[MODALITY]'
# BAMFILE=${cl}_${modal}.bam

bamCoverage --bam $BAMFILE -o ${cl}_${modal}.bigWig \
      --outFileFormat bigwig --normalizeUsing RPGC \
                --effectiveGenomeSize 2913022398 \
                --binSize 1000 --numberOfProcessors 12 \
                --blackListFileName black_list.bed
python data_read.py --cl=${cl} --modal=${modal}
```

## Micro-C and Intact Hi-C
### Dependencies
* hic-straw (1.3.1)
  
#### KR
  ```
  python hic_process.py [CELL NAME] KR [.hic FILE] [OUTPUT DIRECTORY]
  # e.g., python hic_process.py GM12878 KR GM12878.hic .
  ```
#### O/E
  ```
  python hic_process.py [CELL NAME] OE [.hic FILE] [OUTPUT DIRECTORY]
  ```
