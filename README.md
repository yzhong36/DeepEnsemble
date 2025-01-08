# Genome-wide annotation on functional Branchpoints in human genome

## Introduction
We developed an ensemble-based deep learning framework (**DeepEnsemble**) to predict intronic branchpoints, essential for RNA splicing, in the human genome. The model integrates sequence features and genomic distances to identify branchpoints within 70-nucleotide regions upstream of 3' splice sites. Additionally, we prioritized [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/intro/) branchpoint variants and extended the framework to quantify the effects of SNVs on branchpoint functionality.

## Branchpoint annotation
User can download pre-computed annotation files under the folder `data/bp_annotation`. We currently have two versions of annotations based on GENCODE V19 (hg19) or GENCODE V44 (hg38) reference for both predicted (cbp) and experimental-based (ebp) branchpoints. The file format is R-based object and can be accessible when you have
[GenomicRanges](https://bioconductor.org/packages/release/bioc/html/GenomicRanges.html) package installed.

For example 
```r
library(GenomicRanges)

cbp_v44 <- readRDS("data/bp_annotation/gencode_v44_cbp.rds")
cbp_v44
```