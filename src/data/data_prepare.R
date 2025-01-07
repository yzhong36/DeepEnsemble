source("src/utils/feature_extraction.R")

file_path <- "data/raw"
gtf_path <- "data/external/annotations"
save_path <- "data/processed"

######################
# BP Selection
######################

bphunter_BP <- read.delim(file.path(file_path, "Zhang_BP.txt"),sep = "\t")
bphunter_BP <- bphunter_BP[grepl("eBP", bphunter_BP$SOURCE), ]

dbr1_BP <- read.table(file.path(file_path, "Buerer_BP.txt"), header = T)

cola_seq_BP <- read.csv(file.path(file_path, "Zeng_BP.csv"), head = T, skip = 1)

bphunter_BP_gr <- GRanges(seqnames = bphunter_BP$CHROM,
                          IRanges(start = bphunter_BP$BP_POS, width = 1),
                          strand = bphunter_BP$STRAND)
bphunter_BP_gr

dbr1_BP_gr <- GRanges(seqnames = dbr1_BP$chrom,
                      IRanges(start = dbr1_BP$bp_coord + 1, width = 1),
                      strand = dbr1_BP$strand)
dbr1_BP_gr

cola_seq_BP_gr <- GRanges(seqnames = cola_seq_BP$Chromosome,
                          IRanges(start = cola_seq_BP$Branchpoint_position, width = 1),
                          strand = cola_seq_BP$Strand)
chainObject <- import.chain(file.path(gtf_path,
                                      "hg38ToHg19.over.chain"))
cola_seq_BP_gr <- rtracklayer::liftOver(cola_seq_BP_gr, chainObject)
cola_seq_BP_gr <- cola_seq_BP_gr %>% unlist()
cola_seq_BP_gr <- cola_seq_BP_gr[!duplicated(cola_seq_BP_gr)]
cola_seq_BP_gr

getSeq(BSgenome.Hsapiens.UCSC.hg19, bphunter_BP_gr) %>% as.character() %>% table() %>% prop.table()
getSeq(BSgenome.Hsapiens.UCSC.hg19, dbr1_BP_gr) %>% as.character() %>% table() %>% prop.table()
getSeq(BSgenome.Hsapiens.UCSC.hg19, cola_seq_BP_gr) %>% as.character() %>% table() %>% prop.table()

all_BP_gr <- c(bphunter_BP_gr, dbr1_BP_gr, cola_seq_BP_gr)
all_BP_gr <- all_BP_gr[!duplicated(all_BP_gr)]

######################
# Feature Extraction
######################

#####
intronic_bed <- read.table(file.path(gtf_path, 
                           "gencode.v19.bed.iic")) ## this can be replaced with any GTF
head(intronic_bed)
mean(intronic_bed$V5 == ".")
intronic_bed$V5 <- intronic_bed$V5 %>% as.numeric()
intronic_bed$V5 %>% head
summary(intronic_bed$V5)
mean(intronic_bed$V5 >= 90, na.rm = T)
mean(intronic_bed$V5 < 90, na.rm = T)

intronic_gr <- GRanges(seqnames = intronic_bed$V1,
                       IRanges(start = intronic_bed$V2 + 1,
                               end = intronic_bed$V3),
                       strand = intronic_bed$V6)
intronic_gr$intron_type <- "NA"
intronic_gr$intron_type[intronic_bed$V5 >= 90] <- "U12"
intronic_gr$intron_type[intronic_bed$V5 < 90] <- "U2"
table(intronic_gr$intron_type) %>% prop.table()
table(intronic_gr$intron_type)
mean(duplicated(intronic_gr))
#####

intron_tx <- intron_from_annot(annot_path = file.path(gtf_path, "gencode.v19.annotation.gtf.gz"), ## this can be replaced with any GTF
                               min_intron_length = 70)

intron_match <- findOverlaps(intron_tx, intronic_gr, type = "equal")
intron_match
mean(duplicated(queryHits(intron_match)))
mean(duplicated(subjectHits(intron_match)))

intron_tx <- intron_tx[queryHits(intron_match)]
intron_tx$intron_type <- intronic_gr$intron_type[subjectHits(intron_match)]
table(intron_tx$intron_type)
intron_tx$intron_type[intron_tx$intron_type == "NA"] <- "U2"
table(intron_tx$intron_type) %>% prop.table()                               

all_mapped_tx <- map_intron_bp(intron_tx = intron_tx, bp_gr = all_BP_gr)
all_feature_obj <- feature_prepare(intron_tx = all_mapped_tx, bp_gr = all_BP_gr)
all_feature_obj$tx
all_feature_obj$seq %>% dim()
all_feature_obj$five_dist %>% dim()
all_feature_obj$three_dist %>% dim()
all_target <- target_prepare(intron_tx = all_mapped_tx, bp_gr = all_BP_gr)

set.seed(1997)
all_feature_obj$tx$type <- sample(c("train", "valid", "test"), size = length(all_feature_obj$tx), 
                                  replace = T, prob = c(0.7,0.2,0.1))
table(all_feature_obj$tx[all_feature_obj$tx$type == "train"] %>% seqnames()) %>% prop.table()
table(all_feature_obj$tx[all_feature_obj$tx$type == "valid"] %>% seqnames()) %>% prop.table()
table(all_feature_obj$tx[all_feature_obj$tx$type == "test"] %>% seqnames()) %>% prop.table()

write.table(all_feature_obj$seq, file = file.path(save_path, "all_U2_U12_x.txt"),
            row.names = F, col.names = F, quote = F)
write.table(all_target, file = file.path(save_path, "all_U2_U12_y.txt"), 
            row.names = F, col.names = F, quote = F)
write.table(all_feature_obj$five_dist, file = file.path(save_path, "all_U2_U12_x_5_distance.txt"),
            row.names = F, col.names = F, quote = F)
write.table(all_feature_obj$three_dist, file = file.path(save_path, "all_U2_U12_x_3_distance.txt"),
            row.names = F, col.names = F, quote = F)
write.table(mcols(all_feature_obj$tx), file = file.path(save_path, "all_U2_U12_info.txt"),
            row.names = T, col.names = T, quote = F)