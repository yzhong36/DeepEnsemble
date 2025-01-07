require(magrittr)
require(GenomicRanges)
require(BSgenome.Hsapiens.UCSC.hg19)
require(BSgenome.Hsapiens.UCSC.hg38)
require(GenomicFeatures)
require(rtracklayer)
require(dplyr)

intron_from_annot <- function(annot_path, min_intron_length = 70){

    hg19_gtf <- makeTxDbFromGFF(annot_path)
    hg19_full <- import(annot_path)

    tx_id_type <- hg19_full[hg19_full$type == "transcript" & hg19_full$transcript_status == "KNOWN", c("transcript_id", "transcript_type")]

    intron_tx <- intronsByTranscript(hg19_gtf, use.names = T) %>% unlist()
    intron_tx <- intron_tx[names(intron_tx) %in% tx_id_type$transcript_id]
    intron_tx$tx_length <- width(intron_tx)

    intron_tx_id_type <- dplyr::left_join(data.frame(id = names(intron_tx)), 
                                          data.frame(id = tx_id_type$transcript_id, type = tx_id_type$transcript_type),
                                          by = "id")

    intron_tx$tx_type <- intron_tx_id_type$type

    intron_tx <- intron_tx[!duplicated(intron_tx)]
    intron_tx <- intron_tx[width(intron_tx) >= min_intron_length]

    my_ss <- (intron_tx %>% resize(2, fix = "end"))
    my_ss <- my_ss[!duplicated(my_ss)]
    ss_table <- getSeq(BSgenome.Hsapiens.UCSC.hg19, my_ss) %>% as.character() %>% table() %>% prop.table()

    cat("3'ss composition:")
    print(ss_table)

    return(intron_tx)

}

intron_from_annot_genome <- function(annot_path, min_intron_length = 70, genome_version = "hg38"){

    if(genome_version == "hg38"){
        genome <- BSgenome.Hsapiens.UCSC.hg38
    } else{
        genome <- BSgenome.Hsapiens.UCSC.hg19
    }
    cat(paste("Using", genome_version, "for reference ...\n"))

    gtf <- makeTxDbFromGFF(annot_path)
    full <- import(annot_path)

    if(genome_version == "hg38"){
        full <- full[!is.na(full$transcript_support_level)]
        tx_id_type <- full[full$type == "transcript", c("transcript_id", "transcript_type")]
    } else{
        tx_id_type <- full[full$type == "transcript" & full$transcript_status == "KNOWN", c("transcript_id", "transcript_type")]
    }

    intron_tx <- intronsByTranscript(gtf, use.names = T) %>% unlist()
    intron_tx <- intron_tx[names(intron_tx) %in% tx_id_type$transcript_id]
    intron_tx$tx_length <- width(intron_tx)

    intron_tx_id_type <- dplyr::left_join(data.frame(id = names(intron_tx)), 
                                          data.frame(id = tx_id_type$transcript_id, type = tx_id_type$transcript_type),
                                          by = "id")

    intron_tx$tx_type <- intron_tx_id_type$type

    intron_tx <- intron_tx[!duplicated(intron_tx)]
    intron_tx <- intron_tx[width(intron_tx) >= min_intron_length]

    my_ss <- (intron_tx %>% resize(2, fix = "end"))
    my_ss <- my_ss[!duplicated(my_ss)]
    ss_table <- getSeq(genome, my_ss) %>% as.character() %>% table() %>% prop.table()

    cat("3'ss composition:")
    print(ss_table)

    return(intron_tx)

}

map_intron_bp <- function(intron_tx, bp_gr){

    intron_tx <- intron_tx %>% resize(70, fix = "end") %>% resize(66, fix = "start") %>%
                    resize(56, fix = "end")
    intron_tx <- intron_tx[!duplicated(intron_tx)]      

    mapped_tx <- intron_tx[intron_tx %over% bp_gr]
    mapped_tx <- mapped_tx %>% flank(10) %>% resize(70, fix = "start")

    return(mapped_tx)

}

feature_prepare <- function(intron_tx, bp_gr){

    intron_tx_dist_5_matrix <- intron_tx$tx_length - 70 + 1
    intron_tx_dist_5_matrix <- sapply(intron_tx_dist_5_matrix, function(x) return(x:(x+69))) %>% t()
    intron_tx_dist_5_matrix <- intron_tx_dist_5_matrix / intron_tx$tx_length

    intron_tx_dist_3_matrix <- matrix(70:1, ncol = 70, nrow = length(intron_tx), byrow = T)
    intron_tx_dist_3_matrix <- intron_tx_dist_3_matrix / intron_tx$tx_length

    tx_seq <- getSeq(BSgenome.Hsapiens.UCSC.hg19, intron_tx) %>% as.character() %>% 
                     strsplit("")
    tx_seq <- do.call(rbind, tx_seq)

    return(list(tx = intron_tx, 
                seq = tx_seq,
                five_dist = intron_tx_dist_5_matrix,
                three_dist = intron_tx_dist_3_matrix))

}

target_prepare <- function(intron_tx, bp_gr){

    tx_add_map_BP_loc <- mapToTranscripts(bp_gr, intron_tx)
    tx_add_map_BP_loc_list <- tx_add_map_BP_loc %>% split(tx_add_map_BP_loc$transcriptsHits) %>% start()

    tx_add_tx_seq_target <- matrix(0, nrow = length(intron_tx), ncol = 70)

    for (i in seq_along(tx_add_map_BP_loc_list)){
        tx_add_tx_seq_target[i, tx_add_map_BP_loc_list[[i]]] <- 1
    }

    return(tx_add_tx_seq_target)

}
