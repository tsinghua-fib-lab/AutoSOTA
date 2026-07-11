test_lineages_custom <- function(
    GEMLI_items, 
    max_interval = 100, 
    num_spacing = 101,
    valid_fam_sizes = c(1,5)
) {
  lineage_predictions_matrix <- GEMLI_items[['prediction']]
  lineage_dict_bc <- GEMLI_items[['barcodes']]
  valid_fam_sizes_vec <- valid_fam_sizes[1]:valid_fam_sizes[2]
  
  valid_family_dict <- lineage_dict_bc[as.character(lineage_dict_bc) %in% names(table(lineage_dict_bc))[table(lineage_dict_bc) %in% valid_fam_sizes_vec]]
  cell_with_annotation <- intersect(rownames(lineage_predictions_matrix), names(valid_family_dict))
  family_dict_filt <- valid_family_dict[cell_with_annotation]
  real_family_matrix <- outer(family_dict_filt[cell_with_annotation], family_dict_filt[cell_with_annotation], FUN='==')
  diag(real_family_matrix) <- FALSE
  results_repeated_annotated <- lineage_predictions_matrix[cell_with_annotation, cell_with_annotation]
  
  intervals <- unique(round(seq(0,1,length.out=num_spacing)*max_interval,0))
  
  output_matrix <- matrix(NA, ncol=4, nrow=length(intervals))
  colnames(output_matrix) <- c('precision','TP','FP','sensitivity')
  rownames(output_matrix) <- intervals
  
  for (interval in intervals){
    output_matrix[as.character(interval),1] <- sum(real_family_matrix & (results_repeated_annotated>=interval)) / sum(results_repeated_annotated>=interval)
    output_matrix[as.character(interval),2] <- sum(real_family_matrix & (results_repeated_annotated>=interval))
    output_matrix[as.character(interval),3] <- sum((!real_family_matrix) & (results_repeated_annotated>=interval))
  }
  
  # replace this with an apply function
  output_matrix[,"sensitivity"] <- output_matrix[,"TP"]/output_matrix["0","TP"]
  output_matrix <- output_matrix[,c('TP','FP','precision','sensitivity')]
 
  GEMLI_items[['testing_results']] <- output_matrix
  return(GEMLI_items)
}