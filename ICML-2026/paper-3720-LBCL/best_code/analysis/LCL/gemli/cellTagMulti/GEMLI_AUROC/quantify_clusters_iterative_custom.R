quantify_clusters_iterative_custom <- function(
    data_matrix, 
    marker_genes, 
    fast=TRUE,
    check_positive=TRUE,
    max_iterations=25,
    N=2, 
    verbose=0
){
  iterate <- TRUE
  i <- 2
  genes <- marker_genes
  if(check_positive){
    genes <- intersect(marker_genes, rownames(data_matrix)[Matrix::rowMeans(data_matrix)>0])
  }
  data_matrix <- data_matrix[genes,,drop = FALSE]
  
  # this probably can be changed to handle sparse matrices...
  # https://github.com/UPSUTER/GEMLI/blob/main/GEMLI_package_v0/R/calculate_correlations.R
  # oh... for sure...
  corr_expr_raw <- tryCatch(
    {
      GEMLI:::calculate_correlations(t(data_matrix), fast=fast)
    }, error = function(cond){
      GEMLI:::calculate_correlations(t(data_matrix), fast=FALSE)
    }
  )
  corr_expr <- (1 - corr_expr_raw)/2
  corr_expr[is.na(corr_expr)] <- 0
  
  cell_clusters <- data.matrix(matrix(0, nrow=ncol(data_matrix), ncol=1))
  rownames(cell_clusters) <- colnames(data_matrix)
  cell_clusters[,1] <- rep(1, ncol(data_matrix))
  
  while (iterate & ncol(cell_clusters) < max_iterations) {
    if(verbose >= 2) print(paste0("On clustering iteration: ", ncol(cell_clusters)))
    
    cell_clusters <- cbind(cell_clusters, rep(0,nrow(cell_clusters)))
    
    for (cluster in setdiff(unique(cell_clusters[,(i-1)]), 0)) {
      cells_in_cluster <- rownames(cell_clusters)[cell_clusters[,(i-1)] == cluster]
      if (length(cells_in_cluster) >= 4) { 
        # this line ends the sub clustering # min of desired cluster size
        corr_expr_subset <- corr_expr[cells_in_cluster,cells_in_cluster]
        
        # main function
        clustering <- stats::cutree(stats::hclust(stats::as.dist(corr_expr_subset), 
                                                  method = "ward.D2"), 
                                    k = N)
        
        cell_clusters[names(clustering),i] <- as.vector(clustering) + max(c(0, cell_clusters[,i]), na.rm=TRUE)
      }
      else {cell_clusters[cells_in_cluster,i] <- 0}
    }
    
    if (sum(cell_clusters[,i], na.rm=TRUE)==0) {iterate = FALSE}
    i <- i+1
  }
  
  if(verbose == 1) print(paste0("Total iterations: ", ncol(cell_clusters)))
  cell_clusters[cell_clusters==0] <- NA
  return(cell_clusters)
}