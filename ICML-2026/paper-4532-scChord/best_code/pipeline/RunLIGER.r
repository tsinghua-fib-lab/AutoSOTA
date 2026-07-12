options (warn = -1)
library(rliger)
library(Seurat)
library("plyr")
library('Matrix')

train_id = "21"
test_id = "22"
data_path <- "../dataset/" #path to training data ande test data
OutFile <- paste0("../Results/",test_id,"_from_",train_id,"_Liger.csv") #path to results

TRAIN_RNAfile <- paste0(data_path,train_id,"_RNA.h5")
TRAIN_rna <- Read10X_h5(TRAIN_RNAfile)

TRAIN_ADTfile <- paste0(data_path,train_id,"_adtdata.csv")
TRAIN_ADT = as.data.frame(read.csv(TRAIN_ADTfile,quote = "",header = TRUE,row.names=1))

TEST_RNAfile <- paste0(data_path,test_id,"_RNA.h5")
TEST_rna <- Read10X_h5(TEST_RNAfile)

Train_uni_cells <- setdiff(colnames(TRAIN_rna),colnames(TEST_rna))
Train_ADT_cells <- gsub("-", ".",Train_uni_cells)
TRAIN_rna <- TRAIN_rna[,Train_uni_cells]
TRAIN_ADT <- TRAIN_ADT[,Train_ADT_cells]

Ligerex.leaveout <- createLiger(list(train = TRAIN_rna,test = TEST_rna))

Ligerex.leaveout <- normalize(Ligerex.leaveout)
Ligerex.leaveout@norm.data <- Ligerex.leaveout@raw.data
Ligerex.leaveout <- selectGenes(Ligerex.leaveout)
Ligerex.leaveout <- scaleNotCenter(Ligerex.leaveout)
k = (length(Ligerex.leaveout@var.genes)-3)
if (k>20){
  k = 20
}
options (warn = -1)
Ligerex.leaveout <- optimizeALS(Ligerex.leaveout,k = k, lambda = 20)
Ligerex.leaveout <- quantile_norm(Ligerex.leaveout)
options (warn = -1)
Ligerex.leaveout@norm.data$train <- rbind(TRAIN_rna,as(as.matrix(TRAIN_ADT), "dgCMatrix"))
Ligerex.leaveout@raw.data$train <- rbind(TRAIN_rna,as(as.matrix(TRAIN_ADT), "dgCMatrix"))

Imputation <- imputeKNN(Ligerex.leaveout,reference = 'train', queries = list('test'), norm = FALSE, scale = FALSE, knn_k = 30)
Result = as.matrix(Imputation@raw.data$test)[unlist(row.names(TRAIN_ADT)),]
write.table(Result,file = paste0(OutFile),sep=',',quote=F,col.names = TRUE)
