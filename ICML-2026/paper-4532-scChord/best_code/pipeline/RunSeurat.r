options (warn = -1)
library(Seurat)
library(ggplot2)
library('Matrix')

train_id = "21"
test_id = "22"
data_path <- "../dataset/"
ADTfile <- paste0(data_path,train_id,"_adtdata.csv") #path to training adt data
OutFile <- paste0("../Results/",test_id,"_from_",train_id,"_seurat.csv") #path to results

ADT <- read.table(ADTfile,sep = ",",header = TRUE,row.names = 1,quote = "")
# load training data
train_dir = paste0(data_path,train_id)
train_genes = read.delim(paste0(train_dir, "_genes.tsv"), row.names = 1,header = FALSE)
train_barcodes = read.delim(paste0(train_dir, '_barcodes.tsv'), row.names = 1,header = FALSE)
Train <- Matrix::readMM(paste0(train_dir, '_matrix.mtx'))
colnames(Train) = row.names(train_barcodes)
rownames(Train) = row.names(train_genes)
# load test data
test_dir = paste0(data_path,test_id)
test_genes = read.delim(paste0(test_dir, "_genes.tsv"), row.names = 1,header = FALSE)
test_barcodes = read.delim(paste0(test_dir, '_barcodes.tsv'), row.names = 1,header = FALSE)
Test <- Matrix::readMM(paste0(test_dir, '_matrix.mtx'))
colnames(Test) = row.names(test_barcodes)
rownames(Test) = row.names(test_genes)

overlap_genes <- intersect(rownames(Test),rownames(Train))
Test <- Test[overlap_genes,]
Train <- Train[overlap_genes,]

Test <- CreateSeuratObject(counts = Test,project = "Test",min.cells = 0,min.features = 0)
Test <- NormalizeData(Test, normalization.method = "LogNormalize", scale.factor = 10000)

colnames(ADT) <- colnames(Train)
Train <- CreateSeuratObject(counts= Train, project= "Train", min.cells=0, min.features=0)
Train <- NormalizeData(Train, normalization.method = "LogNormalize", scale.factor = 10000)
Train[['ADT']] <- CreateAssayObject(counts = ADT,min.cells = 0,min.features = 0)
DefaultAssay(Train) <- 'ADT'

VariableFeatures(Train) <- rownames(Train[["ADT"]])
Train <- NormalizeData(Train, normalization.method = 'CLR', margin = 2)

DN = 30
if (length(rownames(Train[["ADT"]]))-1<30)
    DN = (length(rownames(Train[["ADT"]])) -1)

overlap_genes <- gsub("_", "-",overlap_genes)

options (warn = -1)
anchors <- FindTransferAnchors(reference = Train,query = Test,reference.assay = 'RNA',query.assay = 'RNA',reduction = 'cca',features = overlap_genes, k.filter = NA, dims = 1:DN)

refdata <- GetAssayData(object = Train,assay = 'ADT',slot = 'data')

imputation <- TransferData(anchorset = anchors,refdata = refdata,query = Test,weight.reduction = 'pca',dims = 1:DN)

options(warn = -1)
Imp_New_proteins = as.data.frame(imputation[['id']]@data)
write.csv(Imp_New_proteins,file = OutFile)