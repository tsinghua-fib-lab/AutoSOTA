# SpaHGC:Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference
## 🔍 Overview
While spatial transcriptomics (ST) has advanced our understanding of gene expression within tissue context, its high experimental cost limits large-scale application. Predicting ST gene expression from pathology images offers a promising, cost-effective alternative, yet existing methods often struggle to capture the complex spatial relationships across slides. 
To address the challenge, we propose SpaHGC, a multi-modal heterogeneous graph-based model that captures both intra-slice and inter-slice spot-spot relationships from histology images. 
It integrates local spatial context within the target slide and cross-slide similarities computed from image embeddings extracted by a pathology foundation model. 
These embeddings enable inter-slice knowledge transfer across slides. Additionally, SpaHGC incorporates Masked Graph Contrastive Learning to enhance feature representation and transfer spatial gene expression knowledge from reference to target slides, enabling it to model complex spatial dependencies and significantly improve prediction accuracy.
We conducted comprehensive benchmarking on seven histology-ST datasets from different platforms, tissues, and cancer subtypes. The results demonstrate that SpaHGC significantly outperforms existing nine state-of-the-art methods across all evaluation metrics. The model’s predicted gene expression profiles closely align with the ground truth data and accurately correspond to tumor regions. Furthermore, the predictions are significantly enriched in multiple cancer-related pathways, highlighting its strong biological relevance and application potential.

![Overview.png](Overview.png)

## ⚙️ Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 4090)
- `pip install -r requiremnt.txt`

## 🔑 Getting access
This study utilizes a pathology foundation model called UNI. You need to request access to the model weights from the Huggingface model page at:[https://huggingface.co/mahmoodlab/UNI](https://huggingface.co/mahmoodlab/UNI). 

## 📁 Data
This study utilizes **seven publicly available ST datasets** from different platforms and tissue types to comprehensively evaluate the performance and generalizability of **SpaHGC**. All datasets are open-access and can be obtained through the corresponding publications or repositories.

### 🔬 Datasets Overview

| Dataset Name   | Tissue / Cancer Type                  | Platform       | Subtype     | #Samples Num | Source      |
|----------------|----------------------------------------|----------------|-------------|----------|-------------|
| cSCC           | Human skin squamous cell carcinoma     | Traditional ST | –           | 12       | [cSCC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240)    |
| HER2+          | Breast cancer                          | Traditional ST | HER2+       | 36       | [HER2+](https://github.com/almaan/her2st/)   |
| Alex           | Breast cancer                          | Visium         | TNBC        | 4        | [Alex](https://doi.org/10.48610/4fb74a9)    |
| Visium BC      | Breast cancer                          | Visium         | HER2+       | 3        | [VisiumBC](https://doi.org/10.48610/4fb74a9)|
| HEST-1K (Lymph node)   | Lymph node                             | Visium         | –           | 4        | [HEST-1K](https://github.com/mahmoodlab/hest)  |
| HEST-1K (Pancreas1)      | Pancreatic cancer                      | Visium         | –           | 4        | [HEST-1K](https://github.com/mahmoodlab/hest)  |
| HEST-1K (Pancreas2)      | Pancreatic cancer                      | Visium         | –           | 3        | [HEST-1K](https://github.com/mahmoodlab/hest)  |


## 🚀 Running demo
We provide an example for predicting gene expression data of the cSCC dataset. Please refer to [ViscSCC.ipynb](ViscSCC.ipynb).

