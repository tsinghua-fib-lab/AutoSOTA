# Federated Genome-Wide Association Study (GWAS) - GA4GH TES Demo

This example demonstrates a privacy-preserving, federated Genome-Wide Association Study (GWAS) pipeline built on the [APPFL](https://github.com/APPFL/APPFL) framework, demonstrated as part of the [Global Alliance for Genomics and Health (GA4GH)](https://www.ga4gh.org/) initiative. The pipeline studies two complex traits - **Type 2 Diabetes (T2D)** and **Body Mass Index (BMI)** - across five geographically distributed sites without ever sharing raw genomic data.

Each site runs GWAS locally and transmits only summary statistics (beta, SE, MAF, N per variant) to the server, which performs inverse-variance weighted fixed-effect meta-analysis. This TES-based deployment uses the GA4GH [Task Execution Service (TES)](https://ga4gh.github.io/task-execution-schemas/docs/) as the compute backend, with S3 as the model exchange medium. Each site's job runs in a Docker container at a TES-compliant endpoint; clients never open inbound ports. The core GWAS algorithms are identical to the gRPC-based version in `resources/configs/gwas/`.

---

## How It Works

The pipeline is built on two custom APPFL classes in `resources/configs/gwas/`:

**`SiteGWASTrainer`** ([`site_gwas_trainer.py`](../../configs/gwas/site_gwas_trainer.py)) extends `BaseTrainer`. At each site it:

1. Loads PLINK binary genotypes and phenotype/covariate CSVs via `SiteGWASDataset`
2. Runs **BMI GWAS** - chunked OLS per variant (`BMI ~ beta*dosage + gamma*covariates + epsilon`), reporting beta, SE, t-stat, p-value
3. Runs **T2D GWAS** - fits a covariate-only logistic null model once, then applies a 1-df score test per variant (more efficient than full logistic regression per variant)
4. Scores local PGS on a held-out evaluation set and computes R-squared (BMI) and AUROC (T2D)
5. Returns all summary statistics as PyTorch tensors for transmission to the server

**`MetaAnalysisAggregator`** ([`meta_analysis_aggregator.py`](../../configs/gwas/meta_analysis_aggregator.py)) extends `BaseAggregator`. On the server it:

1. Receives beta and SE stacks from all sites
2. Applies inverse-variance weighted fixed-effect meta-analysis: `beta_meta = sum(w_i * beta_i) / sum(w_i)`, where `w_i = 1 / SE_i^2`, and `SE_meta = 1 / sqrt(sum(w_i))`
3. Writes Manhattan plots, QQ plots, hit tables (p < 5e-8), and a site PGS metrics summary
4. Broadcasts the aggregated results back to all clients

Supporting files also in `resources/configs/gwas/`:

- **`site_data.py`** - `SiteGWASDataset` validates that PLINK files and CSVs are present and reports sample size for client weighting
- **`gwas_config.py`** / **`gwas_env.env`** - shared parameters; `get_linear_regression()` and `get_logistic_regression()` switch between scikit-learn (CPU) and NVIDIA cuML (GPU) based on `Use_cuML`

### Data

The demo uses synthetic European-ancestry genotype data: 100,000 samples, ~240,000 SNPs (GRCh37), split across five sites. Phenotypes are simulated from PGS Catalog scores ([PGS003443](https://www.pgscatalog.org/score/PGS003443/) for T2D, [PGS004994](https://www.pgscatalog.org/score/PGS004994/) for BMI) with a liability/linear model and age + sex covariates.

| Site | Samples |
|---|---|
| Site 1 | 18,032 |
| Site 2 | 9,237 |
| Site 3 | 25,028 |
| Site 4 | 40,752 |
| Site 5 | 6,951 |
| **Total** | **100,000** |

---

## APPFL Installation

To run the example, first install APPFL in a clean conda environment:

```bash
conda create -n appfl-env python=3.10 -y
conda activate appfl-env
git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
cd APPFL
pip install -e ".[examples]"
```

---

## How to Join the Federation

A template client configuration file [`clients-template.yaml`](./clients-template.yaml) is provided with ten pre-configured clients. Sites 1-5 map to the five real data partitions; clients 6-10 reuse those same partitions to simulate additional participants. Copy the template, then fill in `tes_endpoint: "<your_tes_endpoint>"` for each client you want to activate. Remove entries for any clients you do not need, and update `server_configs.num_clients` in `server.yaml` to match.

> **Note:** The sample data provided covers Site1 through Site5 only. Ensure the `site_id` in each client's `dataset_kwargs` is set to a value within that range. The existing ten entries in `clients-template.yaml` are already configured correctly - do not change the `site_id` values for clients 6-10, as they are already mapped to valid site data directories.

Each client runs as a Docker container (`zilinghan/client-gwas:latest`) at its TES endpoint. The container includes all GWAS dependencies (pandas-plink, scikit-learn, scipy, matplotlib) and the APPFL client runtime. Resource requirements per client:

| Resource | Default |
|---|---|
| CPU cores | 4 |
| RAM | 8 GB |
| Disk | 20 GB |

---

## S3 Bucket for Data Transmission

The `server.yaml` file uses the S3 bucket `appfl-test-demo-bkt` to transfer GWAS summary statistics (beta, SE tensors) between the server and clients. Only the server needs AWS credentials for this bucket; clients receive presigned URLs and do not need direct bucket access.

If you do not have access to `appfl-test-demo-bkt`, create your own bucket and update the bucket name in `server.yaml`:

```bash
aws s3 mb s3://your-own-bucket-name
```

Then update the `file_storage_kwargs` block in `server.yaml`:

```yaml
file_storage_kwargs:
  s3_bucket: "your-own-bucket-name"
  s3_region: "us-east-1"
  presigned_url_expiry: 3600
```

To verify your access to the bucket, run:

```bash
aws s3 ls s3://appfl-test-demo-bkt
```

A successful response lists the bucket contents (or an empty list). An error response means you do not have the required permissions.

---

## Run the Example with TES

Navigate to the `examples/` directory before running (all config paths are relative to it):

```bash
cd examples
python tes/run.py \
  --server_config ./resources/config_tes/gwas/server.yaml \
  --client_config ./resources/config_tes/gwas/clients-template.yaml
```

---

## [Advanced] Run with Map Visualization

> **Note:** Map visualization currently requires TES endpoints to be specified as IP addresses rather than hostnames.

### Install the Visualization Package

In the same conda environment, install `appfl-log`:

```bash
git clone https://github.com/APPFL/appfl-log.git
cd appfl-log
git checkout ga4gh/demo
pip install -e .
```

### Run with Visualization

```bash
cd examples
python tes/run_viz.py \
  --server_config ./resources/config_tes/gwas/server.yaml \
  --client_config ./resources/config_tes/gwas/clients-template.yaml
```

The map visualization will be available at `http://localhost:7070` while the server is running. Once training finishes, you can replay the run at any time in the same folder with the command:

```bash
hivewatch map run --runs-dir runs --port 7070
```

---

## Configuration Reference

Pipeline parameters are set in `resources/configs/gwas/gwas_env.env`:

| Parameter | Default | Description |
|---|---|---|
| `Use_cuML` | `false` | Use NVIDIA cuML GPU backend for regression (vs. scikit-learn CPU) |
| `Variant_Scaling` | `1.0` | Fraction of variants used at GWAS analysis time (1.0 = all) |
| `Hit_P_Threshold` | `5e-8` | P-value threshold for genome-wide significance |

Server parameters in [`server.yaml`](./server.yaml):

| Parameter | Value | Description |
|---|---|---|
| `num_clients` | 5 | Number of federated sites to wait for before aggregation |
| `num_global_epochs` | 4 | Federated rounds (meta-analysis is valid after 1; additional rounds are used for demonstration) |
| `scheduler` | `SyncScheduler` | All clients must report before aggregation proceeds |
| `aggregator` | `MetaAnalysisAggregator` | Fixed-effect inverse-variance weighted meta-analysis |
| `max_message_size` | 1 GB | Required for full-genome summary statistics payloads |

---

## Output Files

The server writes outputs to `./GA4GH_Demo/server/output/`:

```
data/
  appfl_meta_gwas_bmi.csv.gz     Meta-analysis BMI summary statistics
  appfl_meta_gwas_t2d.csv.gz     Meta-analysis T2D summary statistics
  appfl_meta_gwas_hits.csv       Genome-wide significant hits (or top 100 if none pass threshold)
  appfl_site_pgs_metrics.csv     Per-site PGS R-squared (BMI) and AUROC (T2D)
  appfl_meta_summary.csv         High-level run summary
graphs/
  appfl_meta_gwas_bmi_manhattan.png
  appfl_meta_gwas_bmi_qq.png
  appfl_meta_gwas_t2d_manhattan.png
  appfl_meta_gwas_t2d_qq.png
```

Each site writes local outputs to its configured `trainer_output_dirname`:

```
data/
  <SiteN>_local_gwas_bmi.csv.gz
  <SiteN>_local_gwas_t2d.csv.gz
  <SiteN>_local_gwas_hits.csv
  <SiteN>_local_pgs_scores.csv
  <SiteN>_local_pgs_metrics.csv
graphs/
  <SiteN>_local_gwas_bmi_manhattan.png
  <SiteN>_local_gwas_bmi_qq.png
  <SiteN>_local_gwas_t2d_manhattan.png
  <SiteN>_local_gwas_t2d_qq.png
  <SiteN>_local_pgs_bmi_scatter.png
  <SiteN>_local_pgs_t2d_roc.png
```

Summary statistics columns:

| Column | Description |
|---|---|
| CHR | Chromosome |
| SNP | Variant ID (rsID) |
| BP | Base pair position (GRCh37) |
| EA | Effect allele |
| NEA | Non-effect allele |
| BETA | Effect size estimate |
| SE | Standard error |
| STAT | Test statistic (t for BMI, Z for T2D) |
| P | P-value |
| MAF | Minor allele frequency |
| N / N_META | Sample size |
| OR | Odds ratio (T2D only) |
