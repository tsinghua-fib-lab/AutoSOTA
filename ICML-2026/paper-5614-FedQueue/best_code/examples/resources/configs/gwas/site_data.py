from pathlib import Path


class SiteGWASDataset:
    def __init__(self, data_dir, site_id):
        self.data_dir = Path(data_dir).resolve()
        self.site_id = str(site_id)
        self.plink_bed = self.data_dir / "EUR.synthetic.100k.ld.maf.bed"
        self.pheno_gwas = self.data_dir / "phenotypes_gwas.csv"
        self.pheno_eval = self.data_dir / "phenotypes_pgs_eval.csv"
        self.covariates = self.data_dir / "covariates.csv"

        required = [
            self.plink_bed,
            self.plink_bed.with_suffix(".bim"),
            self.plink_bed.with_suffix(".fam"),
            self.pheno_gwas,
            self.pheno_eval,
            self.covariates,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.site_id} missing required inputs:\n  - "
                + "\n  - ".join(missing)
            )

        with self.plink_bed.with_suffix(".fam").open("r", encoding="utf-8") as handle:
            self.sample_size = sum(1 for _ in handle)

    def __len__(self):
        return self.sample_size


def get_site_gwas_dataset(data_dir, site_id):
    return SiteGWASDataset(data_dir=data_dir, site_id=site_id), None
