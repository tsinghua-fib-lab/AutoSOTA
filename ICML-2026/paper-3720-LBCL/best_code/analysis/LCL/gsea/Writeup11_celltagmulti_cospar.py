# from https://cospar.readthedocs.io/en/latest/20210121_reprogramming_static_barcoding_v2.html
import cospar as cs
import numpy as np
import anndata as ad

cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "/home/users/kzlin/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/celltagmulti_data"  # A relative path to save data. If not existed before, create a new one.
cs.settings.figure_path = "/home/users/kzlin/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/celltagmulti_figure"  # A relative path to save figures. If not existed before, create a new one.

adata_orig = ad.read_h5ad("/home/users/kzlin/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/celltagging-multi_fibroblast.h5ad")
print(adata_orig)

# Convert the 'time_info' (or 'Time point') column to string type
adata_orig.obs['time_info'] = adata_orig.obs['time_info'].astype(int).astype(str)
adata_orig.obs['md_fate_coarse_rev1'] = adata_orig.obs['md_fate_coarse_rev1'].astype(str)
adata_orig.obs['state_info'] = adata_orig.obs['md_fate_coarse_rev1'].astype(str)

adata_orig = cs.pp.initialize_adata_object(adata_orig)

cs.hf.update_time_ordering(
    adata_orig, updated_ordering=["3", "12", "21"]
)
cs.hf.check_available_choices(adata_orig)

adata = cs.tmap.infer_Tmap_from_multitime_clones(
    adata_orig,
    clonal_time_points=["3", "12"],
    later_time_point="21",
    smooth_array=[15, 10, 5],
    sparsity_threshold=0.2,
    intraclone_threshold=0.2,
)

adata.write("/home/users/kzlin/kzlinlab/projects/scContrastiveLearn/out/kevin/Writeup11/Writeup11_celltagemulti_cospar.h5ad")