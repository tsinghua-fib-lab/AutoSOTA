# MASS Preprocessing

This folder is a standalone preprocessing package for MASS image pretraining.
It is intentionally separated from the main training and inference environment
because SAM2, TotalSegmentator, and medical-image preprocessing dependencies can
be heavy and version-sensitive.

The goal of preprocessing is to convert raw 3D medical images into the format
used by MASS self-supervised pretraining:

```text
<TRAINING_DATA_ROOT>/
  <dataset_name>/
    <case>_image.npy
    <case>_gt.npy          # optional; only available for labeled datasets
    dataset.h5
      <case>/auto_masks
```

`*_image.npy` is stored uncompressed because images are read at every training
iteration. GT labels are optional and small, so they are stored as individual
`*_gt.npy` files when available. SAM2 auto masks can be numerous, so they are
stored compressed in `dataset.h5`.

For pure SSL pretraining, GT labels are not required. A dataset folder with
`<case>_image.npy` and `dataset.h5/<case>/auto_masks` is sufficient.

## Mask Sources

MASS is not tied to SAM2. The training code only consumes a stack of binary
masks for each image, stored as `dataset.h5/<case>/auto_masks`. These masks can
come from SAM2, another automatic mask generator, classical image processing,
weak labels, atlas proposals, expert annotations, or a mixture of several
sources.

The masks do not need semantic class names. For MASS pretraining, they mainly
need to provide reasonable object or region coverage so the model can learn
in-context segmentation tasks from image-mask examples. Noisy and incomplete
masks are acceptable as long as the collection contains diverse, meaningful
regions across images.

SAM2 is the default mask proposal generator provided in this release because it
scales to unlabeled 3D medical images without manual annotation. Advanced users
can replace it or augment it. For example, auto masks and expert masks can be
mixed by converting all masks for a case into the same binary-mask stack before
training. The final HDF5 key remains `auto_masks` for loader compatibility, even
when some masks are expert-supplied.

## Recommended Workflow

Large-scale SAM2 preprocessing can take a long time, especially when the slice
sampling interval is small or the selected region of interest is large. Before
launching a full dataset, we recommend processing a small subset of cases first,
then visually inspecting the exported `*_image.npy`, optional `*_gt.npy`, and
`dataset.h5` auto masks.

The auto masks do not need to be perfect semantic segmentations. MASS can learn
useful representations when the masks provide reasonably good coverage of the
objects or regions you care about. The important question during QA is whether
the mask proposals cover enough meaningful anatomy, pathology, or subregions for
your pretraining goal.

For quick inspection, use:

```bash
python preprocessing/mass_preprocessing/visualize.py \
  --data_dir /path/to/processed_dataset \
  --save_dir ./visualization \
  --n_samples 3 \
  --n_slices 5
```

If the masks miss the regions of interest or generate too many irrelevant
regions, tune the preprocessing parameters before scaling up. Useful knobs
include the modality/ROI-specific enhancement profile, intensity windows,
`physical_interval_mm`, `min_pixel_count`, `min_voxel_count`, body/label crop
margins, and postprocessing thresholds. The defaults are intended as a generally
reasonable initialization; task-specific tuning often improves auto-mask quality
and preprocessing efficiency.

## File Structure

```text
preprocessing/
  README.md
  requirements.txt
  mass_preprocessing/
    preprocess.py             # raw NIfTI -> standardized NIfTI
    generate_masks.py         # standardized NIfTI -> SAM2 auto masks
    postprocess_masks.py      # SAM2 mask NMS / deduplication
    prepare_training_data.py  # NIfTI + masks -> *_image.npy / *_gt.npy / dataset.h5
    visualize.py              # quick visual inspection of exported datasets
  scripts/
    common.py                 # shared wrappers used by dataset scripts
    bcv.py
    amos.py
    kits.py
    lits.py
    autopet.py
    brats.py
    mnm.py
    structseg.py
    totalsegmentator.py
    unlabeled.py
```

Use `preprocessing/scripts/*.py` as the public entry points. Each dataset has at
most one script. Users edit the path and parameter block at the top of the
script, then run it from the repository root.

The lower-level implementation lives in `preprocessing/mass_preprocessing/`.
Advanced users can call those files directly, but the dataset scripts are the
recommended interface.

## Environment

Create a preprocessing environment separately from the MASS training
environment:

```bash
conda create -n mass-preprocess python=3.10
conda activate mass-preprocess
pip install -r preprocessing/requirements.txt
```

`requirements.txt` pins the core runtime packages from the SAM2 preprocessing
environment we used. It is intentionally slimmer than a full `pip freeze` and
assumes CUDA 12.1 PyTorch wheels:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
SAM2 commit 2b90b9f5ceec907a1c18123530e92e794ad901a4
TotalSegmentator==2.7.0
SimpleITK==2.4.1
nibabel==5.3.2
numpy==1.26.4
scipy==1.15.2
```

If your cluster uses a different CUDA/PyTorch build, install the matching
PyTorch wheels first, then install the remaining packages from the requirements
file. SAM2 is installed from source by the requirements file; alternatively,
install it manually:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install -e .
```

Download the SAM2 checkpoint separately, for example
`sam2.1_hiera_large.pt`, and set `SAM2_CHECKPOINT` in the dataset script.

TotalSegmentator is required for body cropping or `ROI_MODE="body"` in unlabeled preprocessing. If you do not use those paths, it is optional.

## High-Level Pipeline

The preprocessing pipeline has four stages:

1. Standardize raw images at original resolution.
   `preprocess.py` reads raw NIfTI images and optional GT labels, reorients them
   to `RAS`, crops to a label/body/foreground ROI, and writes standardized
   NIfTI files:

   ```text
   <WORK_DIR>/nii/
     <case>.nii.gz
     <case>_gt.nii.gz   # if labels exist
   ```

2. Generate or provide masks at original cropped resolution.
   `generate_masks.py` converts each standardized 3D image into SAM2-friendly
   2D slices, samples seed slices, runs SAM2 automatic mask generation, and
   propagates masks through the volume. This is the default pipeline, but users
   may replace this stage with another mask source if it is converted to the
   same intermediate or final training format.

3. Postprocess generated masks.
   `postprocess_masks.py` applies within-image NMS/deduplication to reduce
   highly overlapping masks. This step is mainly intended for automatic mask
   proposals; curated or expert masks may need lighter postprocessing.

4. Export MASS training data.
   `prepare_training_data.py` resamples images, GT labels, and mask proposals
   to the training spacing. It also applies intensity preprocessing and z-score
   normalization to `*_image.npy`, then writes the final dataset folder:

   ```text
   <TRAINING_DATA_ROOT>/<dataset_name>/
     <case>_image.npy
     <case>_gt.npy          # optional
     dataset.h5
       <case>/auto_masks
   ```

In the default SAM2 pipeline, masks are generated before training-spacing
resampling. This matches the training preprocessing used for MASS: masks are
produced on the cropped image in its original resolution, then resampled into
the final training grid together with the image.

## Quick Start: Labeled Dataset

BCV is the simplest example. Edit the block at the top of
`preprocessing/scripts/bcv.py`:

```python
RAW_DATA_DIR = Path("/path/to/BCV/RawData/Training")
WORK_DIR = Path("/path/to/MASS_preprocessed/bcv")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0
DRY_RUN = False

RUN_STANDARDIZE = True
RUN_SAM = True
RUN_NMS = True
RUN_RESAMPLE = True
IMAGES = None
```

Then run:

```bash
python preprocessing/scripts/bcv.py
```

After a successful run, the final training folder should look like:

```text
/path/to/mass_h5/
  bcv/
    case001_image.npy
    case001_gt.npy
    dataset.h5
      case001/auto_masks
```

Use `DRY_RUN=True` to print commands without executing them. Use `IMAGES` to
process a subset, for example:

```python
IMAGES = ["case001", "case002"]
```

The same pattern is used by the other dataset scripts:

```bash
python preprocessing/scripts/amos.py
python preprocessing/scripts/kits.py
python preprocessing/scripts/lits.py
python preprocessing/scripts/autopet.py
python preprocessing/scripts/brats.py
python preprocessing/scripts/mnm.py
python preprocessing/scripts/structseg.py
python preprocessing/scripts/totalsegmentator.py
```

Some scripts export multiple MASS dataset folders. For example, AMOS creates
`amos_ct/` and `amos_mr/`; AutoPET creates `autopet_ct/` and `autopet_suv/`;
BraTS creates one folder per MR modality.

## Image-Only Pretraining With Unlabeled Data

For arbitrary unlabeled NIfTI collections, use:

```bash
python preprocessing/scripts/unlabeled.py
```

Edit the path and dataset block:

```python
RAW_IMAGE_DIR = Path("/path/to/unlabeled/images")
WORK_DIR = Path("/path/to/MASS_preprocessed/unlabeled_abdomen_ct")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
DATASET_NAME = "unlabeled_abdomen_ct"
SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0

MODALITY = "ct"        # "ct", "mr", or "pet"
ANATOMY = "abdomen"    # "abdomen", "chest", "brain", "cardiac", "whole_body"
ROI_MODE = "threshold" # "threshold", "body", or "none"
```

The unlabeled path performs image-only standardization:

```text
raw image only
  -> reorient to RAS
  -> crop by ROI_MODE
  -> save standardized <case>.nii.gz
  -> generate SAM2 auto masks
  -> NMS
  -> resample + normalize with --skip_gt_labels
  -> save <case>_image.npy + dataset.h5
```

No `<case>_gt.npy` is written for unlabeled data.

### ROI Modes for Unlabeled Data

`ROI_MODE="threshold"` is the default. It creates a foreground mask from image
intensity thresholds and crops around the largest component:

```python
THRESHOLD_LOWER = -500.0
THRESHOLD_UPPER = None
CROP_MARGIN_MM = (20.0, 20.0, 20.0)
KEEP_LARGEST_COMPONENT = True
```

For CT abdomen/chest, `THRESHOLD_LOWER=-500` is a reasonable starting point.
For MR/PET, tune the thresholds after inspecting a few cases, or use
`ROI_MODE="none"` if the images are already tightly cropped.

`ROI_MODE="body"` runs TotalSegmentator's body task and crops around the body
mask. This is useful for whole-body CT/PET but requires TotalSegmentator.

`ROI_MODE="none"` skips cropping. This is safest when the dataset is already
cropped, but it increases SAM2 runtime and may generate masks outside the
anatomy of interest.

`ANATOMY` and `MODALITY` select the SAM2 enhancement profile:

```python
("abdomen", "ct") -> "abdomen_ct"
("abdomen", "mr") -> "abdomen_mr"
("chest", "ct") -> "chest_ct"
("brain", "mr") -> "brain_mr"
("cardiac", "mr") -> "cardiac_mr"
("whole_body", "ct") -> "autopet_ct"
("whole_body", "pet") -> "autopet_suv"
```

If your data uses another anatomy/modality pair, add an entry to
`SAM_DATASET_BY_REGION` and, if necessary, add a corresponding configuration in
`mass_preprocessing/generate_masks.py`.

## Resume and Restart Behavior

The expensive stages are designed to be resumable. Use the same command and the
same output directories to continue after interruption.

Standardization:

- Labeled dataset standardization is relatively cheap and usually overwrites
  standardized NIfTI outputs. After it has completed once, set
  `RUN_STANDARDIZE=False` when resuming later stages.
- `unlabeled.py` skips standardized images that already exist.

SAM2 mask generation:

- `generate_masks.py` writes `processing_checkpoint.json` under `SAM_MASK_DIR`.
- Each task is an image-axis pair, for example `case001_axis0`.
- Each processed seed slice also writes a `*_mapping.json` file.
- On restart, completed tasks are skipped, partial/failed tasks are retried,
  and already processed slices are skipped.
- If the previous run crashed while a task was marked `running`, a new run
  automatically resets that stale state to `partial` and retries it.

NMS/postprocessing:

- `postprocess_masks.py` checks whether the NMS output mapping exists for an
  image. If it exists, that image is skipped.

Training export:

- `prepare_training_data.py` checks for `<case>_image.npy`, optional
  `<case>_gt.npy`, and `dataset.h5/<case>/auto_masks`.
- If all expected outputs exist, the case is skipped.
- It also uses task marker files internally so multiple worker processes do not
  write the same case at the same time.

Useful resume patterns:

```python
# SAM already finished; rerun only NMS and export.
RUN_STANDARDIZE = False
RUN_SAM = False
RUN_NMS = True
RUN_RESAMPLE = True

# NMS already finished; rerun only final export.
RUN_STANDARDIZE = False
RUN_SAM = False
RUN_NMS = False
RUN_RESAMPLE = True

# Reprocess only a few cases.
IMAGES = ["case001", "case002"]
```

To force recomputation, delete the corresponding stage outputs. For example,
delete a case from `dataset.h5` and remove its `*_image.npy`/`*_gt.npy` to force
final export, or remove its SAM mask files and checkpoint entry to force SAM2
mask generation.

Avoid launching two independent SAM2 jobs that write to the same `SAM_MASK_DIR`
at the same time. Multi-GPU processing within one `generate_masks.py` run is
supported through `GPU_IDS`.

## Hyperparameters

The released defaults are reasonable initializations for broad image
pretraining. Advanced users can adjust them for their own modality, anatomy,
image resolution, storage budget, and target ROI size.

`TARGET_SPACING`

- Controls the final training grid.
- The default scripts use `(1.5, 1.5, 1.5)`.
- Smaller spacing keeps more anatomical detail but increases memory, storage,
  and auto-mask size.
- Use the same spacing expected by your MASS training configuration.

`sam_dataset_name`

- Selects CT windowing or MR/PET quantile enhancement for SAM2.
- Examples: `abdomen_ct`, `abdomen_mr`, `chest_ct`, `brain_mr`,
  `cardiac_mr`, `autopet_ct`, `autopet_suv`, `totalseg_ct`,
  `totalseg_mr`.
- The provided enhancement profiles are designed from basic radiology knowledge:
  CT profiles use anatomy-aware windows, while MR/PET profiles use quantile
  ranges that emphasize different contrast levels.
- They are general-purpose starting points, not task-specific optima. For a new
  dataset, choose the closest anatomy/modality profile first, then tune it after
  visual inspection.

SAM2 enhancement profiles:

- Defined in `mass_preprocessing/generate_masks.py` under `DATASET_CONFIGS`.
- CT users can tune `window_ranges` to match the anatomy or pathology of
  interest.
- MR/PET users can tune `quantile_ranges` when the default contrast mapping is
  too flat or too saturated.
- `min_pixel_count` filters small 2D masks before/inside SAM processing. Lower
  it for small organs, vessels, tumors, or thin structures; raise it for large
  organs or noisy backgrounds.
- `min_voxel_count` filters small 3D propagated masks. Lower it for genuinely
  small ROIs; raise it when many tiny false-positive objects survive.
- Task-specific SAM2 tuning often improves auto-mask quality. The released
  values are intended to work reasonably across datasets, but a dedicated
  profile for a specific modality, body region, and target structure size can
  perform better.

`physical_interval_mm`

- Physical distance between sampled seed slices used by SAM2 mask generation.
  The default is `30.0` mm.
- Internally, the slice step is computed from the spacing of the selected axis:
  `slice_step = round(physical_interval_mm / axis_spacing)`.
- This is one of the most important speed/quality knobs. Larger values sample
  fewer SAM2 seed slices, which makes preprocessing faster and produces fewer
  auto masks.
- For datasets dominated by large organs or coarse anatomical regions, a larger
  value such as `30-50` mm is often sufficient and can substantially reduce
  preprocessing time.
- For very small organs, vessels, lesions, or thin structures, use a smaller
  value such as `5-10` mm so SAM2 sees more seed slices. This can improve
  coverage but will slow preprocessing and increase the number of generated
  masks.
- `min_slices_per_sample` still enforces a minimum number of sampled slices, so
  very small cropped volumes will not be reduced to only one or two seeds.

`min_slices_per_sample`

- Ensures a minimum number of seed slices even for small cropped volumes.
- Increase it for very small ROIs or sparse anatomy.

`max_masks_per_slice`

- Caps the number of SAM2 masks kept per seed slice.
- Lower it if masks are noisy or storage is too large.
- Increase it only if important small structures are consistently missing.

`nms_iou_threshold`

- Default is `0.95`, which only removes near-duplicates.
- Lower values are more aggressive and reduce storage, but can remove useful
  overlapping objects.

`CROP_MARGIN_MM`

- Used for unlabeled image-only cropping.
- Increase it if the crop is too tight around anatomy.
- Decrease it if there is too much background and SAM2 produces many irrelevant
  masks.

Intensity preprocessing:

- Implemented in `mass_preprocessing/prepare_training_data.py`.
- CT is clipped by `ct_clip`, default `[-991, 500]`, then z-score normalized.
- MR/PET use per-volume percentile clipping, default `[2, 98]`, then z-score
  normalized.
- Tune these only if the saved `*_image.npy` contrast looks poor or a dataset
  has unusual intensity scaling.

Memory/runtime:

- Use multiple GPUs by setting `GPU_IDS = [0, 1, 2, 3]`.
- If SAM2 propagation runs out of memory, reduce `max_masks_per_slice`, increase
  `physical_interval_mm`, or set `SAM2_MAX_OBJECTS_PER_CHUNK` to a smaller
  value before running.

## Dataset-Specific Notes

The release scripts include small dataset-specific fixes because several public
medical datasets have inconsistent metadata or layouts.

- BCV: fixes the image/label direction matrix before reorienting to `RAS`.
- LiTS: cases 28 and 34 have incorrect spacing metadata in common releases; the
  script sets their spacing to `(0.7, 0.7, 2.5)` before reorientation.
- AMOS: exports CT and MR into separate standardized folders based on case ID.
- AutoPET: reads `CTres.nii.gz`, `SUV.nii.gz`, and `SEG.nii.gz`; uses
  TotalSegmentator body masks for cropping; exports CT and SUV as separate MASS
  datasets.
- MnM: reads 4D cardiac MR, extracts labeled time frames, and saves them as 3D
  volumes.
- BraTS: handles multiple MR modalities and applies a shared crop ROI from the
  segmentation foreground.
- TotalSegmentator: combines per-class binary masks into a compact multi-label
  GT map and optionally crops to the body region.

For new datasets, use `scripts/unlabeled.py` as the template if labels are not
needed, or copy the closest labeled dataset script and add a dataset branch in
`mass_preprocessing/preprocess.py`.

Always inspect a small subset visually before launching a large run. Metadata
issues such as wrong spacing, wrong orientation, empty labels, or mismatched
image/label shapes are common in public releases.

## Direct Commands

The dataset scripts are recommended, but the underlying commands can be run
directly.

Standardize a labeled dataset:

```bash
python preprocessing/mass_preprocessing/preprocess.py \
  --dataset bcv \
  --input_dir /path/to/raw/BCV \
  --output_dir /path/to/work/bcv/nii
```

Standardize TotalSegmentator:

```bash
python preprocessing/mass_preprocessing/preprocess.py \
  --dataset totalsegmentator \
  --input_dir /path/to/raw/TotalSegmentator \
  --output_dir /path/to/work/totalseg_ct/nii \
  --modality CT
```

Generate SAM2 masks:

```bash
python preprocessing/mass_preprocessing/generate_masks.py \
  --image_dir /path/to/work/bcv/nii \
  --output_dir /path/to/work/bcv/sam_masks \
  --dataset_name abdomen_ct \
  --sam2_checkpoint /path/to/sam2.1_hiera_large.pt \
  --gpu_ids 0 \
  --physical_interval_mm 30 \
  --auto_select_axes
```

Run NMS:

```bash
python preprocessing/mass_preprocessing/postprocess_masks.py \
  --mask_dir /path/to/work/bcv/sam_masks \
  --output_dir /path/to/work/bcv/sam_masks_nms \
  --gpu_ids 0
```

Export the final MASS dataset:

```bash
python preprocessing/mass_preprocessing/prepare_training_data.py \
  --data_dir /path/to/work/bcv/nii \
  --auto_label_dir /path/to/work/bcv/sam_masks_nms \
  --output_dir /path/to/mass_h5/bcv \
  --target_spacing 1.5 1.5 1.5 \
  --modality ct \
  --gpu_id 0
```

For unlabeled data, add `--skip_gt_labels` at export time. The
`scripts/unlabeled.py` wrapper already does this.

## Visual Inspection

After export, inspect a few cases:

```bash
python preprocessing/mass_preprocessing/visualize.py \
  --data_dir /path/to/mass_h5/bcv \
  --images case001 \
  --save_dir /path/to/visual_checks
```

Check that the image contrast is reasonable, GT labels align when available,
and auto masks cover meaningful anatomical structures without being dominated
by background.

## Troubleshooting

`ModuleNotFoundError: sam2`

- Install SAM2 from source at the pinned commit or use
  `pip install -r preprocessing/requirements.txt`.

`TotalSegmentator` not found

- Install `TotalSegmentator==2.7.0` in the preprocessing environment.
- This is only needed for paths that explicitly use body segmentation.

CUDA out of memory during SAM2 propagation

- Reduce `max_masks_per_slice`.
- Increase `physical_interval_mm`.
- Set `SAM2_MAX_OBJECTS_PER_CHUNK` to a smaller value.
- Process fewer images at once with `IMAGES`.

Final dataset is missing `*_gt.npy`

- This is expected for unlabeled data.
- For labeled data, check that standardized labels are named
  `<case>_gt.nii.gz` before running `prepare_training_data.py`.

Final dataset is missing auto masks

- Check that `sam_masks_nms` contains NMS outputs for the case.
- If you intentionally want image/GT export without auto masks, pass
  `--skip_auto_labels` to `prepare_training_data.py`.

Unexpected crop or empty crop

- For labeled data, check that the GT label is non-empty.
- For unlabeled data, adjust `THRESHOLD_LOWER`, `THRESHOLD_UPPER`, and
  `CROP_MARGIN_MM`, or switch `ROI_MODE` to `none` for a small test.
