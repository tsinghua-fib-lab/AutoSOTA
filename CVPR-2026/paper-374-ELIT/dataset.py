import os
import json
import random
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import hashlib
import io
import tempfile

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None
import re


# ---------------------------------------------------------------------------
# S3 helper – thin wrapper around boto3 for listing & caching S3 files
# ---------------------------------------------------------------------------

class S3FileCache:
    """
    Lazily downloads files from an S3 prefix into a local cache directory.
    Files are stored under ``<cache_root>/<hex_hash_of_s3_prefix>/...``
    so that multiple datasets can share one cache dir without collision.
    """

    def __init__(self, s3_prefix: str, cache_dir: str | None = None):
        """
        Args:
            s3_prefix: e.g. ``s3://bucket/path/to/data/``
            cache_dir: local directory to cache downloaded files.
                       Defaults to ``/tmp/s3_dataset_cache``.
        """
        import boto3
        self._s3_prefix = s3_prefix.rstrip('/') + '/'
        bucket, key_prefix = self._parse_s3_uri(self._s3_prefix)
        self._bucket = bucket
        self._key_prefix = key_prefix

        # Deterministic cache sub-dir for this particular S3 prefix
        prefix_hash = hashlib.md5(self._s3_prefix.encode()).hexdigest()[:12]
        self._cache_root = os.path.join(
            cache_dir or '/tmp/s3_dataset_cache',
            prefix_hash,
        )
        os.makedirs(self._cache_root, exist_ok=True)

        self._client = boto3.client('s3')

    # ---- public API --------------------------------------------------------

    def list_files(self, sub_prefix: str = '') -> list[str]:
        """
        Return a list of *relative* paths (relative to the dataset root,
        i.e. after ``images/`` or ``vae-sd/``) under ``<s3_prefix>/<sub_prefix>``.
        """
        full_prefix = self._key_prefix + sub_prefix
        paginator = self._client.get_paginator('list_objects_v2')
        rel_paths = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Make path relative to sub_prefix
                rel = key[len(full_prefix):]
                if rel:  # skip the prefix directory marker itself
                    rel_paths.append(rel)
        return rel_paths

    def ensure_local(self, rel_path: str) -> str:
        """
        Return a local filesystem path for *rel_path* (relative to the
        dataset root, e.g. ``images/00000/img00000000.png``).
        Downloads the file on first access; subsequent calls are no-ops.
        """
        local_path = os.path.join(self._cache_root, rel_path)
        if not os.path.exists(local_path):
            s3_key = self._key_prefix + rel_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Download to a temp file then atomically rename to avoid partial reads
            tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(local_path))
            try:
                os.close(tmp_fd)
                self._client.download_file(self._bucket, s3_key, tmp_path)
                os.replace(tmp_path, local_path)
            except BaseException:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        return local_path

    # ---- internal ----------------------------------------------------------

    @staticmethod
    def _parse_s3_uri(uri: str):
        """Return (bucket, key_prefix) from ``s3://bucket/key/prefix/``."""
        assert uri.startswith('s3://'), f"Not an S3 URI: {uri}"
        without_scheme = uri[len('s3://'):]
        bucket, _, key = without_scheme.partition('/')
        return bucket, key


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, data_dir, s3_cache_dir=None):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self._is_s3 = data_dir.startswith('s3://')

        if self._is_s3:
            self._s3_cache = S3FileCache(data_dir, cache_dir=s3_cache_dir)
            self._init_from_s3(supported_ext)
        else:
            self._s3_cache = None
            self._init_from_local(data_dir, supported_ext)

    # ---- local init (original logic) ---------------------------------------

    def _init_from_local(self, data_dir, supported_ext):
        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        all_image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        all_feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )

        self._match_and_build(all_image_fnames, all_feature_fnames, supported_ext,
                              json_loader=lambda: open(os.path.join(self.features_dir, 'dataset.json'), 'rb'))

    # ---- S3 init -----------------------------------------------------------

    def _init_from_s3(self, supported_ext):
        print("Listing S3 files (this may take a minute for large datasets)…")

        # List relative paths under images/ and vae-sd/
        raw_images = self._s3_cache.list_files('images/')
        raw_features = self._s3_cache.list_files('vae-sd/')

        all_image_fnames = sorted(f for f in raw_images if self._file_ext(f) in supported_ext)
        all_feature_fnames = sorted(f for f in raw_features if self._file_ext(f) in supported_ext)

        print(f"  Found {len(all_image_fnames)} images, {len(all_feature_fnames)} features on S3.")

        # Download dataset.json
        json_local = self._s3_cache.ensure_local('vae-sd/dataset.json')
        self._match_and_build(all_image_fnames, all_feature_fnames, supported_ext,
                              json_loader=lambda: open(json_local, 'rb'))

    # ---- shared matching logic ---------------------------------------------

    def _match_and_build(self, all_image_fnames, all_feature_fnames, supported_ext, json_loader):
        _id_re = re.compile(r'(\d+)$')

        def _extract_key(fpath):
            """Return 'subfolder/numeric_id' from a relative path."""
            stem = os.path.splitext(fpath)[0]
            dirname = os.path.dirname(fpath)
            m = _id_re.search(stem)
            return f"{dirname}/{m.group(1)}" if m else stem

        image_by_key = {}
        for f in all_image_fnames:
            image_by_key[_extract_key(f)] = f
        feature_by_key = {}
        for f in all_feature_fnames:
            feature_by_key[_extract_key(f)] = f

        common_keys = sorted(set(image_by_key.keys()) & set(feature_by_key.keys()))

        images_only = set(image_by_key.keys()) - set(feature_by_key.keys())
        features_only = set(feature_by_key.keys()) - set(image_by_key.keys())
        if images_only:
            print(f"WARNING: {len(images_only)} image files have no matching feature file "
                  f"(e.g. {sorted(images_only)[:3]}). These will be skipped.")
        if features_only:
            print(f"WARNING: {len(features_only)} feature files have no matching image file "
                  f"(e.g. {sorted(features_only)[:3]}). These will be skipped.")

        self.image_fnames = [image_by_key[k] for k in common_keys]
        self.feature_fnames = [feature_by_key[k] for k in common_keys]

        # labels
        with json_loader() as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[feature_by_key[k].replace('\\', '/')] for k in common_keys]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    # ---- utilities ---------------------------------------------------------

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def _resolve_path(self, sub_dir, rel_fname):
        """Return a local filesystem path, downloading from S3 if needed."""
        if self._is_s3:
            return self._s3_cache.ensure_local(sub_dir + rel_fname)
        else:
            base = self.images_dir if sub_dir == 'images/' else self.features_dir
            return os.path.join(base, rel_fname)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image_path = self._resolve_path('images/', image_fname)
        with open(image_path, 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        feature_path = self._resolve_path('vae-sd/', feature_fname)
        features = np.load(feature_path)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])

def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset #if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        with open(os.path.join(self.root, f'{index}.png'), 'rb') as f:
            x = np.array(PIL.Image.open(f))
            x = x.reshape(*x.shape[:2], -1).transpose(2, 0, 1)

        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return x, z, c


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, z, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, z, y

class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=True, p_uncond=0.1, mode='train'):
        super().__init__()
        print('Prepare dataset...')
        if mode == 'val':
            self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
            assert len(self.test) == 40504
            self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))
        else:
            self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
            assert len(self.train) == 82783
            self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

            if cfg:  # classifier free guidance
                assert p_uncond is not None
                print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
                self.train = CFGDataset(self.train, p_uncond, self.empty_context)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'