import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest


from jaxcld.models.asr_model import ASRModel
from jaxcld.models.lang_detect_head import CVXNNLangDetectHead, NNLangDetectHead, SklearnLangDetectHead, SVMLangDetectHead


ROOT = Path(__file__).resolve().parents[1]

# Tests pull data from the Hugging Face Hub instead of a local checkout. The
# 5-language (en/hi/id/ms/zh) speech dataset is mirrored at this repo; override
# with CLD_HF_DATASET to point at a fork.
HF_DATASET_REPO = os.environ.get("CLD_HF_DATASET", "williamhtan/cld-multi-dataset")
HF_DATASET_SPLIT = os.environ.get("CLD_HF_DATASET_SPLIT", "test")
# How many clips per language to pull into the lightweight test subset, and the
# cap on rows scanned while streaming to build it (streaming avoids downloading
# the multi-GB dataset in full).
HF_SAMPLES_PER_LANG = int(os.environ.get("CLD_HF_SAMPLES_PER_LANG", "2"))
HF_STREAM_SCAN_LIMIT = int(os.environ.get("CLD_HF_STREAM_SCAN_LIMIT", "2000"))

WHISPER_MODEL_NAME = os.environ.get("CLD_WHISPER_MODEL_NAME", "openai/whisper-small")
MMS_MODEL_NAME = os.environ.get("CLD_MMS_MODEL_NAME", "facebook/mms-1b-all")


@pytest.fixture(scope="session")
def hf_dataset_dir(tmp_path_factory) -> Path:
    """Stream a small, language-balanced subset of the Hub dataset and
    materialize it to a local ``DatasetDict`` dir that ``ASRModel.load_data``
    (which calls ``load_from_disk``) can read.

    Streaming + per-language capping keeps the test fast and avoids pulling the
    full multi-GB dataset. Skips gracefully if the Hub is unreachable.
    """
    datasets = pytest.importorskip("datasets")

    try:
        stream = datasets.load_dataset(HF_DATASET_REPO, split=HF_DATASET_SPLIT, streaming=True)
    except Exception as e:  # network / auth / not-found
        pytest.skip(f"Could not stream {HF_DATASET_REPO} (offline?): {e}")

    rows: List[dict] = []
    per_lang: dict = {}
    try:
        for scanned, sample in enumerate(stream):
            if scanned >= HF_STREAM_SCAN_LIMIT:
                break
            lang = sample.get("lang")
            if lang is None or per_lang.get(lang, 0) >= HF_SAMPLES_PER_LANG:
                continue
            audio = sample["audio"]
            rows.append({
                # Re-wrap to a clean {array, sampling_rate} dict for re-encoding.
                "audio": {
                    "array": np.asarray(audio["array"], dtype=np.float32),
                    "sampling_rate": int(audio["sampling_rate"]),
                },
                "text": sample.get("text") or "",
                "lang": lang,
                "accent": sample.get("accent") or "",
            })
            per_lang[lang] = per_lang.get(lang, 0) + 1
    except Exception as e:
        pytest.skip(f"Failed while streaming {HF_DATASET_REPO}: {e}")

    if not rows:
        pytest.skip(f"No usable rows streamed from {HF_DATASET_REPO} (split={HF_DATASET_SPLIT}).")

    features = datasets.Features({
        "audio": datasets.Audio(sampling_rate=16000),
        "text": datasets.Value("string"),
        "lang": datasets.Value("string"),
        "accent": datasets.Value("string"),
    })
    subset = datasets.Dataset.from_list(rows, features=features)
    # load_data only reads one split; mirror the canonical train/valid/test layout.
    dd = datasets.DatasetDict({"train": subset, "valid": subset, "test": subset})
    out = tmp_path_factory.mktemp("hf_dataset") / "cld_multi_subset"
    dd.save_to_disk(str(out))
    return out


def _random_nn_head_artifact(path: Path, dims: int, n_classes: int, seed: int = 0) -> None:
    """
    Create a pickled state_dict compatible with `NNLangDetectHead.load()`.
    Matches `NNLangDetectHeadModule`: Linear(dims->256) ... Linear(256->n_classes).
    """
    rng = np.random.default_rng(seed)
    state = {
        "classifier.0.weight": (rng.standard_normal((256, dims), dtype=np.float32) * 0.02).astype(np.float32),
        "classifier.0.bias": np.zeros((256,), dtype=np.float32),
        "classifier.3.weight": (rng.standard_normal((n_classes, 256), dtype=np.float32) * 0.02).astype(np.float32),
        "classifier.3.bias": np.zeros((n_classes,), dtype=np.float32),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


class _RandomCVXHead:
    """
    Minimal object to satisfy `CVXNNLangDetectHead.predict()`, which calls:
      head.stacked_predict(pooled, head.theta1, head.theta2)
    where theta1: (C, d, m) and theta2: (C, m) are per-class stacked weights.
    """

    def __init__(self, theta1: np.ndarray, theta2: np.ndarray):
        self.theta1 = theta1
        self.theta2 = theta2

    def stacked_predict(self, X: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
        # theta1: (C, d, m), theta2: (C, m) -> returns (B, C)
        X = np.asarray(X, dtype=np.float32)
        C = theta1.shape[0]
        cols = []
        for c in range(C):
            Z = X @ np.asarray(theta1[c], dtype=np.float32)  # (B, m)
            Z = np.maximum(Z, 0.0)
            cols.append(Z @ np.asarray(theta2[c], dtype=np.float32))  # (B,)
        return np.stack(cols, axis=1)  # (B, C)


def _random_cvxnn_head_artifact(path: Path, dims: int, n_classes: int, seed: int = 0, hidden: int = 64) -> None:
    rng = np.random.default_rng(seed)
    theta1 = (rng.standard_normal((n_classes, dims, hidden), dtype=np.float32) * 0.02).astype(np.float32)
    theta2 = (rng.standard_normal((n_classes, hidden), dtype=np.float32) * 0.02).astype(np.float32)
    head = _RandomCVXHead(theta1=theta1, theta2=theta2)
    with open(path, "wb") as f:
        pickle.dump(head, f)


class _DummySklearnPredictor:
    """Minimal sklearn-like predictor used for serialization tests."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # Deterministic binary prediction from first feature.
        return (X[:, 0] > 0.0).astype(np.int64)


class _NaiveRandomWhisperHead:
    """Naive head for Whisper: expects hidden states shaped (B, T, D) (torch.Tensor)."""

    def __init__(self, n_classes: int, seed: int = 0):
        self.n_classes = int(n_classes)
        self.rng = np.random.default_rng(seed)

    def predict(self, hidden):
        # hidden: torch.Tensor (B, T, D)
        bsz = int(getattr(hidden, "shape", [1])[0])
        return self.rng.integers(low=0, high=self.n_classes, size=(bsz,)).tolist()


class _NaiveRandomMMSHead:
    """Naive head for MMS: expects pooled features shaped (B, D) (numpy.ndarray)."""

    def __init__(self, n_classes: int, seed: int = 0):
        self.n_classes = int(n_classes)
        self.rng = np.random.default_rng(seed)

    def predict(self, pooled: np.ndarray):
        pooled = np.asarray(pooled)
        if pooled.ndim != 2:
            raise ValueError(f"Expected pooled (B, D), got {pooled.shape}")
        bsz = pooled.shape[0]
        return self.rng.integers(low=0, high=self.n_classes, size=(bsz,)).tolist()


def test_sklearn_lang_detect_head_load_and_predict(tmp_path: Path):
    artifact = tmp_path / "sklearn_head.pkl"
    with open(artifact, "wb") as f:
        pickle.dump(_DummySklearnPredictor(), f)

    head = SklearnLangDetectHead.load(str(artifact), asr_model=None)

    # Pooled numpy features
    pooled = np.asarray([[1.0, 0.1], [-1.0, 0.2], [0.0, -0.4]], dtype=np.float32)
    pred_np = head.predict(pooled)
    assert pred_np == [1, 0, 0]

    # Whisper-like hidden states (B, T, D)
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 0.1], [1.0, -0.2]], [[-1.0, 0.2], [-1.0, -0.3]]], dtype=torch.float32)
    pred_torch = head.predict(hidden)
    assert pred_torch == [1, 0]


def test_svm_lang_detect_head_backward_compatible_load(tmp_path: Path):
    artifact = tmp_path / "svm_head.pkl"
    with open(artifact, "wb") as f:
        pickle.dump(_DummySklearnPredictor(), f)

    head = SVMLangDetectHead.load(str(artifact), asr_model=None)
    assert isinstance(head, SVMLangDetectHead)
    pred = head.predict(np.asarray([[0.5, 0.0], [-0.5, 0.0]], dtype=np.float32))
    assert pred == [1, 0]


@pytest.fixture(scope="session")
def dataset_languages(hf_dataset_dir) -> List[str]:
    from datasets import load_from_disk

    ds = load_from_disk(str(hf_dataset_dir))
    langs = sorted({row for row in ds["test"]["lang"] if row is not None})
    if not langs:
        pytest.skip(f"No 'lang' values found in subset from {HF_DATASET_REPO}.")
    return langs


@pytest.fixture(scope="session")
def test_audio(hf_dataset_dir) -> np.ndarray:
    from datasets import load_from_disk

    ds = load_from_disk(str(hf_dataset_dir))
    audio = ds["test"][0]["audio"]
    arr = np.asarray(audio["array"], dtype=np.float32).squeeze()
    if arr.ndim != 1:
        arr = arr.mean(axis=0)
    return arr


@pytest.mark.parametrize("head_kind", ["nn", "cvxnn"])
def test_lang_detect_head_loads_and_predicts_from_whisper_model(tmp_path: Path, head_kind: str, dataset_languages, test_audio):
    """
    LangDetectHead (NN + CVXNN):
    - Test loading (from a generated artifact)
    - Test prediction through WhisperModel.predict() on a real audio file
    """
    languages = list(dataset_languages)
    if len(languages) < 2:
        pytest.skip("Need at least 2 languages for language detection tests.")

    try:
        asr = ASRModel.from_pretrained(WHISPER_MODEL_NAME, config={"languages": languages})
    except Exception as e:
        pytest.skip(f"Whisper model not available locally ({WHISPER_MODEL_NAME}): {e}")

    dims = int(asr.get_dimensions())
    n_classes = len(languages)

    if head_kind == "nn":
        artifact = tmp_path / "nn_head.pkl"
        _random_nn_head_artifact(artifact, dims=dims, n_classes=n_classes, seed=0)
        head = NNLangDetectHead.load(str(artifact), asr)
    elif head_kind == "cvxnn":
        artifact = tmp_path / "cvx_head.pkl"
        _random_cvxnn_head_artifact(artifact, dims=dims, n_classes=n_classes, seed=0, hidden=64)
        head = CVXNNLangDetectHead.load(str(artifact), asr_model=asr)
    else:
        raise AssertionError("unreachable")

    asr.set_lang_detect_head(head)

    langs, texts = asr.predict(test_audio)
    assert isinstance(langs, list) and len(langs) == 1
    assert isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str)
    assert langs[0] in languages


@pytest.mark.parametrize(
    "backend,model_name",
    [
        ("whisper", WHISPER_MODEL_NAME),
        ("mms", MMS_MODEL_NAME),
    ],
)
def test_asr_init_load_data_and_predict_with_naive_head(backend: str, model_name: str, dataset_languages, test_audio, hf_dataset_dir):
    """
    ASR (Whisper + MMS):
    - Init model from default model weights
    - Load data from the Hugging Face Hub dataset (cld-multi-dataset)
    - Predict using a naive random head (to smoke-test predict path only)
    """
    languages = list(dataset_languages)
    if len(languages) < 2:
        pytest.skip("Need at least 2 languages in the dataset to run ASR tests.")

    if backend == "whisper":
        try:
            asr = ASRModel.from_pretrained(model_name, config={"languages": languages})
        except Exception as e:
            pytest.skip(f"Whisper model not available locally ({model_name}): {e}")

        A, y = asr.load_data(str(hf_dataset_dir), dataset_split="test", shuffle=False)
        assert isinstance(A, np.ndarray) and isinstance(y, np.ndarray)
        assert A.ndim == 2 and y.ndim == 1 and A.shape[0] == y.shape[0]

        asr.set_lang_detect_head(_NaiveRandomWhisperHead(n_classes=len(languages), seed=0))
        langs, texts = asr.predict(test_audio)
        assert isinstance(langs, list) and len(langs) == 1 and langs[0] in languages
        assert isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str)

    elif backend == "mms":
        asr = ASRModel.from_pretrained(model_name, config={"languages": languages})

        A, y, n_classes = asr.load_data(str(hf_dataset_dir), dataset_split="test", shuffle=False)
        assert isinstance(A, np.ndarray) and isinstance(y, np.ndarray)
        assert isinstance(n_classes, int) and n_classes == len(languages)
        assert A.ndim == 2 and y.ndim == 1 and A.shape[0] == y.shape[0]

        asr.set_lang_detect_head(_NaiveRandomMMSHead(n_classes=len(languages), seed=0))
        lang, text = asr.predict(test_audio)
        assert isinstance(lang, str) and lang in languages
        assert isinstance(text, str)

    else:
        raise AssertionError(f"Unknown backend={backend}")

