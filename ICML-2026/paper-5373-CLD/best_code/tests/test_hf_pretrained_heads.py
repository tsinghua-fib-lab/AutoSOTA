"""Tests for the published HuggingFace CLD language-detection heads.

Validates the exact usage snippet shown on the model cards
(`williamhtan/cld-*-5lang`):

    import numpy as np
    from huggingface_hub import hf_hub_download
    from jaxcld import ASRModel, CVXNNLangDetectHead

    asr = ASRModel.from_pretrained(base_model, config={"languages": languages})
    head_path = hf_hub_download(repo_id, filename)
    head = CVXNNLangDetectHead.load(head_path, asr)
    asr.set_lang_detect_head(head)
    pred_langs, pred_texts = asr.predict(audio_16k_mono)

The lightweight tests (imports, and download -> load -> forward pass on
synthetic pooled features) run without the multi-GB base ASR models, since
`CVXNNLangDetectHead.load` only unpickles the head and `.predict` accepts
already-pooled `(B, D)` features. The full end-to-end snippet test downloads a
base model and is therefore opt-in via `CLD_RUN_HF_SNIPPET_E2E=1`.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


LANGUAGES = ["en", "hi", "id", "ms", "zh"]

# (repo_id, head_filename, base_model, encoder_dim) — one per published model.
PUBLISHED_HEADS = [
    (
        "williamhtan/cld-whisper-small-5lang",
        "model.pkl",
        "openai/whisper-small",
        768,
    ),
    (
        "williamhtan/cld-whisper-large-v3-5lang",
        "model.pkl",
        "openai/whisper-large-v3",
        1280,
    ),
    (
        "williamhtan/cld-mms-1b-5lang",
        "model.pkl",
        "facebook/mms-1b-all",
        1280,
    ),
]
HEAD_IDS = [h[0].split("/")[-1] for h in PUBLISHED_HEADS]


def _download_head(repo_id: str, filename: str) -> str:
    """hf_hub_download the artifact, skipping the test if the Hub is unreachable."""
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(repo_id, filename)
    except Exception as e:  # network / auth / not-found
        pytest.skip(f"Could not download {repo_id}/{filename} (offline?): {e}")


def test_model_card_imports_resolve():
    """The imports at the top of the model-card snippet must all resolve."""
    import numpy as _np  # noqa: F401
    from huggingface_hub import hf_hub_download
    from jaxcld import ASRModel, CVXNNLangDetectHead

    assert callable(hf_hub_download)
    assert hasattr(ASRModel, "from_pretrained")
    assert hasattr(CVXNNLangDetectHead, "load")


@pytest.mark.parametrize("repo_id,filename,base_model,dim", PUBLISHED_HEADS, ids=HEAD_IDS)
def test_published_head_downloads_loads_and_predicts(repo_id, filename, base_model, dim):
    """Download the published artifact, load it via ``CVXNNLangDetectHead.load``
    (no base model required), and run a forward pass on synthetic pooled features.

    This exercises the novel parts of the model-card snippet end to end:
    ``hf_hub_download`` -> ``CVXNNLangDetectHead.load`` -> ``head.predict``.
    """
    pytest.importorskip("jax")  # artifacts store JAX arrays; loading needs jax
    from jaxcld import CVXNNLangDetectHead

    path = _download_head(repo_id, filename)

    # asr_model is unused by CVXNNLangDetectHead.load (it just unpickles the head).
    head = CVXNNLangDetectHead.load(path, asr_model=None)

    inner = head.head
    assert tuple(inner.theta1.shape) == (len(LANGUAGES), dim, 128)
    assert tuple(inner.theta2.shape) == (len(LANGUAGES), 128)
    assert int(inner.n_classes) == len(LANGUAGES)

    # Forward pass on synthetic pooled encoder features (B, D) -> class indices.
    rng = np.random.default_rng(0)
    pooled = rng.standard_normal((4, dim)).astype(np.float32)
    preds = head.predict(pooled)

    assert isinstance(preds, list) and len(preds) == 4
    assert all(isinstance(p, int) and 0 <= p < len(LANGUAGES) for p in preds)


@pytest.mark.parametrize("repo_id,filename,base_model,dim", PUBLISHED_HEADS, ids=HEAD_IDS)
def test_model_card_snippet_end_to_end(repo_id, filename, base_model, dim):
    """Run the full model-card snippet against the real base model.

    Opt-in (downloads multi-GB ASR weights): set ``CLD_RUN_HF_SNIPPET_E2E=1``.
    Skips gracefully if the base model can't be loaded.
    """
    if os.environ.get("CLD_RUN_HF_SNIPPET_E2E") != "1":
        pytest.skip("Set CLD_RUN_HF_SNIPPET_E2E=1 to run the full end-to-end snippet test.")
    pytest.importorskip("jax")
    from jaxcld import ASRModel, CVXNNLangDetectHead

    try:
        asr = ASRModel.from_pretrained(base_model, config={"languages": LANGUAGES})
    except Exception as e:
        pytest.skip(f"Base model not available ({base_model}): {e}")

    # Encoder dim must match the head this artifact was trained for.
    assert int(asr.get_dimensions()) == dim

    head_path = _download_head(repo_id, filename)
    head = CVXNNLangDetectHead.load(head_path, asr_model=asr)
    asr.set_lang_detect_head(head)

    # 1 second of (synthetic) 16 kHz mono audio.
    audio_16k_mono = np.zeros(16000, dtype=np.float32)
    out = asr.predict(audio_16k_mono)

    assert isinstance(out, tuple) and len(out) == 2
    pred_langs, _pred_texts = out
    # Whisper returns lists; MMS returns scalars.
    if isinstance(pred_langs, list):
        assert len(pred_langs) >= 1
        assert all(lang in LANGUAGES for lang in pred_langs)
    else:
        assert pred_langs in LANGUAGES


def test_mms_masked_pooling_is_invariant_to_batch_padding():
    """Regression test for the MMS padding-mask pooling fix.

    ``MMS.predict`` zero-pads a batch to its longest clip ("padding=longest").
    The pooled encoder features handed to the head must ignore those padded
    frames; otherwise the padding dilutes a short clip's mean pool and its
    prediction starts to depend on what else happens to share its batch. That
    bug is what dropped benchmark accuracy to ~0.85 while the head trains to
    ~0.98 (see the note in ``MMS.predict``).

    The fix guarantees an invariant we can assert directly: a clip's detected
    language must be identical whether it is predicted alone or batched
    alongside a much longer clip (which forces padding on the short one).

    Opt-in (downloads the multi-GB MMS base model): set ``CLD_RUN_HF_SNIPPET_E2E=1``.
    """
    if os.environ.get("CLD_RUN_HF_SNIPPET_E2E") != "1":
        pytest.skip("Set CLD_RUN_HF_SNIPPET_E2E=1 to run the MMS batch-padding invariance test.")
    pytest.importorskip("jax")
    from jaxcld import ASRModel, CVXNNLangDetectHead

    # Pick the published MMS head (the fix is MMS-specific).
    try:
        repo_id, filename, base_model, dim = next(
            h for h in PUBLISHED_HEADS if "mms" in h[2].lower()
        )
    except StopIteration:  # pragma: no cover - guards against table edits
        pytest.skip("No MMS head in PUBLISHED_HEADS.")

    try:
        asr = ASRModel.from_pretrained(base_model, config={"languages": LANGUAGES})
    except Exception as e:
        pytest.skip(f"Base model not available ({base_model}): {e}")
    assert int(asr.get_dimensions()) == dim

    head_path = _download_head(repo_id, filename)
    head = CVXNNLangDetectHead.load(head_path, asr_model=asr)
    asr.set_lang_detect_head(head)

    # Non-trivial audio: with all-zero audio the (zero) padding could not dilute
    # the pool, so the bug would be invisible. Use noise so padded frames differ
    # from real ones, and make the second clip much longer to force padding on
    # the first. Same seed -> the short clip's content is identical in both runs.
    rng = np.random.default_rng(0)
    short = rng.standard_normal(16000).astype(np.float32) * 0.1   # 1 s
    long = rng.standard_normal(48000).astype(np.float32) * 0.1    # 3 s

    # Single clip -> scalar lang; batched [short, long] -> list, index 0 is short.
    lang_single, _ = asr.predict(short)
    langs_batched, _ = asr.predict([short, long])

    assert lang_single in LANGUAGES
    assert langs_batched[0] == lang_single, (
        "MMS prediction for a clip changed when it was batched with a longer "
        "clip -- padded frames are leaking into the mean pool (masked-pooling "
        "regression)."
    )
