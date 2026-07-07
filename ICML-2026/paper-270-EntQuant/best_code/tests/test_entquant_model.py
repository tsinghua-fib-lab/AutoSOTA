"""Integration tests for EntQuantModel.

Requires CUDA. Uses a small model (Qwen/Qwen3-0.6B) for fast testing.

Run: pytest tests/test_entquant_model.py -v
"""

import json
from pathlib import Path

import pytest
import torch
from optimum.quanto import QLinear, QuantizedModelForCausalLM
from torch import nn
from transformers import AutoModelForCausalLM

from entquant import EntQuantModel
from entquant.compression.compressor import BlockCompressor
from entquant.super_weights import (
    detect_fallback_layers,
    detect_super_weights,
    SuperWeightsConfig,
)

MODEL_ID = "Qwen/Qwen3-0.6B"
WEIGHT_QTYPE = "qfloat8"
DTYPE = torch.bfloat16

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

try:
    from optimum.quanto.tensor.weights.marlin import MarlinF8QBytesTensor
except ImportError:
    MarlinF8QBytesTensor = None


@pytest.fixture(scope="module")
def converted_checkpoint(tmp_path_factory) -> Path:
    """Convert model once, shared across tests that need a checkpoint."""
    output_dir = tmp_path_factory.mktemp("converted")
    EntQuantModel.convert(
        MODEL_ID,
        str(output_dir),
        weight_qtype=WEIGHT_QTYPE,
        dtype=DTYPE,
    )
    return output_dir


@requires_cuda
class TestConvert:
    """convert() produces valid checkpoint."""

    def test_checkpoint_files_exist(self, converted_checkpoint):
        d = converted_checkpoint
        assert (d / "config.json").exists()
        assert (d / "model.safetensors.index.json").exists()
        assert (d / "quanto_qmap.json").exists()

    def test_config_has_quantization_metadata(self, converted_checkpoint):
        config = json.loads((converted_checkpoint / "config.json").read_text())
        qc = config["quantization_config"]
        assert qc["quant_method"] == "entquant"
        assert qc["weight_qtype"] == "qfloat8_e4m3fn"

    def test_all_shards_exist(self, converted_checkpoint):
        index = json.loads((converted_checkpoint / "model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            assert (converted_checkpoint / shard).exists()

    def test_qmap_nonempty(self, converted_checkpoint):
        qmap = json.loads((converted_checkpoint / "quanto_qmap.json").read_text())
        assert len(qmap) > 0

    def test_loadable_by_quanto(self, converted_checkpoint):
        qmodel = QuantizedModelForCausalLM.from_pretrained(str(converted_checkpoint))
        assert qmodel is not None
        del qmodel
        torch.cuda.empty_cache()


@requires_cuda
class TestFromPretrained:
    """from_pretrained loads and does forward pass."""

    def test_load_with_compression(self, converted_checkpoint, sample_inputs):
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=True)
        assert isinstance(model, nn.Module)
        assert isinstance(model.compressor, BlockCompressor)

        with torch.no_grad():
            logits = model(**sample_inputs).logits
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

        del model
        torch.cuda.empty_cache()

    def test_quantize_compress_forward(self, sample_inputs):
        model = EntQuantModel.from_pretrained(
            MODEL_ID,
            quantize=True,
            compress=True,
            weight_qtype=WEIGHT_QTYPE,
            dtype=DTYPE,
        )
        assert model.compressor is not None
        stats = model.compression_stats()
        assert stats["injected_ratio"] > 1.0
        assert "avg_entropy" in stats
        assert stats["avg_entropy"] > 0.0

        with torch.no_grad():
            logits = model(**sample_inputs).logits
        assert not torch.isnan(logits).any()

        del model
        torch.cuda.empty_cache()

    def test_decompress_model_matches_original(self, converted_checkpoint, sample_inputs):
        """compress -> decompress_model -> logits match uncompressed."""
        # Uncompressed baseline
        ref = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=False)
        with torch.no_grad():
            logits_ref = ref(**sample_inputs).logits
        del ref
        torch.cuda.empty_cache()

        # Compressed, then decompress_model()
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=True)
        assert model.compressor is not None
        assert len(model.compressor.blocks) > 0

        model.compressor.decompress_model()

        # State should be cleared
        assert len(model.compressor.blocks) == 0

        # No _compressed_weights buffers should remain
        for name, _ in model.model.named_buffers():
            assert not name.endswith("._compressed_weights")

        with torch.no_grad():
            logits = model(**sample_inputs).logits
        assert torch.allclose(logits_ref, logits, atol=1e-4), f"max diff = {(logits_ref - logits).abs().max().item()}"

        del model
        torch.cuda.empty_cache()

    def test_getattr_delegation(self, converted_checkpoint):
        """__getattr__ delegates to inner model for config, etc."""
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=False)
        assert hasattr(model, "config")
        assert model.config is model.model.config

        del model
        torch.cuda.empty_cache()


@requires_cuda
class TestSavePretrained:
    """save_pretrained round-trip tests."""

    def test_save_creates_valid_checkpoint(self, converted_checkpoint, tmp_path):
        """save_pretrained produces config, index, qmap, and shards."""
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=False)
        save_dir = tmp_path / "saved"
        model.save_pretrained(str(save_dir))

        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.safetensors.index.json").exists()
        assert (save_dir / "quanto_qmap.json").exists()

        index = json.loads((save_dir / "model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            assert (save_dir / shard).exists()

        del model
        torch.cuda.empty_cache()

    def test_save_load_roundtrip(self, converted_checkpoint, sample_inputs, tmp_path):
        """save -> load -> logits match."""
        model_a = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=False)
        with torch.no_grad():
            logits_a = model_a(**sample_inputs).logits

        save_dir = tmp_path / "roundtrip"
        model_a.save_pretrained(str(save_dir))
        del model_a
        torch.cuda.empty_cache()

        model_b = EntQuantModel.from_pretrained(str(save_dir), compress=False)
        with torch.no_grad():
            logits_b = model_b(**sample_inputs).logits

        assert torch.allclose(logits_a, logits_b, atol=1e-4), f"max diff = {(logits_a - logits_b).abs().max().item()}"

        del model_b
        torch.cuda.empty_cache()

    def test_save_load_roundtrip_compressed(self, converted_checkpoint, sample_inputs, tmp_path):
        """save compressed -> load+compress -> logits match."""
        model_a = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=True)
        with torch.no_grad():
            logits_a = model_a(**sample_inputs).logits

        save_dir = tmp_path / "roundtrip_compressed"
        model_a.save_pretrained(str(save_dir))
        del model_a
        torch.cuda.empty_cache()

        model_b = EntQuantModel.from_pretrained(str(save_dir), compress=True)
        with torch.no_grad():
            logits_b = model_b(**sample_inputs).logits

        assert torch.allclose(logits_a, logits_b, atol=1e-4), f"max diff = {(logits_a - logits_b).abs().max().item()}"

        del model_b
        torch.cuda.empty_cache()

    def test_saved_checkpoint_loadable_by_quanto(self, converted_checkpoint, tmp_path):
        """QuantizedModelForCausalLM can load save_pretrained output."""
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=False)
        save_dir = tmp_path / "quanto_compat"
        model.save_pretrained(str(save_dir))
        del model
        torch.cuda.empty_cache()

        qmodel = QuantizedModelForCausalLM.from_pretrained(str(save_dir))
        assert qmodel is not None
        del qmodel
        torch.cuda.empty_cache()


@requires_cuda
@pytest.mark.skipif(
    MarlinF8QBytesTensor is None,
    reason="MarlinF8QBytesTensor not importable (quanto internals changed)",
)
class TestMarlinDispatch:
    """Verify that qfloat8 weights are dispatched as MarlinF8QBytesTensor."""

    def _get_qlinear_weights(self, model: nn.Module):
        """Collect all QLinear weight tensors from the inner model."""
        return [(name, m.weight) for name, m in model.model.named_modules() if isinstance(m, QLinear)]

    def test_marlin_tensors_without_compression(self, converted_checkpoint):
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=False)
        qlinear_weights = self._get_qlinear_weights(model)
        assert len(qlinear_weights) > 0, "No QLinear modules found"

        for name, weight in qlinear_weights:
            assert isinstance(weight, MarlinF8QBytesTensor), (
                f"{name}.weight is {type(weight).__name__}, expected MarlinF8QBytesTensor"
            )

        del model
        torch.cuda.empty_cache()

    def test_marlin_tensors_with_compression(self, converted_checkpoint):
        """After decompress, weights should still be MarlinF8QBytesTensor."""
        model = EntQuantModel.from_pretrained(str(converted_checkpoint), compress=True)
        # Decompress first block to materialize weights
        compressor = model.compressor
        assert compressor is not None
        first_block = next(iter(compressor.blocks))
        compressor.decompress(first_block)

        block_module = model.model.get_submodule(first_block)
        marlin_count = 0
        for name, m in block_module.named_modules():
            if isinstance(m, QLinear):
                assert isinstance(m.weight, MarlinF8QBytesTensor), (
                    f"{first_block}.{name}.weight is {type(m.weight).__name__}, expected MarlinF8QBytesTensor"
                )
                marlin_count += 1

        assert marlin_count > 0, "No QLinear modules in decompressed block"

        del model
        torch.cuda.empty_cache()

    def test_marlin_tensors_after_quantize(self):
        """Quantize from base model and verify Marlin dispatch."""
        model = EntQuantModel.from_pretrained(
            MODEL_ID,
            quantize=True,
            compress=False,
            weight_qtype=WEIGHT_QTYPE,
            dtype=DTYPE,
        )
        qlinear_weights = self._get_qlinear_weights(model)
        assert len(qlinear_weights) > 0, "No QLinear modules found"

        for name, weight in qlinear_weights:
            assert isinstance(weight, MarlinF8QBytesTensor), (
                f"{name}.weight is {type(weight).__name__}, expected MarlinF8QBytesTensor"
            )

        del model
        torch.cuda.empty_cache()


@pytest.fixture(scope="class")
def w8a8_model():
    """W8A8 quantized model, shared across TestActivationQtype."""
    model = EntQuantModel.from_pretrained(
        MODEL_ID,
        quantize=True,
        compress=False,
        weight_qtype=WEIGHT_QTYPE,
        activation_qtype="qfloat8",
        dtype=DTYPE,
    )
    yield model
    del model
    torch.cuda.empty_cache()


@requires_cuda
class TestActivationQtype:
    """W8A8 activation quantization tests."""

    def test_convert_w8a8_metadata(self, tmp_path):
        """convert() with activation_qtype writes correct metadata."""
        output_dir = tmp_path / "w8a8_converted"
        EntQuantModel.convert(
            MODEL_ID,
            str(output_dir),
            weight_qtype=WEIGHT_QTYPE,
            activation_qtype="qfloat8",
            dtype=DTYPE,
        )
        config = json.loads((output_dir / "config.json").read_text())
        qc = config["quantization_config"]
        assert qc["activation_qtype"] == "qfloat8_e4m3fn"

        qmap = json.loads((output_dir / "quanto_qmap.json").read_text())
        for entry in qmap.values():
            assert entry["activations"] == "qfloat8_e4m3fn"

    def test_w8a8_marlin(self, w8a8_model):
        """W8A8 model uses MarlinF8QBytesTensor and all QLinear have activation_qtype."""
        if MarlinF8QBytesTensor is None:
            pytest.skip("MarlinF8QBytesTensor not importable")
        qlinear_count = 0
        for _, m in w8a8_model.model.named_modules():
            if isinstance(m, QLinear):
                qlinear_count += 1
                assert isinstance(m.weight, MarlinF8QBytesTensor)
                assert m.activation_qtype is not None
        assert qlinear_count > 0

    def test_w8a8_forward(self, w8a8_model, sample_inputs):
        """W8A8 model produces valid logits."""
        with torch.no_grad():
            logits = w8a8_model(**sample_inputs).logits
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_w8a8_save_load_roundtrip(self, w8a8_model, sample_inputs, tmp_path):
        """W8A8 quantize -> save -> load -> logits match."""
        with torch.no_grad():
            logits_a = w8a8_model(**sample_inputs).logits

        save_dir = tmp_path / "w8a8_roundtrip"
        w8a8_model.save_pretrained(str(save_dir))

        model_b = EntQuantModel.from_pretrained(str(save_dir), compress=False)
        with torch.no_grad():
            logits_b = model_b(**sample_inputs).logits

        assert torch.allclose(logits_a, logits_b, atol=1e-4), f"max diff = {(logits_a - logits_b).abs().max().item()}"

        del model_b
        torch.cuda.empty_cache()


class TestBYOMValidation:
    """BYOM input validation (no CUDA needed)."""

    def test_requires_quantize(self):
        """Passing model without quantize=True raises ValueError."""
        with pytest.raises(ValueError, match="quantize=True is required"):
            EntQuantModel.from_pretrained(model=nn.Linear(10, 10))

    def test_requires_model_or_model_id(self):
        """Neither model nor model_id raises ValueError."""
        with pytest.raises(ValueError, match="Either model_id or model"):
            EntQuantModel.from_pretrained(quantize=True)


@requires_cuda
class TestBYOM:
    """Bring-Your-Own-Model: from_pretrained(model=..., quantize=True)."""

    def test_byom_quantize_forward(self, sample_inputs):
        """BYOM quantize without compression produces valid logits."""
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to("cuda")
        model = EntQuantModel.from_pretrained(
            model=base,
            quantize=True,
            compress=False,
            weight_qtype=WEIGHT_QTYPE,
        )
        assert isinstance(model, EntQuantModel)

        with torch.no_grad():
            logits = model(**sample_inputs).logits
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

        del model
        torch.cuda.empty_cache()

    def test_byom_with_compression(self, sample_inputs):
        """BYOM with ANS compression produces valid compressed model."""
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to("cuda")
        model = EntQuantModel.from_pretrained(
            model=base,
            quantize=True,
            compress=True,
            weight_qtype=WEIGHT_QTYPE,
        )
        assert model.compressor is not None
        stats = model.compression_stats()
        assert stats["injected_ratio"] > 1.0

        with torch.no_grad():
            logits = model(**sample_inputs).logits
        assert not torch.isnan(logits).any()

        del model
        torch.cuda.empty_cache()

    def test_byom_logits_match_stream_build(self, sample_inputs):
        """BYOM logits match the regular stream-build quantize path."""
        # Stream-build path (reference)
        ref = EntQuantModel.from_pretrained(
            MODEL_ID,
            quantize=True,
            compress=False,
            weight_qtype=WEIGHT_QTYPE,
            dtype=DTYPE,
        )
        with torch.no_grad():
            logits_ref = ref(**sample_inputs).logits
        del ref
        torch.cuda.empty_cache()

        # BYOM path
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to("cuda")
        model = EntQuantModel.from_pretrained(
            model=base,
            quantize=True,
            compress=False,
            weight_qtype=WEIGHT_QTYPE,
        )
        with torch.no_grad():
            logits_byom = model(**sample_inputs).logits

        assert torch.allclose(logits_ref, logits_byom, atol=1e-4), (
            f"max diff = {(logits_ref - logits_byom).abs().max().item()}"
        )

        del model
        torch.cuda.empty_cache()

    def test_byom_save_produces_valid_checkpoint(self, tmp_path):
        """BYOM with save_dir produces a loadable checkpoint."""
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to("cuda")
        save_dir = tmp_path / "byom_saved"
        model = EntQuantModel.from_pretrained(
            model=base,
            quantize=True,
            compress=False,
            weight_qtype=WEIGHT_QTYPE,
            save_dir=str(save_dir),
        )

        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.safetensors.index.json").exists()
        assert (save_dir / "quanto_qmap.json").exists()

        del model
        torch.cuda.empty_cache()


class TestSuperWeightDetection:
    """Super weight detection (CPU only)."""

    def test_detect_returns_valid_coordinates(self):
        config = SuperWeightsConfig(
            include="*mlp.down_proj*",
            spike_threshold=50.0,
            top_k=5,
        )
        result = detect_super_weights(MODEL_ID, config=config, device_map="cpu")
        assert isinstance(result, dict)

        for module_name, sw_list in result.items():
            for sw in sw_list:
                assert isinstance(sw.row, int)
                assert isinstance(sw.col, int)

    def test_detect_fallback_layers_returns_set(self):
        config = SuperWeightsConfig(
            include="*mlp.down_proj*",
            spike_threshold=50.0,
            top_k=5,
        )
        result = detect_fallback_layers(MODEL_ID, config=config, device_map="cpu")
        assert isinstance(result, set)
        for name in result:
            assert isinstance(name, str)
