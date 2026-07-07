"""Smoke tests for the paper-style LAD FerretBackbone."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from model.ferret_backbone import FerretBackbone, LADMultiOperator


class TestPaperLADFerretBackbone(unittest.TestCase):
    def test_backbone_exposes_lad_operator_and_shapes(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(dim=96).eval()

        self.assertTrue(hasattr(model, "lad"), "FerretBackbone should expose a paper-style LADOperator")

        x = torch.rand(2, 3, 512, 512)
        with torch.no_grad():
            lad_map = model.lad(x)
            forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(lad_map.shape), (2, 1, 512, 512))
        self.assertEqual(tuple(forensic_features.shape), (2, 384, 64, 64))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))

        for name, tensor in {
            "lad_map": lad_map,
            "forensic_features": forensic_features,
            "coarse_mask": coarse_mask,
            "detection_logit": detection_logit,
        }.items():
            self.assertTrue(torch.isfinite(tensor).all().item(), f"{name} contains NaN/Inf")



    def test_lad_multi_operator_stacks_multiple_tau_maps(self) -> None:
        torch.manual_seed(0)
        op = LADMultiOperator(taus=(0.016, 0.064, 0.128))
        x = torch.rand(2, 3, 64, 64)

        with torch.no_grad():
            maps = op(x)

        self.assertEqual(tuple(maps.shape), (2, 3, 64, 64))
        self.assertTrue(torch.isfinite(maps).all().item())
        self.assertTrue(((maps >= 0.0) & (maps <= 1.0)).all().item())
        self.assertGreater(
            float((maps[:, 0:1] - maps[:, 1:2]).abs().mean()),
            1e-4,
            "different LAD taus should produce non-identical local-detail channels",
        )

    def test_lad_multi_backbone_uses_one_input_channel_per_tau(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.064, 0.128),
        ).eval()

        self.assertTrue(hasattr(model, "lad_multi"))
        self.assertEqual(model.forensic_operator, "lad_multi")
        self.assertEqual(model.cbr1[0].in_channels, 3)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            detail = model.lad_multi(x)
            forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(detail.shape), (2, 3, 128, 128))
        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertTrue(torch.isfinite(forensic_features).all().item())
        self.assertTrue(torch.isfinite(coarse_mask).all().item())

    def test_adaptive_tau_fusion_head_reweights_lad_multi_maps_without_breaking_identity_prompt(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="adaptive_tau_fusion_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.25,
            coarse_prompt_gate_max=0.5,
            coarse_prompt_area_bias=True,
        ).eval()

        self.assertTrue(hasattr(model, "lad_tau_fusion"))
        self.assertEqual(model.cbr1[0].in_channels, 4)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "adaptive tau fusion should initialize to the old dense-prompt behavior",
        )
        self.assertIsNotNone(model._last_lad_tau_weights)
        self.assertEqual(tuple(model._last_lad_tau_weights.shape), (2, 4, 128, 128))
        self.assertTrue(
            torch.allclose(
                model._last_lad_tau_weights,
                torch.ones_like(model._last_lad_tau_weights),
                atol=1e-6,
            ),
            "adaptive tau fusion must start as an exact per-tau identity multiplier",
        )

    def test_adaptive_tau_fusion_receives_gradients_from_dense_prompt_loss(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="adaptive_tau_fusion_multiscale",
            coarse_prompt_hidden=32,
        )

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, detection_logit, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() + detection_logit.mean() * 0.0
        loss.backward()

        tau_fusion_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "lad_tau_fusion" in name and param.requires_grad
        ]
        self.assertTrue(tau_fusion_grads, "adaptive tau fusion parameters should exist")
        self.assertTrue(
            any(
                grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0
                for grad in tau_fusion_grads
            ),
            "adaptive tau fusion should receive non-zero gradients from prompt losses",
        )

    def test_backbone_can_return_forensic_pyramid_for_adapters(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(dim=32, depths=[1, 1], lad_tau=0.064).eval()

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit, dense_prompt, pyramid = model(
                x,
                return_dense_prompt=True,
                return_forensic_pyramid=True,
            )

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertEqual(len(pyramid), 3)
        self.assertEqual(tuple(pyramid[0].shape), tuple(forensic_features.shape))
        self.assertEqual(tuple(pyramid[1].shape), (2, 16, 128, 128))
        self.assertEqual(tuple(pyramid[2].shape), (2, 32, 64, 64))

    def test_coarse_mask_head_outputs_logits_not_sigmoid_probabilities(self) -> None:
        model = FerretBackbone(dim=96).eval()
        final_conv = model.mask_compressor[-1]
        self.assertIsInstance(final_conv, torch.nn.Conv2d)
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(-2.0)

        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            _, coarse_mask, _ = model(x)

        self.assertLess(
            float(coarse_mask.mean()),
            0.0,
            "coarse_mask should be raw logits; a trailing sigmoid would make it non-negative",
        )

    def test_default_coarse_prompt_head_preserves_existing_shapes(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(dim=32, depths=[1, 1], lad_tau=0.064).eval()

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))

    def test_mldc_forensic_operator_uses_three_channel_detail_map(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="mldc",
        ).eval()

        self.assertTrue(hasattr(model, "mldc"))
        self.assertEqual(model.forensic_operator, "mldc")
        self.assertEqual(model.cbr1[0].in_channels, 3)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            detail = model.mldc(x)
            forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(detail.shape), (2, 3, 128, 128))
        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))

    def test_legacy_sam2_mldc_head_matches_old_checkpoint_shapes(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="mldc",
            mask_compressor_kernel_size=1,
            mask_compressor_output="sigmoid",
            legacy_logit_head=True,
        ).eval()

        self.assertEqual(tuple(model.mask_compressor[0].weight.shape), (32, 128, 1, 1))
        self.assertIsInstance(model.mask_compressor[-1], torch.nn.Sigmoid)
        self.assertIsInstance(model.logit[0], torch.nn.Dropout)
        self.assertEqual(tuple(model.logit[1].weight.shape), (1, 128))

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertGreaterEqual(float(coarse_mask.min()), 0.0)
        self.assertLessEqual(float(coarse_mask.max()), 1.0)

    def test_legacy_rgb_operator_keeps_three_channel_image_input_without_detail_module(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="rgb",
            mask_compressor_kernel_size=1,
            mask_compressor_output="sigmoid",
            legacy_logit_head=True,
        ).eval()

        self.assertFalse(hasattr(model, "mldc"))
        self.assertFalse(hasattr(model, "lad"))
        self.assertEqual(model.forensic_operator, "rgb")
        self.assertEqual(model.cbr1[0].in_channels, 3)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))

    def test_lad_mldc_hybrid_operator_keeps_lad_channels_and_adds_zero_init_mldc_fusion(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
        ).eval()

        self.assertTrue(hasattr(model, "lad_multi"))
        self.assertTrue(hasattr(model, "hybrid_mldc"))
        self.assertTrue(hasattr(model, "hybrid_mldc_high_proj"))
        self.assertTrue(hasattr(model, "hybrid_mldc_gate"))
        self.assertTrue(hasattr(model, "lad_tau_fusion"))
        self.assertEqual(model.forensic_operator, "lad_mldc_hybrid")
        self.assertEqual(model.cbr1[0].in_channels, 3)

        hybrid_final = model.hybrid_mldc_high_proj[-1]
        self.assertGreater(float(hybrid_final.weight.detach().abs().sum()), 0.0)
        self.assertLess(float(hybrid_final.weight.detach().abs().max()), 0.001)
        self.assertAlmostEqual(float(hybrid_final.bias.detach().abs().sum()), 0.0, places=7)
        self.assertAlmostEqual(float(model.hybrid_mldc_gate.weight.detach().abs().sum()), 0.0, places=7)
        self.assertAlmostEqual(float(model.hybrid_mldc_gate.bias.detach().abs().sum()), 0.0, places=7)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            detail = model.lad_multi(x)
            forensic_features, coarse_mask, detection_logit, dense_prompt = model(
                x,
                return_dense_prompt=True,
            )

        self.assertEqual(tuple(detail.shape), (2, 3, 128, 128))
        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertIsNotNone(model._last_hybrid_mldc_gate)
        self.assertEqual(tuple(model._last_hybrid_mldc_gate.shape), (2, 1, 128, 128))
        self.assertTrue(
            torch.allclose(
                model._last_hybrid_mldc_gate,
                torch.full_like(model._last_hybrid_mldc_gate, 0.5),
            )
        )

    def test_lad_mldc_hybrid_branch_receives_gradients_from_prompt_loss(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
        )
        with torch.no_grad():
            model.prompt_head_recall.weight.fill_(0.04)
            model.prompt_head_precision.weight.fill_(0.04)
            model.prompt_head_core.weight.fill_(0.02)

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() * 0.1
        loss.backward()

        checks = {
            "hybrid MLDC operator": lambda name: name.startswith("hybrid_mldc."),
            "hybrid high projection": lambda name: name.startswith("hybrid_mldc_high_proj."),
            "hybrid gate": lambda name: name.startswith("hybrid_mldc_gate."),
            "adaptive tau fusion": lambda name: name.startswith("lad_tau_fusion."),
            "prompt detail projection": lambda name: name.startswith("prompt_head_detail_proj."),
        }
        for label, predicate in checks.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if predicate(name) and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from prompt losses",
            )

    def test_fpn_highres_precision_recall_head_exposes_local_small_object_path(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="fpn_highres_precision_recall_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
        ).eval()

        for attr in (
            "prompt_head_fpn_mid_refine",
            "prompt_head_fpn_high_refine",
            "prompt_head_small_detail_proj",
            "prompt_head_small_gate",
        ):
            self.assertTrue(hasattr(model, attr), f"{attr} should exist")

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit, dense_prompt = model(
                x,
                return_dense_prompt=True,
            )

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertIsNotNone(model._last_dense_prompt_small_gate)
        self.assertEqual(tuple(model._last_dense_prompt_small_gate.shape[-2:]), (128, 128))
        self.assertTrue(torch.isfinite(model._last_dense_prompt_small_gate).all().item())

    def test_fpn_highres_precision_recall_head_receives_gradients(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="fpn_highres_precision_recall_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
        )
        with torch.no_grad():
            model.prompt_head_recall.weight.fill_(0.04)
            model.prompt_head_precision.weight.fill_(0.04)
            model.prompt_head_core.weight.fill_(0.02)

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() * 0.1
        loss.backward()

        checks = {
            "FPN mid refine": lambda name: name.startswith("prompt_head_fpn_mid_refine."),
            "FPN high refine": lambda name: name.startswith("prompt_head_fpn_high_refine."),
            "small detail projection": lambda name: name.startswith("prompt_head_small_detail_proj."),
            "small gate": lambda name: name.startswith("prompt_head_small_gate."),
        }
        for label, predicate in checks.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if predicate(name) and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from dense prompt losses",
            )

    def test_direct_signed_highres_prompt_head_exposes_bounded_residual_path(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="direct_signed_highres_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.40,
        ).eval()

        for attr in (
            "prompt_head_fpn_mid_refine",
            "prompt_head_fpn_high_refine",
            "prompt_head_small_detail_proj",
            "prompt_head_small_gate",
            "prompt_head_signed_delta",
            "prompt_head_signed_gate",
        ):
            self.assertTrue(hasattr(model, attr), f"{attr} should exist")

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit, dense_prompt = model(
                x,
                return_dense_prompt=True,
            )

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertIsNotNone(model._last_dense_prompt_signed_delta)
        self.assertIsNotNone(model._last_dense_prompt_signed_gate)
        self.assertEqual(tuple(model._last_dense_prompt_signed_delta.shape[-2:]), (128, 128))
        self.assertEqual(tuple(model._last_dense_prompt_signed_gate.shape[-2:]), (128, 128))
        self.assertLessEqual(float(model._last_dense_prompt_signed_delta.abs().max()), 0.4001)
        self.assertLessEqual(float(model._last_dense_prompt_signed_gate.max()), 0.5001)
        applied_residual = model._last_dense_prompt_signed_delta * model._last_dense_prompt_signed_gate
        self.assertLess(
            float(applied_residual.abs().mean()),
            0.005,
            "direct signed residual path should initialize as a near-identity dense-prompt correction",
        )

    def test_direct_signed_highres_prompt_head_receives_residual_gradients(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="direct_signed_highres_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.40,
        )

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = (dense_prompt - coarse_mask).square().mean() + dense_prompt.square().mean() * 0.01
        loss.backward()

        for label, needle in {
            "signed delta": "prompt_head_signed_delta",
            "signed gate": "prompt_head_signed_gate",
        }.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if needle in name and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from dense prompt losses",
            )

    def test_unet_highres_prompt_head_exposes_context_decoder_path(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="unet_highres_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.60,
        ).eval()

        for attr in (
            "prompt_head_unet_rgb_proj",
            "prompt_head_unet_mldc",
            "prompt_head_unet_mldc_proj",
            "prompt_head_unet_refine",
            "prompt_head_unet_delta",
            "prompt_head_unet_gate",
        ):
            self.assertTrue(hasattr(model, attr), f"{attr} should exist")

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            forensic_features, coarse_mask, detection_logit, dense_prompt = model(
                x,
                return_dense_prompt=True,
            )

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertIsNotNone(model._last_dense_prompt_unet_delta)
        self.assertIsNotNone(model._last_dense_prompt_unet_gate)
        self.assertEqual(tuple(model._last_dense_prompt_unet_delta.shape[-2:]), (128, 128))
        self.assertEqual(tuple(model._last_dense_prompt_unet_gate.shape[-2:]), (128, 128))
        self.assertTrue(torch.isfinite(model._last_dense_prompt_unet_delta).all().item())
        self.assertTrue(torch.isfinite(model._last_dense_prompt_unet_gate).all().item())
        self.assertLessEqual(float(model._last_dense_prompt_unet_delta.abs().max()), 0.6001)
        self.assertLessEqual(float(model._last_dense_prompt_unet_gate.max()), 0.5001)
        applied_residual = model._last_dense_prompt_unet_delta * model._last_dense_prompt_unet_gate
        self.assertLess(
            float(applied_residual.abs().mean()),
            0.01,
            "U-Net residual path should initialize near the existing dense prompt",
        )

    def test_unet_highres_prompt_head_receives_context_gradients(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="unet_highres_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.60,
        )

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = (dense_prompt - coarse_mask).square().mean() + dense_prompt.square().mean() * 0.01
        loss.backward()

        checks = {
            "RGB context projection": lambda name: name.startswith("prompt_head_unet_rgb_proj."),
            "MLDC context projection": lambda name: name.startswith("prompt_head_unet_mldc_proj."),
            "U-Net refinement": lambda name: name.startswith("prompt_head_unet_refine."),
            "U-Net delta": lambda name: name.startswith("prompt_head_unet_delta."),
            "U-Net gate": lambda name: name.startswith("prompt_head_unet_gate."),
        }
        for label, predicate in checks.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if predicate(name) and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from dense prompt losses",
            )

    def test_unet_residual_only_prompt_head_preserves_legacy_dense_prompt_path(self) -> None:
        torch.manual_seed(0)
        legacy = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
        ).eval()
        residual = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="unet_residual_only_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.60,
        ).eval()
        residual.load_state_dict(
            {
                name: tensor
                for name, tensor in legacy.state_dict().items()
                if name in residual.state_dict()
                and residual.state_dict()[name].shape == tensor.shape
            },
            strict=False,
        )

        for attr in (
            "prompt_head_unet_rgb_proj",
            "prompt_head_unet_mldc",
            "prompt_head_unet_mldc_proj",
            "prompt_head_unet_refine",
            "prompt_head_unet_delta",
            "prompt_head_unet_gate",
        ):
            self.assertTrue(hasattr(residual, attr), f"{attr} should exist")

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, _, _, legacy_dense = legacy(x, return_dense_prompt=True)
            _, _, _, residual_dense = residual(x, return_dense_prompt=True)

        self.assertIsNotNone(residual._last_dense_prompt_pre_unet)
        self.assertIsNotNone(residual._last_dense_prompt_unet_delta)
        self.assertIsNotNone(residual._last_dense_prompt_unet_gate)
        pre_unet = F.interpolate(
            residual._last_dense_prompt_pre_unet,
            size=residual_dense.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        self.assertTrue(
            torch.allclose(pre_unet, legacy_dense, atol=1e-4, rtol=1e-4),
            "Residual-only head must preserve the checkpoint-compatible legacy dense prompt before its delta",
        )
        self.assertLess(
            float((residual_dense - legacy_dense).abs().mean()),
            1e-3,
            "Residual-only U-Net branch should initialize as a tiny correction, not rewrite the legacy prompt",
        )

    def test_unet_residual_only_prompt_head_uses_separate_unet_scales(self) -> None:
        torch.manual_seed(0)
        legacy = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.60,
        ).eval()
        residual = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="unet_residual_only_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.60,
            coarse_prompt_unet_gate_init=0.01,
            coarse_prompt_unet_gate_max=0.10,
            coarse_prompt_unet_signed_residual_max_delta=0.15,
        ).eval()
        residual.load_state_dict(
            {
                name: tensor
                for name, tensor in legacy.state_dict().items()
                if name in residual.state_dict()
                and residual.state_dict()[name].shape == tensor.shape
            },
            strict=False,
        )

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, _, _, legacy_dense = legacy(x, return_dense_prompt=True)
            _, _, _, residual_dense = residual(x, return_dense_prompt=True)

        pre_unet = F.interpolate(
            residual._last_dense_prompt_pre_unet,
            size=residual_dense.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        self.assertTrue(
            torch.allclose(pre_unet, legacy_dense, atol=1e-4, rtol=1e-4),
            "Changing U-Net-only scales must not alter the loaded legacy dense prompt",
        )
        self.assertLessEqual(float(residual._last_dense_prompt_unet_gate.max()), 0.10 + 1e-6)
        self.assertLessEqual(float(residual._last_dense_prompt_unet_delta.abs().max()), 0.15 + 1e-6)

    def test_unet_residual_only_prompt_head_receives_context_gradients(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="unet_residual_only_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_signed_residual_max_delta=0.60,
        )

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = (dense_prompt - coarse_mask).square().mean() + dense_prompt.square().mean() * 0.01
        loss.backward()

        checks = {
            "RGB context projection": lambda name: name.startswith("prompt_head_unet_rgb_proj."),
            "MLDC context projection": lambda name: name.startswith("prompt_head_unet_mldc_proj."),
            "U-Net refinement": lambda name: name.startswith("prompt_head_unet_refine."),
            "U-Net delta": lambda name: name.startswith("prompt_head_unet_delta."),
            "U-Net gate": lambda name: name.startswith("prompt_head_unet_gate."),
        }
        for label, predicate in checks.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if predicate(name) and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from dense prompt losses",
            )

    def test_lad_mldc_hybrid_starts_close_to_lad_multi_after_loading_lad_state(self) -> None:
        torch.manual_seed(0)
        lad_model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_area_bias=True,
        ).eval()
        torch.manual_seed(1)
        hybrid_model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_mldc_hybrid",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_area_bias=True,
        ).eval()

        missing, unexpected = hybrid_model.load_state_dict(lad_model.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(key.startswith("hybrid_mldc") for key in missing))

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            lad_outputs = lad_model(x, return_dense_prompt=True)
            hybrid_outputs = hybrid_model(x, return_dense_prompt=True)

        for lad_tensor, hybrid_tensor in zip(lad_outputs, hybrid_outputs):
            self.assertLess(float((lad_tensor - hybrid_tensor).abs().mean()), 0.002)

    def test_multiscale_coarse_prompt_head_outputs_logits_and_trains(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="multiscale",
            coarse_prompt_hidden=32,
        )

        x = torch.rand(2, 3, 128, 128)
        forensic_features, coarse_mask, detection_logit = model(x)

        self.assertEqual(tuple(forensic_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(detection_logit.shape), (2, 1))
        self.assertTrue(coarse_mask.requires_grad)

        loss = coarse_mask.mean() + detection_logit.mean() * 0.0
        loss.backward()

        prompt_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "prompt_head" in name and param.requires_grad
        ]
        self.assertTrue(prompt_grads, "multiscale prompt_head parameters should exist")
        self.assertTrue(
            any(grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0 for grad in prompt_grads),
            "at least one multiscale prompt_head parameter should receive finite non-zero gradients",
        )

    def test_split_multiscale_head_separates_supervised_coarse_from_dense_prompt(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="split_multiscale",
            coarse_prompt_hidden=32,
        ).eval()

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (1, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "split head must initialize to the old coarse prompt behavior",
        )

        with torch.no_grad():
            model.prompt_head_fuse[-1].bias.fill_(1.0)
            _, coarse_after, _, dense_after = model(x, return_dense_prompt=True)

        self.assertTrue(
            torch.allclose(coarse_mask, coarse_after, atol=1e-6),
            "supervised coarse logits should stay on mask_compressor when prompt residual changes",
        )
        self.assertGreater(
            float((dense_after - coarse_after).abs().mean()),
            0.1,
            "dense prompt logits should be able to move independently for SAM prompting",
        )

    def test_gated_split_multiscale_head_starts_identity_and_limits_dense_residual(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="gated_split_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.25,
            coarse_prompt_gate_max=0.5,
            coarse_prompt_area_bias=True,
        ).eval()

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (1, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "gated split head must initialize to the old coarse prompt behavior",
        )

        with torch.no_grad():
            model.prompt_head_fuse[-1].bias.fill_(4.0)
            _, coarse_after, _, dense_after = model(x, return_dense_prompt=True)

        delta = dense_after - coarse_after
        self.assertTrue(torch.allclose(coarse_mask, coarse_after, atol=1e-6))
        self.assertGreater(float(delta.mean()), 0.9)
        self.assertLess(float(delta.mean()), 1.1)
        self.assertLessEqual(float(model._last_dense_prompt_gate.max()), 0.5 + 1e-6)

    def test_highres_gated_split_head_keeps_dense_prompt_residual_at_high_resolution(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="gated_split_multiscale_highres",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.25,
            coarse_prompt_gate_max=0.5,
            coarse_prompt_area_bias=True,
        ).eval()

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            detail = model.lad(x)
            high_feature = model.cbr1(detail)
            mid_feature = model.cbr2(high_feature)
            feature_map = model.final_conv(model.feature(mid_feature))
            coarse_logits, dense_logits = model._make_coarse_and_dense_prompt_logits(
                feature_map=feature_map,
                high_feature=high_feature,
                mid_feature=mid_feature,
            )

        self.assertEqual(tuple(coarse_logits.shape[-2:]), tuple(feature_map.shape[-2:]))
        self.assertEqual(
            tuple(dense_logits.shape[-2:]),
            tuple(high_feature.shape[-2:]),
            "high-res dense prompt head should not downsample the residual to the coarse feature map",
        )

        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (1, 1, 256, 256))
        self.assertLess(
            float((coarse_mask - dense_prompt).abs().mean()),
            0.005,
            "high-res split head should initialize very close to the old coarse prompt behavior",
        )

    def test_gated_split_multiscale_gate_feature_layer_is_not_zero_initialized(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="gated_split_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.25,
            coarse_prompt_gate_max=0.5,
        )

        first = model.prompt_head_gate[0]
        final = model.prompt_head_gate[-1]

        self.assertGreater(
            float(first.weight.detach().abs().sum()),
            0.0,
            "the dense-prompt gate feature layer must be learnable from step one",
        )
        self.assertGreater(
            float(final.weight.detach().abs().sum()),
            0.0,
            "a tiny non-zero final gate weight lets the dense-prompt gate vary by sample",
        )
        self.assertLess(float(final.weight.detach().abs().max()), 0.01)
        self.assertAlmostEqual(float(torch.sigmoid(final.bias.detach()).mean() * 0.5), 0.25, places=5)

    def test_gated_split_dense_gate_exposes_live_tensor_for_supervision(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="gated_split_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.25,
            coarse_prompt_gate_max=0.5,
        )

        x = torch.rand(2, 3, 128, 128)
        _, _, _, dense_prompt = model(x, return_dense_prompt=True)
        self.assertTrue(dense_prompt.requires_grad)
        self.assertIsNotNone(model._last_dense_prompt_gate)
        self.assertTrue(
            model._last_dense_prompt_gate.requires_grad,
            "dense_prompt_gate must stay attached so prompt_gate_supervision can train it",
        )

        loss = model._last_dense_prompt_gate.mean()
        loss.backward()
        gate_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "prompt_head_gate" in name and param.requires_grad
        ]
        self.assertTrue(
            any(grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0 for grad in gate_grads),
            "dense_prompt_gate supervision should produce non-zero gradients for prompt_head_gate",
        )

    def test_dual_branch_multiscale_head_starts_identity_and_exposes_live_branch_gates(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="dual_branch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.1,
            coarse_prompt_gate_max=0.5,
        )

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "dual-branch head must initialize to the old coarse prompt behavior",
        )

        for name in (
            "_last_dense_prompt_fg_gate",
            "_last_dense_prompt_bg_gate",
            "_last_dense_prompt_fg_residual",
            "_last_dense_prompt_bg_residual",
        ):
            value = getattr(model, name, None)
            self.assertIsNotNone(value, f"{name} should be exposed for diagnostics/supervision")
            self.assertTrue(torch.isfinite(value).all().item(), f"{name} contains NaN/Inf")

        self.assertTrue(model._last_dense_prompt_fg_gate.requires_grad)
        self.assertTrue(model._last_dense_prompt_bg_gate.requires_grad)
        self.assertLessEqual(float(model._last_dense_prompt_fg_gate.detach().max()), 0.5 + 1e-6)
        self.assertLessEqual(float(model._last_dense_prompt_bg_gate.detach().max()), 0.5 + 1e-6)

        gate_loss = model._last_dense_prompt_fg_gate.mean() + model._last_dense_prompt_bg_gate.mean()
        gate_loss.backward()
        gate_grads = [
            param.grad
            for name, param in model.named_parameters()
            if ("prompt_head_fg_gate" in name or "prompt_head_bg_gate" in name)
            and param.requires_grad
        ]
        self.assertTrue(
            any(
                grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0
                for grad in gate_grads
            ),
            "branch gate supervision should produce non-zero gradients for FG/BG gate parameters",
        )

    def test_dual_branch_residual_magnitudes_are_non_negative_even_for_negative_branch_logits(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="dual_branch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.1,
            coarse_prompt_gate_max=0.5,
        ).eval()
        with torch.no_grad():
            model.prompt_head_fg.bias.fill_(-2.0)
            model.prompt_head_bg.bias.fill_(-2.0)

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            model(x, return_dense_prompt=True)

        self.assertGreaterEqual(float(model._last_dense_prompt_fg_residual.min()), 0.0)
        self.assertGreaterEqual(float(model._last_dense_prompt_bg_residual.min()), 0.0)

    def test_signed_tribranch_multiscale_head_uses_spatial_fg_bg_gates(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="signed_tribranch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.1,
            coarse_prompt_gate_max=0.5,
        )

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "signed tribranch head must initialize to the old coarse prompt behavior",
        )

        self.assertIsNotNone(model._last_dense_prompt_fg_gate)
        self.assertIsNotNone(model._last_dense_prompt_bg_gate)
        self.assertGreater(
            model._last_dense_prompt_fg_gate.numel(),
            model._last_dense_prompt_fg_gate.shape[0],
            "FG gate should be spatial rather than one scalar per sample",
        )
        self.assertEqual(
            tuple(model._last_dense_prompt_fg_gate.shape[-2:]),
            tuple(model._last_dense_prompt_fg_residual.shape[-2:]),
        )
        self.assertTrue(model._last_dense_prompt_fg_gate.requires_grad)
        self.assertTrue(model._last_dense_prompt_bg_gate.requires_grad)
        self.assertLessEqual(float(model._last_dense_prompt_fg_gate.detach().max()), 0.5 + 1e-6)
        self.assertLessEqual(float(model._last_dense_prompt_bg_gate.detach().max()), 0.5 + 1e-6)

        loss = model._last_dense_prompt_fg_gate.mean() + model._last_dense_prompt_bg_gate.mean()
        loss.backward()
        gate_grads = [
            param.grad
            for name, param in model.named_parameters()
            if ("prompt_head_fg_gate" in name or "prompt_head_bg_gate" in name)
            and param.requires_grad
        ]
        self.assertTrue(
            any(
                grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0
                for grad in gate_grads
            ),
            "spatial branch gate supervision should train the spatial gate convolutions",
        )

    def test_signed_tribranch_multiscale_can_locally_expand_and_suppress(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="signed_tribranch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.2,
            coarse_prompt_gate_max=0.5,
        ).eval()

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
            model.prompt_head_fg.bias.fill_(2.0)
            _, coarse_expand, _, dense_expand = model(x, return_dense_prompt=True)
            model.prompt_head_fg.bias.zero_()
            model.prompt_head_bg.bias.fill_(2.0)
            _, coarse_suppress, _, dense_suppress = model(x, return_dense_prompt=True)

        self.assertTrue(torch.allclose(coarse_mask, dense_prompt, atol=1e-6))
        self.assertTrue(torch.allclose(coarse_mask, coarse_expand, atol=1e-6))
        self.assertTrue(torch.allclose(coarse_mask, coarse_suppress, atol=1e-6))
        self.assertGreater(float((dense_expand - coarse_expand).mean()), 0.1)
        self.assertLess(float((dense_suppress - coarse_suppress).mean()), -0.1)

    def test_highres_signed_tribranch_keeps_branch_maps_at_high_resolution(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            lad_tau=0.064,
            coarse_prompt_head="signed_tribranch_multiscale_highres",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.1,
            coarse_prompt_gate_max=0.5,
            coarse_prompt_area_bias=True,
        ).eval()

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            detail = model.lad(x)
            high_feature = model.cbr1(detail)
            mid_feature = model.cbr2(high_feature)
            feature_map = model.final_conv(model.feature(mid_feature))
            coarse_logits, dense_logits = model._make_coarse_and_dense_prompt_logits(
                feature_map=feature_map,
                high_feature=high_feature,
                mid_feature=mid_feature,
            )

        self.assertEqual(tuple(coarse_logits.shape[-2:]), tuple(feature_map.shape[-2:]))
        self.assertEqual(
            tuple(dense_logits.shape[-2:]),
            tuple(high_feature.shape[-2:]),
            "high-res signed tribranch should compose the dense prompt at high-feature resolution",
        )
        self.assertEqual(tuple(model._last_dense_prompt_fg_gate.shape[-2:]), tuple(high_feature.shape[-2:]))
        self.assertEqual(tuple(model._last_dense_prompt_bg_gate.shape[-2:]), tuple(high_feature.shape[-2:]))
        self.assertEqual(tuple(model._last_dense_prompt_core_gate.shape[-2:]), tuple(high_feature.shape[-2:]))
        self.assertEqual(tuple(model._last_dense_prompt_fg_residual.shape[-2:]), tuple(high_feature.shape[-2:]))
        self.assertEqual(tuple(model._last_dense_prompt_bg_residual.shape[-2:]), tuple(high_feature.shape[-2:]))
        self.assertEqual(tuple(model._last_dense_prompt_core_residual.shape[-2:]), tuple(high_feature.shape[-2:]))

        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (1, 1, 256, 256))
        self.assertLess(
            float((coarse_mask - dense_prompt).abs().mean()),
            0.005,
            "high-res signed tribranch should initialize very close to the old coarse prompt behavior",
        )

    def test_detail_guided_signed_tribranch_uses_lad_context_without_perturbing_resume(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064),
            coarse_prompt_head="detail_guided_signed_tribranch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.1,
            coarse_prompt_gate_max=0.5,
        ).eval()

        self.assertTrue(
            hasattr(model, "prompt_head_detail_proj"),
            "detail-guided head should expose a zero-initialized LAD-context projection",
        )
        last_detail_conv = model.prompt_head_detail_proj[-1]
        self.assertIsInstance(last_detail_conv, torch.nn.Conv2d)
        self.assertAlmostEqual(float(last_detail_conv.weight.detach().abs().sum()), 0.0, places=7)
        self.assertAlmostEqual(float(last_detail_conv.bias.detach().abs().sum()), 0.0, places=7)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "detail-guided signed tribranch must initialize to the old dense-prompt behavior",
        )
        self.assertIsNotNone(model._last_dense_prompt_fg_gate)
        self.assertIsNotNone(model._last_dense_prompt_bg_gate)
        self.assertIsNotNone(model._last_dense_prompt_core_gate)

    def test_detail_guided_signed_tribranch_detail_projection_can_receive_prompt_gradients(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064),
            coarse_prompt_head="detail_guided_signed_tribranch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.2,
            coarse_prompt_gate_max=0.5,
        )
        with torch.no_grad():
            model.prompt_head_core.weight.fill_(0.05)

        x = torch.rand(2, 3, 128, 128)
        _, _, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean()
        loss.backward()

        detail_grads = [
            param.grad
            for name, param in model.named_parameters()
            if "prompt_head_detail_proj" in name and param.requires_grad
        ]
        self.assertTrue(detail_grads, "detail context projection parameters should exist")
        self.assertTrue(
            any(
                grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0
                for grad in detail_grads
            ),
            "dense-prompt losses should be able to train the LAD-context projection",
        )

    def test_adaptive_detail_guided_signed_head_combines_tau_fusion_and_lad_context_safely(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="adaptive_detail_guided_signed_tribranch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.1,
            coarse_prompt_gate_max=0.5,
        ).eval()

        self.assertTrue(hasattr(model, "lad_tau_fusion"))
        self.assertTrue(hasattr(model, "prompt_head_detail_proj"))
        self.assertEqual(model.cbr1[0].in_channels, 4)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "adaptive detail-guided signed head should initialize to the resumed checkpoint behavior",
        )
        self.assertIsNotNone(model._last_lad_tau_weights)
        self.assertTrue(
            torch.allclose(
                model._last_lad_tau_weights,
                torch.ones_like(model._last_lad_tau_weights),
                atol=1e-6,
            ),
            "adaptive tau fusion must start as an exact per-tau identity multiplier",
        )
        self.assertIsNotNone(model._last_dense_prompt_fg_gate)
        self.assertIsNotNone(model._last_dense_prompt_bg_gate)
        self.assertIsNotNone(model._last_dense_prompt_core_gate)

    def test_adaptive_detail_guided_signed_head_trains_tau_fusion_and_detail_context(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="adaptive_detail_guided_signed_tribranch_multiscale",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.2,
            coarse_prompt_gate_max=0.5,
        )
        with torch.no_grad():
            model.prompt_head_core.weight.fill_(0.05)

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() * 0.1
        loss.backward()

        for label, needle in {
            "adaptive tau fusion": "lad_tau_fusion",
            "detail context projection": "prompt_head_detail_proj",
        }.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if needle in name and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from prompt losses",
            )

    def test_precision_recall_adaptive_prompt_head_exposes_local_router_and_endpoints(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.15,
            coarse_prompt_gate_max=0.6,
            coarse_prompt_area_bias=True,
        ).eval()

        self.assertTrue(hasattr(model, "lad_tau_fusion"))
        self.assertTrue(hasattr(model, "prompt_head_detail_proj"))
        self.assertTrue(hasattr(model, "prompt_head_precision"))
        self.assertTrue(hasattr(model, "prompt_head_recall"))
        self.assertTrue(hasattr(model, "prompt_head_router_gate"))

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "new precision/recall head should initialize without perturbing resumed prompt behavior",
        )
        self.assertIsNotNone(model._last_lad_tau_weights)
        self.assertTrue(torch.allclose(model._last_lad_tau_weights, torch.ones_like(model._last_lad_tau_weights)))
        self.assertIsNotNone(model._last_dense_prompt_fg_gate)
        self.assertIsNotNone(model._last_dense_prompt_bg_gate)
        self.assertIsNotNone(model._last_dense_prompt_core_gate)
        self.assertIsNotNone(model._last_dense_prompt_fg_residual)
        self.assertIsNotNone(model._last_dense_prompt_bg_residual)
        self.assertEqual(tuple(model._last_dense_prompt_fg_gate.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(model._last_dense_prompt_bg_gate.shape), (2, 1, 128, 128))

    def test_precision_recall_adaptive_prompt_head_trains_tau_detail_router_and_endpoints(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="precision_recall_adaptive_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.2,
            coarse_prompt_gate_max=0.5,
        )
        with torch.no_grad():
            model.prompt_head_recall.weight.fill_(0.04)
            model.prompt_head_precision.weight.fill_(0.04)
            model.prompt_head_core.weight.fill_(0.02)

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() * 0.1
        loss.backward()

        for label, needle in {
            "adaptive tau fusion": "lad_tau_fusion",
            "detail context projection": "prompt_head_detail_proj",
            "recall endpoint": "prompt_head_fg",
            "precision endpoint": "prompt_head_bg",
            "local router": "prompt_head_router_gate",
        }.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if needle in name and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from prompt losses",
            )

    def test_uncertainty_guided_precision_recall_head_preserves_resume_and_exposes_context(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="uncertainty_guided_precision_recall_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.15,
            coarse_prompt_gate_max=0.6,
            coarse_prompt_area_bias=True,
        ).eval()

        self.assertTrue(hasattr(model, "lad_tau_fusion"))
        self.assertTrue(hasattr(model, "prompt_head_detail_proj"))
        self.assertTrue(hasattr(model, "prompt_head_uncertainty_proj"))
        self.assertTrue(hasattr(model, "prompt_head_uncertainty_balance_gate"))
        self.assertTrue(hasattr(model, "prompt_head_router_gate"))

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertTrue(
            torch.allclose(coarse_mask, dense_prompt, atol=1e-6),
            "uncertainty-guided head should initialize without perturbing resumed prompt behavior",
        )
        self.assertIsNotNone(model._last_dense_prompt_fg_gate)
        self.assertIsNotNone(model._last_dense_prompt_bg_gate)
        self.assertEqual(tuple(model._last_dense_prompt_fg_gate.shape), (2, 1, 128, 128))

    def test_uncertainty_guided_precision_recall_head_trains_context_adapter(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="uncertainty_guided_precision_recall_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.2,
            coarse_prompt_gate_max=0.5,
        )
        with torch.no_grad():
            model.prompt_head_recall.weight.fill_(0.04)
            model.prompt_head_precision.weight.fill_(0.04)
            model.prompt_head_core.weight.fill_(0.02)

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() * 0.1
        loss.backward()

        for label, needle in {
            "uncertainty context adapter": "prompt_head_uncertainty_proj",
            "uncertainty balance gate": "prompt_head_uncertainty_balance_gate",
            "adaptive tau fusion": "lad_tau_fusion",
            "detail context projection": "prompt_head_detail_proj",
        }.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if needle in name and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from prompt losses",
            )

    def test_contextual_highres_precision_recall_head_uses_rgb_mldc_context_and_stays_near_resume(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="contextual_highres_precision_recall_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.10,
            coarse_prompt_gate_max=0.50,
            coarse_prompt_area_bias=True,
        ).eval()

        self.assertTrue(hasattr(model, "lad_tau_fusion"))
        self.assertTrue(hasattr(model, "prompt_head_detail_proj"))
        self.assertTrue(hasattr(model, "prompt_head_rgb_proj"))
        self.assertTrue(hasattr(model, "prompt_head_mldc"))
        self.assertTrue(hasattr(model, "prompt_head_mldc_proj"))
        self.assertTrue(hasattr(model, "prompt_head_router_gate"))

        rgb_final = model.prompt_head_rgb_proj[-1]
        mldc_final = model.prompt_head_mldc_proj[-1]
        self.assertAlmostEqual(float(rgb_final.weight.detach().abs().sum()), 0.0, places=7)
        self.assertAlmostEqual(float(mldc_final.weight.detach().abs().sum()), 0.0, places=7)

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            detail = model.lad_multi(x)
            if hasattr(model, "lad_tau_fusion"):
                detail, _ = model.lad_tau_fusion(detail)
            high_feature = model.cbr1(detail)
            mid_feature = model.cbr2(high_feature)
            feature_map = model.final_conv(model.feature(mid_feature))
            coarse_logits, dense_logits = model._make_coarse_and_dense_prompt_logits(
                feature_map=feature_map,
                high_feature=high_feature,
                mid_feature=mid_feature,
                detail_map=detail,
                image_context=x,
            )

        self.assertEqual(tuple(coarse_logits.shape[-2:]), tuple(feature_map.shape[-2:]))
        self.assertEqual(
            tuple(dense_logits.shape[-2:]),
            tuple(high_feature.shape[-2:]),
            "contextual PR head should compose the SAM dense prompt at high-feature resolution",
        )
        self.assertEqual(tuple(model._last_dense_prompt_fg_gate.shape[-2:]), tuple(high_feature.shape[-2:]))
        self.assertEqual(tuple(model._last_dense_prompt_bg_gate.shape[-2:]), tuple(high_feature.shape[-2:]))

        with torch.no_grad():
            _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)

        self.assertEqual(tuple(coarse_mask.shape), (2, 1, 256, 256))
        self.assertEqual(tuple(dense_prompt.shape), (2, 1, 256, 256))
        self.assertLess(
            float((coarse_mask - dense_prompt).abs().mean()),
            0.005,
            "contextual high-res PR head should initialize near the old prompt behavior",
        )

    def test_contextual_highres_precision_recall_head_trains_context_paths(self) -> None:
        torch.manual_seed(0)
        model = FerretBackbone(
            dim=32,
            depths=[1, 1],
            forensic_operator="lad_multi",
            lad_multi_taus=(0.016, 0.032, 0.064, 0.128),
            coarse_prompt_head="contextual_highres_precision_recall_prompt_head",
            coarse_prompt_hidden=32,
            coarse_prompt_gate_init=0.20,
            coarse_prompt_gate_max=0.50,
        )
        with torch.no_grad():
            model.prompt_head_recall.weight.fill_(0.04)
            model.prompt_head_precision.weight.fill_(0.04)
            model.prompt_head_core.weight.fill_(0.02)

        x = torch.rand(2, 3, 128, 128)
        _, coarse_mask, _, dense_prompt = model(x, return_dense_prompt=True)
        loss = dense_prompt.square().mean() + coarse_mask.square().mean() * 0.1
        loss.backward()

        for label, needle in {
            "adaptive tau fusion": "lad_tau_fusion",
            "detail context projection": "prompt_head_detail_proj",
            "RGB context projection": "prompt_head_rgb_proj",
            "MLDC context projection": "prompt_head_mldc_proj",
            "local router": "prompt_head_router_gate",
        }.items():
            grads = [
                param.grad
                for name, param in model.named_parameters()
                if needle in name and param.requires_grad
            ]
            self.assertTrue(grads, f"{label} parameters should exist")
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all().item()
                    and grad.abs().sum().item() > 0
                    for grad in grads
                ),
                f"{label} should receive non-zero gradients from prompt losses",
            )


if __name__ == "__main__":
    unittest.main()
