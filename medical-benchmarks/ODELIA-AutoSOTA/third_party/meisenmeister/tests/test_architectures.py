from __future__ import annotations

import unittest

import torch

from meisenmeister.architectures import (
    BaseArchitecture,
    PrimusMClsNetwork,
    ResidualEncoderClsNetwork,
    ResNet3D18,
    get_architecture_class,
    get_architecture_registry,
)


class ArchitectureTests(unittest.TestCase):
    def test_architecture_registry_finds_resnet3d18(self) -> None:
        registry = get_architecture_registry()

        self.assertIn("ResNet3D18", registry)
        self.assertIn("ResidualEncoderClsNetwork", registry)
        self.assertIn("PrimusMClsNetwork", registry)

    def test_unknown_architecture_error_lists_available_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown architecture 'missing'.*PrimusMClsNetwork.*ResNet3D18.*ResidualEncoderClsNetwork",
        ):
            get_architecture_class("missing")

    def test_resnet3d18_satisfies_base_interface(self) -> None:
        architecture = ResNet3D18(in_channels=3, num_classes=2)

        self.assertIsInstance(architecture, BaseArchitecture)

    def test_resnet3d18_forward_shape(self) -> None:
        architecture = ResNet3D18(in_channels=3, num_classes=4)
        x = torch.randn(2, 3, 32, 32, 32)

        y = architecture(x)

        self.assertEqual(tuple(y.shape), (2, 4))

    def test_residual_encoder_cls_network_satisfies_base_interface(self) -> None:
        architecture = ResidualEncoderClsNetwork(in_channels=1, num_classes=2)

        self.assertIsInstance(architecture, BaseArchitecture)

    def test_residual_encoder_cls_network_forward_shape(self) -> None:
        architecture = ResidualEncoderClsNetwork(in_channels=1, num_classes=3)
        x = torch.randn(2, 1, 32, 32, 32)

        y = architecture(x)

        self.assertEqual(tuple(y.shape), (2, 3))

    def test_primus_m_cls_network_satisfies_base_interface(self) -> None:
        architecture = PrimusMClsNetwork(
            in_channels=1,
            num_classes=2,
            input_shape=(16, 16, 16),
        )

        self.assertIsInstance(architecture, BaseArchitecture)

    def test_primus_m_cls_network_forward_shape(self) -> None:
        architecture = PrimusMClsNetwork(
            in_channels=1,
            num_classes=3,
            input_shape=(16, 16, 16),
        )
        x = torch.randn(1, 1, 16, 16, 16)

        y = architecture(x)

        self.assertEqual(tuple(y.shape), (1, 3))

    def test_primus_m_cls_network_grad_cam_not_supported(self) -> None:
        architecture = PrimusMClsNetwork(
            in_channels=1,
            num_classes=2,
            input_shape=(16, 16, 16),
        )

        with self.assertRaisesRegex(NotImplementedError, "Grad-CAM is not available"):
            architecture.get_grad_cam_target_layer()

    def test_residual_encoder_cls_network_loads_nnunet_encoder_weights(self) -> None:
        architecture = ResidualEncoderClsNetwork(in_channels=1, num_classes=2)
        source_state = architecture.state_dict()
        pretrained_state = {}
        source_weight_key = "encoder.stages.0.blocks.0.conv1.conv.weight"
        source_bias_key = "encoder.stages.0.blocks.0.conv1.norm.weight"
        pretrained_state[f"network.{source_weight_key}"] = torch.full_like(
            source_state[source_weight_key], 0.125
        )
        pretrained_state[f"network.{source_bias_key}"] = torch.full_like(
            source_state[source_bias_key], 0.75
        )
        pretrained_state["network.decoder.fake.weight"] = torch.ones(1)

        with unittest.mock.patch(
            "torch.load",
            return_value={"state_dict": pretrained_state},
        ):
            architecture.load_initial_weights(
                path=unittest.mock.Mock(is_file=lambda: True),
                device=torch.device("cpu"),
            )

        loaded_state = architecture.state_dict()
        self.assertTrue(
            torch.allclose(
                loaded_state[source_weight_key],
                torch.full_like(loaded_state[source_weight_key], 0.125),
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_state[source_bias_key],
                torch.full_like(loaded_state[source_bias_key], 0.75),
            )
        )


if __name__ == "__main__":
    unittest.main()
