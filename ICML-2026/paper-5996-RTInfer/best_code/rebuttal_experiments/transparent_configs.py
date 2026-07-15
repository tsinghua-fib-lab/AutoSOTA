from __future__ import annotations

from common import PRUNING_TIERS


CONFIGS = [
    ("service_robot", "MobileNetv2-SSDLite", "[3, 300, 300]", 1, 2),
    ("service_robot", "ResNet50", "[3, 64, 64]", 1, 2),
    ("service_robot", "VGG16", "[3, 32, 32]", 1, 4),
    ("service_robot", "ResNet152", "[1, 48, 48]", 1, 4),
    ("uav_ground_station", "MobileNetv2-SSDLite", "[3, 300, 300]", 1, 2),
    ("uav_ground_station", "GoogLeNet", "[3, 64, 64]", 1, 4),
    ("uav_ground_station", "ResNet50", "[1, 128, 128]", 1, 4),
    ("smart_traffic", "MobileNetv2-SSDLite", "[3, 300, 300]", 1, 4),
    ("smart_traffic", "MobileNetv2", "[3, 32, 32]", 1, 8),
]


def main() -> None:
    print("application,model,input_shape,batch_size,early_exits,pruning_tiers")
    tiers = "|".join(str(tier) for tier in PRUNING_TIERS)
    for row in CONFIGS:
        print(",".join(str(value) for value in (*row, tiers)))
    print("\nlimitation_note")
    print(
        "The CNN-only memory-pressure setup relies on non-standard stress-test configurations; "
        "the stronger practical motivation is mixed modern CNN/ViT/LLM workloads with high-resolution inputs."
    )


if __name__ == "__main__":
    main()

