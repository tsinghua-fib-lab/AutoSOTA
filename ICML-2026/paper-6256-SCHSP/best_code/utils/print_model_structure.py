#!/usr/bin/env python3

import timm

def print_model_structure():
    """Print the complete structure of the EVA02 model."""
    
    model_name = 'timm/eva02_base_patch14_448.mim_in22k_ft_in22k'
    print(f"🔍 Complete Model Structure for: {model_name}")
    print("=" * 100)
    
    # Load the model
    model = timm.create_model(model_name, pretrained=False)
    model.eval()
    
    print(f"📋 Full Model Structure:")
    print(model)
    
    print(f"\n" + "=" * 100)
    print(f"📋 Named Modules (recursive):")
    print("=" * 100)
    
    for name, module in model.named_modules():
        print(f"{name}: {type(module)}")
    
    print(f"\n" + "=" * 100)
    print(f"📋 Top-level Children:")
    print("=" * 100)
    
    for name, child in model.named_children():
        print(f"{name}: {type(child)}")
        print(f"   {child}")
        print("-" * 50)

if __name__ == "__main__":
    print_model_structure()
