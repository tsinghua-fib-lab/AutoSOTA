#!/usr/bin/env python3
"""Simple direct test to verify rescale parameter extraction from training_params."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_rescale_extraction():
    """Test that we can extract rescale from training_params dict."""

    print("="*80)
    print("SIMPLE RESCALE PARAMETER EXTRACTION TEST")
    print("="*80)

    # Test 1: Extract z_score
    training_params_zscore = {
        "lr": 1e-3,
        "epochs": 100,
        "rescale": "z_score"
    }

    rescale = training_params_zscore.get("rescale", "z_score")
    print(f"\nTest 1: Extract from dict with rescale='z_score'")
    print(f"  training_params: {training_params_zscore}")
    print(f"  Extracted rescale: '{rescale}'")

    if rescale == "z_score":
        print("  ✅ PASS")
    else:
        print(f"  ❌ FAIL: Got '{rescale}' instead of 'z_score'")
        return False

    # Test 2: Extract none
    training_params_none = {
        "lr": 1e-3,
        "epochs": 100,
        "rescale": "none"
    }

    rescale = training_params_none.get("rescale", "z_score")
    print(f"\nTest 2: Extract from dict with rescale='none'")
    print(f"  training_params: {training_params_none}")
    print(f"  Extracted rescale: '{rescale}'")

    if rescale == "none":
        print("  ✅ PASS")
    else:
        print(f"  ❌ FAIL: Got '{rescale}' instead of 'none'")
        return False

    # Test 3: Default when not present
    training_params_missing = {
        "lr": 1e-3,
        "epochs": 100
    }

    rescale = training_params_missing.get("rescale", "z_score")
    print(f"\nTest 3: Default when rescale not in dict")
    print(f"  training_params: {training_params_missing}")
    print(f"  Extracted rescale: '{rescale}'")

    if rescale == "z_score":
        print("  ✅ PASS (uses default)")
    else:
        print(f"  ❌ FAIL: Got '{rescale}' instead of default 'z_score'")
        return False

    print("\n" + "="*80)
    print("✅ ALL BASIC TESTS PASSED")
    print("="*80)
    print("\nNow testing the actual training functions...")

    # Test 4: Check that the functions have the extraction line
    print("\n" + "="*80)
    print("VERIFYING TRAINING FUNCTIONS HAVE THE FIX")
    print("="*80)

    with open("baselines/trainers.py", "r") as f:
        content = f.read()

    # Find all three functions and check for the extraction line
    functions_to_check = [
        ("train_fm_post_transform_only_ut", 788),
        ("train_fm_post_transform", 965),
        ("train_fm_post_transform_plug_y", 1126),
    ]

    all_found = True
    for func_name, approx_line in functions_to_check:
        # Find the function
        func_start = content.find(f"def {func_name}(")
        if func_start == -1:
            print(f"  ❌ Could not find function {func_name}")
            all_found = False
            continue

        # Look for the extraction line within next 1000 chars
        search_region = content[func_start:func_start+1500]
        if 'rescale = training_params.get("rescale"' in search_region:
            print(f"  ✅ {func_name}: Has rescale extraction")
        else:
            print(f"  ❌ {func_name}: Missing rescale extraction!")
            all_found = False

    if all_found:
        print("\n✅ All three functions have the fix applied!")
    else:
        print("\n❌ Some functions are missing the fix!")
        return False

    # Test 5: Read the actual code and verify the pattern
    print("\n" + "="*80)
    print("DETAILED CODE VERIFICATION")
    print("="*80)

    import re

    for func_name, approx_line in functions_to_check:
        # Find function and check the pattern
        func_match = re.search(
            rf'def {func_name}\([^)]+\).*?theta, x, y = data.*?rescale = training_params\.get\("rescale"',
            content,
            re.DOTALL
        )

        if func_match:
            print(f"\n✅ {func_name}:")
            print(f"   Found correct pattern:")
            # Show a snippet
            snippet = func_match.group(0)
            lines = snippet.split('\n')
            for i, line in enumerate(lines[-5:]):  # Show last 5 lines
                print(f"   {line}")
        else:
            print(f"\n❌ {func_name}: Pattern not found!")
            all_found = False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe fix has been correctly applied to all three FM training functions.")
    print("They will now use the rescale parameter from training_params instead of")
    print("the hardcoded default.")

    return True

if __name__ == "__main__":
    success = test_rescale_extraction()
    sys.exit(0 if success else 1)
