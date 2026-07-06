#!/usr/bin/env python3
"""
Test New Architecture - Validate Refactored Code
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path so 'from src import ...' works
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src import (
    CONFIG,
    JailbreakEnvironment,
    JailbreakProbeRunner,
    LLMClient,
    CognitiveFitter,
    batch_fit_cognitive_model,
)


def test_config():
    """Test configuration system"""
    print("=" * 60)
    print("Test 1: Configuration System")
    print("=" * 60)

    assert 'alpha_pos' in CONFIG.default_params
    assert 'theta' in CONFIG.param_bounds
    assert 'BASELINE' in CONFIG.strategy_map

    print("✓ Configuration system normal")
    print(f"  Default parameters: {len(CONFIG.default_params)}")
    print(f"  Parameter bounds: {len(CONFIG.param_bounds)}")
    print(f"  Strategy mappings: {len(CONFIG.strategy_map)}")
    return True


def test_mab_environment():
    """Test MAB environment"""
    print("\n" + "=" * 60)
    print("Test 2: MAB Environment")
    print("=" * 60)

    env = JailbreakEnvironment(
        model_name="test-model",
        instruction="test instruction",
        num_trials=5,
        save_path=None
    )

    # Test scenario retrieval
    scenario_id = env.get_current_scenario_id()
    scenario = env.get_current_scenario()
    assert scenario_id in env.registry
    assert 'group' in scenario

    # Test step (simulated response)
    arm_mapping = {'Option A': 'Compliance', 'Option B': 'Refusal'}
    reward, action = env.step("Option A", arm_mapping)

    assert action in ['Compliance', 'Refusal']
    assert env.t == 1

    print("✓ MAB environment normal")
    print(f"  Scenarios: {len(env.registry)}")
    print(f"  Current trial: {env.t}")
    print(f"  Action: {action}, Reward: {reward}")
    return True


def test_llm_client():
    """Test LLM client"""
    print("\n" + "=" * 60)
    print("Test 3: LLM Client")
    print("=" * 60)

    # Mock mode
    client = LLMClient("ollama", "test-model", mock=True)
    response = client.call([{"role": "user", "content": "test"}])

    assert response in ["Option A", "Option B"]

    print("✓ LLM client normal")
    print(f"  Mock response: {response}")
    return True


def test_cognitive_fitter():
    """Test cognitive model fitter"""
    print("\n" + "=" * 60)
    print("Test 4: Cognitive Model Fitter")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'trial': range(1, n+1),
        'group': ['Baseline'] * 50 + ['Optimism'] * 50,
        'action': ['Refusal'] * 30 + ['Compliance'] * 20 + ['Refusal'] * 20 + ['Compliance'] * 30,
        'reward': np.random.choice([0.0, 1.0], n),
        'is_full_feedback': False,
        'forgone_reward': 0.0,
        'file_id': [1] * n
    })

    # Standard mode (now all modes use 9 parameters)
    fitter = CognitiveFitter(df)
    result = fitter.fit_scenario('Baseline', 'BASELINE')

    assert result is not None
    assert 'alpha_pos' in result.params
    assert result.nll > 0

    print("✓ CognitiveFitter normal")
    print(f"  NLL: {result.nll:.4f}")
    print(f"  BIC: {result.bic:.4f}")

    # All 9 parameters are always available
    result_ext = fitter.fit_scenario('Optimism', 'OPTIMISM')

    if result_ext:
        assert 'R_perc' in result_ext.params
        assert 'lambda_LA' in result_ext.params
        print("✓ All 9 parameters available")

    return True


def test_batch_fit():
    """Test batch fitting"""
    print("\n" + "=" * 60)
    print("Test 5: Batch Fitting")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'trial': list(range(1, 51)) * 4,
        'group': (['Baseline'] * 50 + ['Optimism'] * 50 +
                 ['Threat'] * 50 + ['Authority'] * 50),
        'action': ['Refusal', 'Compliance'] * 100,
        'reward': np.random.choice([0.0, 1.0], n),
        'is_full_feedback': False,
        'forgone_reward': 0.0,
        'file_id': [1] * n
    })

    results = batch_fit_cognitive_model(df, extended_mode=False)

    assert len(results) >= 3  # At least have some of Baseline, Optimism, Threat, Authority
    assert 'Baseline' in results

    print("✓ Batch fitting normal")
    print(f"  Processed scenario groups: {len(results)}")
    for group, result in results.items():
        print(f"    {group}: NLL={result.nll:.4f}")

    return True


def test_integration():
    """Test integration workflow"""
    print("\n" + "=" * 60)
    print("Test 6: Integration Workflow")
    print("=" * 60)

    # Simulate complete workflow
    # 1. Create environment
    env = JailbreakEnvironment("test-model", "explain account recovery safety", 5, None)

    # 2. Create client
    client = LLMClient("ollama", "test-model", mock=True)

    # 3. Create runner
    runner = JailbreakProbeRunner(client, "test-model", env)

    # 4. Run enough steps (at least 5 to fit)
    for _ in range(10):
        runner.run_step("explain account recovery safety")

    # 5. Get data
    df = pd.DataFrame(env.logs)

    # 6. Fit parameters
    from src.core.utils import prepare_for_fitting
    df_prepared = prepare_for_fitting(df)
    results = batch_fit_cognitive_model(df_prepared, extended_mode=False)

    # Must have at least Baseline to fit
    assert len(results) >= 1

    print("✓ Integration workflow normal")
    print(f"  Generated records: {len(df)}")
    print(f"  Fitting results: {len(results)} scenario groups")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("New Architecture Test Suite")
    print("="*70)

    tests = [
        ("Configuration System", test_config),
        ("MAB Environment", test_mab_environment),
        ("LLM Client", test_llm_client),
        ("Cognitive Fitter", test_cognitive_fitter),
        ("Batch Fitting", test_batch_fit),
        ("Integration Workflow", test_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {name} exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print("="*70)

    if failed == 0:
        print("\n🎉 All tests passed! New architecture working correctly.")
        return 0
    else:
        print(f"\n⚠ {failed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
