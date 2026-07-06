import unittest

from decoupledmarket.content import our_run_gpt_prompt as prompt_module


class TestPromptModuleSmoke(unittest.TestCase):
    def test_required_entrypoints_exist(self):
        required_funcs = [
            "analysis",
            "technical_analysis",
            "run_gpt_prompt_trading_stock",
            "run_llm_trading_stock",
            "pre_reflect",
            "long_reflect",
        ]
        for func_name in required_funcs:
            self.assertTrue(
                hasattr(prompt_module, func_name),
                f"missing function: {func_name}",
            )


if __name__ == "__main__":
    unittest.main()
