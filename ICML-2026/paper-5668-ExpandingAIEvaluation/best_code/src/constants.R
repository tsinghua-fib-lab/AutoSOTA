# Defines LLM labels, featured LLMs for smaller figures, and color palettes

LLM_LABELS <- c(
  "phi_4" = "Phi 4",
  "o3_mini_medium" = "O3 Mini Medium",
  "o3_medium" = "O3 Medium",
  "o4_mini_medium" = "O4 Mini Medium",
  "gpt_4_1" = "GPT 4.1",
  "o1_mini_medium" = "O1 Mini Medium",
  "phi-4-mini-instruct" = "Phi-4 Mini Instruct",
  "o1_medium" = "O1 Medium",
  "o1_preview_medium" = "O1 Preview Medium",
  "gpt_4o_old" = "GPT-4o (Old)",
  "gpt_4o_new" = "GPT-4o (New)",
  "gpt_4o_mini" = "GPT-4o Mini",
  "claude_opus_4_16k" = "Claude 4 Opus (16k)",
  "claude_sonnet_4_32k" = "Claude 4 Sonnet (32k)",
  "claude_3_7_sonnet_32k" = "Claude 3.7 Sonnet (32k)",
  "claude_3_5_sonnet_old" = "Claude 3.5 Sonnet (Old)",
  "claude_3_5_sonnet_new" = "Claude 3.5 Sonnet (New)",
  "claude_3_haiku" = "Claude 3 Haiku",
  "Meta_llama_31_405" = "Llama 3.1 (40B)",
  "claude_3_sonnet" = "Claude 3 Sonnet",
  "claude_3_5_haiku" = "Claude 3.5 Haiku",
  "Meta_llama_3_8" = "Llama 3 (8B)",
  "gemini_25_pro_8192" = "Gemini 2.5 Pro (8192)",
  "gemini_25_flash_8192" = "Gemini 2.5 Flash (8192)",
  "gemini_20_flash" = "Gemini 2.0 Flash",
  "Meta_llama_31_70" = "Llama 3.1 (70B)",
  "Meta_llama_3_70" = "Llama 3 (70B)"
)

FEATURED_LLM_LABELS <- c(
  "Meta_llama_31_405" = "Llama 3.1 \n (40B)",
  "phi_4" = "Phi 4",
  "o4_mini_medium" = "O4 Mini \n Medium",
  "gemini_25_pro_8192" = "Gemini 2.5 \n Pro (8192)"
)

PALETTE <- palette.colors(palette = "Okabe-Ito")
GLMM_COLOR <- PALETTE[[7]]
REGRESSION_FREE_COLOR <- PALETTE[[6]]
SIMPLE_AVERAGE_COLOR <- PALETTE[[8]]

PLOT_COLORS <- c(
  "GLMM" = GLMM_COLOR,
  "Regression-Free" = REGRESSION_FREE_COLOR,
  "avg_single_epoch" = SIMPLE_AVERAGE_COLOR
)

PLOT_SHAPES <- c(
  "GLMM" = 16,
  "Regression-Free" = 17,
  "avg_single_epoch" = 15
)
