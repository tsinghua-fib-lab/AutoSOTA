DATA_NAME_LATEX = {
    "dolma17": r"\dolmaSeventeen{}",
    "no_code": r"\noCode{}",
    "no_math_no_code": r"\noMathNoCode{}",
    "no_reddit": r"\noReddit{}",
    "no_flan": r"\noFlan{}",
    "dolma-v1-6-and-sources-baseline": r"\dolmaVSixteenAndSourcesBaseline{}",
    "c4": r"\cFour{}",
    "prox_fineweb_pro": r"\proxFinewebPro{}",
    "fineweb_edu_dedup": r"\finewebEduDedup{}",
    "falcon": r"\falcon{}",
    "falcon_and_cc": r"\falconAndCc{}",
    "falcon_and_cc_eli5_oh_top10p": r"\falconAndCcEliFiveOhTopTenP{}",
    "falcon_and_cc_eli5_oh_top20p": r"\falconAndCcEliFiveOhTopTwentyP{}",
    "falcon_and_cc_og_eli5_oh_top10p": r"\falconAndCcOgEliFiveOhTopTenP{}",
    "falcon_and_cc_tulu_qc_top10": r"\falconAndCcTuluQcTopTen{}",
    "DCLM-baseline": r"\DCLMBaseline{}",
    "dolma17-75p-DCLM-baseline-25p": r"\dolmaSeventeenSeventyFivePDCLMBaselineTwentyFiveP{}",
    "dolma17-50p-DCLM-baseline-50p": r"\dolmaSeventeenFiftyPDCLMBaselineFiftyP{}",
    "dolma17-25p-DCLM-baseline-75p": r"\dolmaSeventeenTwentyFivePDCLMBaselineSeventyFiveP{}",
    "dclm_ft7percentile_fw2": r"\dclmFtSevenPercentileFwTwo{}",
    "dclm_ft7percentile_fw3": r"\dclmFtSevenPercentileFwThree{}",
    "dclm_fw_top10": r"\dclmFwTopTen{}",
    "dclm_fw_top3": r"\dclmFwTopThree{}",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": r"\posEliFiveOhNegDclmRefinedwebStepsTwoThousandLrThreeEFourTopTenP{}",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": r"\posEliFiveOhNegDclmRefinedwebStepsTwoThousandLrThreeEFourTopTwentyP{}"
}

DATA_NAME_CLEAN = {
    # "dolma": "Dolma1.7",
    "dolma17": "Dolma1.7",
    "no_code": "Dolma1.7 (no code)", 
    "no_math_no_code": "Dolma1.7 (no math, code)",
    "no_reddit": "Dolma1.7 (no Reddit)",
    "no_flan": "Dolma1.7 (no Flan)",
    "dolma-v1-6-and-sources-baseline": "Dolma1.6++",
    "c4": "C4",
    "prox_fineweb_pro": "FineWeb-Pro",
    "fineweb_edu_dedup": "FineWeb-Edu", 
    "falcon": "Falcon",
    "falcon_and_cc": "Falcon+CC",
    "falcon_and_cc_eli5_oh_top10p": "Falcon+CC (QC 10%)",
    "falcon_and_cc_eli5_oh_top20p": "Falcon+CC (QC 20%)",
    "falcon_and_cc_og_eli5_oh_top10p": "Falcon+CC (QC Orig 10%)",
    "falcon_and_cc_tulu_qc_top10": "Falcon+CC (QC Tulu 10%)",
    "DCLM-baseline": "DCLM-Baseline",
    "dolma17-75p-DCLM-baseline-25p": "DCLM-Baseline 25% / Dolma 75%",
    "dolma17-50p-DCLM-baseline-50p": "DCLM-Baseline 50% / Dolma 50%", 
    "dolma17-25p-DCLM-baseline-75p": "DCLM-Baseline 75% / Dolma 25%",
    "dclm_ft7percentile_fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm_ft7percentile_fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm_fw_top10": "DCLM-Baseline (QC FW 10%)",
    "dclm_fw_top3": "DCLM-Baseline (QC FW 3%)",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "DCLM-Baseline (QC 10%)",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "DCLM-Baseline (QC 20%)",
}


PREFIXES = [
    '\\midrule\n\\textbf{\\citet{bhagia2024establishingtaskscalinglaws} using all models} & ~ & ~ \\\\\n',
    '\\midrule\n\\textbf{\\citet{bhagia2024establishingtaskscalinglaws} without 750M} & ~ & ~ \\\\\n',
    '\\midrule\n\\textbf{\\citet{bhagia2024establishingtaskscalinglaws} without 530M, 750M} & ~ & ~ \\\\\n',
]

SETUP_NAME_LATEX = {
    '3_param-default':                         PREFIXES[0] + 'FLOPs $\\rightarrow$ Task Loss $\\rightarrow$ Metric (2 step, FLOPs)',
    '3_param-default-helper_points':           '  + helper point',
    # '3_param-default-step2=0.5':               '  + using final 50% of checkpoints for step 2 prediction',
    # '3_param-default-helper_points-step2=0.5': '  + helper point + final 50% of checkpoints',
    '2_param-default':                         '  + remove the irreducible error term $E$',
    '3_param-1_step':                          'FLOPs $\\rightarrow$ Metric (1 step)',
    '5_param-ai2':                             '$(N, D)$ $\\rightarrow$ Task Loss $\\rightarrow$ Metric (2 step, $(N, D)$)',
    # '5_param-1_step-ai2':                      '$(N, D)$ $\\rightarrow$ Metric (1 step, $(N, D)$)',
    # '3_param-intermediate-default':            '$(N, D)$ $\\rightarrow$ Metric $\\rightarrow$ Primary metric',
    # '3_param-intermediate-default-helper_points': '  + helper point',
    
    # No 750M point
    '3_param-no_750M':                         PREFIXES[1] + 'FLOPs $\\rightarrow$ Task Loss $\\rightarrow$ Metric (2 step, FLOPs)',
    '3_param-no_750M-helper_points':           '  + helper point',
    # '3_param-no_750M-step2=0.5':               '  + using final 50% of checkpoints for step 2 prediction',
    # '3_param-no_750M-helper_points-step2=0.5': '  + helper point + final 50% of checkpoints',
    '2_param-no_750M':                         '  + remove the irreducible error term $E$',
    '3_param-1_step-no_750M':                  'FLOPs $\\rightarrow$ Metric (1 step)',
    '5_param-ai2-no_750M':                     '$(N, D)$ $\\rightarrow$ Task Loss $\\rightarrow$ Metric (2 step, $(N, D)$)',
    # '5_param-1_step-ai2-no_750M':              '$(N, D)$ $\\rightarrow$ Metric (1 step, $(N, D)$)',
    # '3_param-intermediate-no_750M':            '$(N, D)$ $\\rightarrow$ Metric $\\rightarrow$ Primary metric',
    # '3_param-intermediate-no_750M-helper_points': '  + helper point',
    
    # No 750M and 530M point
    '3_param-no_750M_no_530M':                 PREFIXES[2] + 'FLOPs $\\rightarrow$ Task Loss $\\rightarrow$ Metric (2 step, FLOPs)',
    '3_param-no_750M_no_530M-helper_points':   '  + helper point',
    # '3_param-no_750M_no_530M-step2=0.5':       '  + using final 50% of checkpoints for step 2 prediction',
    # '3_param-no_750M_no_530M-helper_points-step2=0.5': '  + helper point + final 50% of checkpoints',
    '2_param-no_750M_no_530M':                 '  + remove the irreducible error term $E$',
    '3_param-1_step-no_750M_no_530M':          'FLOPs $\\rightarrow$ Metric (1 step)',
    '5_param-ai2-no_750M_no_530M':             '$(N, D)$ $\\rightarrow$ Task Loss $\\rightarrow$ Metric (2 step, $(N, D)$)',
    # '5_param-1_step-ai2-no_750M_no_530M':      '$(N, D)$ $\\rightarrow$ Metric (1 step, $(N, D)$)',
    # '3_param-intermediate-no_750M_no_530M':    '$(N, D)$ $\\rightarrow$ Metric $\\rightarrow$ Primary metric',
    # '3_param-intermediate-no_750M_no_530M-helper_points': '  + helper point',
}

TASK_NAME_LATEX = {
    'olmes_10_macro_avg': 'OLMES Avg.',
    'mmlu': 'MMLU',
    'arc_challenge': 'ARC-Challenge',
    'arc_easy': 'ARC-Easy',
    'boolq': 'BoolQ',
    'csqa': 'CommonsenseQA',
    'hellaswag': 'HellaSwag',
    'openbookqa': 'OpenBookQA',
    'piqa': 'PIQA',
    'socialiqa': 'SocialIQA',
    'winogrande': 'WinoGrande'
}


# Two-class prediction results
MIX_TWO_CLASS_RESULTS = {
    'mmlu': 0.90590395480226,
    'arc_challenge': 0.9156497175141245,
    'hellaswag': 0.8522598870056499,
    'piqa': 0.7932768361581923,
    'boolq': 0.5670056497175143,
    'openbookqa': 0.8060169491525425,
    'winogrande': 0.7324011299435031,
    'csqa': 0.7608192090395481,
    'arc_easy': 0.9374011299435027,
    'socialiqa': 0.7687853107344634
}