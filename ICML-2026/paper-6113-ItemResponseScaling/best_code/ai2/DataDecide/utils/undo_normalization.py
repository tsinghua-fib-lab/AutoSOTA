def undo_changes_to_df(df):
    """
    Reverts the changes made to the dataframe by the process_and_upload_results function.

    Args:
        df (pd.DataFrame): The modified dataframe.

    Returns:
        pd.DataFrame: The dataframe with changes reverted.
    """
    # Reverse column renaming
    df.rename(columns={"params": "model", "data": "group"}, inplace=True)

    # Reverse data name mapping
    DATA_NAME_CLEAN_REVERSE = {v: k for k, v in {
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
    }.items()}
    df['group'] = df['group'].map(DATA_NAME_CLEAN_REVERSE)

    # Reverse seed mapping
    SEED_MAPPING_REVERSE = {v: k for k, v in {
        6198: "default",
        14: "small aux 2",
        15: "small aux 3",
        4: "large aux 2",
        5: "large aux 3"
    }.items()}
    df['seed'] = df['seed'].map(SEED_MAPPING_REVERSE)

    return df