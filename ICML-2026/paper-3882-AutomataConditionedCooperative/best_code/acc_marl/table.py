import pandas as pd

# # read CSV
# df = pd.read_csv("storage/exp_test_results_n_10000.csv", sep=", ")

csv_files = [
    "storage/exp_test_results_n_1000_2buttons_2agents.csv",
    "storage/exp_test_results_n_1000_2rooms_2agents.csv",
    "storage/exp_test_results_n_1000_4buttons_4agents.csv",
    "storage/exp_test_results_n_1000_4rooms_4agents.csv",
]

dfs = []
for f in csv_files:
    df_temp = pd.read_csv(f, sep=", ", engine='python')
    dfs.append(df_temp)

# Concatenate into one dataframe
df = pd.concat(dfs, ignore_index=True)

# === Parse the "value +/- error" columns into numeric parts ===
def parse_val_err(cell):
    if pd.isna(cell):
        return None, None
    try:
        val, err = cell.split(" +/- ")
        return float(val), float(err)
    except Exception:
        return None, None

df[["SuccVal", "SuccErr"]] = df["Success Probability"].apply(
    lambda x: pd.Series(parse_val_err(x))
)

# === Human-readable labels for rows (policies) and columns (samplers + OOD) ===
policy_labels = {
    "rad_pbrs": "RAD Embd; PBRS",
    # "rad_no_pbrs": "RAD; no PBRS",
    "no_rad_pbrs": "no RAD Embd; PBRS",
    # "no_rad_no_pbrs": "no RAD; no PBRS",
}

col_labels = [
    ("R", False, "\\reach"),
    ("RA", False, "\\reachavoid"),
    ("RAD", False, "\\rad"),
    ("R", True, "\\reach (OOD)"),
    ("RA", True, "\\reachavoid (OOD)"),
    ("RAD", True, "\\rad (OOD)"),
]

config_labels = {
    "config/2buttons_2agents.yaml": "Buttons-2",
    "config/2rooms_2agents.yaml": "Rooms-2",
    "config/4buttons_4agents.yaml": "Buttons-4",
    "config/4rooms_4agents.yaml": "Rooms-4"
}

def format_val_err(val, err):
    if val is None:
        return "--"
    return f"{val:.3f} $\\pm$ {err:.3f}"

def make_table_all_configs_vertical_cline_configs():
    tex = []
    tex.append("\\begin{table*}[t]")
    tex.append("\\centering")
    tex.append("\\renewcommand{\\arraystretch}{1.5}")
    tex.append("\\setlength{\\tabcolsep}{4pt}")

    # Column preamble: 1 for vertical config, 1 for policy, rest for sampler columns
    tex.append("\\begin{tabular}{|p{15pt}|p{80pt}||" +
               ">{\\centering\\arraybackslash}m{50pt}|"*(len(col_labels)//2) +
               "|" +
               ">{\\centering\\arraybackslash}m{50pt}|"*(len(col_labels)//2) +
               "}")
    tex.append("\\hline")
    tex.append(f"\\multicolumn{{{2+len(col_labels)}}}{{|c|}}{{\\textbf{{Success Probability}}}}\\\\")
    tex.append("\\hline")
    header = " & ".join(["Env", "Policy"] + [lbl for _, _, lbl in col_labels]) + " \\\\"
    tex.append(header)
    tex.append("\\hline\\hline")

    for cidx, (config, config_label) in enumerate(config_labels.items()):
        df_config = df[df["Config"] == config]
        num_policies = len(policy_labels)

        for idx, (pol, pol_label) in enumerate(policy_labels.items()):
            row = []

            # Vertical config label using multirow, only on first policy
            if idx == 0:
                row.append(f"\\multirow{{{num_policies}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{config_label}}}}}}}")
            else:
                row.append("")

            # Policy label
            row.append(pol_label)
            df_pol = df_config[df_config["Policy"] == pol]

            # Sampler/OOD values
            for sampler, ood, _ in col_labels:
                cell_vals = []
                for assign in [False, True]:
                    df_entry = df_pol[
                        (df_pol["Sampler"] == sampler) &
                        (df_pol["OOD"] == ood) &
                        (df_pol["Assign"] == assign)
                    ]
                    if not df_entry.empty:
                        val = df_entry.iloc[0]["SuccVal"]
                        err = df_entry.iloc[0]["SuccErr"]
                        cell_vals.append(format_val_err(val, err))
                    else:
                        cell_vals.append("--")
                cell = "\\shortstack{\\strut " + " \\\\[-2pt] ".join(cell_vals) + "}"
                row.append(cell)

            # Add row
            tex.append(" & ".join(row) + " \\\\")

            # Draw horizontal line for policy + sampler columns
            col_start = 2  # policy column
            col_end = 1 + len(col_labels) + 1  # last column
            tex.append(f"\\cline{{{col_start}-{col_end}}}")

        # After each config, draw a full horizontal line across all columns
        if cidx < len(config_labels.items()) - 1:
            tex.append("\\hline\\hline")
        else:
            tex.append("\\hline")

    tex.append("\\end{tabular}")
    tex.append("\\caption{Results are for random and optimal assignments, respectively, and averaged over 5 seeds, each run for 1,000 episodes.}")
    tex.append("\\label{table:probs}")
    tex.append("\\end{table*}")

    return "\n".join(tex)


# === Generate the table ===
print(make_table_all_configs_vertical_cline_configs())
