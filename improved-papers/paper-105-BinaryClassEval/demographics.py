import pandas as pd
import numpy as np
from pathlib import Path

def generate_table1(csv_file_path, apache_file_path):
    """
    Generate Table 1: Demographics and outcomes by sex and race/ethnicity
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the eICU patient CSV file
    apache_file_path : str
        Path to the APACHE patient results CSV file
    """
    
    # Read the data
    df = pd.read_csv(csv_file_path)
    
    # Read APACHE scores
    apache_df = pd.read_csv(apache_file_path)
    
    # Join with APACHE scores (keep only relevant columns and take first APACHE score per patient)
    apache_scores = apache_df[['patientunitstayid', 'apachescore']].groupby('patientunitstayid').first().reset_index()
    df = df.merge(apache_scores, on='patientunitstayid', how='left')
    
    # Create binary mortality outcome (died in hospital)
    df['hospital_mortality'] = (df['hospitaldischargestatus'] == 'Expired').astype(int)
    df['unit_mortality'] = (df['unitdischargestatus'] == 'Expired').astype(int)
    
    # Use ethnicity as-is from the data
    df['ethnicity_clean'] = df['ethnicity'].fillna('Other/Unknown')
    
    # Convert age to numeric (handle "> 89" cases)
    df['age_numeric'] = df['age'].apply(lambda x: 90 if x == '> 89' else float(x) if pd.notna(x) else np.nan)
    
    # Calculate hospital length of stay (in days)
    df['hospital_los_days'] = (df['hospitaldischargeoffset'] - df['hospitaladmitoffset']) / 1440
    
    # Calculate overall statistics
    print("=" * 100)
    print("TABLE 1: Baseline Characteristics and Outcomes by Sex and Race/Ethnicity")
    print("=" * 100)
    print()
    
    # Overall cohort
    print_group_statistics("OVERALL COHORT", df)
    print()
    
    # By Sex
    print("-" * 100)
    print("STRATIFIED BY SEX")
    print("-" * 100)
    for sex in ['Female', 'Male']:
        sex_df = df[df['gender'] == sex]
        if len(sex_df) > 0:
            print_group_statistics(f"Sex: {sex}", sex_df)
            print()
    
    # By Race/Ethnicity (African American and Caucasian only)
    print("-" * 100)
    print("STRATIFIED BY RACE/ETHNICITY")
    print("-" * 100)
    for ethnicity in ['African American', 'Caucasian']:
        eth_df = df[df['ethnicity_clean'] == ethnicity]
        if len(eth_df) > 0:
            print_group_statistics(f"Race: {ethnicity}", eth_df)
            print()
    
    # Create summary table as DataFrame
    summary_df = create_summary_dataframe(df)
    
    return summary_df


def print_group_statistics(group_name, group_df, indent=False):
    """Print statistics for a specific group"""
    prefix = "  " if indent else ""
    
    # Make a copy to avoid SettingWithCopyWarning
    group_df = group_df.copy()
    
    n = len(group_df)
    
    # Age statistics
    age_mean = group_df['age_numeric'].mean()
    age_std = group_df['age_numeric'].std()
    age_median = group_df['age_numeric'].median()
    age_q1 = group_df['age_numeric'].quantile(0.25)
    age_q3 = group_df['age_numeric'].quantile(0.75)
    
    # APACHE score statistics
    apache_mean = group_df['apachescore'].mean()
    apache_std = group_df['apachescore'].std()
    apache_median = group_df['apachescore'].median()
    apache_q1 = group_df['apachescore'].quantile(0.25)
    apache_q3 = group_df['apachescore'].quantile(0.75)
    
    # Mortality
    hosp_deaths = group_df['hospital_mortality'].sum()
    hosp_mort_rate = (hosp_deaths / n * 100) if n > 0 else 0
    
    unit_deaths = group_df['unit_mortality'].sum()
    unit_mort_rate = (unit_deaths / n * 100) if n > 0 else 0
    
    # ICU Length of stay (unitdischargeoffset is in minutes from ICU admission, convert to days)
    icu_los_median = group_df['unitdischargeoffset'].median() / 1440
    icu_los_q1 = group_df['unitdischargeoffset'].quantile(0.25) / 1440
    icu_los_q3 = group_df['unitdischargeoffset'].quantile(0.75) / 1440
    
    # Hospital Length of stay
    hosp_los_median = group_df['hospital_los_days'].median()
    hosp_los_q1 = group_df['hospital_los_days'].quantile(0.25)
    hosp_los_q3 = group_df['hospital_los_days'].quantile(0.75)
    
    print(f"{prefix}{group_name}")
    print(f"{prefix}{'─' * (len(group_name) + 2)}")
    print(f"{prefix}N: {n}")
    print(f"{prefix}Age (years):")
    print(f"{prefix}  Mean (SD): {age_mean:.1f} ({age_std:.1f})")
    print(f"{prefix}  Median [IQR]: {age_median:.0f} [{age_q1:.0f}-{age_q3:.0f}]")
    print(f"{prefix}APACHE Score:")
    print(f"{prefix}  Mean (SD): {apache_mean:.1f} ({apache_std:.1f})")
    print(f"{prefix}  Median [IQR]: {apache_median:.0f} [{apache_q1:.0f}-{apache_q3:.0f}]")
    print(f"{prefix}ICU Length of Stay (days):")
    print(f"{prefix}  Median [IQR]: {icu_los_median:.1f} [{icu_los_q1:.1f}-{icu_los_q3:.1f}]")
    print(f"{prefix}Hospital Length of Stay (days):")
    print(f"{prefix}  Median [IQR]: {hosp_los_median:.1f} [{hosp_los_q1:.1f}-{hosp_los_q3:.1f}]")
    print(f"{prefix}Hospital Mortality:")
    print(f"{prefix}  Deaths: {hosp_deaths} ({hosp_mort_rate:.1f}%)")
    print(f"{prefix}ICU Mortality:")
    print(f"{prefix}  Deaths: {unit_deaths} ({unit_mort_rate:.1f}%)")


def create_summary_dataframe(df):
    """Create a structured summary table as a DataFrame"""
    
    summary_data = []
    
    # Overall
    summary_data.append(get_row_statistics("Overall", df))
    
    # By Sex (Female and Male only)
    for sex in ['Female', 'Male']:
        sex_df = df[df['gender'] == sex]
        if len(sex_df) > 0:
            summary_data.append(get_row_statistics(sex, sex_df))
    
    # By Ethnicity (African American and Caucasian only)
    for ethnicity in ['African American', 'Caucasian']:
        eth_df = df[df['ethnicity_clean'] == ethnicity]
        if len(eth_df) > 0:
            summary_data.append(get_row_statistics(ethnicity, eth_df))
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def get_row_statistics(label, group_df):
    """Get statistics for a row in the summary table"""
    # Make a copy to avoid SettingWithCopyWarning
    group_df = group_df.copy()
    
    n = len(group_df)
    
    # Calculate age
    age_mean = group_df['age_numeric'].mean()
    age_std = group_df['age_numeric'].std()
    
    # APACHE score
    apache_mean = group_df['apachescore'].mean()
    apache_std = group_df['apachescore'].std()
    
    # Calculate mortality
    hosp_deaths = group_df['hospital_mortality'].sum()
    hosp_mort_rate = (hosp_deaths / n * 100) if n > 0 else 0
    
    unit_deaths = group_df['unit_mortality'].sum()
    unit_mort_rate = (unit_deaths / n * 100) if n > 0 else 0
    
    # ICU Length of stay (unitdischargeoffset is in minutes from ICU admission, convert to days)
    icu_los_median = group_df['unitdischargeoffset'].median() / 1440
    
    # Hospital Length of stay
    hosp_los_median = group_df['hospital_los_days'].median()
    
    return {
        'Group': label,
        'N': n,
        'Age_Mean': f"{age_mean:.1f}",
        'Age_SD': f"{age_std:.1f}",
        'APACHE_Mean': f"{apache_mean:.1f}",
        'APACHE_SD': f"{apache_std:.1f}",
        'ICU_LOS_days_Median': f"{icu_los_median:.1f}",
        'Hospital_LOS_days_Median': f"{hosp_los_median:.1f}",
        'Hospital_Deaths_N': hosp_deaths,
        'Hospital_Mortality_%': f"{hosp_mort_rate:.1f}",
        'ICU_Deaths_N': unit_deaths,
        'ICU_Mortality_%': f"{unit_mort_rate:.1f}"
    }


def create_latex_table(df):
    """Create a formatted LaTeX table from the summary dataframe"""
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Baseline Characteristics and Outcomes by Sex and Race/Ethnicity}")
    latex.append("\\label{tab:demographics}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\hline")
    latex.append("\\textbf{Group} & \\textbf{N} & \\textbf{Age} & \\textbf{APACHE} & \\textbf{ICU LOS} & \\textbf{Hospital LOS} & \\textbf{Hospital Mortality} & \\textbf{ICU Mortality} \\\\")
    latex.append(" & & \\textbf{Mean (SD)} & \\textbf{Mean (SD)} & \\textbf{Median (days)} & \\textbf{Median (days)} & \\textbf{N (\\%)} & \\textbf{N (\\%)} \\\\")
    latex.append("\\hline")
    
    for _, row in df.iterrows():
        group = row['Group']
        n = row['N']
        age = f"{row['Age_Mean']} ({row['Age_SD']})"
        apache = f"{row['APACHE_Mean']} ({row['APACHE_SD']})"
        icu_los = f"{row['ICU_LOS_days_Median']}"
        hosp_los = f"{row['Hospital_LOS_days_Median']}"
        hosp_mort = f"{row['Hospital_Deaths_N']} ({row['Hospital_Mortality_%']}\\%)"
        icu_mort = f"{row['ICU_Deaths_N']} ({row['ICU_Mortality_%']}\\%)"
        
        latex.append(f"{group} & {n} & {age} & {apache} & {icu_los} & {hosp_los} & {hosp_mort} & {icu_mort} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


if __name__ == "__main__":
    # Example usage
    script_dir = Path(__file__).resolve().parent
    csv_file = script_dir / "data" / "demo" / "patient.csv"
    apache_file = script_dir / "data" / "demo" / "apachePatientResult.csv"
    
    # Generate Table 1
    table1_df = generate_table1(csv_file, apache_file)
    
    # Create and save LaTeX table
    latex_table = create_latex_table(table1_df)
    
    with open("table1_demographics.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    
    print("\n" + "=" * 100)
    print("LaTeX table saved to 'table1_demographics.tex'")
    print("=" * 100)
    
    # Display the LaTeX table
    print("\nLATEX TABLE:")
    print(latex_table)