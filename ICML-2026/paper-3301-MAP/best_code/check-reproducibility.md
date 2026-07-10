---
argument-hint: [github-repo-folder]
description: Evaluate GitHub repository against SISC Reproducibility Badge criteria
---

Evaluate the GitHub repository in the specified folder (or current directory if none specified) for Reproducibility.

**Target folder:** $1

I need to assess this repository for Reproducibility and check if code and data available based on these specific criteria:

## Core Requirements Assessment

### 1. Code and Data Availability
- [ ] All computer code that implements computational methods is publicly available
- [ ] All data needed to reproduce results is publicly available
- [ ] All parameter settings are included (in code or separate data files)
- [ ] Code allows reproduction of ALL numerical results (tables, figures, etc.)

### 2. README File Quality
Check if the README file includes:
- [ ] Detailed description of all code and data files
- [ ] Clear explanation of how to use the code to reproduce paper results
- [ ] Paper title and authors clearly identified
- [ ] Purpose statement indicating this provides reproducibility information

### 3. Repository Organization
- [ ] Files are well-organized and easy to navigate
- [ ] Clear folder structure with logical naming
- [ ] Separate dedicated repository OR clearly identified subfolder
- [ ] Direct URL pointing to relevant files/folder

### 4. Completeness Coverage
- [ ] Code covers all computational methods described in the paper
- [ ] Code covers all computational tests/experiments in the paper
- [ ] Parameter files/settings for all experiments are provided
- [ ] Dependencies and requirements are clearly specified

### 5. Usability and Documentation
- [ ] Installation/setup instructions are provided
- [ ] Usage examples or scripts are included
- [ ] Code is commented appropriately
- [ ] License information is provided

## Evaluation Process

Please use the code-quality-auditor agent to:

1. **Analyze the repository structure** and identify all code, data, and documentation files
2. **Review the README file** against the specific SISC requirements
3. **Assess code completeness** relative to computational methods that should be implemented
4. **Check reproducibility infrastructure** (build scripts, parameter files, example runs)
5. **Evaluate documentation quality** for enabling reproduction

## Output Requirements

Generate a detailed assessment report saved as `[folder_name]_reproducibility_assessment.md` with:

### Executive Summary
- Overall recommendation: **APPROVE BADGE** / **CONDITIONAL APPROVAL** / **DENY BADGE**
- Key strengths and deficiencies

### Detailed Checklist Results
- Each criterion marked as ✅ Met, ⚠️ Partially Met, or ❌ Not Met
- Specific evidence and examples for each assessment

### Required Improvements (if any)
- Specific actionable recommendations to meet badge criteria
- Missing files, documentation, or code components
- Suggested README improvements

### Editor Notes
- Summary for handling editor's high-level validation
- Assessment of whether repository "sufficiently covers computational methods and tests"
- Evaluation of README adequacy for reproduction guidance

Focus on objective assessment against the published SISC criteria, not general code quality judgments.