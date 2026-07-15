.PHONY: all gpqa bbh mmlu simulations

.renv_restored: renv.lock
	Rscript -e "renv::restore(prompt = FALSE)"
	touch .renv_restored

gpqa: analysis_scripts/gpqa_analysis.R
	Rscript analysis_scripts/gpqa_analysis.R

bbh: analysis_scripts/bbh_analysis.R .renv_restored
	Rscript analysis_scripts/bbh_analysis.R

mmlu: analysis_scripts/mmlu_analysis.R .renv_restored
	Rscript analysis_scripts/mmlu_analysis.R

simulations: analysis_scripts/simulation_analysis.R .renv_restored
	Rscript analysis_scripts/simulation_analysis.R

all: gpqa bbh mmlu simulations