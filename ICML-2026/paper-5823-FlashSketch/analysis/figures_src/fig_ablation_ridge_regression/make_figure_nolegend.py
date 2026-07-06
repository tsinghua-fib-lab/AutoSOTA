from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from analysis.figures_src.ablation.plotting import generate_figures
from analysis.figures_src.fig_ablation_ridge_regression.config_nolegend import CONFIG


if __name__ == "__main__":
    generate_figures(CONFIG, "fig.ablation.ridge-regression.nolegend", __file__)
