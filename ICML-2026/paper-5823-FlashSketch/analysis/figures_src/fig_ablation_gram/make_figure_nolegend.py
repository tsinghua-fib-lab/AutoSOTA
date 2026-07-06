from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from analysis.figures_src.ablation.plotting import generate_figures
from analysis.figures_src.fig_ablation_gram.config_nolegend import CONFIG


if __name__ == "__main__":
    generate_figures(CONFIG, "fig.ablation.gram.nolegend", __file__)
