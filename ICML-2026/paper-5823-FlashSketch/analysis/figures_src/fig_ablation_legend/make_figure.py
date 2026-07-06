from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from analysis.figures_src.ablation.plotting import generate_legends
from analysis.figures_src.fig_ablation_legend.config import CONFIG


if __name__ == "__main__":
    generate_legends(CONFIG, "fig.ablation.legend", __file__)
