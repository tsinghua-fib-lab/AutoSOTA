from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from analysis.figures_src.camera_ready.plotting import generate_figures
from analysis.figures_src.fig_e2e_camera_ready.config import CONFIG


if __name__ == "__main__":
    generate_figures(CONFIG, "fig.camera-ready.gram", __file__)
