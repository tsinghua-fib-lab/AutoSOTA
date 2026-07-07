from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store, ZenField
from omegaconf import OmegaConf

from run.hydra_zen import make_config

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
OmegaConf.register_new_resolver("choice", lambda x: HydraConfig.get().runtime.choices[x])

hydra = HydraConf()
hydra.run.dir = "${run.path}/hydra/runs/${run.timestamp}"
hydra.verbose = "${merge:['__main__'],${run.verbose}}"
hydra.job.name = "${run.identifier}"
store(hydra)

run = make_config(
    identifier=ZenField(str, "${cfg.model.identifier}"),
    series=ZenField(str, "default"),
    timestamp=ZenField(str, "${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    root_dir=ZenField(str, "${oc.env:PROJECT_ROOT}"),
    artifact_dir=ZenField(str, "${run.root_dir}/artifacts"),
    path=ZenField(str, "${run.artifact_dir}/runs/${run.series}/${run.identifier}"),
    save_results=ZenField(bool, True),
    save_model_dir=ZenField(str | None, None),
    verbose=ZenField(list[str], ["hydra", "run", "entquant"]),
)
store(run, group="run", name="default")

run_save = run(save_model_dir="${run.path}/model")
store(run_save, group="run", name="save")
