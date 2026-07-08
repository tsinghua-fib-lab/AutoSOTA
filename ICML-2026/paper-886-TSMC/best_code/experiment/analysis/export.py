from typing import NamedTuple
import os

import wandb
import yaml

import pandas as pd

import tqdm
from tqdm.contrib.concurrent import thread_map

from experiment.analysis.utils import flatten_dict


DEFAULT_MAX_WORKERS: int = 10
CONFIG_PATH: str = 'configs'
DATA_PATH: str = 'data'
CACHE_PATH: str = 'cache'

api = wandb.Api(timeout=500)


class ExportSpec(NamedTuple):
    entity: str
    project_name: str
    out_path: str
    metrics: list[str]
    sweep_id: str | None = None

    @property
    def path(self):
        return os.path.join(self.entity, self.project_name)

    def runs(self):
        # See issues: https://github.com/wandb/wandb/issues/6614
        return api.runs(
            path=self.path,
            filters={} if self.sweep_id is None else {'sweep': self.sweep_id},
            order='+created_at'
        )

    def get_api_ids(self):
        # Get run identifiers to enable fetching: `api.run('id')`
        ids = get_ids(self.runs())
        return [os.path.join(self.path, n) for n in ids]

    def export(
            self,
            *,
            min_step: int | None = None,
            max_step: int | None = None,
            workers: int = DEFAULT_MAX_WORKERS
    ) -> list[str | None]:
        return export_raw(
            self,
            min_step=min_step,
            max_step=max_step,
            max_workers=workers
        )


def get_ids(runs):
    return [r.id for r in tqdm.tqdm(runs, desc='Collecting runs')]


def download_raw_csv(
        identifier: str,
        out_path: str,
        *args,
        warn_unfinished: bool = True,
        **kwargs
) -> None | str:
    run = api.run(identifier)

    if run.state != "finished":
        if warn_unfinished:
            print(f"Warning, run {run.name} is unfinished")
        return identifier  # Failure; retry?

    # Export Config
    config = run.config
    with open(os.path.join(out_path, 'configs', f'{run.id}.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Export data
    scan = run.scan_history(*args, **kwargs)
    data = [r for r in scan]

    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(out_path, 'data', f'{run.id}.csv'),
        header=True
    )

    return None


def export_raw(
        spec: ExportSpec,
        min_step: int,
        max_step: int,
        warn_unfinished: bool = True,
        max_workers: int = DEFAULT_MAX_WORKERS
) -> list[None | str]:

    ids = spec.get_api_ids()

    os.makedirs(os.path.join(spec.out_path, CONFIG_PATH), exist_ok=True)
    os.makedirs(os.path.join(spec.out_path, DATA_PATH), exist_ok=True)

    results = thread_map(
        lambda identifier: download_raw_csv(
            identifier,
            out_path=spec.out_path,
            warn_unfinished=warn_unfinished,
            keys=spec.metrics,
            min_step=min_step, max_step=max_step  # scan kwargs
        ),
        ids,
        max_workers=max_workers,
        desc='Exporting data'
    )

    return results


def get_configs(spec: ExportSpec) -> dict[str, ...]:

    # Collect configs
    cfg_prefix = os.path.join(spec.out_path, CONFIG_PATH)

    ids = set(get_ids(spec.runs()))

    cfg_list = list(filter(lambda x: x.removesuffix('.yaml') in ids,
                           os.listdir(cfg_prefix)))

    cfgs = {}
    for f in tqdm.tqdm(cfg_list, desc='Loading Configs'):
        with open(os.path.join(cfg_prefix, f), 'r') as stream:
            cfgs[f.removesuffix('.yaml')] = yaml.safe_load(stream)

    return cfgs


def format_config(
        spec: ExportSpec,
        configs: dict[str, dict[str, ...]],
        group_id: str = 'environment.name.value'
) -> dict[str, pd.DataFrame]:
    cfg = {k: flatten_dict(v, separator='.') for k, v in configs.items()}
    cfg = {k: v | {
        'sweep_id': api.run(os.path.join(spec.path, k)).sweep.id,
        'sweep_name': api.run(os.path.join(spec.path, k)).sweep.name
    } for k, v in tqdm.tqdm(cfg.items(), desc='Annotating sweep ID')}

    # Filter out annotation
    cfg = {k: {name: value for name, value in v.items() if 'desc' not in name}
           for k, v in cfg.items()}

    df = pd.DataFrame(cfg)

    df_dict = {}
    groups = df.T.groupby(group_id)
    for env, subdf in groups:

        # Drop duplicate runs (exact same config)
        subdf = subdf.loc[~(subdf.astype(str).nunique(axis=1) <= 1)]

        sweep_id = subdf.sweep_id  # Prevent dropping sweep IDs
        sweep_name = subdf.sweep_name  # Prevent dropping sweep IDs

        # Drop duplicate columns (uninformative separators)
        subdf = subdf.loc[:, ~(subdf.astype(str).nunique(axis=0) <= 1)]

        # Reannotate if dropped
        subdf.loc[:, 'sweep_id'] = sweep_id
        subdf.loc[:, 'sweep_name'] = sweep_name

        df_dict[str(env)] = subdf

    return df_dict


def cache_config(spec: ExportSpec):
    cfgs = get_configs(spec)
    dfs = format_config(spec, cfgs)

    outdir = os.path.join(spec.out_path, CACHE_PATH, spec.path)
    os.makedirs(outdir, exist_ok=True)

    for name, df in tqdm.tqdm(dfs.items(), desc='Writing config-cache'):
        df.to_csv(
            os.path.join(outdir, name.replace(' ', '_') + '.csv'),
            index_label='run'
        )
