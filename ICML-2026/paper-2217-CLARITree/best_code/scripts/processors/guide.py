# benchmark/processors/guide.py
from __future__ import annotations
import numpy as np
from pathlib import Path
import subprocess, time
import pandas as pd
from .base import Processor, FitArtifacts

GUIDE_EXE = Path("./bin/guide")  

class GuideProcessor(Processor):
    name = "guide"

    def build(self, **hparams):
        return dict(**hparams)

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> FitArtifacts:
        csv_path: Path = Path(model["csv_path"])
        depth: int = model.get("depth", 3)
        max_nodes: int | None = model.get("max_nodes", None)
        work_dir: Path = Path(model.get("work_dir", "./guide_work"))
        work_dir.mkdir(parents=True, exist_ok=True)

        dsc_path = work_dir / "data.dsc"
        inp_path = work_dir / "data.inp"
        out_path = work_dir / "data.out"
        r_path  = work_dir / "guide_model.R"

        self._make_dsc(csv_path, dsc_path)
        self._make_inp(dsc_path, inp_path, out_path, r_path, depth, max_nodes)

        t0 = time.perf_counter()
        result = subprocess.run(
            [str(GUIDE_EXE.resolve())],
            input=inp_path.read_text(),
            text=True,
            capture_output=True,
        )
        t1 = time.perf_counter()

        fit_time = t1 - t0
        if result.returncode != 0:
            raise RuntimeError(f"GUIDE failed, see logs:\n{result.stderr[:200]}")

        # Check if guide_model.R was created successfully
        # GUIDE may return 0 even if it fails, so we need to check if the output file exists
        if not r_path.exists():
            error_msg = f"GUIDE did not create guide_model.R in {work_dir}"
            if out_path.exists():
                error_text = out_path.read_text(errors="ignore")
                # Check for common error patterns in GUIDE output
                if "Error" in error_text:
                    error_lines = [line for line in error_text.splitlines() if "Error" in line]
                    error_msg += f"\nGUIDE output errors:\n" + "\n".join(error_lines[-5:])
            raise RuntimeError(error_msg)

        complexity = max_nodes if max_nodes is not None else None
        if out_path.exists():
            text = out_path.read_text(errors="ignore")
            for line in text.splitlines():
                if "Number of terminal nodes" in line:
                    try:
                        complexity = int(line.strip().split(":")[-1])
                    except Exception:
                        pass
        model["r_model_path"] = r_path.as_posix()
        model["work_dir"] = work_dir.as_posix()
        model["csv_path"] = csv_path.as_posix()
        return FitArtifacts(model=model, complexity=complexity, extras={"fit_time": fit_time})

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GUIDE processor does not support in-memory predict")

    
    # ===== Helpers =====
    @staticmethod
    def _infer_predictor_roles(csv_path: Path) -> list[tuple[str, str]]:
        """
        Infer GUIDE predictor roles from CSV:
        - binary 0/1 feature -> 's' (split-only categorical/string-like)
        - otherwise          -> 'n' (numeric: split + linear regression)
        """
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(f"{csv_path} must contain at least one feature and one target column")
        X_df = df.iloc[:, :-1]
        roles: list[tuple[str, str]] = []
        for col_name in X_df.columns:
            col = pd.to_numeric(X_df[col_name], errors="coerce")
            non_na = col.dropna()
            if non_na.empty:
                # Degenerate all-NA column: keep numeric to avoid GUIDE type mismatch surprises.
                roles.append((str(col_name), "n"))
                continue
            uniq = set(np.unique(non_na.to_numpy(dtype=float)))
            is_binary_01 = uniq.issubset({0.0, 1.0})
            roles.append((str(col_name), "s" if is_binary_01 else "n"))
        return roles

    def _make_dsc(self, csv_path: Path, dsc_path: Path):
        base = Path.cwd()
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        predictor_roles = self._infer_predictor_roles(csv_path)
        if len(predictor_roles) != len(cols) - 1:
            raise RuntimeError("GUIDE DSC role inference failed: predictor count mismatch")
        with dsc_path.open("w", newline="\n") as f:
            rel_csv = csv_path.resolve().relative_to(base)  # 相对 base 的路径
            f.write(f"'./{rel_csv.as_posix()}'\n")
            f.write("NA\n")
            f.write("2\n")  
            for i, (name, role) in enumerate(predictor_roles, start=1):
                f.write(f"{i} {name} {role}\n")
            f.write(f"{len(cols)} {cols[-1]} d\n")

    def _make_inp(
        self,
        dsc_path: Path,
        inp_path: Path,
        out_path: Path,
        r_path: Path,
        depth: int,
        max_nodes: int | None,
    ):
        # dsc_abs = dsc_path.resolve().as_posix()
        # out_abs = out_path.resolve().as_posix()
        # r_abs   = r_path.resolve().as_posix()
        base = Path.cwd()
        dsc_abs = f"./{dsc_path.resolve().relative_to(base).as_posix()}"
        out_abs = f"./{out_path.resolve().relative_to(base).as_posix()}"
        r_abs   = f"./{r_path.resolve().relative_to(base).as_posix()}"

        # #======================bucket 20 version==========================
        # lines = [
        #     'GUIDE       (do not edit this file unless you know what you are doing)',
        #     ' 45.0      (version of GUIDE that generated this file)',
        #     ' 1          (1=model fitting, 2=importance or DIF scoring, 3=data conversion)',
        #     f'"{out_abs}"  (name of output file)',
        #     ' 1          (1=one tree, 2=ensemble)',
        #     ' 2          (1=classification, 2=regression, 3=propensity score tree)',
        #     ' 1          (1=linear, 2=quantile, 3=Poisson, 4=censored response, 5=multiresponse or itemresponse, 6=longitudinal with T vars, 7=logistic)',
        #     ' 1          (1=least squares, 2=least median of squares)',
        #     ' 1          (0=stepwise, 1=multiple linear, 2=best simple polynomial, 3=constant, 4=ANCOVA)',
        #     ' 1          (1=intercept included, 2=intercept excluded)',
        #     ' 0          (0=no truncation, 1=node range, 2=+10% node range, 3=global range)',
        #     ' 2          (1=interaction tests, 2=skip them)',
        #     ' 0          (0=tree with fixed no. of nodes, 1=prune by CV, 2=no pruning)',
        #     f' {max_nodes if max_nodes is not None else 10}          (maximum number of terminal nodes of pruned tree)',
        #     f'"{dsc_abs}"  (name of DSC file)',
        #     ' 1          (1=split point from quantiles, 2=use exhaustive search)',
        #     ' 2          (1=default max number of splits on N and S variables, 2=specify number in next line)',
        #     ' 5         (max number of splits on N and S variables)',
        #     ' 2          (1=default max. number of split levels, 2=specify no. in next line)',
        #     f' {depth}   (max. no. split levels)',
        #     ' 1          (1=default min. node size, 2=specify min. value in next line)',
        #     ' 0          (0=no LaTeX code, 1=tree without node numbers, 2=tree with node numbers)',
        #     ' 1          (1=no storage, 2=store fit and split variables, 3=store split variables and values)',
        #     ' 1          (1=do not save, 2=save regression coefs in a file)',
        #     ' 1          (1=do not save fitted values and node IDs, 2=save in a file)',
        #     ' 2          (1=do not write R function, 2=write R function)',
        #     f'"{r_abs}" (R code file)',
        #     ' 1          (rank of top variable to split root node)',
        # ]

        # ======================full version==========================
        lines = [
            'GUIDE       (do not edit this file unless you know what you are doing)',
            ' 45.0      (version of GUIDE that generated this file)',
            ' 1          (1=model fitting, 2=importance or DIF scoring, 3=data conversion)',
            f'"{out_abs}"  (name of output file)',
            ' 1          (1=one tree, 2=ensemble)',
            ' 2          (1=classification, 2=regression, 3=propensity score tree)',
            ' 1          (1=linear, 2=quantile, 3=Poisson, 4=censored response, 5=multiresponse or itemresponse, 6=longitudinal with T vars, 7=logistic)',
            ' 1          (1=least squares, 2=least median of squares)',
            ' 1          (0=stepwise, 1=multiple linear, 2=best simple polynomial, 3=constant, 4=ANCOVA)',
            ' 1          (1=intercept included, 2=intercept excluded)',
            ' 0          (0=no truncation, 1=node range, 2=+10% node range, 3=global range)',
            ' 2          (1=interaction tests, 2=skip them)',
            ' 0          (0=tree with fixed no. of nodes, 1=prune by CV, 2=no pruning)',
            f' {max_nodes if max_nodes is not None else 10}          (maximum number of terminal nodes of pruned tree)',
            f'"{dsc_abs}"  (name of DSC file)',
            ' 2          (1=split point from quantiles, 2=use exhaustive search)',
            ' 2          (1=default max. number of split levels, 2=specify no. in next line)',
            f' {depth}   (max. no. split levels)',
            ' 1          (1=default min. node size, 2=specify min. value in next line)',
            ' 0          (0=no LaTeX code, 1=tree without node numbers, 2=tree with node numbers)',
            ' 1          (1=no storage, 2=store fit and split variables, 3=store split variables and values)',
            ' 1          (1=do not save, 2=save regression coefs in a file)',
            ' 1          (1=do not save fitted values and node IDs, 2=save in a file)',
            ' 2          (1=do not write R function, 2=write R function)',
            f'"{r_abs}" (R code file)',
            ' 1          (rank of top variable to split root node)',
        ]

        inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
