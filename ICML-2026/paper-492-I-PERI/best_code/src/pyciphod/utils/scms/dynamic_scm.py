import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Any, Tuple, Union

from pyciphod.utils.scms.scm import SCM
from pyciphod.utils.time_series.data_format import wide_timevar_to_ts_df, wide_timevar_to_lagged_df
from pyciphod.utils.time_series.data_format import TimeVar, DTimeVar, CTimeVar
from pyciphod.utils.graphs.temporal_graphs import FtAcyclicDirectedMixedGraph, FtDirectedAcyclicGraph, create_random_ft_admg, create_random_ft_dag


class DynamicSCM(SCM):
    """Abstract base class for time-evolving SCMs."""
    def __init__(self, v: Iterable[TimeVar], u: Iterable[TimeVar], f: Dict[TimeVar, Dict[str, Any]], u_dist: Optional[Any] = None):
        if not all(isinstance(node, TimeVar) for node in v):
            raise TypeError("All observed variables in a DynamicSCM must be TimeVar objects.")

        if not all(isinstance(node, TimeVar) for node in u):
            raise TypeError("All latent variables in a DynamicSCM must be TimeVar objects.")

        if not all(isinstance(node, TimeVar) for node in f.keys()):
            raise TypeError("All keys of f must be TimeVar objects.")

        super().__init__(v, u, f, u_dist)

    def induced_ft_dag(self) -> FtDirectedAcyclicGraph:
        """
        Return the FtDirectedAcyclicGraph induced by the DynamicSCM (observed + latent variables).

        If there is unobserved confounding (a latent that is a parent of two or more observed
        variables), raise ValueError and suggest using `induced_ft_admg()` which can represent
        such confounding via bidirected edges.
        """
        # Check for latent-induced confounding: any latent with >=2 observed descendants
        for latent in self.u:
            obs_children = set(self._dag.get_children(latent)).intersection(self.v)
            if len(obs_children) >= 2:
                raise ValueError(
                    f"Unobserved confounding detected: latent '{latent}' is an ancestor of observed variables {sorted(obs_children)}. "
                    "Use `induced_admg()` to obtain an ADMG that represents such confounding."
                )

        # Build a DAG containing only the observed variables and the directed edges among them
        new_dag = FtDirectedAcyclicGraph()
        for ov in self.v:
            new_dag.add_vertex(ov)

        for (src, tgt) in self._dag.get_directed_edges():
            if src in self.v and tgt in self.v:
                new_dag.add_directed_edge(src, tgt)

        return new_dag

    def induced_ft_admg(self) -> FtAcyclicDirectedMixedGraph:
        """
        Return an AcyclicDirectedMixedGraph induced by the SCM over the observed variables.

        Directed edges among observed variables are preserved. Latent variables that are
        parents of two or more observed variables are converted into bidirected (confounded)
        edges between every pair of their observed descendants. If multiple latents induce the
        same confounding, the confounded edge is added once.
        """
        admg = FtAcyclicDirectedMixedGraph()
        # add observed vertices
        for ov in self.v:
            admg.add_vertex(ov)

        # preserve directed edges among observed variables
        for (src, tgt) in self._dag.get_directed_edges():
            if src in self.v and tgt in self.v:
                admg.add_directed_edge(src, tgt)

        # convert latent ancestors into bidirected confounding among their observed descendants
        for latent in self.u:
            obs_children = set(self._dag.get_children(latent)).intersection(self.v)
            obs_list = sorted(obs_children, key=lambda n: (n.name, n.time))
            for i in range(len(obs_list)):
                for j in range(i + 1, len(obs_list)):
                    a, b = obs_list[i], obs_list[j]
                    # add a bidirected/confounded edge between a and b
                    admg.add_confounded_edge(a, b)

        return admg

    def induced_dag(self) -> FtDirectedAcyclicGraph:
       return self.induced_ft_dag()

    def induced_admg(self) -> FtAcyclicDirectedMixedGraph:
       return self.induced_ft_admg()


class DtDynamicSCM(DynamicSCM):
    def __init__(self, v: Iterable[DTimeVar], u: Iterable[DTimeVar], f: Dict[DTimeVar, Dict[str, Any]], u_dist: Optional[Any] = None):
        super().__init__(v, u, f, u_dist)
        if not all(isinstance(node, DTimeVar) for node in self.v):
            raise TypeError("DtDynamicSCM requires DTimeVar observed variables.")
        if not all(isinstance(node, DTimeVar) for node in self.u):
            raise TypeError("DtDynamicSCM requires DTimeVar latent variables.")

    def get_max_delay(self, include_latent: bool = False) -> int:
        max_delay = 0

        for child, spec in self.f.items():
            if child in self.u:
                continue

            parents = spec.get("parents", [])
            for p in parents:
                if (not include_latent) and (p in self.u):
                    continue

                delay = child.time - p.time

                if delay < 0:
                    raise ValueError(
                        f"Parent {p} is in the future of child {child}; negative delay found."
                    )

                max_delay = max(max_delay, delay)

        return max_delay

    def is_structurally_causally_stationary(self) -> bool:
        """
        Check whether the finite DtDynamicSCM is compatible with causal stationarity
        in the template sense.

        Rule used:
        - For each variable name separately, the latest available structural equation
          is treated as the transition template.
        - Earlier equations for that variable must be exactly the feasible truncation
          of that template:
              * same parent names
              * same lags
              * parents whose lag would require times before the start are omitted
        - Exogenous/latent parents are treated the same way via their name and lag.

        This is a structural compatibility check, not a distributional stationarity test.
        """
        observed_nodes = list(self.v)
        latent_nodes = set(self.u)
        all_nodes = set(observed_nodes) | latent_nodes

        # Group observed nodes by variable name
        by_name = {}
        for node in observed_nodes:
            by_name.setdefault(node.name, []).append(node)

        def signature(child, parents):
            """
            Return sorted signature [(parent_name, lag), ...]
            where lag = child.time - parent.time.
            """
            sig = []
            for p in parents:
                if p not in all_nodes:
                    return None
                if not isinstance(p, DTimeVar):
                    return None
                lag = child.time - p.time
                if lag < 0:
                    return None
                sig.append((p.name, lag))
            return tuple(sorted(sig))

        for var_name, nodes in by_name.items():
            nodes = sorted(nodes, key=lambda n: n.time)

            # Need at least one equation
            latest = nodes[-1]
            if latest not in self.f:
                return False
            latest_spec = self.f[latest]
            if "parents" not in latest_spec:
                return False

            latest_sig = signature(latest, latest_spec["parents"])
            if latest_sig is None:
                return False

            # Check every earlier equation against the feasible truncation
            for node in nodes[:-1]:
                if node not in self.f:
                    return False
                spec = self.f[node]
                if "parents" not in spec:
                    return False

                actual_sig = signature(node, spec["parents"])
                if actual_sig is None:
                    return False

                # Build expected truncation of latest template at this earlier time
                expected_sig = []
                for parent_name, lag in latest_sig:
                    # Parent at time node.time - lag must exist at positive time
                    parent_time = node.time - lag
                    if parent_time >= 1:
                        expected_sig.append((parent_name, lag))
                expected_sig = tuple(sorted(expected_sig))

                if actual_sig != expected_sig:
                    return False

        return True

    def _draw_noise(self, latent_name: DTimeVar, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n_samples innovations for one latent process name.

        Supports:
        - self.u_dist callable with size argument
        - self.u_dist callable without args
        - self.u_dist dict keyed by latent variable name
        - self.u_dist scalar
        """
        dist = self._u_dist_map

        if dist is None:
            return np.zeros(n_samples, dtype=float)

        if isinstance(dist, dict):
            if latent_name not in dist:
                raise KeyError(f"No distribution specified for latent variable name {latent_name!r}")
            dist = dist[latent_name]

        if callable(dist):
            try:
                out = dist(size=n_samples)
                arr = np.asarray(out, dtype=float)
                if arr.shape == ():
                    arr = np.repeat(arr, n_samples)
                return arr
            except TypeError:
                pass

            vals = [dist() for _ in range(n_samples)]
            return np.asarray(vals, dtype=float)

        return np.repeat(float(dist), n_samples)

    def _call_func(self, func, par_vals, noise):
        try:
            return func(par_vals, noise)
        except TypeError:
            return func(par_vals)

    # def old_generate_causally_stationary_data(
    #     self,
    #     n_timepoints: int = 10,
    #     n_samples: int = 1000,
    #     burn_in: int = 200,
    #     include_latent: bool = False,
    #     seed: Optional[int] = None,
    #     initial_value: float = 0.0,
    #     reindex_time: bool = True,
    # ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    #     """
    #     Generate data from the causally stationary template implied by the SCM.
    #
    #     Procedure
    #     ---------
    #     1. Check that the SCM is causally stationary.
    #     2. Take the latest available equation for each observed variable name as
    #        the transition template.
    #     3. Simulate forward for burn_in + n_timepoints steps.
    #     4. Discard burn_in and return the remaining time points.
    #
    #     Parameters
    #     ----------
    #     n_timepoints : int
    #         Number of retained time points after burn-in.
    #     n_samples : int
    #         Number of independent trajectories.
    #     burn_in : int
    #         Number of initial simulated time points to discard.
    #     include_latent : bool
    #         Whether to also return latent innovations.
    #     seed : int, optional
    #         Random seed.
    #     initial_value : float
    #         Value used for unavailable times before t=1.
    #     reindex_time : bool
    #         If True, returned columns are indexed from 1..n_timepoints.
    #         Otherwise they keep absolute simulated times after burn-in.
    #
    #     Returns
    #     -------
    #     pd.DataFrame or (pd.DataFrame, pd.DataFrame)
    #     """
    #     if not self.is_structurally_causally_stationary():
    #         raise ValueError(
    #             "This DtDynamicSCM is not causally stationary, so a stationary template "
    #             "cannot be extracted."
    #         )
    #
    #     if n_timepoints <= 0:
    #         raise ValueError("n_timepoints must be positive")
    #     if n_samples <= 0:
    #         raise ValueError("n_samples must be positive")
    #     if burn_in < 0:
    #         raise ValueError("burn_in must be non-negative")
    #
    #     rng = np.random.default_rng(seed)
    #
    #     observed_nodes = list(self.v)
    #     latent_nodes = list(self.u)
    #
    #     # Group observed nodes by variable name
    #     obs_by_name: Dict[str, list[DTimeVar]] = {}
    #     for node in observed_nodes:
    #         obs_by_name.setdefault(node.name, []).append(node)
    #
    #     # Latest equation for each observed variable = transition template
    #     transition_specs: Dict[str, Dict[str, Any]] = {}
    #     contemp_graph = nx.DiGraph()
    #
    #     observed_set = set(observed_nodes)
    #     latent_set = set(latent_nodes)
    #
    #     for var_name, nodes in obs_by_name.items():
    #         latest = max(nodes, key=lambda n: n.time)
    #         spec = self.f[latest]
    #         parents = spec["parents"]
    #         func = spec["func"]
    #
    #         observed_parent_specs = []
    #         latent_parent_specs = []
    #
    #         for p in parents:
    #             lag = latest.time - p.time
    #             if lag < 0:
    #                 raise ValueError(f"Negative lag found for parent {p} of child {latest}")
    #
    #             if p in observed_set:
    #                 observed_parent_specs.append((p, p.name, lag))
    #                 if lag == 0:
    #                     contemp_graph.add_edge(p.name, latest.name)
    #             elif p in latent_set:
    #                 latent_parent_specs.append((p, p.name, lag))
    #             else:
    #                 raise ValueError(f"Parent {p} is neither observed nor latent in this SCM")
    #
    #         transition_specs[var_name] = {
    #             "template_child": latest,
    #             "func": func,
    #             "observed_parents": observed_parent_specs,
    #             "latent_parents": latent_parent_specs,
    #         }
    #         contemp_graph.add_node(var_name)
    #
    #     if not nx.is_directed_acyclic_graph(contemp_graph):
    #         raise ValueError(
    #             "Contemporaneous dependencies among transition equations contain a cycle."
    #         )
    #
    #     eval_order = list(nx.topological_sort(contemp_graph))
    #     total_time = burn_in + n_timepoints
    #
    #     obs_hist: Dict[str, np.ndarray] = {
    #         name: np.full((n_samples, total_time), np.nan, dtype=float)
    #         for name in obs_by_name
    #     }
    #
    #     latent_names = sorted({u.name for u in latent_nodes})
    #     lat_hist: Dict[str, np.ndarray] = {
    #         name: np.full((n_samples, total_time), np.nan, dtype=float)
    #         for name in latent_names
    #     }
    #
    #     # Draw all latent processes first
    #     latent_templates = {}
    #     for u in latent_nodes:
    #         latent_templates.setdefault(u.name, u)
    #     for t in range(total_time):
    #         for lname in latent_names:
    #             latent_template = latent_templates[lname]
    #             lat_hist[lname][:, t] = self._draw_noise(latent_template, n_samples, rng)
    #
    #     def get_obs(name: str, t: int) -> np.ndarray:
    #         if t < 0:
    #             return np.full(n_samples, initial_value, dtype=float)
    #         return obs_hist[name][:, t]
    #
    #     def get_lat(name: str, t: int) -> np.ndarray:
    #         if t < 0:
    #             return np.full(n_samples, initial_value, dtype=float)
    #         return lat_hist[name][:, t]
    #
    #     # Simulate forward
    #     for t in range(total_time):
    #         for var_name in eval_order:
    #             spec = transition_specs[var_name]
    #             func = spec["func"]
    #
    #             vals = np.empty(n_samples, dtype=float)
    #
    #             for i in range(n_samples):
    #                 par_vals = {}
    #
    #                 for template_parent, parent_name, lag in spec["observed_parents"]:
    #                     par_vals[template_parent] = float(get_obs(parent_name, t - lag)[i])
    #
    #                 for template_parent, latent_name, lag in spec["latent_parents"]:
    #                     par_vals[template_parent] = float(get_lat(latent_name, t - lag)[i])
    #
    #                 vals[i] = float(self._call_func(func, par_vals, 0.0))
    #
    #             obs_hist[var_name][:, t] = vals
    #
    #     keep_times = list(range(burn_in, total_time))
    #
    #     obs_cols = {}
    #     for var_name in sorted(obs_hist):
    #         for j, t in enumerate(keep_times, start=1):
    #             col = DTimeVar(var_name, j) if reindex_time else DTimeVar(var_name, t + 1)
    #             obs_cols[col] = obs_hist[var_name][:, t]
    #
    #     df_observed = pd.DataFrame(obs_cols)
    #
    #     if include_latent:
    #         lat_cols = {}
    #         for lat_name in sorted(lat_hist):
    #             for j, t in enumerate(keep_times, start=1):
    #                 col = DTimeVar(lat_name, j) if reindex_time else DTimeVar(lat_name, t + 1)
    #                 lat_cols[col] = lat_hist[lat_name][:, t]
    #         df_latent = pd.DataFrame(lat_cols)
    #         return df_observed, df_latent
    #
    #     return df_observed

    def generate_causally_stationary_data_from_latest_mechanisms(
            self,
            n_timepoints: int = 10,
            n_samples: int = 1000,
            burn_in: int = 200,
            include_latent: bool = False,
            seed: Optional[int] = None,
            initial_value: float = 0.0,
            reindex_time: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate data by using only the most recent structural mechanism of each
        observed time series as a repeating transition template.

        Semantics
        ---------
        For each observed variable name V:
        - find the observed node V_t with largest time index t in the SCM;
        - use its structural equation as the transition template for all future times.

        Earlier structural equations in the SCM are ignored.

        This function therefore constructs a time-homogeneous dynamic extension from
        the latest available mechanisms. It does NOT check whether the original SCM
        itself is causally stationary.

        Parameters
        ----------
        n_timepoints : int
            Number of retained time points after burn-in.
        n_samples : int
            Number of independent trajectories.
        burn_in : int
            Number of initial simulated time points to discard.
        include_latent : bool
            Whether to also return latent innovations.
        seed : int, optional
            Random seed.
        initial_value : float
            Value used for unavailable times before t=1.
        reindex_time : bool
            If True, returned columns are indexed from 1..n_timepoints.
            Otherwise they keep absolute simulated times after burn-in.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        """
        if n_timepoints <= 0:
            raise ValueError("n_timepoints must be positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")

        rng = np.random.default_rng(seed)

        observed_nodes = list(self.v)
        latent_nodes = list(self.u)

        obs_by_name: Dict[str, List[DTimeVar]] = {}
        for node in observed_nodes:
            obs_by_name.setdefault(node.name, []).append(node)

        observed_set = set(observed_nodes)
        latent_set = set(latent_nodes)

        transition_specs: Dict[str, Dict[str, Any]] = {}
        contemp_graph = nx.DiGraph()

        # latent families are indexed by (latent_name, lag)
        latent_families: set[Tuple[str, int]] = set()

        for var_name, nodes in obs_by_name.items():
            latest = max(nodes, key=lambda n: n.time)

            if latest not in self.f:
                raise ValueError(f"Missing structural equation for latest node {latest!r}")

            spec = self.f[latest]
            if "parents" not in spec or "func" not in spec:
                raise ValueError(
                    f"Structural specification for {latest!r} must contain 'parents' and 'func'"
                )

            parents = spec["parents"]
            func = spec["func"]

            observed_parent_specs = []
            latent_parent_specs = []

            for p in parents:
                if not isinstance(p, DTimeVar):
                    raise TypeError(f"Parent {p!r} is not a DTimeVar")

                lag = latest.time - p.time
                if lag < 0:
                    raise ValueError(f"Negative lag found for parent {p} of child {latest}")

                if p in observed_set:
                    observed_parent_specs.append((p, p.name, lag))
                    if lag == 0:
                        contemp_graph.add_edge(p.name, latest.name)
                elif p in latent_set:
                    latent_parent_specs.append((p, p.name, lag))
                    latent_families.add((p.name, lag))
                else:
                    raise ValueError(f"Parent {p!r} is neither observed nor latent in this SCM")

            transition_specs[var_name] = {
                "template_child": latest,
                "func": func,
                "observed_parents": observed_parent_specs,
                "latent_parents": latent_parent_specs,
            }
            contemp_graph.add_node(var_name)

        if not nx.is_directed_acyclic_graph(contemp_graph):
            raise ValueError(
                "The latest mechanisms induce a contemporaneous cycle, so they cannot be "
                "used as a recursive transition template."
            )

        eval_order = list(nx.topological_sort(contemp_graph))
        total_time = burn_in + n_timepoints

        obs_hist: Dict[str, np.ndarray] = {
            name: np.full((n_samples, total_time), np.nan, dtype=float)
            for name in obs_by_name
        }

        # one latent history per latent family (latent_name, lag)
        lat_hist: Dict[Tuple[str, int], np.ndarray] = {
            lf: np.full((n_samples, total_time), np.nan, dtype=float)
            for lf in sorted(latent_families)
        }

        # representative template latent node for each latent family
        latent_template_nodes: Dict[Tuple[str, int], DTimeVar] = {}
        for spec in transition_specs.values():
            for template_parent, latent_name, lag in spec["latent_parents"]:
                key = (latent_name, lag)
                if key not in latent_template_nodes:
                    latent_template_nodes[key] = template_parent

        # draw latent innovations
        for t in range(total_time):
            for lf in lat_hist:
                template_latent = latent_template_nodes[lf]
                lat_hist[lf][:, t] = self._draw_noise(template_latent, n_samples, rng)

        def get_obs(name: str, t: int) -> np.ndarray:
            if t < 0:
                return np.full(n_samples, initial_value, dtype=float)
            return obs_hist[name][:, t]

        def get_lat(lat_name: str, lag: int, t: int) -> np.ndarray:
            key = (lat_name, lag)
            if key not in lat_hist or t < 0:
                return np.full(n_samples, initial_value, dtype=float)
            return lat_hist[key][:, t]

        # simulate forward
        for t in range(total_time):
            for var_name in eval_order:
                spec = transition_specs[var_name]
                func = spec["func"]

                vals = np.empty(n_samples, dtype=float)

                for i in range(n_samples):
                    par_vals = {}

                    for template_parent, parent_name, lag in spec["observed_parents"]:
                        par_vals[template_parent] = float(get_obs(parent_name, t - lag)[i])

                    for template_parent, latent_name, lag in spec["latent_parents"]:
                        par_vals[template_parent] = float(get_lat(latent_name, lag, t - lag)[i])

                    vals[i] = float(self._call_func(func, par_vals, 0.0))

                obs_hist[var_name][:, t] = vals

        keep_times = list(range(burn_in, total_time))

        obs_cols = {}
        for var_name in sorted(obs_hist):
            for j, t in enumerate(keep_times, start=1):
                col = DTimeVar(var_name, j) if reindex_time else DTimeVar(var_name, t + 1)
                obs_cols[col] = obs_hist[var_name][:, t]

        df_observed = pd.DataFrame(obs_cols)

        if include_latent:
            lat_cols = {}
            for (lat_name, lag), hist in sorted(lat_hist.items()):
                col_name = f"{lat_name}_lag{lag}"
                for j, t in enumerate(keep_times, start=1):
                    col = DTimeVar(col_name, j) if reindex_time else DTimeVar(col_name, t + 1)
                    lat_cols[col] = hist[:, t]
            df_latent = pd.DataFrame(lat_cols)
            return df_observed, df_latent

        return df_observed

    def generate_causally_stationary_time_series_from_latest_mechanisms(self, n_timepoints: int = 100, burn_in: int = 200, include_latent: bool = False, time_series_format: str = "wide_row", seed: Optional[int] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate time series data from the causally stationary template implied by the SCM.
        Parameters
        ----------
        n_timepoints : int
            Number of retained time points after burn-in.
        burn_in : int
            Number of initial simulated time points to discard.
        include_latent : bool
            Whether to also return latent innovations.
        time_series_format : str
            Format of the returned time series data. Currently only "wide_row",  "ts", and "ts_lagged" is supported, which returns a DataFrame with one row and columns as DTimeVar.."""
        if time_series_format not in {"wide_row", "ts", "ts_lagged"}:
            raise ValueError(f"Unsupported time_series_format: {time_series_format!r}")

        if include_latent:
            df_observed, df_latent = self.generate_causally_stationary_data_from_latest_mechanisms(n_timepoints=n_timepoints, n_samples=1, burn_in=burn_in,
                                               include_latent=include_latent, seed=seed, reindex_time=True)
            if time_series_format == "ts":
                ts_observed = wide_timevar_to_ts_df(df_observed)
                ts_latent = wide_timevar_to_ts_df(df_latent)
            elif time_series_format == "ts_lagged":
                max_delay = self.get_max_delay(include_latent=include_latent)
                ts_observed = wide_timevar_to_lagged_df(df_observed, window_size=2*(max_delay + 1))
                ts_latent = wide_timevar_to_lagged_df(df_latent, window_size=2*(max_delay + 1))
            else:
                ts_observed, ts_latent = df_observed, df_latent
            return ts_observed, ts_latent
        else:
            df_observed = self.generate_causally_stationary_data_from_latest_mechanisms(n_timepoints=n_timepoints, n_samples=1, burn_in=burn_in,
                                               include_latent=include_latent, seed=seed, reindex_time=True)
            if time_series_format == "ts":
                ts_observed = wide_timevar_to_ts_df(df_observed)
            elif time_series_format == "ts_lagged":
                max_delay = self.get_max_delay(include_latent=include_latent)
                ts_observed = wide_timevar_to_lagged_df(df_observed, window_size=2*(max_delay + 1))
            else:
                ts_observed = df_observed
        return ts_observed


class CtDynamicSCM(DynamicSCM):
    def __init__(self, v: Iterable[CTimeVar], u: Iterable[CTimeVar], f: Dict[CTimeVar, Dict[str, Any]], u_dist: Optional[Any] = None):
        super().__init__(v, u, f, u_dist)
        if not all(isinstance(node, CTimeVar) for node in self.v):
            raise TypeError("CtDynamicSCM requires CTimeVar observed variables.")
        if not all(isinstance(node, CTimeVar) for node in self.u):
            raise TypeError("CtDynamicSCM requires CTimeVar latent variables.")


def create_random_linear_dt_dynamic_scm_from_ftadmg(ftadmg: FtAcyclicDirectedMixedGraph,
                                      weight_low: float = -1.0,
                                      weight_high: float = 1.0,
                                      intercept_low: float = 0.0,
                                      intercept_high: float = 0.0,
                                      u_dist: Optional[Any] = None,
                                      causally_stationary: bool = True,
                                      seed: Optional[int] = None) -> DtDynamicSCM:
    """
    :param ftadmg:
    :param weight_low:
    :param weight_high:
    :param intercept_low:
    :param intercept_high:
    :param u_dist:
    :param seed:
    :return:
    """
    rng = np.random.default_rng(seed)
    # Build observed vertex list (DTimeVar objects)
    v = sorted(ftadmg.get_vertices(), key=lambda n: (n.name, n.time))

    # Create latent variables for confounded edges and per-target latents
    # sort confounded edges deterministically by endpoint names and times
    confounded = sorted(
        ftadmg.get_confounded_edges(),
        key=lambda ab: (ab[0].name, ab[0].time, ab[1].name, ab[1].time)
    )

    # latent_map: map (a, b) -> DTimeVar latent (for confounding between a and b)
    latent_map: Dict[Tuple[DTimeVar, DTimeVar], DTimeVar] = {}
    u_list: List[DTimeVar] = []

    for (a, b) in confounded:
        base_name = f"U_{a.name}_{b.name}"
        # Use time of the confounded event (a.time)
        latent = DTimeVar(base_name, a.time)
        # ensure uniqueness: if name conflict, append suffix
        i = 1
        while latent in u_list:
            latent = DTimeVar(f"{base_name}_{i}", a.time)
            i += 1
        latent_map[(a, b)] = latent
        u_list.append(latent)

    # Create a per-target latent for each observed DTimeVar (like idiosyncratic noise)
    per_target_latent: Dict[DTimeVar, DTimeVar] = {}
    for tgt in v:
        name = f"U_{tgt.name}"
        latent = DTimeVar(name, tgt.time)
        i = 1
        while latent in u_list:
            latent = DTimeVar(f"{name}_{i}", tgt.time)
            i += 1
        per_target_latent[tgt] = latent
        u_list.append(latent)

    # Finalize u
    u = u_list

    # Build coefficients mapping (src, tgt) -> weight for all edges
    coefficients: Dict[Tuple[DTimeVar, DTimeVar], float] = {}

    # Add confounding edges coefficients: latent -> each confounded observed (weight=1)
    for (a, b), latent in latent_map.items():
        coefficients[(latent, a)] = 1.0
        coefficients[(latent, b)] = 1.0

    # Add per-target latent coefficients
    for tgt, latent in per_target_latent.items():
        coefficients[(latent, tgt)] = 1.0

    # Directed edges from the FT-ADMG
    # sort directed edges deterministically by src/tgt names and times
    directed_edges = sorted(
        ftadmg.get_directed_edges(),
        key=lambda st: (st[0].name, st[0].time, st[1].name, st[1].time)
    )
    if causally_stationary:
        # assign one weight per (src_name, tgt_name, lag) template and reuse for all matching edges
        weight_template: Dict[Tuple[str, str, int], float] = {}
        for (src, tgt) in directed_edges:
            lag = tgt.time - src.time
            key = (src.name, tgt.name, lag)
            if key not in weight_template:
                weight_template[key] = float(rng.uniform(weight_low, weight_high))
            coefficients[(src, tgt)] = weight_template[key]
    else:
        for (src, tgt) in directed_edges:
            w = float(rng.uniform(weight_low, weight_high))
            coefficients[(src, tgt)] = w

    # Build intercepts for all variables (observed + latent)
    all_vars = list(u) + list(v)
    intercepts: Dict[DTimeVar, float] = {var: float(rng.uniform(intercept_low, intercept_high)) for var in all_vars}

    # Build f mapping: for each variable collect parents and linear function
    f_map: Dict[DTimeVar, Dict[str, Any]] = {}

    # Precompute directed parents from admg
    directed_parent_map: Dict[DTimeVar, List[DTimeVar]] = {var: [] for var in v}
    for (src, tgt) in directed_edges:
        directed_parent_map[tgt].append(src)

    for var in all_vars:
        parents: List[DTimeVar] = []
        if var in v:
            # observed variable: add directed parents
            parents.extend(directed_parent_map.get(var, []))
            # add confounding latents that involve this variable (match by exact DTimeVar)
            for (a, b), latent in latent_map.items():
                if var == a or var == b:
                    parents.append(latent)
            # add its per-target latent
            parents.append(per_target_latent[var])
        else:
            # latent variables: no parents
            parents = []

        b = float(intercepts.get(var, 0.0))

        def make_linear_func(target, parents_list, b_val):
            def func(par_vals, noise):
                s = b_val
                for p in parents_list:
                    s += float(par_vals.get(p, 0.0)) * float(coefficients.get((p, target), 0.0))
                s += noise
                return s
            return func

        if len(parents) > 0:
            f_map[var] = {'parents': parents, 'func': make_linear_func(var, parents, b)}
        else:
            # latent or exogenous variable -> func None (treated as pure noise)
            f_map[var] = {'parents': parents, 'func': None}

    scm = DtDynamicSCM(v=v, u=u, f=f_map, u_dist=u_dist)
    return scm


def create_random_linear_dt_dynamic_scm(
    num_ts: int = 3,
    p_edge: float = 0.3,
    causally_stationary: bool = True,
    max_delay: int = 2,
    allow_instantaneous: bool = True,
    allow_unmeasured_confounding: bool = True,
    weight_low: float = -1.0,
    weight_high: float = 1.0,
    intercept_low: float = 0.0,
    intercept_high: float = 0.0,
    u_dist: Optional[Any] = None,
    seed: Optional[int] = None
) -> DtDynamicSCM:
    """
    Create a random linear DtDynamicSCM with specified parameters.

    Parameters
    ----------
    num_ts : int
        Number of observed time series (per time step).
    max_delay : int
        Maximum lag for directed edges.
    weight_low : float
        Minimum edge weight.
    weight_high : float
        Maximum edge weight.
    intercept_low : float
        Minimum intercept value.
    intercept_high : float
        Maximum intercept value.
    u_dist : Optional[Any]
        Distribution for latent variables (see DtDynamicSCM._draw_noise).
    causally_stationary : bool
        Whether to enforce causal stationarity in the generated SCM.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    ftadmg = create_random_ft_admg(num_ts=num_ts, p_edge=p_edge, causally_stationary=causally_stationary, max_delay=max_delay, allow_instantaneous=allow_instantaneous, allow_unmeasured_confounding=allow_unmeasured_confounding, seed=seed)
    scm = create_random_linear_dt_dynamic_scm_from_ftadmg(ftadmg=ftadmg, weight_low=weight_low, weight_high=weight_high, intercept_low=intercept_low, intercept_high=intercept_high, u_dist=u_dist, causally_stationary=causally_stationary, seed=seed)
    return scm


if __name__ == '__main__':
    X_1 = DTimeVar('X', 1)
    X_2 = DTimeVar('X', 2)
    X_3 = DTimeVar('X', 3)
    Y_1 = DTimeVar('Y', 1)
    Y_2 = DTimeVar('Y', 2)
    U_1 = DTimeVar('U', 1)
    U_2 = DTimeVar('U', 2)
    U_3 = DTimeVar('U', 3)


    def f_x(par_vals, noise):
        return 0.4 * par_vals[X_1] + 0.8 * par_vals[X_2] + 0.5 * noise

    scm = DtDynamicSCM(
        v=[X_1, X_2, X_3, Y_1, Y_2],
        u=[U_1, U_2, U_3],
        f={
            X_1: {
                "parents": [U_1],
                "func": lambda par_vals, noise: 0.5 * par_vals[U_1]
            },
            X_2: {
                "parents": [X_1, U_2],
                "func": lambda par_vals, noise: 0.8 * par_vals[X_1] + 0.5 * par_vals[U_2]
            },
            X_3: {
                "parents": [X_1, X_2, U_3],
                "func": lambda par_vals, noise: (
                        0.4 * par_vals[X_1] + 0.8 * par_vals[X_2] + 0.5 * par_vals[U_3]
                )
            },
            Y_1: {
                "parents": [X_1, U_1],
                "func": lambda par_vals, noise: 0.7 * par_vals[X_1] + 0.5 * par_vals[U_1]
            },
            Y_2: {
                "parents": [Y_1, U_2, X_1],
                "func": lambda par_vals, noise: (
                        0.6 * par_vals[Y_1] + 0.7 * par_vals[X_1] + 0.5 * par_vals[U_2]
                )
            },
        },
        u_dist=lambda: np.random.normal(0, 1)
    )

    df, df_l = scm.generate_data(n_samples=100, include_latent=True, seed=42)
    # print(df)
    # print(df_l)
    # print(type(scm.f))
    # print(scm.f)
    print(scm.get_max_delay())
    print(scm.is_structurally_causally_stationary())

    # data = scm.generate_causally_stationary_time_series(n_timepoints=2000, burn_in=5, include_latent=False, time_series_format="wide_row", seed=42)
    # print(data)
    # data1= wide_timevar_to_ts_df(data)
    # print(data1)
    # w = 2*(scm.get_max_delay(include_latent=False) + 1)
    # w=1
    # data2 = wide_timevar_to_lagged_df(data, window_size=w, )
    # print(data2)

    # scm = create_random_linear_dt_dynamic_scm(num_ts=3, max_delay=2, weight_low=-1.0, weight_high=1.0, intercept_low=0.0, intercept_high=0.0, u_dist=lambda: np.random.normal(0, 1), causally_stationary=True, seed=42)
    g = scm.induced_admg()
    g.draw_graph()
    print(g.get_directed_edges())
    data = scm.generate_causally_stationary_time_series_from_latest_mechanisms(n_timepoints=2000, burn_in=5, include_latent=False, time_series_format="ts_lagged", seed=42)
    # print(data.to_string())
    scg = g.get_summary_causal_graph()
    scg.draw_graph()
    print(scg.get_directed_edges())
