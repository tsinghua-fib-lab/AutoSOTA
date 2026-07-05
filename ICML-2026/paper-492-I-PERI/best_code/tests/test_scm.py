# Run tests with pytest; they are designed to be deterministic when seeds are provided.

import pytest
import pandas as pd

from pyciphod.utils.scms.scm import SCM, LinearSCM, create_random_linear_scm, create_random_linear_scm_from_admg
from pyciphod.utils.graphs.graphs import AcyclicDirectedMixedGraph


def test_linear_scm_generation_and_shapes():
    v = ['X', 'Y']
    u = ['U']
    coeffs = {('U', 'X'): 2.0, ('X', 'Y'): 3.0}
    scm = LinearSCM(v=v, u=u, coefficients=coeffs)

    df = scm.generate_data(100, include_latent=False, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 2)
    assert list(df.columns) == v

    df_obs, df_lat = scm.generate_data(10, include_latent=True, seed=0)
    assert isinstance(df_obs, pd.DataFrame) and isinstance(df_lat, pd.DataFrame)
    assert df_obs.shape == (10, 2)
    assert df_lat.shape == (10, 1)


def test_generate_data_reproducibility():
    v = ['X']
    u = ['U']
    coeffs = {('U', 'X'): 1.0}
    scm = LinearSCM(v=v, u=u, coefficients=coeffs)
    a = scm.generate_data(50, include_latent=False, seed=7)
    b = scm.generate_data(50, include_latent=False, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_forbidden_observed_parent_of_latent_via_f():
    v = ['X']
    u = ['L']
    # f says latent L has parent X (forbidden)
    f = {'L': {'parents': ['X'], 'func': lambda parents, noise: parents['X'] + noise}}
    with pytest.raises(ValueError):
        SCM(v=v, u=u, f=f)


def test_forbidden_observed_parent_of_latent_via_coeffs():
    v = ['X']
    u = ['L']
    coeffs = {('X', 'L'): 1.5}
    with pytest.raises(ValueError):
        LinearSCM(v=v, u=u, coefficients=coeffs)


def test_induced_dag_no_confounding():
    # U -> X -> Y (U only affects X)
    v = ['X', 'Y']
    u = ['U']
    coeffs = {('U', 'X'): 1.0, ('X', 'Y'): 2.0}
    scm = LinearSCM(v=v, u=u, coefficients=coeffs)
    dag = scm.induced_dag()
    # expected directed edge among observed: X->Y
    directed = dag.get_directed_edges()
    assert ('X', 'Y') in directed
    # dag should not contain latent nodes
    assert set(dag.get_vertices()) == set(v)


def test_induced_dag_raises_with_confounding():
    # U -> X, U -> Y causes unobserved confounding
    v = ['X', 'Y']
    u = ['U']
    coeffs = {('U', 'X'): 1.0, ('U', 'Y'): 1.0}
    scm = LinearSCM(v=v, u=u, coefficients=coeffs)
    with pytest.raises(ValueError):
        _ = scm.induced_dag()


def test_induced_admg_with_confounding():
    v = ['X', 'Y']
    u = ['U']
    coeffs = {('U', 'X'): 1.0, ('U', 'Y'): 1.0, ('X', 'Y'): 0.0}
    scm = LinearSCM(v=v, u=u, coefficients=coeffs)
    admg = scm.induced_admg()
    conf = admg.get_confounded_edges()
    # confounded edges stored undirected; check either order
    assert any((('X', 'Y') == e or ('Y', 'X') == e) for e in conf)


def test_create_random_linear_scm_with_confounding_and_generation():
    scm, coeffs, intercepts = create_random_linear_scm(num_v=4, p_edge=0.6, unmeasured_confounding=True, seed=1)
    # sample
    df_obs, df_lat = scm.generate_data(n_samples=20, include_latent=True, seed=2)
    assert df_obs.shape[0] == 20
    assert df_lat.shape[0] == 20
    # observed columns should be named X0..X3
    assert set(df_obs.columns).issuperset({f'X{i}' for i in range(4)})


def test_create_random_linear_scm_from_admg_generates_latents():
    admg = AcyclicDirectedMixedGraph()
    admg.add_vertex('X')
    admg.add_vertex('Y')
    admg.add_confounded_edge('X', 'Y')
    admg.add_directed_edge('X', 'Y')

    scm, coeffs, intercepts = create_random_linear_scm_from_admg(admg, seed=3)
    # coefficients should include keys starting with 'U_' for latent variables
    assert any(k[0].startswith('U_') for k in coeffs.keys())
    # generating data should work
    df_obs, df_lat = scm.generate_data(5, include_latent=True, seed=0)
    assert df_obs.shape[0] == 5
    assert df_lat.shape[0] == 5
