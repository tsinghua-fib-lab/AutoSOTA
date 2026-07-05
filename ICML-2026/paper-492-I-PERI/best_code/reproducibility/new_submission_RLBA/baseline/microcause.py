import pandas as pd
import numpy as np
import networkx as nx

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp

from .estimation import FisherZ
import random


def find_transition_matrix(g, df, backward_step=0.1):
    transition_df = pd.DataFrame(
        np.zeros([len(df.columns), len(df.columns)]),
        columns=df.columns,
        index=df.columns,
    )

    for edge in g.edges:
        cause = edge[0]
        effect = edge[1]

        if cause != effect:
            par_cause = list(g.predecessors(cause))
            par_effect = list(g.predecessors(effect))

            if cause in par_cause:
                par_cause.remove(cause)
            if effect in par_effect:
                par_effect.remove(effect)
            if cause in par_effect:
                par_effect.remove(cause)

            cond_list = list(set(par_cause + par_effect))

            corr = FisherZ(cause, effect, cond_list=cond_list)
            stat = abs(corr.get_dependence(df))

            print(cause, effect, cond_list, stat)

            # Forward step: cause -> effect
            transition_df.loc[cause, effect] = stat

            # Backward step: effect -> cause
            transition_df.loc[effect, cause] = backward_step * stat

    # Self step
    adj = nx.to_numpy_array(g)

    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                adj[j, i] = 1

    for node in g.nodes:
        for i, col_i in enumerate(list(df.columns)):
            P_pc_max = []

            for k in adj[:, i].nonzero()[0]:
                col_k = df.columns[k]

                if node != col_k:
                    par_cause = list(g.predecessors(col_k))
                    par_effect = list(g.predecessors(node))

                    if col_k in par_cause:
                        par_cause.remove(col_k)
                    if node in par_effect:
                        par_effect.remove(node)
                    if col_k in par_effect:
                        par_effect.remove(col_k)

                    cond_list = list(set(par_cause + par_effect))

                    corr = FisherZ(node, col_k, cond_list=cond_list)
                    stat = abs(corr.get_dependence(df))
                    P_pc_max.append(stat)

            P_pc_max = np.max(P_pc_max) if len(P_pc_max) > 0 else 0

            if node != col_i:
                par_cause = list(g.predecessors(col_i))
                par_effect = list(g.predecessors(node))

                if col_i in par_cause:
                    par_cause.remove(col_i)
                if node in par_cause:
                    par_cause.remove(node)
                if node in par_effect:
                    par_effect.remove(node)
                if col_i in par_effect:
                    par_effect.remove(col_i)

                cond_list = list(set(par_cause + par_effect))

                corr = FisherZ(node, col_i, cond_list=cond_list)
                stat = abs(corr.get_dependence(df))
                q_ii = stat

                if q_ii > P_pc_max:
                    value = q_ii - P_pc_max
                    if value > transition_df.loc[col_i, col_i]:
                        transition_df.loc[col_i, col_i] = value

    print(transition_df)

    # Normalizing columns
    for effect in g.nodes:
        total_corr = transition_df[effect].sum()

        if total_corr > 0:
            for node in g.nodes:
                transition_df.loc[node, effect] = (
                    transition_df.loc[node, effect] / total_corr
                )

    return transition_df


def random_walk(g, transition_df, start_idx=0, walkLength=10):
    transition_matrix = transition_df.values

    i = random.randint(0, len(g.nodes) - 1)

    visited = [transition_df.columns[i]]
    I = np.arange(len(g.nodes))

    visited.append(transition_df.columns[i])

    for _ in range(walkLength):
        if sum(transition_matrix[:, i]) > 0.99:
            i = np.random.choice(I, p=transition_matrix[:, i])
        else:
            i = np.random.choice(I)

        visited.append(transition_df.columns[i])

    return visited


def micro_cause(
    data,
    anomalous_nodes,
    anomalies_start_time=None,
    anomaly_length=200,
    gamma_max=1,
    sig_threshold=0.05,
):
    last_start_time_anomaly = 0

    for node in anomalous_nodes:
        last_start_time_anomaly = max(
            last_start_time_anomaly,
            anomalies_start_time[node],
        )

    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]

    dataframe = pp.DataFrame(
        anomalous_data.values,
        datatime=np.arange(len(anomalous_data)),
        var_names=anomalous_data.columns,
    )

    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=gamma_max,
        pc_alpha=sig_threshold,
    )

    g = nx.DiGraph()
    g.add_nodes_from(data.columns)

    for name_y in pcmci.all_parents.keys():
        for name_x, t_xy in pcmci.all_parents[name_y]:
            edge = (data.columns[name_x], data.columns[name_y])

            if edge not in g.edges:
                g.add_edge(*edge)

    transition_df = find_transition_matrix(g, data, backward_step=0.1)

    visited = random_walk(g, transition_df, walkLength=1000)
    freq_dict = {x: visited.count(x) for x in visited}

    rc_list = []

    if len(freq_dict.keys()) > 1:
        for _ in range(2):
            rc = max(freq_dict, key=freq_dict.get)

            if rc not in rc_list:
                rc_list.append(rc)

            del freq_dict[rc]
    else:
        rc_list.append(list(freq_dict.keys())[0])

    return rc_list


def micro_cause0(
    graph,
    data,
    anomalous_nodes,
    anomalies_start_time=None,
    anomaly_length=200,
    gamma_max=1,
    sig_threshold=0.05,
):
    last_start_time_anomaly = 0

    for node in anomalous_nodes:
        last_start_time_anomaly = max(
            last_start_time_anomaly,
            anomalies_start_time[node],
        )

    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    _ = data.loc[last_start_time_anomaly:first_end_time_anomaly]

    g = graph

    transition_df = find_transition_matrix(g, data, backward_step=0.1)

    visited = random_walk(g, transition_df, walkLength=1000)
    freq_dict = {x: visited.count(x) for x in visited}

    rc_list = []

    if len(freq_dict.keys()) > 1:
        for _ in range(2):
            rc = max(freq_dict, key=freq_dict.get)

            if rc not in rc_list:
                rc_list.append(rc)

            del freq_dict[rc]
    else:
        rc_list.append(list(freq_dict.keys())[0])

    return rc_list