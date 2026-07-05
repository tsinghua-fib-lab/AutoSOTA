import pandas as pd
import numpy as np

from causal_reasoning.summary_causal_graph.micro_queries import direct_effect
from utils.stat_tests.dependency_measures import LinearRegressionCoefficient
from utils.stat_tests.dependency_measures import grubb_test
from utils.graphs.partially_specified_graphs import SummaryCausalGraph


class SGRCA:
    def __init__(self, g: SummaryCausalGraph, anomalous_vertices, anomalies_start_time=None, anomaly_length=200, max_delay=1,
                 sig_threshold=0.05):
        self.g = g
        self.anomalous_vertices = anomalous_vertices
        self.anomalies_start_time = anomalies_start_time
        self.anomaly_length = anomaly_length
        self.max_delay = max_delay
        self.sig_threshold = sig_threshold



        self.maximally_connected_anomalous_components = self._get_maximally_connected_anomalous_components()

        self.root_causes = dict()
        for subgraph_id in self.maximally_connected_anomalous_components.keys():
            self.root_causes[subgraph_id] = {"lineage_defying": [], "time_defying": [], "effect_defying": [],
                                             "noise_defying": []}

        # indicates if data was used or not
        self.search_rc_from_graph = False
        # indicates if data was used or not
        self.search_rc_from_data = False

    def _get_maximally_connected_anomalous_components(self):
        # find maximally connected components in the anomalous subgraph
        maximally_connected_components = list()
        anomalous = set(self.anomalous_vertices)
        visited = set()

        for start in anomalous:
            if start in visited:
                continue

            component = set()
            stack = [start]

            while stack:
                v = stack.pop()
                if v in visited:
                    continue

                visited.add(v)
                component.add(v)

                for u in anomalous:
                    if u not in visited and self.g.is_adjacent(v, u):
                        stack.append(u)

            maximally_connected_components.append(component)

        # assign a unique id to each maximally connected component and return as a dict
        mcac_dict = dict()
        for i, component in enumerate(maximally_connected_components):
            g_anom = SummaryCausalGraph()
            for v in component:
                g_anom.add_vertex(v)
                for u in component:
                    if (v, u) in self.g.directed_edges:
                        g_anom.add_directed_edge(v, u)
                    if (v, u) in self.g.confounded_edges:
                        g_anom.add_confounded_edge(v, u)
                    if (v, u) in self.g.uncertain_edges:
                        g_anom.add_uncertain_edge(v, u)
            mcac_dict[i] = g_anom
        return mcac_dict

    def get_lineage_defying(self):
        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.vertices:
                if not subgraph.get_parents(v):
                    self.root_causes[subgraph_id]["lineage_defying"].append(v)

    def get_time_defying(self):
        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.vertices:
                par = subgraph.get_parents(v)
                par.remove(v)
                test_passed = True
                for u in par:
                    if self.anomalies_start_time(v)>self.anomalies_start_time(u):
                        test_passed = False
                        break
                if test_passed:
                    self.root_causes[subgraph_id]["lineage_defying"].append(v)

    def _split_data_by_regime(self, data):
        # TODO
        # split data into regime before anomaly and regime during anomaly
        data_before_anomaly = data[data["timestamp"] < self.anomalies_start_time]
        data_during_anomaly = data[(data["timestamp"] >= self.anomalies_start_time) & (data["timestamp"] < self.anomalies_start_time + self.anomaly_length)]
        return data_before_anomaly, data_during_anomaly

    def get_effect_defying(self, data):
        # TODO
        data_before_anomaly, data_during_anomaly = self._split_data_by_regime(data)
        normal_data_batchs = np.array_split(data_before_anomaly, 10)
        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.vertices:
                if v not in self.root_causes[subgraph_id]["lineage_defying"] and v not in self.root_causes[subgraph_id]["time_defying"]:
                    for u in subgraph.get_parents(v):
                        adj_set = 1
                        lr = LinearRegressionCoefficient(u, v, cond_list=adj_set, drop_na=False)
                        dep_before = lr.get_dependence(data_before_anomaly)
                        dep_during = lr.compute_direct_effect(data_during_anomaly)
                        dep_list = [dep_before, dep_during]
                        grubb_test_res = grubb_test(dep_list, confidence_level=self.sig_threshold)
                        if grubb_test_res > self.sig_threshold:
                            self.root_causes[subgraph_id]["effect_defying"].append(v)


    def get_noise_defying(self, data):
        # TODO
        1

    def run(self, data):
        self.get_lineage_defying()
        self.get_time_defying()
        self.get_effect_defying(data)
        self.get_noise_defying(data)
        return self.root_causes