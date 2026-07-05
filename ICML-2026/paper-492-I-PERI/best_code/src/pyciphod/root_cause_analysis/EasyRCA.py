import networkx as nx
import pandas as pd
import numpy as np
from causal_reasoning.summary_causal_graph.micro_queries import adjutment_set_for_direct_effect_in_ascgl_using_parentsY
from utils.stat_tests.dependency_measures import LinearRegressionCoefficient
from utils.stat_tests.dependency_measures import grubb_test
from utils.graphs.partially_specified_graphs import SummaryCausalGraph


class EasyRCA:
    def __init__(self, summary_graph:SummaryCausalGraph, anomalous_nodes, anomalies_start_time=None, anomaly_length=200, gamma_max=1,
                 sig_threshold=0.05):
        # TODO add a parameter distinguish_anomaly_type which chooses between one of the strategies for when to distinguish between structural and parametric:
        # always, except_subroots, except_time_defying, except_subroots_and_time_defying, never.
        # In some cases (probably all but "never"), every direct effect should be estimated so that an intervention which is structural *and* parametric is counted as structural.
        # This might be implemented as a level of confidence of each type of root cause (subroot, time defying, structural, parametric) so that EasyRCA tries to find the ones with the most confidence first.
        """
        :param summary_graph: networkx graph
        :param anomalous_nodes: list
        :param anomalies_start_time: dict
        """
        # test if summary graph is acyclic without self loops
        dag = summary_graph.copy()
        for node in summary_graph.get_vetrices():
            if node in dag.get_adjacencies(node):
                dag.remove_directed_edge(node, node)
        for node in summary_graph.get_vetrices():
            if node in dag.get_adjacencies(node):
                print("Summary causal graph has self loops, please remove them before using EasyRCA")
                exit(0)
        if not dag.is_acyclic():
            print("Summary causal graph is not acylic if self loops are omited")
            exit(0)

        self.summary_graph = summary_graph
        self.anomalous_nodes = anomalous_nodes
        self.anomalies_start_time = anomalies_start_time
        self.anomaly_length = anomaly_length
        self.gamma_max = gamma_max
        self.sig_threshold = sig_threshold

        self.dict_linked_anomalous_graph = dict()
        self._find_linked_anomalous_graphs()

        # self._process_anomalies_intervals()

        self.root_causes = dict()
        for subgraph_id in self.dict_linked_anomalous_graph.keys():
            self.root_causes[subgraph_id] = {"roots": [], "time_defying": [], "structure_defying": [],
                                             "param_defying": []}

        self.get_recommendations = pd.DataFrame()

        # indicates if data was used or not
        self.search_rc_from_graph = False
        # indicates if data was used or not
        self.search_rc_from_data = False

        # Minimum lag between each edge in the graph: if both nodes in an edges are anomalous then detect the min lag by
        # looking at the time of the apearence of anomalies, if one the nodes is not anomalous then min lag is 0
        self.gamma_min_dict = dict()
        self.d_sep_by_empty_in_manip_graph = dict()
        self._get_gamma_min()

        self.nodes_to_temporal_nodes = dict()
        self.temporal_nodes = []
        for node in self.summary_graph.nodes:
            self.nodes_to_temporal_nodes[node] = []
            for gamma in range(2 * self.gamma_max + 1):
                if gamma == 0:
                    temporal_node = str(node) + "_t"
                    self.nodes_to_temporal_nodes[node].append(temporal_node)
                    self.temporal_nodes.append(temporal_node)
                else:
                    temporal_node = str(node) + "_t_" + str(gamma)
                    self.nodes_to_temporal_nodes[node].append(temporal_node)
                    self.temporal_nodes.append(temporal_node)

    def _get_gamma_min(self):
        """
            Find Minimum lag between each edge in the graph: if both nodes in an edges are anomalous then detect the min lag by
        # looking at the time of the apearence of anomalies, if one the nodes is not anomalous then min lag is 0
        :return: 1
        """
        dag = remove_self_loops(self.summary_graph)
        for edge in self.summary_graph.edges:
            if (edge[0] in self.anomalous_nodes) and (edge[1] in self.anomalous_nodes):
                if edge[0] == edge[1]:
                    # self.d_sep_by_empty_in_manip_graph[edge] = False
                    self.gamma_min_dict[edge] = 1
                else:
                    manip_dag = dag.copy()
                    manip_dag.remove_edge(edge[0], edge[1])
                    x_set = set([edge[0]])
                    y_set = set([edge[1]])
                    z_set = set()
                    if nx.d_separated(manip_dag, x_set, y_set, z_set):
                        self.d_sep_by_empty_in_manip_graph[edge] = True
                        self.gamma_min_dict[edge] = self.anomalies_start_time[edge[1]] - self.anomalies_start_time[edge[0]]
                    else:
                        # todo apply for nodes that do not have an undirected (activated) path
                        self.d_sep_by_empty_in_manip_graph[edge] = False
                        # all_path_x_y = nx.all_simple_paths(manip_dag, edge[0], edge[1])
                        # list_possible_min_gamma = [self.anomalies_start_time[edge[1]] -
                        #                            self.anomalies_start_time[edge[0]]]
                        # for path in all_path_x_y:
                        #     path_graph = manip_dag.subgraph(path)
                        #     if not nx.d_separated(path_graph, x_set, y_set, z_set):
                        #         possible_min_gamma = 0
                        #         for new_edge in path_graph.edges:
                        #             possible_min_gamma = possible_min_gamma + self.anomalies_start_time[new_edge[1]] - \
                        #                                  self.anomalies_start_time[new_edge[0]]
                        #         list_possible_min_gamma.append(possible_min_gamma)
                        # self.gamma_min_dict[edge] = min(list_possible_min_gamma)
                        if self.anomalies_start_time[edge[1]] - self.anomalies_start_time[edge[0]] < 0:
                            self.gamma_min_dict[edge] = 0
                        else:
                            self.gamma_min_dict[edge] = self.anomalies_start_time[edge[1]] - self.anomalies_start_time[
                                edge[0]]

            else:
                self.gamma_min_dict[edge] = 0

    def _find_linked_anomalous_graphs(self):
        """
            Find linked anomalous graphs, given the initial summary causal graph by looking grouping all anomalous
            nodes that have an undirected path between them
        :return: The linked anomalous graphs dict
        """

        undirected_graph = self.summary_graph.to_undirected()
        undirected_acyclic_graph = remove_self_loops(undirected_graph)
        temp_anomalous_nodes = self.anomalous_nodes

        treated_nodes = []
        id_subgraph = 0
        for node in temp_anomalous_nodes:
            if node not in treated_nodes:
                linked_anomalous_nodes = [node]
                list_neighbors = [node]
                while len(list_neighbors) > 0:
                    new_list_neighbors = []
                    for neighbor in list_neighbors:
                        # Add only anomalous nodes and the new nodes not seen before
                        new_list_neighbors += [element for element in
                                               list(undirected_acyclic_graph.neighbors(neighbor)) if element not in
                                               linked_anomalous_nodes and element in self.anomalous_nodes]

                        linked_anomalous_nodes = linked_anomalous_nodes + new_list_neighbors
                    list_neighbors = new_list_neighbors
                linked_anomalous_nodes = list(set(linked_anomalous_nodes))
                treated_nodes = treated_nodes + linked_anomalous_nodes

                if len(linked_anomalous_nodes) > 0:
                    sub_g = self.summary_graph.subgraph(linked_anomalous_nodes).copy()
                    self.dict_linked_anomalous_graph[id_subgraph] = sub_g
                    id_subgraph = id_subgraph + 1

    def _search_roots(self, subgraph_id):
        """
        Find roots of a given linked anomalous graph
        :param subgraph_id: the id of linked anomalous graph
        :return: Void
        """
        linked_anomalous_graph = self.dict_linked_anomalous_graph[subgraph_id]
        dag = remove_self_loops(linked_anomalous_graph)
        for node in dag.nodes:
            parents_of_node = list(dag.predecessors(node))
            if len(parents_of_node) == 0:
                self.root_causes[subgraph_id]["roots"].append(node)

    def _search_time_defiance(self, subgraph_id):
        """
        Use time about the first appearece of anomalies to find nodes that temporally defy the causal structure
        :param subgraph_id: the id of linked anomalous graph
        :return: Void
        """
        linked_anomalous_graph = self.dict_linked_anomalous_graph[subgraph_id]
        dag = remove_self_loops(linked_anomalous_graph)
        for edge in dag.edges:
            node_c = edge[0]
            node_e = edge[1]
            if node_c != node_e:
                if self.d_sep_by_empty_in_manip_graph[edge]:
                    if self.anomalies_start_time[node_c] > self.anomalies_start_time[node_e]:
                        self.root_causes[subgraph_id]["time_defying"].append(node_e)

    def return_other_possible_root_causes(self, subgraph_id):
        """
        Save all nodes that are potentially root causes but were not detected root nodes nor time defying nodes
        :param subgraph_id: the id of linked anomalous graph
        :return: Void
        """
        r = self.root_causes[subgraph_id]["roots"]
        td = self.root_causes[subgraph_id]["time_defying"]
        all_nodes = self.dict_linked_anomalous_graph[subgraph_id].nodes
        possible_root_causes = [node for node in all_nodes if (node not in r) and (node not in td)]
        return possible_root_causes

    def _process_data(self, data):
        new_data = pd.DataFrame()
        for gamma in range(0, 2 * self.gamma_max + 1):
            shifteddata = data.shift(periods=-2 * self.gamma_max + gamma)

            new_columns = []
            for node in data.columns:
                new_columns.append(self.nodes_to_temporal_nodes[node][gamma])
            shifteddata.columns = new_columns
            new_data = pd.concat([new_data, shifteddata], axis=1, join="outer")
        new_data.dropna(axis=0, inplace=True)

        # devide new data into two: normal and anomalous
        last_start_time_normal = 0
        first_end_time_normal = self.anomalies_start_time[self.anomalous_nodes[0]] - 1
        for node in self.anomalous_nodes:
            first_end_time_normal = min(first_end_time_normal, self.anomalies_start_time[node] - 1)
        normal_data = new_data.loc[last_start_time_normal:first_end_time_normal]

        last_start_time_anomaly = 0
        for node in self.anomalous_nodes:
            last_start_time_anomaly = max(last_start_time_anomaly, self.anomalies_start_time[node])
        first_end_time_anomaly = last_start_time_anomaly + self.anomaly_length
        anomalous_data = new_data.loc[last_start_time_anomaly:first_end_time_anomaly]

        return normal_data, anomalous_data


    def _search_structure_defiance_after_param(self, subgraph_id, normal_data, anomalous_data):
        """
        Search for parameteric defying nodes
        :param subgraph_id: the id of linked anomalous graph
        :param normal_data: data without anomalies
        :param anomalous_data: anomalous data
        :return: Void
        """
        linked_anomalous_graph = self.dict_linked_anomalous_graph[subgraph_id]
        dag = remove_self_loops(linked_anomalous_graph)
        for edge in dag.edges:
            x = edge[0]
            y = edge[1]
            all_root_causes_except_param = self.root_causes[subgraph_id]["roots"] + \
                                           self.root_causes[subgraph_id]["time_defying"] + \
                                           self.root_causes[subgraph_id]["structure_defying"]

            if y in self.root_causes[subgraph_id]["param_defying"]:
                if y not in all_root_causes_except_param:
                    # cond_dict, _ = backdoor_from_summary_graph(self.summary_graph, x, y, gamma_max=self.gamma_max,
                    #                                        gamma_min_dict=self.gamma_min_dict,
                    #                                        xy_d_sep_by_empty_in_manip_graph=
                    #                                        self.d_sep_by_empty_in_manip_graph[edge])
                    # cond_dict2, _ = singledoor_from_summary_graph(self.summary_graph, x, y, gamma_max=self.gamma_max,
                    #                                               gamma_min_dict=self.gamma_min_dict,
                    #                                               xy_d_sep_by_empty_in_manip_graph=
                    #                                               self.d_sep_by_empty_in_manip_graph[edge])

                    cond_dict, _ = adjutment_set_for_direct_effect_in_ascgl_using_parentsY(self.summary_graph, x, y,
                                                                            gamma_max=self.gamma_max,
                                                                            gamma_min_dict=self.gamma_min_dict)
                    for gamma in cond_dict.keys():
                        cond_set = cond_dict[gamma]
                        # cond_set = cond_dict[gamma] + cond_dict2[gamma]
                        yt = self.nodes_to_temporal_nodes[y][0]
                        xt = self.nodes_to_temporal_nodes[x][gamma]
                        ci = LinearRegressionCoefficient(xt, yt, cond_set)
                        pval_normal = ci.test_zeo_coef(normal_data)
                        pval_anomalous = ci.test_zeo_coef(anomalous_data)
                        if (pval_anomalous >= self.sig_threshold) and (pval_normal < self.sig_threshold):
                        # if (pval_anomalous >= self.sig_threshold):
                            if y not in self.root_causes[subgraph_id]["structure_defying"]:
                                self.root_causes[subgraph_id]["structure_defying"].append(y)
                                self.root_causes[subgraph_id]["param_defying"].remove(y)

    def _search_param_defiance(self, subgraph_id, normal_data, anomalous_data):
        """
        sds
        :param subgraph_id: the id of linked anomalous graph
        :param data:
        :return:
        """
        # split data
        batch_size = anomalous_data.shape[0]
        split_nb = int(normal_data.shape[0]/batch_size)
        normal_data_batchs = np.array_split(normal_data, split_nb)
        linked_anomalous_graph = self.dict_linked_anomalous_graph[subgraph_id]
        dag = remove_self_loops(linked_anomalous_graph)
        for edge in dag.edges:
            x = edge[0]
            y = edge[1]
            all_root_causes_except_param = self.root_causes[subgraph_id]["roots"] + \
                                           self.root_causes[subgraph_id]["time_defying"] + \
                                           self.root_causes[subgraph_id]["structure_defying"] + \
                                           self.root_causes[subgraph_id]["param_defying"]
            if y not in all_root_causes_except_param:
                # cond_dict, _ = backdoor_from_summary_graph(self.summary_graph, x, y, gamma_max=self.gamma_max,
                #                                         gamma_min_dict=self.gamma_min_dict,
                #                                         xy_d_sep_by_empty_in_manip_graph=
                #                                         self.d_sep_by_empty_in_manip_graph[edge])
                # cond_dict2, _ = singledoor_from_summary_graph(self.summary_graph, x, y, gamma_max=self.gamma_max,
                #                                         gamma_min_dict=self.gamma_min_dict,
                #                                         xy_d_sep_by_empty_in_manip_graph=
                #                                         self.d_sep_by_empty_in_manip_graph[edge])

                cond_dict, _ = adjutment_set_for_direct_effect_in_ascgl_using_parentsY(self.summary_graph, x, y,
                                                                         gamma_max=self.gamma_max,
                                                                         gamma_min_dict=self.gamma_min_dict)
                # print("Cond dict", x, y)
                # print(cond_dict)
                for gamma in cond_dict.keys():
                    cond_set = cond_dict[gamma]
                    # cond_set = cond_dict[gamma] + cond_dict2[gamma]
                    # gamma = self.gamma_min_dict[edge]
                    yt = self.nodes_to_temporal_nodes[y][0]
                    xt = self.nodes_to_temporal_nodes[x][gamma]
                    # ci = FisherZ(xt, yt, cond_set)
                    print(xt, yt, cond_set)
                    ci = LinearRegressionCoefficient(xt, yt, cond_set)
                    pval_normal = ci.test_zeo_coef(normal_data)
                    if (pval_normal < self.sig_threshold):
                        coeff_anomalous = ci.get_coeff(anomalous_data)
                        # _, pval_anomalous = ci.get_pvalue(anomalous_data)
                        pval_list = []
                        pval_list.append(coeff_anomalous)
                        # ref_normal = ci.get_coeff(normal_data_batchs[split_nb-1])
                        for i in range(split_nb - 1):
                            coeff_normal = ci.get_coeff(normal_data_batchs[i])
                            pval_list.append(coeff_normal)
                        print(pval_list)
                        grubb_res = grubb_test(pval_list, confidence_level=self.sig_threshold)
                        if grubb_res["anomaly_position"] == 0:
                            if y not in self.root_causes[subgraph_id]["param_defying"]:
                                self.root_causes[subgraph_id]["param_defying"].append(y)

    @staticmethod
    def _sorted_by_reach(rc_list, dag):
        """
        sds
        :return:
        """
        list_nb_descendants = []
        for node in rc_list:
            list_nb_descendants.append(len(list(nx.descendants(dag, node))))
        sorted_id = sorted(range(len(list_nb_descendants)), key=list_nb_descendants.__getitem__, reverse=True)
        new_rc_list = []
        count_iter = 0
        for i in range(len(sorted_id)):
            if i == 0:
                new_rc_list.append([rc_list[sorted_id[i]]])
                count_iter = count_iter + 1
            else:
                if list_nb_descendants[sorted_id[i]] == list_nb_descendants[sorted_id[i-1]]:
                    new_rc_list[count_iter - 1].append(rc_list[sorted_id[i]])
                else:
                    new_rc_list.append([rc_list[sorted_id[i]]])
                    count_iter = count_iter + 1
        return new_rc_list

    def action_recommendation(self):
        """
        Recommend actions to the user
        :return: ranking of actions that needs to be done on root causes
        """
        for subgraph_id in self.dict_linked_anomalous_graph.keys():
            linked_anomalous_graph = self.dict_linked_anomalous_graph[subgraph_id]
            dag = remove_self_loops(linked_anomalous_graph)

            rc_from_graph = ["roots", "time_defying"]
            sorted_list = []
            for rc in rc_from_graph:
                rc_list = self.root_causes[subgraph_id][rc]
                sorted_list = sorted_list + self._sorted_by_reach(rc_list, dag)

            if self.search_rc_from_data:
                rc_from_data = ["structure_defying", "param_defying"]
                for rc in rc_from_data:
                    rc_list = self.root_causes[subgraph_id][rc]
                    sorted_list = sorted_list + self._sorted_by_reach(rc_list, dag)
            else:
                # Construct other possible root causes
                prc_list = self.return_other_possible_root_causes(subgraph_id)
                sorted_list = sorted_list + self._sorted_by_reach(prc_list, dag)

            self.get_recommendations["LinkedAnomalousGraph_" + str(subgraph_id)] = pd.Series(sorted_list)

    def run_without_data(self):
        self.search_rc_from_graph = True
        for subgraph_id in self.dict_linked_anomalous_graph.keys():
            self._search_roots(subgraph_id)
            self._search_time_defiance(subgraph_id)
        self.action_recommendation()

    def run(self, data):
        self.search_rc_from_data = True
        normal_data, anomalous_data = self._process_data(data)
        for subgraph_id in self.dict_linked_anomalous_graph.keys():
            if not self.search_rc_from_graph:
                self._search_roots(subgraph_id)
                self._search_time_defiance(subgraph_id)

            # self._search_structure_defiance(subgraph_id, normal_data, anomalous_data)
            self._search_param_defiance(subgraph_id, normal_data, anomalous_data)
            self._search_structure_defiance_after_param(subgraph_id, normal_data, anomalous_data)
        self.action_recommendation()


if __name__ == '__main__':
    np.random.seed(1)
    graph = nx.DiGraph()
    graph.add_edges_from([("y", "z"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("d", "f"), ("e", "f")])
    graph.add_edges_from([("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e"), ("f", "f"), ("y", "z"), ("z", "z")])
    # anomalous = list(graph.nodes)
    anomalous = ["a", "b", "c", "e", "f", "z", "y", "d"]
    anomalies_start = dict()
    for node in anomalous:
        if node == "z":
            anomalies_start[node] = 10549
        else:
            anomalies_start[node] = 10550

    anomaly_size = 400

    # find some root causes using only the graph
    AG = EasyRCA(graph, anomalous, anomalies_start_time=anomalies_start, anomaly_length=anomaly_size)
    AG.run_without_data()
    print(AG.root_causes)
    AG.action_recommendation()
    print(AG.get_recommendations)

    # Simulate data
    import numpy as np
    data_size = 100000

    # a = np.random.normal(size=data_size)
    # b = 2*a + 0.1 * np.random.normal(size=data_size)
    # c = 5*a + 0.2 * np.random.normal(size=data_size)
    # d = 5*b + 0.2 * np.random.normal(size=data_size)
    # e = 5*c + 0.2 * np.random.normal(size=data_size)
    # f = 5 * d + 5 * e + 0.2 * np.random.normal(size=data_size)
    # y = np.random.normal(1, 1, size=data_size)
    # z = 3 * y + 0.2 * np.random.normal(size=data_size)

    a = np.random.normal(size=data_size)
    xi_b = 0.1 * np.random.normal(size=data_size)
    xi_c = 0.2 * np.random.normal(size=data_size)
    xi_d = 0.2 * np.random.normal(size=data_size)
    xi_e = 0.2 * np.random.normal(size=data_size)
    xi_f = 0.2 * np.random.normal(size=data_size)
    y = np.random.normal(size=data_size)
    xi_z = 0.2 * np.random.normal(size=data_size)

    b = np.zeros(shape=data_size)
    c = np.zeros(shape=data_size)
    d = np.zeros(shape=data_size)
    e = np.zeros(shape=data_size)
    f = np.zeros(shape=data_size)
    z = np.zeros(shape=data_size)

    for t in range(1, data_size):
        b[t] = b[t-1] + 2*a[t] + xi_b[t]
        c[t] = c[t-1] + 5 * a[t] + xi_c[t]
        d[t] = d[t-1] + 5 * b[t] + xi_d[t]
        e[t] = e[t-1] + 5 * c[t] + xi_e[t]
        f[t] = f[t-1] + 5 * d[t] + 5 * e[t] + xi_f[t]
        z[t] = z[t-1] + 3 * y[t] + xi_z[t]

    data = pd.DataFrame(np.array([a, b, c, d, e, f, y, z]).T, columns=["a", "b", "c", "d", "e", "f", "y", "z"])

    # generate intervention
    # ad = np.array(4 * e[anomalies_start["e"]: anomalies_start["e"] + anomaly_size] + 0.2 * np.random.normal(size=anomaly_size))
    # data["f"].loc[anomalies_start["e"]: anomalies_start["e"] + anomaly_size - 1] = ad

    # # generate intervention on a
    # data["a"].loc[anomalies_start["a"]: anomalies_start["a"] + anomaly_size - 1] = np.random.normal(size=anomaly_size)
    # # propagate intervention to all descendants of a
    # data["b"].loc[anomalies_start["b"]: anomalies_start["b"] + anomaly_size - 1] = 2*data["a"].loc[anomalies_start["a"]: anomalies_start["a"] + anomaly_size - 1] + 0.1 * np.random.normal(size=anomaly_size)
    # data["c"].loc[anomalies_start["c"]: anomalies_start["c"] + anomaly_size - 1] = 5*data["a"].loc[anomalies_start["a"]: anomalies_start["a"] + anomaly_size - 1] + 0.2 * np.random.normal(size=anomaly_size)
    # data["d"].loc[anomalies_start["d"]: anomalies_start["d"] + anomaly_size - 1] = 5*data["b"].loc[anomalies_start["b"]: anomalies_start["b"] + anomaly_size - 1] + 0.2 * np.random.normal(size=anomaly_size)
    # data["e"].loc[anomalies_start["e"]: anomalies_start["e"] + anomaly_size - 1] = 5*data["c"].loc[anomalies_start["c"]: anomalies_start["c"] + anomaly_size - 1] + 0.2 * np.random.normal(size=anomaly_size)
    # # generate intervention on f
    # ad = np.array(10 * d[anomalies_start["e"]: anomalies_start["e"] + anomaly_size] +10 * e[anomalies_start["e"]: anomalies_start["e"] + anomaly_size] + 0.2 * np.random.normal(size=anomaly_size))
    # data["f"].loc[anomalies_start["f"]: anomalies_start["f"] + anomaly_size - 1] = ad
    # # generate intervention on y
    # data["y"].loc[anomalies_start["y"]: anomalies_start["y"] + anomaly_size - 1] = np.random.normal(size=anomaly_size)
    # # propagate intervention to all descendants of y
    # data["z"].loc[anomalies_start["z"]: anomalies_start["z"] + anomaly_size - 1] = 3*data["y"].loc[anomalies_start["y"]: anomalies_start["y"] + anomaly_size - 1] + 0.2 * np.random.normal(size=anomaly_size)

    # generate intervention on a
    data["a"].loc[anomalies_start["a"]: anomalies_start["a"] + anomaly_size - 1] = np.random.normal(size=anomaly_size)
    # generate intervention on f
    data["y"].loc[anomalies_start["y"]: anomalies_start["y"] + anomaly_size - 1] = np.random.normal(size=anomaly_size)
    for t in range(anomalies_start["a"], anomalies_start["a"] + anomaly_size - 1):
        # propagate intervention to all descendants of a
        data["b"].loc[t] = b[t-1] + 2*data["a"].loc[t] + xi_b[t]
        data["c"].loc[t] = c[t-1] + 5 * data["a"].loc[t] + xi_c[t]
        data["d"].loc[t] = d[t-1] + 5 * data["b"].loc[t] + xi_d[t]
        data["e"].loc[t] = e[t-1] + 5 * data["c"].loc[t] + xi_e[t]
        # generate parametric intervention on f
        data["f"].loc[t] = xi_f[t]
        # propagate intervention to all descendants of y
        data["z"].loc[t-1] = z[t-2] + 3 * data["y"].loc[t-1] + xi_z[t-1]


    # find all root causes using graph and data
    AG.run(data)
    print(AG.root_causes)
    AG.action_recommendation()
    print(AG.get_recommendations)
