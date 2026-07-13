
#ifndef MWU_H
#define MWU_H

#include <graph.hpp>

std::tuple<bool,std::vector<int>,Bitset> oracle(Graph const & graph, std::vector<double> const & edge_weights, double rho);
double mwu(Graph const & graph, double epsilon, double rho, bool verbose = false);

double mwu_SU(Graph const & graph, double epsilon, double timeout, bool verbose = false);

int greedy_triangle_packing(Graph const & graph);

double balkanski(Graph const & graph);

int LSCut(Graph const & graph, int repetitions);

int randomized_pivot(Graph const & graph, int repetitions);

int setcovergreedy(Graph const & graph);

#endif