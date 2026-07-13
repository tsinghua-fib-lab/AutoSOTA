#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <cassert>
#include <timer.hpp>
#include <graph.hpp>
#include <bounds.hpp>
// #include <execution> -- removed: not used and requires TBB on gcc 9.x
#include <utils.hpp>

std::tuple<bool,std::vector<int>,Bitset> oracle(Graph const & graph,std::vector<double> const & edge_weights, double rho ) {

    ScopedTimer t1("MWU::Oracle");

    static std::vector<std::pair<int,double>> sorted_triangles;

    
    sorted_triangles.resize(graph.num_triangles());

    // Fill weights sequentially
    for (size_t i = 0; i < sorted_triangles.size(); ++i) {
        const auto& tri = graph.triangles[i];
        double w = edge_weights[tri.edge1_idx] +
                    edge_weights[tri.edge2_idx] +
                    edge_weights[tri.edge3_idx];
        sorted_triangles[i] = {static_cast<int>(i), w};
    }

    std::sort(sorted_triangles.begin(), sorted_triangles.end(),
        [&](std::pair<int,double>& p1, std::pair<int,double>& p2) {
            return p1.second < p2.second;
        });
    

    double W = 0;
    for (size_t e_idx = 0; e_idx < edge_weights.size(); ++e_idx) {
        W += edge_weights[e_idx];
    }
    
    double used_weight = 0;
    Bitset used_edges(graph.num_edges());
    std::vector<int> packing;
    for (auto [idx,weight] : sorted_triangles) {
        const auto& tri = graph.triangles[idx];
        int e1 = tri.edge1_idx, e2 = tri.edge2_idx, e3 = tri.edge3_idx;
        bool covered = (used_edges[e1] | used_edges[e2] | used_edges[e3]);
        if (covered) continue;

        if (used_weight + weight > W/rho) break;
        packing.push_back(idx);
        used_edges.set(e1);
        used_edges.set(e2);
        used_edges.set(e3);
        used_weight +=  weight;
    }
    return {true, packing, used_edges};
}


double mwu(Graph const & graph, double epsilon, double rho, bool verbose) {
    if (verbose) std::cout << " Starting MWU_eps="<<epsilon<<" rho="<<rho<< std::endl;

    size_t m = graph.num_edges();
    double best_feasible_sol_so_far = 0;
    double best_dual_cover_so_far = m;

    
    double mwu_objective = 0;
    std::vector<double> edge_weights(m, 1.0);
    std::vector<double> edge_loads(graph.num_edges(), 0.0);
    //max num iterations:
    int T = 2 * rho * log(graph.num_edges()) / epsilon / epsilon;
    
    double sum_of_weights;

    int iteration;
    double max_edge_usage = 0;
    double min_triangle_weight;
    for (iteration = 1; iteration<T;iteration++){
        auto [oracle_feasible, packing, used_edges] = oracle(graph, edge_weights, rho);
        double objective_this_round = 1.0 * rho * packing.size();
        mwu_objective += objective_this_round;
        int unsatisfied_constraints=0;
        int notnearsatisfied_constraints=0;
        auto & min_triangle = graph.triangles[packing[0]];
        min_triangle_weight = edge_weights[min_triangle.edge1_idx]+edge_weights[min_triangle.edge2_idx]+edge_weights[min_triangle.edge3_idx];
        max_edge_usage = 0;
        sum_of_weights = 0;
        for (size_t i = 0; i < m; ++i) {
            edge_weights[i] *= 1+ epsilon/rho*used_edges[i];
            sum_of_weights += edge_weights[i];
            edge_loads[i] += rho * used_edges[i];
            if (edge_loads[i]/iteration > 1) unsatisfied_constraints++;
            if (edge_loads[i]/iteration > (1+epsilon)) notnearsatisfied_constraints++;
            max_edge_usage = std::max({max_edge_usage, edge_loads[i]/iteration});
        }
        best_feasible_sol_so_far = std::max({ (mwu_objective / iteration /max_edge_usage), best_feasible_sol_so_far});
        best_dual_cover_so_far = std::min({ (sum_of_weights/min_triangle_weight), best_dual_cover_so_far});

        double best_packing = best_feasible_sol_so_far;
        double new_packing = (mwu_objective / iteration /max_edge_usage);
        double best_cover = best_dual_cover_so_far;
        double new_cover = (sum_of_weights/min_triangle_weight);
        if (verbose) {
            std::cout << "  Iteration " << iteration  << ". New packing: " << new_packing << " Best packing: " << best_packing << "  New cover: " << new_cover << " Best cover: " << best_cover <<" Current apx: "<<(best_cover/best_packing)<<"\n";
        }
        if (best_cover/best_packing<1+epsilon) break;

    }
    if (verbose) std::cout << " Finished after " << iteration << " iterations. "  << "Final Objective " << best_feasible_sol_so_far  << std::endl;

    return best_feasible_sol_so_far;
}

double mwu_SU(Graph const & graph, double epsilon, double timeout, bool verbose){
    if (verbose) std::cout << " Starting MWU_Single_eps="<<epsilon<< std::endl;
    auto start_time = std::chrono::steady_clock::now();

    size_t m = graph.num_edges();

    std::vector<std::vector<int>> edge_to_triangles(m);
    for (int t_idx = 0; t_idx < (int)graph.triangles.size(); ++t_idx) {
        const auto& tri = graph.triangles[t_idx];
        edge_to_triangles[tri.edge1_idx].push_back(t_idx);
        edge_to_triangles[tri.edge2_idx].push_back(t_idx);
        edge_to_triangles[tri.edge3_idx].push_back(t_idx);
    }
    double best_primal = m;
    double best_dual = 1;

    std::vector<double> edge_weights(m, 1.0);
    std::vector<double> triangle_packings(graph.num_triangles(), 0.0);
    std::vector<double> edge_loads(m, 0.0);
    double sum_of_weights = m;
    double sum_of_packings = 0;
    size_t min_tri_idx = 0;
    double min_tri_cost = 3;
    double max_edge_load = 1;
    int iteration=0;
    while(best_primal/best_dual>1+epsilon){
        iteration++;
        if (verbose and (iteration%500==0 or iteration<10)) std::cout << " It "<<iteration <<"  sum_of_weights "<<sum_of_weights << "  new primal "<<sum_of_weights/min_tri_cost << "  best primal "<<best_primal<< "  best dual "<<best_dual<<"  current apx "<<(best_primal/best_dual)<<std::endl;
        if (iteration % 100 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            if (elapsed.count() > timeout) {
                // Timeout reached
                return 0; 
            }
        }

        const auto& min_tri = graph.triangles[min_tri_idx];
        edge_weights[min_tri.edge1_idx]*=(1+epsilon);
        edge_weights[min_tri.edge2_idx]*=(1+epsilon);
        edge_weights[min_tri.edge3_idx]*=(1+epsilon);
        triangle_packings[min_tri_idx]++;

        sum_of_weights += epsilon * (min_tri_cost);
        sum_of_packings++;
        edge_loads[min_tri.edge1_idx]++;
        edge_loads[min_tri.edge2_idx]++;
        edge_loads[min_tri.edge3_idx]++;
        max_edge_load = std::max({max_edge_load, edge_loads[min_tri.edge1_idx], edge_loads[min_tri.edge2_idx], edge_loads[min_tri.edge3_idx]});

        min_tri_cost=std::numeric_limits<double>::infinity();
        for (size_t t_idx = 0; t_idx < graph.num_triangles(); ++t_idx){
            const auto& tri = graph.triangles[t_idx];
            double tri_cost = (edge_weights[tri.edge1_idx] 
                                + edge_weights[tri.edge2_idx] 
                                + edge_weights[tri.edge3_idx]);  
            if (tri_cost<min_tri_cost) {
                min_tri_cost = tri_cost;
                min_tri_idx = t_idx;
            }
        }

        best_primal = std::min({sum_of_weights/min_tri_cost,best_primal});
        best_dual = std::max({sum_of_packings/max_edge_load,best_dual});
    }
    if (verbose) std::cout << " Finished after " << iteration << " iterations. "  << "Final Objective " << best_dual  << std::endl;
    return best_dual;
}