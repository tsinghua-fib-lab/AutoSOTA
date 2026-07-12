
#ifndef COVERAGE_BFM_SWM_H
#define COVERAGE_BFM_SWM_H
#include <algorithm>
#include "Deng-Distorted.h"

Result BFM_SWM(double Budget , double eps) {

    cout<<"BFM-SWM & Max Budget: "<<Budget<<"-------------------start-------------"<<endl;

    auto start = std::chrono::high_resolution_clock::now();

    int l=1;
    int t=0;
    long long int oracle_times = 0;
    double alpha = 1 + sqrt(6)/2;
    double beta = 3;
    double p_t = eps / alpha;
    S_class A_u = S_class();


    vector<float> pi(node_num,0.0);
    vector<bool> is_active(node_num,false);


    //************** R active users****************
    vector<int> R;

    for (int u = 0; u < node_num; u++){

        if(Groundset[u].cost <= Budget){
            is_active[u] = true;
            R.push_back(u);
            pi[u]=Budget;
        }

    }

    // Sort active nodes by descending degree for better selection
    sort(R.begin(), R.end(), [](int a, int b) {
        return Edge[a].size() > Edge[b].size();
    });

    vector<S_class> S_prev(l, S_class());
    vector<S_class> S_curr(l,S_class());
    vector<S_class> S_sets;

    S_class A_1 = S_class();
    S_class A_1_temp = S_class();

    bool stop = false;
    while(!stop) {
        A_1 = A_1_temp;


        t = t + 1;
        p_t = alpha * p_t;


        for (int i = 0; i < l; i++) {
            S_curr[i] = S_class();
        }


        for (auto it = R.begin(); it != R.end();) {
            int u = *it;


            bool in_prev = false;
            for (int i = 0; i < l; i++) {
                if (S_prev[i].selected[u]==1||A_u.selected[u]==1) {
                    in_prev = true;
                    break;
                }
            }
            if (in_prev) {
                ++it;
                continue;
            }

            //*********************best_j********************
            int best_j = 0;
            double max_mg = -1;
            for(int j = 0; j < l; j++){
                double mg = S_curr[j].marginal_gain_g(u);
                oracle_times++;
                if(mg > max_mg){
                    max_mg = mg;
                    best_j = j;
                }
            }


            //******************update price*******************
            if(max_mg / (beta + p_t/Budget) < pi[u]){
                pi[u] = max_mg / (beta + p_t/Budget);
            }

            if(Groundset[u].cost <=  pi[u]){
                double curr_f = S_curr[best_j].S_revenue;

                if((curr_f + max_mg) - (S_curr[best_j].S_price + pi[u]) > p_t ){
                    A_u.clear();
                    A_1_temp = S_curr[best_j];
                    A_u.add_element_truth(A_u.marginal_gain_g(u),u,pi[u]);
                    oracle_times++;

                    break;
                }
                else{

                    S_curr[best_j].add_element_truth(max_mg, u,pi[u]);

                    ++it;
                }

            }
            else{

                is_active[u] = false;
                it = R.erase(it);

            }

        }


        stop = true;
        for(int u = 0; u < node_num ;u++){
            if(!is_active[u]) continue;
            bool covered = false;
            for(int i = 0; i < l ;i++){
                if(S_prev[i].selected[u]==1 || S_curr[i].selected[u]==1||A_u.selected[u]==1){
                    covered = true;
                    break;}

            }
            if(!covered){
                stop = false;
                break;
            }
        }

        S_sets.clear();
        for (int i = 0; i < l; i++) {
            S_sets.push_back(S_prev[i]);

            S_sets.push_back(S_curr[i]);

        }

        S_prev = S_curr;

    }
    int M = t;
 //   cout <<"M: "<<M<<endl;

    S_class S_star = S_class();

    double max_v = -1e18;

    for (auto& A : S_sets) {
        double w_S = A.S_revenue - A.S_price ;

        if (w_S > max_v) {
            max_v = w_S;
            S_star = A;
        }
    }
    if(S_star.S_revenue - S_star.S_price < A_u.S_revenue - A_u.S_price){
        cout<<"A_u"<<endl;
        S_star = A_u;
    }


    S_star.S_revenue = S_star.f_S();
    oracle_times++;

    // Post-processing: greedy fill remaining budget with best marginal/cost nodes
    double remaining_budget = Budget - S_star.S_cost;
    if (remaining_budget > 0) {
        // Collect candidate nodes: unselected active nodes within budget, with positive marginal gain
        vector<pair<double,int>> candidates;
        for (int u : R) {
            if (!S_star.selected[u] && Groundset[u].cost <= remaining_budget) {
                double mg = S_star.marginal_gain_g(u);
                oracle_times++;
                if (mg > 0) {
                    candidates.push_back({mg / Groundset[u].cost, u});
                }
            }
        }
        // Sort by descending marginal_gain/cost
        sort(candidates.rbegin(), candidates.rend());
        for (auto& cand : candidates) {
            int u = cand.second;
            if (Groundset[u].cost <= remaining_budget) {
                double mg = S_star.marginal_gain_g(u);
                oracle_times++;
                S_star.add_element_truth(mg, u, 0.0);
                remaining_budget -= Groundset[u].cost;
            }
        }
        // Recompute final revenue after additions
        S_star.S_revenue = S_star.f_S();
        oracle_times++;
    }

    auto end = std::chrono::high_resolution_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    long long int total_time_ns = duration.count();


//    cout<<"running time: "<<total_time_ns<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"  revenue: "<<S_star.S_revenue << " Price: "<<S_star.S_price<<" Cost: "<<S_star.S_cost<<" size: "<<S_star.solution.size()<<endl;
//    for(const auto &p:S_star.solution)
//        cout<<p<<" ";
//    cout<<endl;
//
//    cout<<"oracle times: "<<oracle_times<<endl;

     cout<<"objective values: "<<S_star.S_revenue<<endl;
     cout<<"oracle queries: "<<oracle_times<<endl;
     cout<<"BFM-SWM ---------end--------- "<<endl<<endl;

    return Result(S_star.S_revenue,S_star.S_cost,S_star.S_price, S_star.solution.size(),oracle_times,total_time_ns);

}

#endif //COVERAGE_BFM_SWM_H
