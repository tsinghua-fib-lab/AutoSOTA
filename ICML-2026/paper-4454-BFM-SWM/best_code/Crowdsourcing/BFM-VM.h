

#ifndef CS_BFM_VM_H
#define CS_BFM_VM_H
#include "TripleEagleNm.h"
Result BFM_VM(double Budget , double eps) {

    cout<<"BFM-VM & Budget: "<<Budget<<"-------------------start-------------"<<endl;

    auto start = std::chrono::high_resolution_clock::now();

    int l=2;
    int t=1;
    long long int oracle_times = 0;
    double alpha = 1 + sqrt(3);

    vector<float> pi(node_num,0.0);
    vector<bool> is_active(node_num,false);


    //************** R active users****************
    vector<int> R;

    for (int u = 0; u < node_num; u++){

        if(contrast_cost[u] <= Budget){
            is_active[u] = true;
            R.push_back(u);
            pi[u]=Budget;
        }
    }

    //**************** p_t = max(f(u)) ***************
    int best_node = -1;
    double p_t = -1e18;

    for (int u = 0; u < node_num; u++) {
        double delta =  f_u(u);
        oracle_times++;
        if(delta > p_t){
            p_t = delta;
            best_node = u;
        }
    }


    vector<S_class> S_prev(l, S_class());
    vector<S_class> S_curr(l,S_class());
    vector<S_class> S_sets;

    if(best_node != -1){
        S_prev[0].add_element_truth(S_prev[0].marginal(best_node),best_node,pi[best_node]);
        oracle_times++;
    }

    bool stop = false;
    while(!stop) {
        t = t + 1;
        p_t = alpha * p_t;

        for (int i = 0; i < l; i++) {
            S_curr[i] = S_class();
        }

        for (auto it = R.begin(); it != R.end();) {
            int u = *it;

            bool in_prev = false;
            for (int i = 0; i < l; i++) {
                if (S_prev[i].selected[u]==1) {
                    in_prev = true;
                    break;
                }
            }
            if (in_prev) {
                ++it;
                continue;
            }

            //******************best_j**************
            int best_j = 0;
            double max_mg = -1;
            for(int j = 0; j < l; j++){
                double mg = S_curr[j].marginal(u);
                oracle_times++;
                if(mg > max_mg){
                    max_mg = mg;
                    best_j = j;
                }
            }

            //******************update price*******************
            if(max_mg / ( p_t/Budget) < pi[u]){
                pi[u] = max_mg / (p_t/Budget);
            }

            if(contrast_cost[u] <=  pi[u]){
                double curr_f = S_curr[best_j].f_S();
                oracle_times++;
                if(curr_f + max_mg > p_t ){
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
                if(S_prev[i].selected[u]==1 || S_curr[i].selected[u]==1){
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
        double w_S = A.S_revenue  ;

        if (w_S > max_v) {
            max_v = w_S;
            S_star = A;
        }
    }

    S_star.S_revenue = S_star.f_S();
    oracle_times++;

    auto end = std::chrono::high_resolution_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    long long int total_time_ns = duration.count();

//    cout<<"running time: "<<total_time_ns<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"  revenue: "<<S_star.S_revenue << " Price: "<<S_star.S_price<<" Cost: "<<S_star.S_cost<<" size: "<<S_star.Set.size()<<endl;
//    for(const auto &p:S_star.Set)
//        cout<<p<<" ";
//    cout<<endl;
//
//    cout<<"oracle times: "<<oracle_times<<endl;
    cout<<"Objective Values: "<<S_star.S_revenue<<endl;
    cout<<"Oracle Queries: "<<oracle_times<<endl;

    cout<<"BFM-VM ----------------end---------------"<<endl<<endl;
    return Result(S_star.S_revenue,S_star.S_cost,S_star.S_price, S_star.Set.size(),oracle_times,total_time_ns);

}




#endif //CS_BFM_VM_H
