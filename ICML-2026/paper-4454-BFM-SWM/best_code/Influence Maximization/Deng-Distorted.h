

#ifndef COVERAGE_DENG_DISTORTED_H
#define COVERAGE_DENG_DISTORTED_H
#include "Deng-CostScaled.h"



vector<double> precompute_gammas_sto(int n) {

    vector<double> gammas(n + 1);

    double base = 1.0 - 1.0 / (double)n;

    for (int k = 1; k <= n; k++) {

        gammas[k] = pow(base, (double)(n - k));
    }

    return gammas;
}


S_class Meta(const vector<int>& r_sequence, const vector<double>& gammas, long long int& oracle_times){

    S_class S = S_class();

    int n = node_num;

    for(int k=1; k<=n; k++){
        int u = r_sequence[k-1];
        if(S.selected[u]==1) continue;

      //  double gamma = pow(1.0 - 1.0 /(double) n, (double) (n-k));
        double marginal_gain = S.marginal_gain_g(u);
        oracle_times++;

        double score = gammas[k]*marginal_gain - Groundset[u].cost;
        if(score > 1e-9){
            S.add_element(marginal_gain,u);
        }
    }

    return S;

}



Result Deng_Distorted(double Budget){

    cout << "Deng-Distorted & Budget: "<<Budget<<"-------------------start---------------" << endl;

    auto start = std::chrono::high_resolution_clock::now();

    long long int oracle_times = 0;

    static default_random_engine e(12345);
    uniform_int_distribution<int> dist(0, node_num - 1);
    vector<int> r_sequence;
    r_sequence.reserve(node_num);

    for(int k = 0; k < node_num; k++){
        r_sequence.push_back(dist(e));
    }


    vector<double> gammas = precompute_gammas_sto(node_num);
    S_class S_star = Meta(r_sequence, gammas, oracle_times);

    vector<double> payments(node_num, 0.0);


    for(auto& i : S_star.solution){
        S_class current_S;
        double p_i = 0;

        for(int k = 1; k <= node_num; k++){
            int u = r_sequence[k-1];

            if(u == i){

                double mg_i = current_S.marginal_gain_g(i);
                oracle_times++;

                double threshold = gammas[k] * mg_i;
                if (threshold > p_i) {
                    p_i = threshold;
                }

            } else {

                if(current_S.selected[u] == 1) continue;

                double mg_u = current_S.marginal_gain_g(u);
                oracle_times++;

                double score = gammas[k] * mg_u - Groundset[u].cost;
                if(score > 1e-9){
                    current_S.add_element(mg_u, u);
                }
            }
        }
        payments[i] = p_i;
    }

    // **********************check budget and cut off******************
    double total_payment = 0.0;
    for(int i : S_star.solution){
        total_payment += payments[i];
    }

    while (total_payment > Budget + 1e-9 && !S_star.solution_order.empty()) {
        int last_id = S_star.solution_order.back();
        total_payment -= payments[last_id];
        S_star.S_cost-=Groundset[last_id].cost;
        payments[last_id] = 0.0;

        S_star.solution.erase(last_id);
        S_star.solution_order.pop_back();

        if(last_id < node_num) S_star.selected[last_id] = 0;
    }

    S_star.S_price = total_payment;
    S_star.S_revenue = S_star.f_S();
    oracle_times++;


    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    long long int total_time_ns = duration.count();

//    cout<<"running time: "<<total_time_ns<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"  revenue: "<<S_star.S_revenue<<" cost: "<<S_star.S_cost<<"Price: "<<S_star.S_price<<" size: "<<S_star.solution.size()<<endl;
//    for(const auto &p:S_star.solution)
//        cout<<p<<" ";
//    cout<<endl;
//    cout<<"oracle times: "<<oracle_times<<endl;

    cout<<"objective values: "<<S_star.S_revenue<<endl;
    cout<<"oracle queries: "<<oracle_times<<endl;
    cout<<"Deng-Distorted ---------end--------- "<<endl<<endl;

    return Result(S_star.S_revenue,S_star.S_cost,S_star.S_price,S_star.solution.size(),oracle_times,total_time_ns);
}





#endif //COVERAGE_DENG_DISTORTED_H
