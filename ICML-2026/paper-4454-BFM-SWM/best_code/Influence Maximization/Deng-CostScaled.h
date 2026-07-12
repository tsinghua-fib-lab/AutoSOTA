

#ifndef COVERAGE_DENG_COSTSCALED_H
#define COVERAGE_DENG_COSTSCALED_H
#include "Deng-ROI.h"


Result Deng_CostScaled(double Budget) {

    cout << "Deng-CostScaled & Budget: " << Budget << "---------start---------" << endl;

    auto start = std::chrono::high_resolution_clock::now();

    long long int oracle_times = 0;

    int type = 2;
    vector<S_class> snapshots;
    S_class S_star = DistortedGreedy_lazy(oracle_times,type,snapshots);

    vector<double> payments(node_num, 0.0);


    for (int idx = 0; idx<S_star.solution_order.size();++idx) {
        int i = S_star.solution_order[idx];
        S_class current_S = snapshots[idx];
        double p_i = Groundset[i].cost;


        priority_queue<LazyNode> pq;
        for (int u = 0; u < node_num; u++) {
            if (u == i || current_S.selected[u]) continue;

            double mg = current_S.marginal_gain_g(u);
            oracle_times++;
            double score=0;
            if (type==1) score = (mg - Groundset[u].cost)/Groundset[u].cost;
            else score = mg - 2 * Groundset[u].cost;
            pq.push({u, score, (int)current_S.solution.size()});

        }

        for (int k = current_S.solution.size(); k < node_num - 1; k++) {

            double mg_i = current_S.marginal_gain_g(i);
            oracle_times++;


            double max_other_G = -1e18;
            int best_other_u = -1;

            while (!pq.empty()) {
                LazyNode top = pq.top();
                pq.pop();


                if (current_S.selected[top.id]) continue;


                if (top.iteration < current_S.solution.size()) {
                    double mg_u = current_S.marginal_gain_g(top.id);
                    oracle_times++;

                    if (type==1) top.score =(mg_u - Groundset[top.id].cost)/Groundset[top.id].cost;
                    else top.score = mg_u - 2 * Groundset[top.id].cost;
//                    top.iteration = k;
                    top.iteration = current_S.solution.size();
                    pq.push(top);
                    continue;
                }


                max_other_G = top.score;
                best_other_u = top.id;
                pq.push(top);
                break;
            }

            //************* compute price ***************

            if(max_other_G < 1e-9) break;

            double threshold = 0;
            double max_other = max(0.0, max_other_G);
            if(type ==1) threshold = mg_i / (1.0 + max_other);
            else  threshold = (mg_i - max_other)/2.0;

            if (threshold > p_i) p_i = threshold;


            if (best_other_u != -1 && max_other_G > 1e-9) {
                current_S.add_element(0.0, best_other_u);

            }

        }
        payments[i] = p_i;
    }


    // ********************** check budget and cut off ******************
    double total_payment = 0.0;
    for (int i : S_star.solution) {
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

    S_star.S_revenue = S_star.f_S() ;
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
    cout<<"Deng-CostScaled ---------end--------- "<<endl<<endl;

    return Result(S_star.S_revenue,S_star.S_cost,S_star.S_price,S_star.solution.size(),oracle_times,total_time_ns);
}

#endif //COVERAGE_DENG_COSTSCALED_H
