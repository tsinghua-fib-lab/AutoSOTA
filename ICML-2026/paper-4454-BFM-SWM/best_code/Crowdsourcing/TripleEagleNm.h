

#ifndef IMAGE_BFM_NM_H
#define IMAGE_BFM_NM_H
#include "read_data.h"
#include <chrono>
#include <iomanip>
//default_random_engine e_BFM(5);
default_random_engine e_BFM(time(NULL));
uniform_real_distribution<double> dis(0.0,1.0);

Result TripleEagleNm(double eps,double B)
{

    auto start = std::chrono::high_resolution_clock::now();

    cout<<"TripleEagleNm & Budget: "<<B<<"-------------------start-------------"<<endl;
    long long int oracle_times=0;
    float alpha=2.521;
    float beta=1.315;

    /***********INITIALIZE*************/
    vector<float> pi(node_num,B);
    vector<int> C;//store the node that accepts the max budget
    double f_u_max=-1.0;
    int v_star;
    for (int iter = 0; iter < node_num; iter++) {
        if (!node_budget_feasible(iter, B))
            continue;

        C.push_back(iter);
        oracle_times++;
        double value = f_u(iter);
        if (value >= f_u_max) {
            f_u_max = value;
            v_star = iter;
        }
    }
    /***********INITIALIZE*************/
    vector<S_class> A;
    A.emplace_back();
    A.emplace_back();

    /********reverse access*************/
//    for (auto u = C.rbegin(); u != C.rend(); ++u)
//    {
//        if(*u==v_star)
//            continue;
//        /************argmax f(u|Ai)***********/
//        int max_solution=-1;
//        double max_marginal=-999.0;
//        for(int iter=0;iter<A.size();iter++)
//        {
//            oracle_times++;
//            double temp_marginal=A[iter].marginal(*u);
//            if(temp_marginal>max_marginal)
//            {
//                max_marginal=temp_marginal;
//                max_solution=iter;
//            }
//        }
//        /************argmax f(u|Ai)***********/
//        double price=B*max_marginal/(A[max_solution].S_revenue+alpha*f_u_max);
//        pi[*u]=price;
//        if(price>=contrast_cost[*u])
//        {
//            A[max_solution].add_element_truth(max_marginal,*u,price);
//        }
//    }
    /********reverse access*************/

    /********order access*************/
    for(auto u:C)//order access
    {
        if(u==v_star)
            continue;
        /************argmax f(u|Ai)***********/
        int max_solution=-1;
        double max_marginal=-999.0;
        for(int iter=0;iter<A.size();iter++)
        {
            oracle_times++;
            double temp_marginal=A[iter].marginal(u);
            if(temp_marginal>max_marginal)
            {
                max_marginal=temp_marginal;
                max_solution=iter;
            }
        }
        /************argmax f(u|Ai)***********/
        double price=B*max_marginal/(beta*A[max_solution].S_revenue+alpha*f_u_max);
        pi[u]=price;
        if(price>=contrast_cost[u])
        {
            A[max_solution].add_element_truth(max_marginal,u,price);
        }
    }
    /********order access*************/

    S_class S;
    if(max(A[0].S_revenue,A[1].S_revenue)>=f_u_max)
    {
        int u=v_star;
        /************argmax f(u|Ai)***********/
        int max_solution=-1;
        double max_marginal=-999.0;
        for(int iter=0;iter<A.size();iter++)
        {
            oracle_times++;
            double temp_marginal=A[iter].marginal(u);
            if(temp_marginal>max_marginal)
            {
                max_marginal=temp_marginal;
                max_solution=iter;
            }
        }
        /************argmax f(u|Ai)***********/
        double price=B*max_marginal/(beta*A[max_solution].S_revenue+alpha*f_u_max);
        pi[u]=price;
        if(price>=contrast_cost[u])
        {
            A[max_solution].add_element_truth(max_marginal,u,price);
        }

        /************argmax f(X)***********/
        double max_revenue=-999.0;
        for(int iter=0;iter<A.size();iter++)
        {
            if(A[iter].S_revenue>max_revenue)
            {
                max_revenue=A[iter].S_revenue;
                max_solution=iter;
            }


//            cout<<"A"<<iter<<endl;
//            cout<<"  revenue: "<<A[iter].S_revenue<<" size: "<<A[iter].Set.size()<<" price: "<<A[iter].S_cost<<endl;
//            cout<<"  all nodes: "<<endl;
//
//            for(const auto &p:A[iter].Set)
//                cout<<p<<'\t';
//            cout<<endl;
//            cout<<"real revenue: "<<A[iter].f_S()<<endl;


        }
        /************argmax f(X)***********/
        for (auto e = A[max_solution].Set.rbegin(); e != A[max_solution].Set.rend(); ++e)
        {
            if(S.S_price+pi[(*e)]>B)
            {
                break;
            }
            else
            {
                S.add_element_truth(0.0,(*e),pi[(*e)]);
            }
        }
        S.S_revenue=S.f_S();//calculate the revenue in the end
    }
    else
    {
//        cout<<"randomness exists !"<<endl;
        double Z=dis(e_BFM);
        if(Z<=alpha/(2.0+alpha+beta))
        {
            S.add_element_truth(f_u_max,v_star,pi[v_star]);
        }
        else
        {
            /************argmax f(X)***********/
            int max_solution=-1;
            double max_marginal=-999.0;
            for(int iter=0;iter<A.size();iter++)
            {
                double m = A[iter].marginal(v_star);
                oracle_times++;
                if(m>max_marginal){
                    max_marginal = m;
                    max_solution = iter;
                }
//                if(A[iter].S_revenue>max_revenue)
//                {
//                    max_revenue=A[iter].S_revenue;
//                    max_solution=iter;
//                }
            }
            /************argmax f(X)***********/
            pi[v_star]=B-A[max_solution].S_price;
            if(pi[v_star]>=contrast_cost[v_star]&& max_marginal>=0)
            {
                A[max_solution].add_element_truth(max_marginal,v_star,pi[v_star]);
            }

            max_solution=-1;
            double max_revenue=-999.0;
            for(int iter=0;iter<A.size();iter++)
            {
                if(A[iter].S_revenue>max_revenue)
                {
                    max_revenue=A[iter].S_revenue;
                    max_solution=iter;
                }
            }
            S=A[max_solution];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    long long int total_time_ns = duration.count();

//    cout<<"running time: "<<total_time_ns<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"  revenue: "<<S.S_revenue<<" size: "<<S.Set.size()<<" price: "<<S.S_price<<endl;
//    cout<<"  all nodes: "<<endl;
//
//    for(const auto &p:S.Set)
//        cout<<p<<'\t';
//    cout<<endl;
//    cout<<"oracle times: "<<oracle_times<<endl;
    cout<<"Objective Values: "<<S.S_revenue<<endl;
    cout<<"Oracle Queries: "<<oracle_times<<endl;

    cout<<"TripleEagleNm ---------end--------- "<<endl<<endl;
    return Result(S.S_revenue,S.S_cost,S.S_price,S.Set.size(),oracle_times,total_time_ns);
}


#endif //IMAGE_BFM_NM_H
