

#ifndef IMAGE_SIMULTANEOUS_H
#define IMAGE_SIMULTANEOUS_H
#include "BFM-VM.h"

class Ai_node
{
public:
    Ai_node(const int &node_temp)
    {
        node=node_temp;
        //weight=f_u(node);
        l=1;
    }
    int node;
    //double weight;
    int l;
};

pair<int,double> Boost(multimap<double,Ai_node> &A,S_class &Si,const vector<int> &available,const double &B)
{
/********check this when pop a node*************/
    double m=0.0;//always record weight of aij
    int ai=-1;
    while(!A.empty())
    {
        //check the end node, i.e., the node with biggest weight
        auto it=A.end();
        it--;

        double old_value=it->first;

        /*******no need this?*****/
        //if the value of maximum element is less than 0, then all element has value less than 0 due to the diminish property of submodularity, so we break and return empty element.
/*        if(old_value<=0)
        {
            A.clear();
            ai.user=-1;
            ai.product=-1;
            m=0.0;
            break;
        }*/

        //if not satisfy the k-system constraint or has been selected by other solution, then delete it and pop the next element
        //if(!Si.all_feasible(it->second.node,B)||available[it->second.node.product][it->second.node.user]==0)
        if(available[it->second.node]==0)
        {
            A.erase(it);
            continue;
        }
        //if the element is useful, then we compute its new weight, and let its update numbers +1
        //oracle_times++;
        double new_value=Si.marginal(it->second.node);
        //the value of element in multimap can be change directly, but the key can not be change, we need to erase it and re-insert it
        it->second.l++;
        //if the value of the element diminishes not much, we can return it
        if(new_value>=old_value)
        {
            ai=it->second.node;
            m=new_value;
            break;
        }
            //or we re-insert now element and then check the next element
        else
        {
            Ai_node temp=it->second;
            //erase the element to update its weight
            A.erase(it);
            //if its update numbers is not greater than max, then we re-insert it into the queue
            //if(temp.l<=(log(node_num*product_types*2.0/eps)/log(1.0+eps))) {
            A.insert(pair<double, Ai_node>(new_value, temp));
            //}
        }
    }
    return pair<int,double>(ai,m);
}


S_class USM(const S_class &N,long long int &oracle_times)
{
    S_class X;
    S_class Y;
    Y.Set.assign(N.Set.begin(),N.Set.end());
    //Y.selected.assign(N.selected.begin(),N.selected.end());
    Y.S_price=N.S_price;
    Y.S_revenue=N.S_revenue;

    default_random_engine e(1234);
    for(auto u=N.Set.begin();u!=N.Set.end();u++)
        //for(auto u=N.Set.rbegin();u!=N.Set.rend();u++)
    {
        //calculate ai
        double f_Xi_1=X.S_revenue;
        //X.selected[*u]=1;
        X.Set.push_back(*u);

        oracle_times++;

        double f_Xi_1_and_u=X.f_S();
        X.Set.pop_back();
        //X.selected[*u]=0;
        double ai=f_Xi_1_and_u-f_Xi_1;

        //calculate bi
        double f_Yi_1=Y.S_revenue;
        //Y.selected[*u]=0;

        oracle_times++;

        double f_Yi_1_sub_u=Y.S_sub_u(*u);
        //Y.selected[*u]=1;
        double bi=f_Yi_1_sub_u-f_Yi_1;

        double ai1=max(ai,0.0);
        double bi1=max(bi,0.0);
        double probability=0.0;
        if(ai1==0&&bi1==0)
            probability=1.0;
        else
        {
            probability=ai1/(ai1+bi1);
        }
        bernoulli_distribution r(probability);

      //  if(bi>0) cout<<"bi>0 !!!"<<endl;
        if(r(e)==1)
        {
            //X.selected[*u]=1;
            X.Set.push_back(*u);
            X.S_revenue=f_Xi_1_and_u;

            X.S_price+=N.node_price[*u];
            X.node_price[*u]=N.node_price[*u];
        }
        else
        {
          //  cout<<"ai<bi !!!"<<endl;
            //Y.selected[*u]=0;
            //delete u_i
            for(vector<int>::iterator p=Y.Set.begin();p!=Y.Set.end();)
            {
                if((*p)==(*u))
                {
                    p=Y.Set.erase(p);
                    break;
                }
                else{
                    p++;
                }
            }
            Y.S_revenue=f_Yi_1_sub_u;
            Y.S_price-=N.node_price[*u];
        }
    }
    return X;
}

Result Simultaneous(double eps,double B)
{

    auto start = std::chrono::high_resolution_clock::now();

    cout<<"SIP & Budget: "<<B<<"---------start---------"<<endl;

    long long int oracle_times=0;
    vector<double> pi(node_num,B);

    vector<int> A(node_num,0);//mark all nodes are available
    double f_u_max=-1.0;
    int u_star;
    for (int iter = 0; iter < node_num; iter++) {

//        if (!node_budget_feasible(iter, B))
//            continue;

        oracle_times++;
        double value = f_u(iter);

        if (value >= f_u_max) {
            f_u_max = value;
            u_star = iter;
        }

        A[iter] = 1;//single element with feasible budget is set to 1
    }

    double OPT=f_u_max;

    vector<vector<S_class>> S(2,vector<S_class>(2,S_class()));
    S[0][1].add_element(f_u_max,u_star);

    int t=0;
    while(true)
    {
        t++;
        for(auto &it:S[t%2])
            it.clear();

        OPT*=2.0;

        //initialize C and Boost_array
        vector<int> now_avaliable(node_num,0);//mark all nodes are available now
        //bool empty=true;

        vector<multimap<double,Ai_node>> Boost_array(2,multimap<double,Ai_node>());

        for (int iter = 0; iter < node_num; iter++) {
            //available in D and not in S_{t-1}
            if (A[iter] == 1 && S[(t + 1) % 2][0].selected[iter] == 0&& S[(t + 1) % 2][1].selected[iter] == 0) {
                //empty=false;//has available element
                now_avaliable[iter] = 1;

                Ai_node temp(iter);
                double value = f_u(iter);
                Boost_array[0].insert(pair<double, Ai_node>(value, temp));
                Boost_array[1].insert(pair<double, Ai_node>(value, temp));
            }
        }

//        cout<<"now OPT: "<<OPT<<endl;
        bool avaliable_empty=false;
        while (S[t%2][0].S_revenue<OPT&&S[t%2][1].S_revenue<OPT) {
            for (int j = 0; j < S[t%2].size(); j++)
            {
                S[t%2][j].max_marginal=-999999999.0;
                S[t%2][j].max_element=-1;

                //A=empty, then return element=-1 and marginal gain =0
                pair<int,double> temp=Boost(Boost_array[j], S[t%2][j], now_avaliable, B);
                S[t%2][j].max_marginal=temp.second;
                S[t%2][j].max_element=temp.first;

            }
            double max_marginal = 0.0;
            int max_element=-1;
            int max_solution = -1;
            for(int j=0;j<S.size();j++)
            {
                if (S[t%2][j].max_marginal > max_marginal) {
                    max_solution=j;
                    max_marginal=S[t%2][j].max_marginal;
                    max_element=S[t%2][j].max_element;
                }
            }
            //if (max_solution == -1||max_marginal<=0)//non element or marginal gain<=0
            if (max_solution == -1)//non element or marginal gain<=0
            {
                avaliable_empty=true;
                break;
            }

            //only used for calculate oracle times
            for (int iter = 0; iter < node_num; iter++)
            {
                if (now_avaliable[iter] == 0)
                    continue;
                oracle_times+=2;
            }

            //S<-S\cup {u}
            pi[max_element]=min(pi[max_element],max_marginal*B/OPT);

            double price=pi[max_element];
            if(price>=contrast_cost[max_element])
                S[t % 2][max_solution].add_element_truth(max_marginal, max_element, price);
            else
                A[max_element]=0;//discarded the node who doesn't accept our price
            now_avaliable[max_element] = 0;//discarded or selected
        }

//        cout<<"t: "<<t<<endl;
//        for(const auto& first:S) {
//            for (const auto &it:first) {
//                cout << "S: " << endl;
//                cout << "  revenue: " << it.S_revenue << " size: " << it.Set.size() << " cost: " << it.S_cost << endl;
//                cout << "  all nodes: " << endl;
//                for (const auto &p:it.Set)
//                    cout << p << '\t';
//                cout << endl;
//            }
//        }
        if(avaliable_empty)
            break;
    }
    //call USM
    S_class S_star;
    for(const auto& i:S) {
        for (const auto &j:i) {
            S_class temp=USM(j,oracle_times);
            if(temp.S_revenue>S_star.S_revenue)
                S_star=temp;
        }
    }
    for(const auto& it:S[(t + 1)%2]) {
        if (it.S_revenue >= S_star.S_revenue)
            S_star = it;
    }

    if(S_star.S_price>B)
    {
//        cout<<"this happen !"<<endl;
//        cout<<"S revenue 1: "<<S_star.S_sub_u(S_star.Set.back())<<endl;
        S_star.S_price-=S_star.node_price[S_star.Set.back()];
        S_star.Set.pop_back();
        S_star.S_revenue=S_star.f_S();
        oracle_times++;
//        cout<<"S revenue 2: "<<S_star.S_revenue<<endl;
    }


    S_star.S_cost = 0;
    for(const auto &p:S_star.Set){
        S_star.S_cost+=contrast_cost[p];
    }


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    long long int total_time_ns = duration.count();


//    cout<<"running time: "<<total_time_ns<<endl;
//    cout<<"S*:"<<endl;
//    cout<<"  revenue: "<<S_star.S_revenue <<" size: "<<S_star.Set.size()<<" cost "<< S_star.S_cost<<" Price: "<<S_star.S_price<<endl;
//    cout<<"  all nodes: "<<endl;
//
//    for(const auto &p:S_star.Set)
//        cout<<p<<'\t';
//    cout<<endl;
//
//    cout<<"oracle times: "<<oracle_times<<endl;
    cout<<"Objective Values: "<<S_star.S_revenue<<endl;
    cout<<"Oracle Queries: "<<oracle_times<<endl;

    cout<<"SIP ---------end--------- "<<endl<<endl;
    return Result(S_star.S_revenue ,S_star.S_cost,S_star.S_price,S_star.Set.size(),oracle_times,total_time_ns);


}
#endif //IMAGE_SIMULTANEOUS_H
