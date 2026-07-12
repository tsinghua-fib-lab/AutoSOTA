#ifndef STREAMING_ALGORITHM_UTILITY_FUNCTIONS_H
#define STREAMING_ALGORITHM_UTILITY_FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <sys/types.h>
#include <sstream>
#include <climits>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "list"
#include "random"
using namespace std;
const int d=1;

string edge_text="./Graph/slashdot/renum_edge.txt";
string node_text="./Graph/slashdot/renum_node.txt";
int node_num=0;
int edge_num=0;
const int k=2;
double min_cost=999999999.0;
double max_cost=-1.0;
const int group_num=5;
int group_limit=200;
const int q=6;
class Node
{
public:
    Node(){};
    Node(int num, int g,double w,double c)
    {
        id=num;
        group=g;
        weight=w;
        cost=c;
    }
    int id;
    int group;
    double weight;
    double cost;
};
vector<Node> Groundset;
vector<vector<int>> Edge;
void read_data()
{
    ifstream in1(node_text);
    int node_temp;
    int group_temp;
    double weight_temp;
    int degree_temp;
    double node_cost;


    default_random_engine e(12345);
    uniform_real_distribution<double> distribution(10.0, 100.0);
//    uniform_real_distribution<double> ran(0.00001,1.0);
//    default_random_engine e(123);

    while(!in1.eof())
    {
        in1>>node_temp>>weight_temp>>degree_temp>>group_temp;
        //calculate cost
        int cost_temp=degree_temp-q;
        int cost=1+max(cost_temp,0);

        //double norm_cost=1.0-exp(-0.2*node_cost);
        double random_cost = distribution(e);
        if (in1.fail())
            break;
        Groundset.emplace_back(node_temp,group_temp,1.0,1.0+sqrt(degree_temp));
       // Groundset.emplace_back(node_temp,group_temp,1.0,random_cost);
      //  Groundset.emplace_back(node_temp,group_temp,1.0,1.0 + max(0,degree_temp -q));

      //  Groundset.emplace_back(node_temp,group_temp,1.0,1.0 );
      //  Groundset.emplace_back(node_temp,group_temp,1.0,1.0 + max(0.0, sqrt(degree_temp)-q));
       // Groundset.emplace_back(node_temp,group_temp,1.0,1.0 + max(0.0, sqrt(degree_temp - q)));
    }
    in1.close();

    for(int i=0;i<Groundset.size();i++)
        Edge.emplace_back();

    ifstream in2(edge_text);
    int v1,v2;
    edge_num=0;
    unordered_set<int> node;
    while(!in2.eof())
    {
        in2>>v1>>v2;
        if (in2.fail())
            break;
        edge_num++;
        node.insert(v1);
        node.insert(v2);
        Edge[v1].push_back(v2);
    }
    node_num=node.size();

    cout<<"map node size: "<<node_num<<endl;
    cout<<"map edge size: "<<edge_num<<endl;
    in2.close();
}
double f_u_distorted(const int &node_id,const double &ratio)
{
    double sum_weight=0.0;
    for(const auto &v: Edge[node_id])
    {
        sum_weight+=Groundset[v].weight;
    }
    return sum_weight-ratio*Groundset[node_id].cost;
}
double g_u(const int &node_id)
{
    double sum_weight=0.0;
    for(const auto &v: Edge[node_id])
    {
        sum_weight+=Groundset[v].weight;
    }
    return sum_weight;
}
double f_u(const int &node_id)
{
    double sum_weight=0.0;
    for(const auto &v: Edge[node_id])
    {
        sum_weight+=Groundset[v].weight;
    }

    return sum_weight-Groundset[node_id].cost;
}

bool node_budget_feasible(const int &node_id,double  Budget)
{
    if(Groundset[node_id].cost>Budget)
        return false;
    else
        return true;
}

//////wait to fix
/*double Cup_marginal_gain(const Edge &edge)
{
    return edge.weight;
}*/
class S_class {
public:
    S_class() {
        S_cost=0.0;
        S_revenue = 0.0;
        S_price=0.0;
        for(int i=0;i<group_num;i++)
            group_exist.push_back(0);
        max_marginal=-999999999.0;
        max_element=-1;
        selected.resize(node_num,0);
        node_price.resize(node_num,0.0);
    }
    //value
    double S_cost;
    double S_revenue;
    double S_price;
    vector<int> selected;

    vector<double> node_price;
    vector<int> solution_order;

    vector<int> group_exist;
    //set
    unordered_set<int> solution;
    unordered_set<int> node_covered;
    //used for offline algorithm
    double max_marginal;
    int max_element;
    void clear()
    {
        S_cost=0.0;
        S_revenue = 0.0;
        S_price = 0.0;
        for(int i=0;i<group_num;i++)
            group_exist[i]=0;

        fill(selected.begin(), selected.end(), 0);
        solution.clear();

        solution_order.clear();
        node_covered.clear();
    }
    void replace_with_singleton(const double &marginal,const int &node)
    {
        clear();
        add_element(marginal,node);
    }
    //1. value==false, last time we find it is feasible, then it will never be feasible.
    //2. value==true, last time is feasible, but this time we still need to check
    //vector<int> edge_feasible;
    double f_S()
    {
        unordered_set<int> covered_vertex;

        double sum_weight=0.0;
        double sum_cost=0.0;
        for(const auto &u:solution)
        {
            //sum weight of N(S)
            for(const auto &v:Edge[u])
            {
                if(!covered_vertex.count(v))
                {
                    sum_weight+=Groundset[v].weight;
                    covered_vertex.emplace(v);
                }
            }
/*            //sum weight of S
            if(!covered_vertex.count(u))
            {
                sum_weight+=Groundset[u].weight;
                covered_vertex.emplace(u);
            }*/

            sum_cost+=Groundset[u].cost;

        }
      //  cout<< " cost  "<< sum_cost<<endl;
        return sum_weight-sum_cost;
    }
    double g_S()
    {
        unordered_set<int> covered_vertex;

        double sum_weight=0.0;
        double sum_cost=0.0;
        for(const auto &u:solution)
        {
            //sum weight of N(S)
            for(const auto &v:Edge[u])
            {
                if(!covered_vertex.count(v))
                {
                    sum_weight+=Groundset[v].weight;
                    covered_vertex.emplace(v);
                }
            }
/*            //sum weight of S
            if(!covered_vertex.count(u))
            {
                sum_weight+=Groundset[u].weight;
                covered_vertex.emplace(u);
            }*/

            //sum_cost+=Groundset[u].cost;
        }
        return sum_weight;
    }

    void add_element(const double &marginal,const int &node_id)
    {
        selected[node_id] = 1;
        S_revenue+=marginal;
        S_cost+=Groundset[node_id].cost;
        group_exist[Groundset[node_id].group]++;

        solution.emplace(node_id);
        solution_order.push_back(node_id);
        for(const auto &v:Edge[node_id])
            node_covered.emplace(v);
        //node_covered.emplace(node_id);
    }
    void add_element_truth(const double &marginal,const int &node_id, const double &price)
    {
        if(solution.count(node_id)) return;
        selected[node_id] = 1;


        S_revenue+=marginal;
        S_cost+=Groundset[node_id].cost;
        group_exist[Groundset[node_id].group]++;

        S_price +=price;

        solution.emplace(node_id);
        solution_order.push_back(node_id);
        for(const auto &v:Edge[node_id])
            node_covered.emplace(v);
        //node_covered.emplace(node_id);
    }

    double marginal_gain_g(const int &node_id)
    {
        double sum_weight=0.0;
        for(const auto &v:Edge[node_id])
        {
            if(!node_covered.count(v))
            {
                sum_weight+=Groundset[v].weight;
            }
        }
        return sum_weight;
    }
    double marginal_gain(const int &node_id)
    {
        double sum_weight=0.0;
        for(const auto &v:Edge[node_id])
        {
            if(!node_covered.count(v))
            {
                sum_weight+=Groundset[v].weight;
            }
        }
        return sum_weight-Groundset[node_id].cost;
    }
    double marginal_gain_distorted(const int &node_id,const double &ratio)
    {
        double sum_weight=0.0;
        for(const auto &v:Edge[node_id])
        {
            if(!node_covered.count(v))
            {
                sum_weight+=Groundset[v].weight;
            }
        }
        return sum_weight-ratio*Groundset[node_id].cost;
    }

    bool node_budget_feasible(const int &node_id,double  Budget)
    {
        if(Groundset[node_id].cost>Budget)
            return false;
        else
            return true;
    }


    bool is_feasible(const int &node_id, const int &max_node)
    {
        if(solution.count(node_id))
            return false;
        if(solution.size()>=max_node)
            return false;
        if(group_exist[Groundset[node_id].group]>=group_limit)
            return false;
        return true;
    }
};
class Result
{
public:
    Result(){}
    Result(double rev,double cos,int siz,long long int ora)
    {
        revenue=rev;
        cost=cos;
        size=siz;
        oracle=ora;
    }
    Result(double rev,double cos,double pi, int siz,long long int ora,long long int total_time)
    {
        revenue=rev;
        cost=cos;
        price=pi;
        size=siz;
        oracle=ora;
        running_time=total_time;
    }
    double revenue;
    double price;
    long long int oracle;
    long long int round;
    long long int running_time;
    double cost;
    int size;
    long long int time=0;
    long long int max_query=0;
    long long int memory=0;
};
class SolutionWithStore
{
public:
    SolutionWithStore()
    {
        for(int i=0;i<edge_num;i++)
            is_planar.push_back(-1);
    };
    S_class solution;
    //1. is_planar=-1, we don't know
    //2. is_planar=0, not planar;
    //3. is_planar=1, is planar.
    vector<int> is_planar;
};

#endif //STREAMING_ALGORITHM_UTILITY_FUNCTIONS_H