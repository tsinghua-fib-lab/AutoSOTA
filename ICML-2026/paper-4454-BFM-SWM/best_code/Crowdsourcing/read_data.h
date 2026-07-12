


#ifndef IMAGE_READ_DATA_H
#define IMAGE_READ_DATA_H
#include <iostream>
#include "fstream"
#include "time.h"
#include "vector"
#include "random"
#include "set"
#include "algorithm"
#include "list"
#include "map"
#include <iomanip>
#include <fstream>
#include <iostream>
using namespace std;
const int node_num=3000;
const int m=3072;
const int category=10;
const int per_category_max=10;
const int k=1;
vector<int> lable;
vector<vector<int>> feature;
vector<double> contrast_cost(node_num,0.0);
vector<vector<double>> similarity;
vector<double> sim_sum;
int K;
const int d=1;
double min_cost=999999999;
double max_cost=-999999999;
double lambda_f = 1.0;
double lambda = 0.2;
string cost_text = "../rdata/category/rd_contrast.txt";
string feature_text = "../rdata/category/rd_feature.txt";
const double ave_num=10;
const int max_image=10;

void read()
{

    feature.resize(node_num, vector<int>(m));
    ifstream in1(feature_text);
    //ifstream in1("feature.txt");
    int i=0;
    int templable=-1;
//    while(i < node_num && in1 >> templable){
//        lable.push_back(templable);
//        for(int j=0;j<m;j++)
//           in1 >> feature[i][j];
//        i++;
//    }
    while(!in1.eof())
    {
        in1>>templable;
        if (in1.fail())
            break;
        lable.push_back(templable);
        for(int j=0;j<m;j++)
            in1 >> feature[i][j];

        i++;
    }
    in1.close();

    cout<<"lable num: "<<lable.size()<<" | "<<i<<endl;

    ifstream in2(cost_text);
    //ifstream in2("contrast.txt");
    i=0;
    double temp_cost;
    double sum_cost=0.0;

    while(!in2.eof())
    {
        in2>>temp_cost;
        if (in2.fail())
            break;

        if(temp_cost<min_cost)
            min_cost=temp_cost;
        if(temp_cost>max_cost)
            max_cost=temp_cost;

        contrast_cost[i]=temp_cost;
        sum_cost+=temp_cost;
        i++;
        //cout<<temp_cost<<endl;
    }
//while(i <node_num && in2 >> temp_cost){
//    contrast_cost[i]=temp_cost;
//            if(temp_cost<min_cost)
//            min_cost=temp_cost;
//        if(temp_cost>max_cost)
//            max_cost=temp_cost;
//        i++;
//
//}
//    in2.close();
//
//    cout<<"normalize: "<<endl;
//  //  double ave_cost=sum_cost/node_num;
    if (sum_cost > 0) {

        double scale_factor = (node_num / ave_num) / sum_cost;

        for (int j = 0; j < node_num; j++) {
            contrast_cost[j] *= scale_factor;


            if (contrast_cost[j] < 1e-6) contrast_cost[j] = 1e-6;
        }


        min_cost *= scale_factor;
        max_cost *= scale_factor;
    }
    //*
//    for(auto &x:contrast_cost)
//    {
//        x=(node_num*x)/(sum_cost*ave_num);
//        //cout<<x<<endl;
//    }
    //*/


}

//void generate_raw_cost_file() {
//    string output_filename = "rd_contrast_raw.txt";
//    ofstream out(output_filename);
//
//    if (!out.is_open()) {
//        cerr << "error " << output_filename << endl;
//        return;
//    }
//
//
//    out << std::fixed << std::setprecision(8);
//
//    for (int i = 0; i < node_num; ++i) {
//        double sum = 0.0;
//        for (int j = 0; j < m; ++j) {
//            sum += feature[i][j];
//        }
//        double mean = sum / m;
//
//        double sum_sq_diff = 0.0;
//        for (int j = 0; j < m; ++j) {
//            double diff = (double)feature[i][j] - mean;
//            sum_sq_diff += diff * diff;
//        }
//
//        double sigma = sqrt(sum_sq_diff / m);
//        out << sigma << "\n";
//        contrast_cost[i] = sigma;
//    }
//
//    out.close();
//    cout << "success" << endl;
//
//}



void cal_similarity()
{
    similarity.resize(node_num, vector<double>(node_num));
    sim_sum = vector<double>(node_num,0.0);

    //inner product
    for(int i = 0;i < feature.size();i++)
    {
        for(int j = i;j < feature.size();j++)
        {
            double product = 0.0;
            for(int k = 0;k < feature[0].size();k++)
            {
                product+=(feature[i][k]*feature[j][k]);
            }
            similarity[i][j] = product;
            similarity[j][i] = product;
        }
    }
    for(int it=0;it < node_num;it++)
    {
        for (int itv = 0; itv < node_num; itv++) {
            sim_sum[it] += similarity[it][itv];
            //cout<<sim_mat_sum[it]<<endl;
        }
    }
    //distance
/*
    for(int i = 0;i < feature.size();i++)
    {
        for(int j = i;j < feature.size();j++)
        {
            double distance = 0.0;
            for(int k = 0;k < feature[0].size();k++)
            {
                distance += pow((feature[i][k]-feature[j][k]),2);
            }
            distance = sqrt(distance);
            similarity[i][j] = distance;
            similarity[j][i] = distance;
        }
    }
    for(int it=0;it < node_num;it++)
    {
        for (int itv = 0; itv < node_num; itv++) {
            sim_sum[it] += similarity[it][itv];
            //cout<<sim_mat_sum[it]<<endl;
        }
    }
*/
    //consine
    /*
    for(int i = 0;i < feature.size();i++)
    {
        for(int j = i;j < feature.size();j++)
        {
            double product = 0.0;
            double module_a=0.0;
            double module_b=0.0;
            for(int k = 0;k < feature[0].size();k++)
            {
                product+=(feature[i][k]*feature[j][k]);
                module_a+=pow(feature[i][k],2);
                module_b+=pow(feature[j][k],2);
            }
            module_a=sqrt(module_a);
            module_b=sqrt(module_b);
            similarity[i][j] = product/(module_a*module_b);
            similarity[j][i] = product/(module_a*module_b);
        }
    }
    for(int it=0;it < node_num;it++)
    {
    for (int itv = 0; itv < node_num; itv++) {
        sim_sum[it] += similarity[it][itv];
        //cout<<sim_mat_sum[it]<<endl;
    }
    }

     */
}

//fixed
//double f_u(int node)
//{
//    double sum_value=0.0;
//    for(int i=0;i<node_num;i++)
//    {
//        sum_value += similarity[i][node];
//    }
//    return sum_value-similarity[node][node]/(node_num);
//}
double f_u(const int &node)
{
    double sum_value=sim_sum[node];
    return (sum_value-lambda_f*similarity[node][node])/node_num;
}

bool node_budget_feasible(const int &node,double Budget)
{
    /**********check single node Budget constraint**********/
    if(contrast_cost[node]>Budget)
        return false;
    else
        return true;
}
class S_class {
public:
    S_class() {
        max_marginal=-999999999.0;
        max_element=-1;
        S_cost=0.0;
        S_revenue=0.0;
        S_price=0.0;
        for(int iter=0;iter<category;iter++)
            category_sum.push_back(0);

        selected.resize(node_num,0);
        node_price.resize(node_num,0.0);
    }
    double S_cost;
    double S_revenue;
    double S_price;
    vector<int> Set;

    double max_marginal;
    int max_element;
    vector<int> selected;
    vector<int> category_sum;
    /***********only for SIM***************/
    vector<double> node_price;
    bool is_feasible(const int &e,int max_image)
    {
        if(Set.size()>=max_image||category_sum[lable[e]]>=per_category_max)
            return false;
        else
            return true;
    }
    void clear()
    {
        S_revenue = 0.0;
        S_cost=0.0;
        S_price=0.0;
        Set.clear();
        fill(selected.begin(), selected.end(), 0);
        fill(node_price.begin(), node_price.end(), 0.0);
        for(int iter=0;iter<category;iter++)
            category_sum[iter]=0;

        max_marginal=-999999999.0;
        max_element=-1;
    }
    void replace_with_singleton(const double &marginal,const int &node)
    {
        clear();
        add_element(marginal,node);
    }

    void add_element(const double &marginal,const int &node)
    {
        selected[node] = 1;
        Set.push_back(node);
        S_revenue += marginal;
        S_cost+=contrast_cost[node];

        category_sum[lable[node]]++;//S+e\in I

    }
    void add_element_truth(const double &marginal,const int &node,const double &price)
    {
        selected[node] = 1;
        Set.push_back(node);
        S_revenue += marginal;
        S_price +=price;
        S_cost+=contrast_cost[node];
        category_sum[lable[node]]++;//S+e\in I
        node_price[node]=price;
    }

    bool budget_feasible(const int &node,double Budget)
    {
        if(S_cost+contrast_cost[node]>Budget)
            return false;
        else
            return true;
    }

    double f_S()
    {
        double sum_value=0.0;
        float M1 = 0,M2 = 0;
        for(int it=0;it < Set.size();it++)
        {
            M1 += sim_sum[Set[it]];
            for(int its=0;its<Set.size();its++)
                M2 += similarity[Set[it]][Set[its]];

            //sum cost
            //sum_cost+=modular_cost[Set[it]];
        }
        sum_value = M1-lambda_f*M2;
        return sum_value/node_num;
    }
    /*
    double f_S()
    {
        if(Set.empty())
            return 0.0;

        double v1=0.0;
        double v2=0.0;
        for(int i=0;i<node_num;i++)
        {
            double max_temp=-999999999;
            for(int j=0;j<Set.size();j++)
            {
                if(similarity[i][Set[j]]>max_temp)
                    max_temp=similarity[i][Set[j]];
            }
            v1+=max_temp;
        }
        for(int i=0;i<Set.size();i++)
        {
            for(int j=0;j<Set.size();j++)
            {
                v2+=similarity[Set[i]][Set[j]];
            }
        }
        return v1-v2/(node_num);
    }
     */


    double marginal(int e)
    {
        //Set.push_back(node);
        //selected[node]=1;
        //double f_S_and_u=f_S();
        //Set.pop_back();
        //selected[node]=0;
        float M1=0,M2=0;
        M1 += sim_sum[e];
        for(int it=0;it < Set.size();it++)
        {
            if(e==Set[it]) return 0;
            M2 += similarity[e][Set[it]];
            M2 += similarity[Set[it]][e];
        }
        M2 += similarity[e][e];

        return (M1-lambda_f*M2)/node_num;
    }

//    double marginal(int e)
//    {
//        if(Set.empty())
//        {
//            return f_u(e);
//        }
//        //slow
//        double v1=0.0;
//        double v2=0.0;
//
//        for(int i=0;i<Set.size();i++)
//        {
//            if(e==Set[i]) return 0.0;
//
//            v2+=similarity[e][Set[i]];
//            v2+=similarity[Set[i]][e];
//        }
//        v2+=similarity[e][e];
//
//        for(int i=0;i<node_num;i++)
//        {
//            double max_similarity=-999999999;
//            for(auto j:Set)
//            {
//                if(similarity[i][j]>max_similarity)
//                    max_similarity=similarity[i][j];
//            }
//
//            if(similarity[i][e]>max_similarity)
//            {
//                v1+=(similarity[i][e]-max_similarity);
//            }
//        }
//
//        return v1-v2/(node_num);
//    }

   double S_sub_u(int e)
    {
        double sum_value=0.0;
        float M1 = 0,M2 = 0;
        for(int it=0;it < Set.size();it++)
        {
            if(Set[it]==e)
                continue;
            M1 += sim_sum[Set[it]];
            for(int its=0;its<Set.size();its++)
            {
                if(Set[its]==e)
                    continue;
                M2 += similarity[Set[it]][Set[its]];
            }
        }
        sum_value = M1-lambda_f*M2;
        return sum_value/node_num;
    }

    void copy(const S_class &temp)
    {
        S_revenue=temp.S_revenue;
        S_cost=temp.S_cost;
        Set.assign(temp.Set.begin(),temp.Set.end());
        selected.assign(temp.selected.begin(),temp.selected.end());
        //category_sum.assign(temp.category_sum.begin(),temp.category_sum.end());
    }
};
class Result
{
public:
    Result(){}
    Result(double rev,double cos,double pi,int siz,long long int ora,long long int total_time)
    {
        revenue=rev;
        cost=cos;
        price=pi;
        size=siz;
        oracle=ora;
        running_time=total_time;
    }
    Result(double rev,double cos,int siz,long long int ora)
    {
        revenue=rev;
        cost=cos;
        size=siz;
        oracle=ora;
    }
    Result(double rev,long long int ora,double uti_der,double ora_der,int round)
    {
        revenue=rev;

        oracle=ora;
        utility_deviation=uti_der;
        oracle_deviation=ora_der;
    }
    double revenue;
    double  price;
    long long int oracle;
    long long int round;
    long long int running_time;
    double cost;
    int size;
    long long int time=0;
    long long int max_query=0;
    double utility_deviation;
    double oracle_deviation;
};

#endif //IMAGE_READ_DATA_H
