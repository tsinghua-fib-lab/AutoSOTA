#include "time.h"
#include "BFM-SWM.h"
int main(int argc,char *argv[]) {

    read_data();

    time_t nowtime;
    struct tm *p;;
    time(&nowtime);
    p = localtime(&nowtime);
    string::size_type pos1, pos2, posend;
    pos1 = edge_text.find_last_of("/");
    pos2 = edge_text.rfind("/", pos1 - 1);
    posend = edge_text.find_last_not_of("/");
    string name1 = edge_text.substr(pos2 + 1, pos1 - pos2 - 1);
    string name2 = edge_text.substr(pos1 + 1, posend);
    string result_name = name1 + "_" + name2;
    //cout<<result_name<<endl;
    string outtext =
            "./result/result_" + result_name + "_" + to_string(p->tm_mon + 1) + "." + to_string(p->tm_mday) + "_" +
            to_string(p->tm_hour) + "_" + to_string(p->tm_min) + "_" + to_string(p->tm_sec) + ".txt";



      vector<Result>  BFM_SWM_result;
      vector<Result>  Deng_CostScaled_result;
      vector<Result>  Deng_ROI_result;
      vector<Result>  Deng_Distorted_result;


    vector<int> ground_set;
    for(int i=0;i<node_num;i++)
        ground_set.push_back(i);

    double eps=0.1;

    int B_start=400;
    int B_end=400;
    int B_step=100;


    int num_B = (B_end -B_start)/B_step +1;
    vector<double> sum_revenue1(num_B,0.0);
    vector<long long> sum_oracle1(num_B,0.0);
    vector<double> sum_revenue2(num_B,0.0);
    vector<long long> sum_oracle2(num_B,0.0);

    for(int B = B_start;B <= B_end;B += B_step)

    {
        BFM_SWM_result.push_back(BFM_SWM(B,eps));
        Deng_Distorted_result.push_back(Deng_Distorted(B));
        Deng_ROI_result.push_back(Deng_ROI(B));
        Deng_CostScaled_result.push_back(Deng_CostScaled(B));
    }


    ofstream out(outtext);
    out<<"eps: "<<eps<<endl;
    out<<"Budget: "<<endl;
    for(int B=B_start;B<=B_end;B+=B_step)
    {
        out<<B<<"\t";
    }
    out<<endl;

    out<<"BFM-SWM "<<endl;
    out<<"objective values: "<<endl;
    for(auto p:BFM_SWM_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto p:BFM_SWM_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
    out<<"running times: "<<endl;
    for(auto p:BFM_SWM_result)
    {
        out<<p.running_time<<"\t";
    }
    out<<endl;


    out<<"Deng-Distorted "<<endl;
    out<<"objective values: "<<endl;
    for(auto p:Deng_Distorted_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto p:Deng_Distorted_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
    out<<"running times: "<<endl;
    for(auto p:Deng_Distorted_result)
    {
        out<<p.running_time<<"\t";
    }
    out<<endl;

    out<<"Deng-ROI "<<endl;
    out<<"objective values: "<<endl;
    for(auto p:Deng_ROI_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto p:Deng_ROI_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
    out<<"running times: "<<endl;
    for(auto p:Deng_ROI_result)
    {
        out<<p.running_time<<"\t";
    }
    out<<endl;

    out<<"Deng-CostScaled "<<endl;
    out<<"objective values: "<<endl;
    for(auto p:Deng_CostScaled_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto p:Deng_CostScaled_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
    out<<"running times: "<<endl;
    for(auto p:Deng_CostScaled_result)
    {
        out<<p.running_time<<"\t";
    }
    out<<endl;

    return 0;
}