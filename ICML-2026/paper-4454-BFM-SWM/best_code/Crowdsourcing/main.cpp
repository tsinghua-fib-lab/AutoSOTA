
#include "read_data.h"
#include "time.h"
#include "SIP.h"

int main(int argc,char *argv[]) {

    read();

    cal_similarity();

    double eps=0.1;


    time_t nowtime;
    struct tm* p;;
    time(&nowtime);
    p = localtime(&nowtime);

    string outtext = "../result/result_" + to_string(p->tm_mon + 1) + "_" + to_string(p->tm_mday) + "_" + to_string(p->tm_hour) + "_" + to_string(p->tm_min) + "_" + to_string(p->tm_sec) + ".txt";

    vector<Result> simultaneous_result;
    vector<Result> TripleEagleNm_result;
    vector<Result> BFM_VM_result;

    double B_start=200;
    double B_end=400;
    double B_step=20;


    for(double B=B_start;B<=B_end;B+=B_step)
    {
        BFM_VM_result.push_back(BFM_VM(B,eps));
        TripleEagleNm_result.push_back(TripleEagleNm(eps,B));
        simultaneous_result.push_back(Simultaneous(eps,B));
    }

    ofstream out(outtext);
    out<<"eps: "<<eps<<endl;
    out<<"Budget: "<<endl;
    for(double B=B_start;B<=B_end;B+=B_step)
    {
        out<<B<<"\t";
    }
    out<<endl;


    out<<"BFM-VM "<<endl;
    out<<"objective values: "<<endl;
    for(auto p:BFM_VM_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto p:BFM_VM_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
//    out<<"running times: "<<endl;
//    for(auto p:BFM_VM_result)
//    {
//        out<<p.running_time<<"\t";
//    }
//    out<<endl;

    out<<"BFM_NM"<<endl;
    out<<"objective values: "<<endl;
    for(auto &p:TripleEagleNm_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto &p:TripleEagleNm_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
//    out<<"running times: "<<endl;
//    for(auto &p:TripleEagleNm_result)
//    {
//        out<<p.running_time<<"\t";
//    }
//    out<<endl;

    out<<"SIP "<<endl;
    out<<"objective values: "<<endl;
    for(auto &p:simultaneous_result)
    {
        out<<p.revenue<<"\t";
    }
    out<<endl;
    out<<"oracle queries: "<<endl;
    for(auto &p:simultaneous_result)
    {
        out<<p.oracle<<"\t";
    }
    out<<endl;
//    out<<"running times: "<<endl;
//    for(auto &p:simultaneous_result)
//    {
//        out<<p.running_time<<"\t";
//    }
//    out<<endl;


    return 0;
}
