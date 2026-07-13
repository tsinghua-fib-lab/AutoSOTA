#include <graph.hpp>
#include <bounds.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <timer.hpp>
#include <cmath>
#include <cxxopts.hpp>


int main(int argc, char **argv) {


    cxxopts::Options options("ISAR", "Instance-specific approximation ratios for MaxCut and Correlation Clustering");
    options.add_options()
        ("file", "Graph file(s)", cxxopts::value<std::vector<std::string>>())
        ("p,problem", "Problem", cxxopts::value<std::string>())
        ("v,verbose", "Enable verbose output", cxxopts::value<bool>()->default_value("false"))
        ("c,convergence", "Enable output for convergence plot", cxxopts::value<bool>()->default_value("false"));
    options.parse_positional({"file"});
    options.positional_help("file");

    auto result = options.parse(argc, argv);
    
    std::vector<std::string> file;
    if (result.count("file")) {
        file = result["file"].as<std::vector<std::string>>();
    }

    bool verbose = result["verbose"].as<bool>();
    bool convergence = result["convergence"].as<bool>();
    if (convergence) verbose=true;
    std::string problem = result["problem"].as<std::string>();
    double timeout = 7200;

    bool MC =  (problem == "MC");
    bool CC =  (problem == "CC");
    bool CE =  (problem == "CE");
    if (not (CE or CC or MC)){
        std::cout << "Unknown problem\n";
        return 0;
    }
    if (file.empty()) {
        return 0;
    }

    for (const std::string& graph_file : file) {
        std::cout << "Processing " << graph_file << std::endl;

        Graph G;
        try {
            G.readFromFile(graph_file, CE, 1);
        } catch (const std::exception& e) {
            G = Graph();
            G.readFromFile(graph_file, CE, 0);
        }
        int m = G.num_edges();

        std::string max_problem = "MAXCUT", min_problem = "MINTRIANGLECOVER";
        if (CE) {
            max_problem = "CLUSTER EDITING AGREEMENT", min_problem = "CLUSTER EDITING DISAGREEMENT";
        }
        if (CC) {
            max_problem = "CORRELATION CLUSTERING AGREEMENT", min_problem = "CORRELATION CLUSTERING DISAGREEMENT";
        }

        std::cout << "-----------"<<min_problem<<"------------"<<std::endl;

        if (MC) {
            int algo = setcovergreedy(G);
            std::cout << "ALGO: SetCoverGreedy    "<<algo <<std::endl;
        }
        if (CE) {
            int algo = randomized_pivot(G,50);
            std::cout << "ALGO: RandomizedPivot   "<<algo <<std::endl;
        }

        int Balkanski = std::ceil(balkanski(G));
        std::cout << "CERT: Balkanski "<<Balkanski << std::endl;

        int greedyPacking = greedy_triangle_packing(G);
        std::cout << "CERT: GreedyPacking "<<greedyPacking << std::endl;

        int MWU1, MWU5, MWU05, MWU025, MWU001, MWU0005, MWU0002, MWU_rho2, MWU_rho4, MWU_rho5, MWU_rho2_0005, MWU_rho2_0002;
        {
            ScopedTimer t2("MWU_eps=0.5");
            MWU5 = std::ceil(mwu(G, 0.5, 3, verbose));
            std::cout << "CERT: MWU_eps=0.5 "<<MWU5 <<std::endl;
        }
        {
            ScopedTimer t2("MWU_eps=0.1");
            MWU1 = std::ceil(mwu(G, 0.1, 3, verbose));
            std::cout << "CERT: MWU_eps=0.1 "<<MWU1 <<std::endl;
        }{
            ScopedTimer t2("MWU_eps=0.05");
            MWU05 = std::ceil(mwu(G, 0.05, 3, verbose));
            std::cout << "CERT: MWU_eps=0.05 "<<MWU05 <<std::endl;
}{            ScopedTimer t2("MWU_eps=0.025");            MWU025 = std::ceil(mwu(G, 0.025, 3, verbose));            std::cout << "CERT: MWU_eps=0.025 "<<MWU025 <<std::endl;        }{            ScopedTimer t2("MWU_eps=0.01");            MWU001 = std::ceil(mwu(G, 0.01, 3, verbose));            std::cout << "CERT: MWU_eps=0.01 "<<MWU001 <<std::endl;
}{            ScopedTimer t2("MWU_eps=0.005");            MWU0005 = std::ceil(mwu(G, 0.005, 3, verbose));            std::cout << "CERT: MWU_eps=0.005 "<<MWU0005 <<std::endl;
}{            ScopedTimer t2("MWU_eps=0.002");            MWU0002 = std::ceil(mwu(G, 0.002, 3, verbose));            std::cout << "CERT: MWU_eps=0.002 "<<MWU0002 <<std::endl;
}{            ScopedTimer t2("MWU_eps=0.01_rho=2");            MWU_rho2 = std::ceil(mwu(G, 0.01, 2, verbose));            std::cout << "CERT: MWU_eps=0.01_rho=2 "<<MWU_rho2 <<std::endl;        }{            ScopedTimer t2("MWU_eps=0.01_rho=4");            MWU_rho4 = std::ceil(mwu(G, 0.01, 4, verbose));            std::cout << "CERT: MWU_eps=0.01_rho=4 "<<MWU_rho4 <<std::endl;        }{            ScopedTimer t2("MWU_eps=0.01_rho=5");            MWU_rho5 = std::ceil(mwu(G, 0.01, 5, verbose));            std::cout << "CERT: MWU_eps=0.01_rho=5 "<<MWU_rho5 <<std::endl;
}{            ScopedTimer t2("MWU_eps=0.005_rho=2");            MWU_rho2_0005 = std::ceil(mwu(G, 0.005, 2, verbose));            std::cout << "CERT: MWU_eps=0.005_rho=2 "<<MWU_rho2_0005 <<std::endl;        }{            ScopedTimer t2("MWU_eps=0.002_rho=2");            MWU_rho2_0002 = std::ceil(mwu(G, 0.002, 2, verbose));            std::cout << "CERT: MWU_eps=0.002_rho=2 "<<MWU_rho2_0002 <<std::endl;
        }

        int MWU_singleupdate1,MWU_singleupdate5,MWU_singleupdate05,MWU_singleupdate025,MWU_singleupdate01;
        {
            ScopedTimer t2("MWU_Single_eps=0.5");
            MWU_singleupdate5 = std::ceil(mwu_SU(G, 0.5, timeout, verbose));
            std::cout << "CERT: MWU_Single_eps=0.5 "<<MWU_singleupdate5 <<std::endl;
}{            ScopedTimer t2("MWU_Single_eps=0.1");            MWU_singleupdate1 = std::ceil(mwu_SU(G, 0.1, timeout, verbose));            std::cout << "CERT: MWU_Single_eps=0.1 "<<MWU_singleupdate1 <<std::endl;        }{            ScopedTimer t2("MWU_Single_eps=0.05");            MWU_singleupdate05 = std::ceil(mwu_SU(G, 0.05, timeout, verbose));            std::cout << "CERT: MWU_Single_eps=0.05 "<<MWU_singleupdate05 <<std::endl;        }{            ScopedTimer t2("MWU_Single_eps=0.025");            MWU_singleupdate025 = std::ceil(mwu_SU(G, 0.025, timeout, verbose));            std::cout << "CERT: MWU_Single_eps=0.025 "<<MWU_singleupdate025 <<std::endl;        }{            ScopedTimer t2("MWU_Single_eps=0.01");            MWU_singleupdate01 = std::ceil(mwu_SU(G, 0.01, timeout, verbose));            std::cout << "CERT: MWU_Single_eps=0.01 "<<MWU_singleupdate01 <<std::endl;
        }
        
        if (CE or convergence) {
            ScopedTimer::print_timers();
            ScopedTimer::reset_timers();  
            continue;
        }

        std::cout << "------------------"<<max_problem<<"--------------------"<<std::endl;

        if (MC) {
            int algo_LS = LSCut(G,10);
            std::cout << "ALGO: LocalSearch "<<algo_LS<< std::endl;
        }

        std::cout << "CERT: num_edges  "<<m <<std::endl;

        int Balkanski_compl = m - Balkanski;
        std::cout << "CERT: Balkanski  "<<Balkanski_compl <<std::endl;

        int greedyPacking_compl = m - greedyPacking;
        std::cout << "CERT: GreedyPacking  "<<greedyPacking_compl <<std::endl;

        int MWU5_compl = m - MWU5;
        std::cout << "CERT: MWU_eps=0.5 "<<MWU5_compl <<std::endl;

        int MWU1_compl = m - MWU1;
        std::cout << "CERT: MWU_eps=0.1 "<<MWU1_compl <<std::endl;
        
        int MWU05_compl = m - MWU05;
        std::cout << "CERT: MWU_eps=0.05 "<<MWU05_compl <<std::endl;

        int MWU5_Single_compl = m - MWU_singleupdate5;
        if (MWU_singleupdate5==0) MWU5_Single_compl=0;
        std::cout << "CERT: MWU_Single_eps=0.5 "<<MWU5_Single_compl <<std::endl;

        ScopedTimer::print_timers();
        ScopedTimer::reset_timers();   
    }
    
    return 0;
}
