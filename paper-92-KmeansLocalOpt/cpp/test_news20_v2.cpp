// Test with different random seeds
#include <bits/stdc++.h>
using namespace std;
#include "LO_K_means.hpp"
#include "Setting.hpp"

const long double eps = 1e-10;
using namespace std::chrono;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    string input_file = "../data/news20_2000_1089.txt";
    ifstream fin(input_file);
    int N, D;
    fin >> N >> D;
    vector<vector<long double>> data(N, vector<long double>(D));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) fin >> data[i][j];
    fin.close();
    auto [unique_data, weight] = make_unique(data);
    int K = 5;

    cout << "Running with fixed seed 42..." << endl;
    srand(42);
    
    {
        vector<long double> Md_loss, Md_time, Md_step;
        for (int iter = 0; iter < 20; ++iter) {
            LO_K_Means<long double> kmeans(unique_data, weight, K, "rand", "squared", rand(), eps);
            auto t1 = high_resolution_clock::now();
            auto [assign_min_d, centers_min_d, loss_min_d] = kmeans.Min_D_LO_K_means();
            auto t2 = high_resolution_clock::now();
            long double time_min_d = duration<long double>(t2 - t1).count();
            Md_loss.push_back(loss_min_d);
            Md_time.push_back(time_min_d);
            Md_step.push_back(kmeans.step_num);
            cout << "Trial " << iter+1 << ": loss=" << fixed << setprecision(0) << loss_min_d 
                 << ", time=" << setprecision(4) << time_min_d << "s, steps=" << kmeans.step_num << endl;
        }
        
        auto stats_loss = computeStats(Md_loss);
        auto stats_time = computeStats(Md_time);
        auto stats_step = computeStats(Md_step);
        cout << fixed << setprecision(4);
        cout << "\nMin-D-LO with seed 42:" << endl;
        cout << "  Mean loss:  " << stats_loss.first << endl;
        cout << "  Min loss:   " << *min_element(Md_loss.begin(), Md_loss.end()) << endl;
        cout << "  Avg time:   " << stats_time.first << " sec" << endl;
        cout << "  Avg steps:  " << stats_step.first << endl;
    }
    
    return 0;
}
