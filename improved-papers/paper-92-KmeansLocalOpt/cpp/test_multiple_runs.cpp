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

    // Run 5 independent sets of 20 trials
    for (int run = 0; run < 5; ++run) {
        vector<long double> Md_loss, Md_time, Md_step;
        srand(run * 12345 + 1);
        for (int iter = 0; iter < 20; ++iter) {
            LO_K_Means<long double> kmeans(unique_data, weight, K, "rand", "squared", rand(), eps);
            auto t1 = high_resolution_clock::now();
            auto [assign, centers, loss] = kmeans.Min_D_LO_K_means();
            auto t2 = high_resolution_clock::now();
            long double t = duration<long double>(t2 - t1).count();
            Md_loss.push_back(loss);
            Md_time.push_back(t);
            Md_step.push_back(kmeans.step_num);
        }
        auto sl = computeStats(Md_loss);
        auto st = computeStats(Md_time);
        auto ss = computeStats(Md_step);
        cout << fixed << setprecision(2);
        cout << "Run " << run+1 << ": Mean loss=" << sl.first 
             << ", Min loss=" << *min_element(Md_loss.begin(), Md_loss.end()) 
             << ", Avg time=" << st.first 
             << "s, Avg steps=" << ss.first << endl;
    }
    return 0;
}
