// Compare K-means, C-LO_K_means, D-LO_K_means, and Min_D_LO_K_means by loss, time, and steps.

#include <bits/stdc++.h>
using namespace std;

#include "LO_K_means.hpp"
#include "Setting.hpp"

const long double eps = 1e-10;
using namespace std::chrono;

void print(const string& str, const vector<long double>& loss, const vector<long double>& time, const vector<long double>& step) {
    pair<long double, long double> loss_stats = computeStats(loss);
    pair<long double, long double> time_stats = computeStats(time);
    pair<long double, long double> step_stats = computeStats(step);

    cout << fixed << setprecision(4);
    cout << str << ":" << endl;
    cout << "  Loss (avg ± std):  " << loss_stats.first << " ± " << loss_stats.second << endl;
    cout << "  Loss (min):        " << *min_element(loss.begin(), loss.end()) << endl;
    cout << "  Time (avg):        " << time_stats.first << " sec" << endl;
    cout << "  Steps (avg):       " << step_stats.first << endl;
}
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // Read dataset
    string input_file = "../data/Iris.txt";
    ifstream fin(input_file);

    int N, D;
    fin >> N >> D;
    vector<vector<long double>> data(N, vector<long double>(D));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) fin >> data[i][j];

    fin.close();
    auto [unique_data, weight] = make_unique(data);
    int K = 50;

    vector<long double> k_loss, k_time, k_step;
    vector<long double> c_loss, c_time, c_step;
    vector<long double> d_loss, d_time, d_step;
    vector<long double> Md_loss, Md_time, Md_step;
    vector<long double> P_loss, P_time, P_step;

    for (int iter = 0; iter < 20; ++iter) {
        // K-means setting
        LO_K_Means<long double> kmeans(unique_data, weight, K, "rand", "squared", rand(), eps);

        // Standard K-means
        auto t1 = high_resolution_clock::now();
        auto [assign_k, centers_k, loss_k] = kmeans.K_means();
        auto t2 = high_resolution_clock::now();
        long double time_k = duration<long double>(t2 - t1).count();

        k_loss.push_back(loss_k);
        k_time.push_back(time_k);
        k_step.push_back(kmeans.step_num);

        // C-LO_K_means (Function 1)
        t1 = high_resolution_clock::now();
        auto [assign_c, centers_c, loss_c] = kmeans.C_LO_K_means();
        t2 = high_resolution_clock::now();
        long double time_c = duration<long double>(t2 - t1).count();

        c_loss.push_back(loss_c);
        c_time.push_back(time_c);
        c_step.push_back(kmeans.step_num);

        // D-LO_K_means (Function 2)
        t1 = high_resolution_clock::now();
        auto [assign_d, centers_d, loss_d] = kmeans.D_LO_K_means();
        t2 = high_resolution_clock::now();
        long double time_d = duration<long double>(t2 - t1).count();

        d_loss.push_back(loss_d);
        d_time.push_back(time_d);
        d_step.push_back(kmeans.step_num);

        // Min-D-LO_K_means (Function 3)
        t1 = high_resolution_clock::now();
        auto [assign_min_d, centers_min_d, loss_min_d] = kmeans.Min_D_LO_K_means();
        t2 = high_resolution_clock::now();
        long double time_min_d = duration<long double>(t2 - t1).count();

        Md_loss.push_back(loss_min_d);
        Md_time.push_back(time_min_d);
        Md_step.push_back(kmeans.step_num);

        // D-LO-P&X
        t1 = high_resolution_clock::now();
        auto [assign_p, centers_p, loss_p] = kmeans.D_LO_P_X();
        t2 = high_resolution_clock::now();
        long double time_p = duration<long double>(t2 - t1).count();

        P_loss.push_back(loss_p);
        P_time.push_back(time_p);
        P_step.push_back(kmeans.step_num);
    }

    print("K-means", k_loss, k_time, k_step);
    print("C-LO", c_loss, c_time, c_step);
    print("D-LO", d_loss, d_time, d_step);
    print("Min-D-LO", Md_loss, Md_time, Md_step);
    print("D-LO-P&X", P_loss, P_time, P_step);
}
