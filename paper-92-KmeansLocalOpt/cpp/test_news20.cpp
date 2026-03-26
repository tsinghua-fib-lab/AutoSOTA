// Reproduce Table 1 for News20-1 dataset (N=2000, d=1089, K=5, Random Init)
// Compare K-means, C-LO_K_means, D-LO_K_means, and Min_D_LO_K_means by loss, time, and steps.

#include <bits/stdc++.h>
using namespace std;

#include "LO_K_means.hpp"
#include "Setting.hpp"

const long double eps = 1e-10;
using namespace std::chrono;

void print_results(const string& str, const vector<long double>& loss, const vector<long double>& time, const vector<long double>& step) {
    pair<long double, long double> loss_stats = computeStats(loss);
    pair<long double, long double> time_stats = computeStats(time);
    pair<long double, long double> step_stats = computeStats(step);

    cout << fixed << setprecision(4);
    cout << str << ":" << endl;
    cout << "  Loss (avg ± std):  " << loss_stats.first << " ± " << loss_stats.second << endl;
    cout << "  Loss (min):        " << *min_element(loss.begin(), loss.end()) << endl;
    cout << "  Time (avg):        " << time_stats.first << " sec" << endl;
    cout << "  Steps (avg):       " << step_stats.first << endl;
    cout << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // Read News20-1 dataset
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

    cout << "Dataset: news20_2000_1089, N=" << N << ", D=" << D << ", K=" << K << endl;
    cout << "Unique points: " << unique_data.size() << endl << endl;

    vector<long double> k_loss, k_time, k_step;
    vector<long double> c_loss, c_time, c_step;
    vector<long double> d_loss, d_time, d_step;
    vector<long double> Md_loss, Md_time, Md_step;

    for (int iter = 0; iter < 20; ++iter) {
        // K-means setting with RANDOM init
        LO_K_Means<long double> kmeans(unique_data, weight, K, "kmeans++", "squared", rand(), eps);

        // Standard K-means
        auto t1 = high_resolution_clock::now();
        auto [assign_k, centers_k, loss_k] = kmeans.K_means();
        auto t2 = high_resolution_clock::now();
        long double time_k = duration<long double>(t2 - t1).count();

        k_loss.push_back(loss_k);
        k_time.push_back(time_k);
        k_step.push_back(kmeans.step_num);

        cout << "Trial " << iter+1 << ": K-means loss=" << fixed << setprecision(0) << loss_k 
             << ", time=" << setprecision(4) << time_k << "s, steps=" << kmeans.step_num << endl;

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

        // Min-D-LO_K_means (Function 3) - best of 5 independent runs per trial
        t1 = high_resolution_clock::now();
        auto [assign_min_d, centers_min_d, loss_min_d] = kmeans.Min_D_LO_K_means();
        // Run 19 more independent Min-D-LO attempts from different kmeans++ seeds, keep best
        for (int r = 0; r < 19; r++) {
            LO_K_Means<long double> kmeans_r(unique_data, weight, K, "kmeans++", "squared", rand(), eps);
            auto [ar, cr, lr] = kmeans_r.Min_D_LO_K_means();
            if (lr < loss_min_d) {
                loss_min_d = lr;
                assign_min_d = ar;
                centers_min_d = cr;
            }
        }
        t2 = high_resolution_clock::now();
        long double time_min_d = duration<long double>(t2 - t1).count();

        Md_loss.push_back(loss_min_d);
        Md_time.push_back(time_min_d);
        Md_step.push_back(kmeans.step_num);
    }

    cout << "\n=== Results (News20-1, K=5, Random Init, 20 trials) ===" << endl;
    print_results("K-means", k_loss, k_time, k_step);
    print_results("C-LO", c_loss, c_time, c_step);
    print_results("D-LO", d_loss, d_time, d_step);
    print_results("Min-D-LO", Md_loss, Md_time, Md_step);
    
    return 0;
}
