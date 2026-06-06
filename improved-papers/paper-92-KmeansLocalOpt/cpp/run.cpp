/*
Execution example :
Compares Standard K-means, C-LO_K_means, D-LO_K_means, and Min_D_LO_K_means by loss and steps.
*/
#include <bits/stdc++.h>
using namespace std;

#include "LO_K_means.hpp"
#include "Setting.hpp"

const long double eps = 1e-10;

int main() {
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
    // K-means setting
    LO_K_Means<long double> kmeans(unique_data, weight, K, "kmeans++", "squared", rand(), eps);

    // Standard K-means
    auto [assignment, centers, loss] = kmeans.K_means();
    // C_LO_K_means (Function 1)
    auto [assignment_c, centers_c, loss_c] = kmeans.C_LO_K_means();
    // D_LO_K_means (Function 2)
    auto [assignment_d, centers_d, loss_d] = kmeans.D_LO_K_means();
    // Min_D_LO_K_means (Function 3)
    auto [assignment_min_d, centers_min_d, loss_min_d] = kmeans.Min_D_LO_K_means();

    cout << "Loss: " << loss << endl;
    cout << "Loss C: " << loss_c << endl;
    cout << "Loss D: " << loss_d << endl;
    cout << "Loss Min D: " << loss_min_d << endl;
}