#pragma once

#include <bits/stdc++.h>
using namespace std;

template <typename T>
pair<vector<vector<T>>, vector<T>> make_unique(const vector<vector<T>>& points) {
    vector<vector<T>> sorted_points = points;

    sort(sorted_points.begin(), sorted_points.end());

    vector<vector<T>> unique_points;
    vector<T> weight;

    unique_points.push_back(sorted_points[0]);
    weight.push_back((T)1);
    for (int i = 1; i < sorted_points.size(); i++) {
        if (sorted_points[i] == sorted_points[i - 1]) {
            weight.back() += ((T)1);
        } else {
            unique_points.push_back(sorted_points[i]);
            weight.push_back((T)1);
        }
    }

    return make_pair(unique_points, weight);
}

template <typename T>
pair<T, T> computeStats(const vector<T>& arr) {
    int N = arr.size();
    T sum = 0;
    for (const auto& x : arr) sum += x;
    T mean = sum / (T)N;
    T sq_sum = 0;
    for (const auto& x : arr) {
        T diff = x - mean;
        sq_sum += diff * diff;
    }
    T variance = sq_sum / (T)N;

    return {mean, sqrt(variance)};
}