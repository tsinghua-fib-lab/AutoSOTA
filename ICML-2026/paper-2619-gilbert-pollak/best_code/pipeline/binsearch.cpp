#include <bits/stdc++.h>
using namespace std;

bool run(int mid) {
    // call `./plot mid` (e.g. ./plot 8500) and return its exit code
    string cmd = string("./plot ") + to_string(mid) + string(" > plot_result.txt");
    int status = system(cmd.c_str());
    assert(status == 0);
    ifstream fin("plot_result.txt");
    int certified_size, que_size;
    fin >> certified_size >> que_size;
    return que_size == 0;
}

int main() {
    int low = 8240;
    int high = 8670;
    
    const double precision = 0.0001;
        
    while (high > low) {
        int mid = (low + high + 1) / 2;
        double mid_double = mid / 10000.0;
        cout << "Testing rho = " << fixed << setprecision(4) << mid_double << " ... ";
        int result = run(mid);
        cout << (result == 1 ? "Success" : "Failure") << endl;
        if (result == 1) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    
    double final_x = low / 10000.0;
    cout << fixed << setprecision(4) << final_x << endl;
    ofstream frho("rho.txt");
    frho << fixed << setprecision(4) << final_x << endl;
    int result = run(low + 1);
    assert(result == 0);
    // rename plot_result.txt to bottleneck.txt
    rename("plot_result.txt", "bottleneck.txt");
    return 0;
}