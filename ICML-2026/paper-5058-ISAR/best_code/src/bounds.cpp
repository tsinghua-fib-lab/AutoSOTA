#include <timer.hpp>
#include <bounds.hpp>

#include <random>
#include <algorithm>
#include <unordered_map>
#include <queue>

// An updatable priority queue
template<typename T, typename U>
class UpdatableHeap {
private:
    std::vector<std::pair<T, U>> heap;
    std::unordered_map<U, int> position;

    void sift_up(int i) {
        while (i > 0) {
            int p = (i - 1) / 2;
            if (heap[p].first > heap[i].first) {
                std::swap(heap[p], heap[i]);
                position[heap[p].second] = p;
                position[heap[i].second] = i;
                i = p;
            } else {
                break;
            }
        }
    }

    void sift_down(int i) {
        int n = heap.size();
        while (2 * i + 1 < n) {
            int j = 2 * i + 1;
            if (j + 1 < n && heap[j + 1].first < heap[j].first) {
                j++;
            }
            if (heap[i].first > heap[j].first) {
                std::swap(heap[i], heap[j]);
                position[heap[i].second] = i;
                position[heap[j].second] = j;
                i = j;
            } else {
                break;
            }
        }
    }

public:
    void push(T priority, U value) {
        heap.push_back({priority, value});
        int i = heap.size() - 1;
        position[value] = i;
        sift_up(i);
    }

    std::pair<T, U> top() {
        return heap[0];
    }

    void pop() {
        int n = heap.size();
        position.erase(heap[0].second);
        heap[0] = heap[n - 1];
        position[heap[0].second] = 0;
        heap.pop_back();
        if (!heap.empty()) {
            sift_down(0);
        }
    }

    void update(T priority, U value) {
        if (position.find(value) == position.end()) {
            push(priority, value);
            return;
        }
        int i = position[value];
        T old_priority = heap[i].first;
        heap[i].first = priority;
        if (priority < old_priority) {
            sift_up(i);
        } else {
            sift_down(i);
        }
    }

    bool empty() {
        return heap.empty();
    }
    
    size_t size() {
        return heap.size();
    }
};

int greedy_triangle_packing(Graph const & graph){
    ScopedTimer t1("GreedyPacking");
    auto triangle_edge_weight_sum = [&](int triangle_idx) {
        const auto& tri = graph.triangles[triangle_idx];
        int e1 = tri.edge1_idx, e2 = tri.edge2_idx, e3 = tri.edge3_idx;
        return graph.edge_triangle_count[e1] + graph.edge_triangle_count[e2] + graph.edge_triangle_count[e3];
    };

    std::vector<int> triangle_indices(graph.num_triangles());
    std::iota(triangle_indices.begin(), triangle_indices.end(), 0);

    std::sort(triangle_indices.begin(), triangle_indices.end(),
        [&](int a, int b) {
            return triangle_edge_weight_sum(a) < triangle_edge_weight_sum(b);
        });

    // Greedily pick triangles with disjoint edges
    std::vector<int> used_edges(graph.num_edges(), 0);
    std::vector<int> packing;
    for (int idx : triangle_indices) {
        const auto& tri = graph.triangles[idx];
        int e1 = tri.edge1_idx, e2 = tri.edge2_idx, e3 = tri.edge3_idx;
        if (used_edges[e1] || used_edges[e2] || used_edges[e3]) continue;
        packing.push_back(idx);
        used_edges[e1]=1;
        used_edges[e2]=1;
        used_edges[e3]=1;
    }
    return packing.size();
}

double balkanski(Graph const & graph){
    ScopedTimer t1("Balkanski");
    double sol=0;
    for (size_t triangle_idx = 0; triangle_idx < graph.triangles.size(); ++triangle_idx) {
        const auto& tri = graph.triangles[triangle_idx];
        int e1 = tri.edge1_idx, e2 = tri.edge2_idx, e3 = tri.edge3_idx;
        double a1= 1.0 /graph.edge_triangle_count[e1];
        double a2= 1.0 /graph.edge_triangle_count[e2];
        double a3= 1.0 /graph.edge_triangle_count[e3];
        sol+= std::min({a1,a2,a3});
    }
    return sol;
}

int LSCut(Graph const & graph, int repetitions) {
    int bestcut = 0;
    for (size_t i = 0; i < repetitions; i++)
    {
        ScopedTimer t1("LocalSearch");
        std::vector<int> nodeassignment(graph.num_nodes());
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 1);

        for (size_t i = 0; i < graph.num_nodes(); ++i) {
            nodeassignment[i] = dist(rng) ? 1 : -1;
        }

        bool improved = true;
        while (improved) {
            improved = false;
            for (size_t v = 0; v < graph.num_nodes(); ++v) {
                const auto& neighbors = graph.adj[v];
                int degree = neighbors.size();
                int otherSideNeighbors = 0;
                for (const Neighbor& neighbor : neighbors) {
                    if (nodeassignment[v] != nodeassignment[neighbor.node]) {
                        otherSideNeighbors++;
                    }
                }
                if (otherSideNeighbors < degree / 2.0) {
                    nodeassignment[v] *= -1;
                    improved = true;
                }
            }
        }

        int maxCut = 0;
        for (size_t v = 0; v < graph.num_nodes(); ++v) {
            const auto& neighbors = graph.adj[v];
            for (const Neighbor& neighbor : neighbors) {
                if (nodeassignment[v] != nodeassignment[neighbor.node]) {
                    maxCut++;
                }
            }
        }
        int thiscut = maxCut / 2;
        bestcut = std::max(bestcut,thiscut);
    }
    return bestcut;
}


int setcovergreedy(Graph const & graph){
    ScopedTimer t1("SetCoverGreedy");
    int m = graph.num_edges();

    std::vector<int> delta(m);
    for (int e = 0; e < m; e++) {
        delta[e] = graph.edge_triangle_count[e];
    }

    UpdatableHeap<int,int> heap; // (priority = -Δ_e, edge id)
    for (int e = 0; e < m; e++) {
        heap.push(-delta[e], e);
    }

    std::vector<bool> edge_removed(m, false);
    std::vector<long long> t(m+1, 0);

    int ell=1;
    for (ell = 1; ell <= m; ell++) {
        if (heap.empty()) throw std::runtime_error("empty heap");

        auto [neg_val, e] = heap.top();
        heap.pop();
        int val = -neg_val; // restore Δ

        if (val==0){
            break;
        }

        edge_removed[e] = true;
        t[ell] = t[ell-1] + val;

        
        // Destroy triangles containing e
        int u = graph.edges[e].first;
        int v = graph.edges[e].second;
        int lower = graph.degree(u) < graph.degree(v) ? u : v;
        int higher = lower==u ? v:u;
        for (const auto& w : graph.adj[lower]) {
            if (edge_removed[w.edge_idx]) continue;
            // Since adj is sorted by node, use binary search to check if 'u' is a neighbor
            auto neighbor_it = std::lower_bound(
                graph.adj[w.node].begin(), graph.adj[w.node].end(), higher,
                [](const Neighbor& n, int value) { return n.node < value; });
            if (neighbor_it != graph.adj[w.node].end() && neighbor_it->node == higher) {
                if (edge_removed[neighbor_it->edge_idx]) continue;
                //Found triangle containing e
                delta[w.edge_idx]--;
                heap.update(-delta[w.edge_idx], w.edge_idx);
                delta[neighbor_it->edge_idx]--;
                heap.update(-delta[neighbor_it->edge_idx], neighbor_it->edge_idx);
            }
        }
    }
    return ell-1;
}

int randomized_pivot(Graph const & graph, int repetitions) {
    ScopedTimer t1("RandomizedPivot");
    int best_objective = graph.num_edges();

    std::vector<int> nodes(graph.num_nodes());
    for(int i=0; i<nodes.size(); ++i) nodes[i] = i;

    std::mt19937 rng(std::random_device{}()); 

    for (int r = 0; r < repetitions; ++r) {
        std::shuffle(nodes.begin(), nodes.end(), rng);
        std::vector<int> cluster_id(nodes.size(), -1);
        int current_cluster = 0;

        for (int u : nodes) {
            if (cluster_id[u] != -1) continue;

            cluster_id[u] = current_cluster;
            for (const auto& neighbor : graph.adj[u]) {
                if (graph.edge_signs[neighbor.edge_idx]!=1) continue;
                int v = neighbor.node;
                if (cluster_id[v] == -1) {
                    cluster_id[v] = current_cluster;
                }
            }
            current_cluster++;
        }

        int cost = 0;
        for (int i = 0; i < graph.num_edges(); ++i) {
            int u = graph.edges[i].first;
            int v = graph.edges[i].second;
            int sign = graph.edge_signs[i];

            if (cluster_id[u] == cluster_id[v] and sign == -1) {
                cost++;
            } 
            if (cluster_id[u] != cluster_id[v] and sign == 1) {
                cost++;
            }
        }
        best_objective = std::min(best_objective,cost);
    }

    return best_objective;
}

