#include <graph.hpp>
#include <timer.hpp>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>

    void Graph::add_edge(int u, int v, int sign) {
        if (u == v) return;
        if (u >= adj.size() || v >= adj.size()) {
            adj.resize(std::max({u+1,v+1}));
        }
        // Check for duplicate edge
        auto it = std::lower_bound(
            adj[u].begin(), adj[u].end(), v,
            [](const Neighbor& n, int value) { return n.node < value; });
        if (it != adj[u].end() && it->node == v) return;
        
        edges.push_back({std::min(u, v), std::max(u, v)});
        int edge_idx = edges.size() - 1;
        adj[u].push_back({v,edge_idx});
        adj[v].push_back({u,edge_idx});
        if (_is_signed) {
            if (sign!=1 and sign!=-1) throw std::runtime_error("Edge " + std::to_string(u) + " " + std::to_string(v) + " with invalid sign " + std::to_string(sign));
            edge_signs.push_back(sign);
        }
    }

    void Graph::initialize() {
        //ScopedTimer t1("Graph::initialize");
        for(int i=0; i<num_nodes(); ++i) {
            std::sort(adj[i].begin(), adj[i].end(), [](const Neighbor& a, const Neighbor& b) {
            return a.node < b.node;
            });
        }
        edge_triangle_count.resize(edges.size(),0);
        if (_is_positive_only) {
            CE_triangle_enumeration();
        } else{
            ChibaNishizeki();
        }
        for(int i=0; i<num_nodes(); ++i) {
            std::sort(adj[i].begin(), adj[i].end(), [](const Neighbor& a, const Neighbor& b) {
            return a.node < b.node;
            });
        }
    }

    size_t Graph::CE_triangle_enumeration(){
        //We enumerate all bad triangles and add negative edges part of such triangles to the graph
        for (int u = 0; u < num_nodes(); ++u) {
            for (size_t i = 0; i < adj[u].size(); ++i) {
                for (size_t j = i + 1; j < adj[u].size(); ++j) {
                    int v = adj[u][i].node;
                    int e1_idx = adj[u][i].edge_idx;
                    int w = adj[u][j].node;
                    int e2_idx = adj[u][j].edge_idx;
                    auto it = std::lower_bound(
                        adj[v].begin(), adj[v].end(), w,
                        [](const Neighbor& n, int value) { return n.node < value; });
                    if (it == adj[v].end() || it->node != w) {
                        triangles.push_back({u, v, w, e1_idx, e2_idx, 0});
                    }
                }
            }
        }
        std::sort(triangles.begin(), triangles.end(), [](const Triangle& a, const Triangle& b) {
            if (a.v != b.v) return a.v < b.v;
            return a.w < b.w;
        });
        for (size_t i = 0; i < triangles.size(); ++i) {
            auto [ u, v, w, e1, e2, _] = triangles[i];
            // Create new edge index only for unique (v, w) pairs
            if (i == 0 || triangles[i].v != triangles[i-1].v || triangles[i].w != triangles[i-1].w) {
                add_edge(v, w, -1);
                edge_triangle_count.push_back(0);
            }
            int e3 = num_edges() - 1;
            triangles[i].edge3_idx = e3;
            edge_triangle_count[e1]++;
            edge_triangle_count[e2]++;
            edge_triangle_count[e3]++;
        }
        return triangles.size();
    }

    size_t Graph::ChibaNishizeki()
    {
        if (not triangles.empty()){
            return 0;
        }
        int all_triangles = 0;

        //Sort nodes by degree
        std::vector<int> nodes(num_nodes());
        {
            //ScopedTimer t2("Graph::ChibaNishizeki::sort");
            std::iota(nodes.begin(), nodes.end(), 0);
            std::sort(nodes.begin(), nodes.end(), [this](int a, int b) {
                return (degree(a) > degree(b)) || (degree(a) == degree(b) && a > b);
            });
        }

        // Create a copy of the adjacency list if needed by removing the reference.
        std::vector<std::vector<Neighbor>> adjListCopy = adj;

        for (int u : nodes) {
            for (auto [v,uv_idx] : adjListCopy[u]) {
                for (auto it = adjListCopy[v].begin(); it != adjListCopy[v].end();) {
                    if (it->node == u) {
                        it = adjListCopy[v].erase(it); // We can safely remove u already here. Erase returns it to next item.
                        continue;
                    }
                    // Since adjListCopy[it->node] is sorted by node, use binary search to check if 'u' is a neighbor
                    auto neighbor_it = std::lower_bound(
                        adjListCopy[it->node].begin(), adjListCopy[it->node].end(), u,
                        [](const Neighbor& n, int value) { return n.node < value; });
                    if (neighbor_it != adjListCopy[it->node].end() && neighbor_it->node == u) {

                        all_triangles++;
                        int e1_idx = uv_idx;
                        int e2_idx = it->edge_idx;
                        int e3_idx = neighbor_it->edge_idx;
                        if (_is_signed){
                            // Check for ++-
                            if ( edge_signs[e1_idx]+edge_signs[e2_idx]+edge_signs[e3_idx] != 1){
                                it++;
                                continue;
                            } 
                        }
                        triangles.push_back({u, v, it->node, e1_idx, e2_idx, e3_idx});
                        edge_triangle_count[e1_idx]++;
                        edge_triangle_count[e2_idx]++;
                        edge_triangle_count[e3_idx]++;
                    }
                    it++;
                }
            }
        }
        std::cout << " Found "<<all_triangles << " triangles in total of which " << triangles.size() << " are relevant"<<std::endl;
        return triangles.size();
    }

    size_t Graph::degree(int v) const { return adj[v].size(); }
    size_t Graph::num_nodes() const { return adj.size(); }
    size_t Graph::num_edges() const { return edges.size(); }
    size_t Graph::num_triangles() const { return triangles.size(); }
    bool Graph::is_signed() const { return _is_signed; }
    bool Graph::is_positive_only() const { return _is_positive_only; }

    void Graph::readFromFile(const std::string& filePath, bool as_positive_edges, int smallest_nodeindex) {
        // C++17 compatible suffix check (gcc 9.x lacks std::string::ends_with)
        auto ends_with = [](const std::string& s, const std::string& suffix) -> bool {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        bool input_is_signed = ends_with(filePath, ".cc");
        bool nextline_is_nm = ends_with(filePath, ".mtx");
        _is_positive_only = as_positive_edges and not input_is_signed;
        _is_signed = input_is_signed or _is_positive_only;

        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("File cannot be opened: " + filePath);
        }
        std::string line;
        int n,m;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#' || line == "MULTICUT") {
                continue;
            }
            std::istringstream iss(line);
            int u, v, sign=0;   //sign is either 1 or -1.
            if (nextline_is_nm) {
                iss >> n >> n >> m;
                nextline_is_nm = false;
                continue;
            }
            if (!(iss >> u >> v)) continue;
            if (input_is_signed) iss>>sign;
            if (_is_positive_only) sign=1; //input edges are declared positive
            if (smallest_nodeindex==1 ){ 
                if  (u==0 or v==0) throw std::runtime_error("zero-indexed file");
                u--;
                v--;
            }
            add_edge(u, v, sign);
        }
        if (not _is_positive_only) {
            std::cout << "Finished reading " << filePath << " with " << num_nodes() << " nodes and " << num_edges()<<" edges." << std::endl;
        } else{
            std::cout << "Finished reading " << filePath << " with " << num_nodes() << " nodes and " << num_edges()<<" positive edges and "<<(num_nodes()*(num_nodes()-1)/2-num_edges())<<" implicit negative edges." << std::endl;
        }

        initialize();
        if (_is_positive_only) {
            std::cout << "Initialized graph has " << num_edges() << " edges." << std::endl;
        }
        
        if (not _is_signed){
            std::cout << "Initialized graph has " << num_triangles() << " triangles." << std::endl;
        }
        else{
            std::cout << "Initialized graph is signed and has " << num_triangles() << " ++- triangles." << std::endl;
        }
    }