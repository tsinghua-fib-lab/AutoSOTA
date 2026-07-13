
#ifndef GRAPH_H
#define GRAPH_H


#include <vector>
#include <unordered_map>
#include <utility>
#include <functional>
#include <string>
#include <iostream>
#include <utils.hpp>

using Edge = std::pair<int,int>;

// Custom hash for pairs
struct EdgeHash {
    std::size_t operator () (const std::pair<int,int> &p) const {
        auto h1 = std::hash<int>{}(std::min(p.first,p.second));
        auto h2 = std::hash<int>{}(std::max(p.first,p.second));
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct EdgeEqual {
    bool operator()(const Edge& lhs, const Edge& rhs) const {
        return (lhs.first == rhs.first && lhs.second == rhs.second) ||
               (lhs.first == rhs.second && lhs.second == rhs.first);
    }
};

struct Triangle {
    int u, v, w;
    int edge1_idx, edge2_idx, edge3_idx;
};
struct Neighbor {
    int node, edge_idx;
};


class Graph {
public:
    std::vector<std::vector<Neighbor>> adj;
    std::vector<Edge> edges;
    std::vector<Triangle> triangles;
    std::vector<int> edge_triangle_count; //Delta_e
    bool _is_signed;
    bool _is_positive_only;
    std::vector<int> edge_signs;

    //Handling CE: While enumerating triangles in initialization: 
    //Add any negative edges to G that are part of ++- triangles as edges. We don't need any others.
    void add_edge(int u, int v, int sign = 0);

    void initialize();
    size_t ChibaNishizeki();
    size_t CE_triangle_enumeration();

    size_t degree(int v) const;
    size_t num_nodes() const;
    size_t num_edges() const;
    size_t num_triangles() const;
    bool is_signed() const;
    bool is_positive_only() const;

    void readFromFile(const std::string& filePath, bool as_positive_edges=false, int smallest_nodeindex = 1);

};


#endif