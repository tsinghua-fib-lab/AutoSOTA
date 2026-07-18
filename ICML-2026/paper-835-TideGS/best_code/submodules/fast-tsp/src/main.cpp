#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tsp.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


PYBIND11_MODULE(_core, m) {
    m.def("find_tour", &find_tour, R"pbdoc(
        Find a good TSP tour

        Run a local solver to find a near-optimal TSP tour. For small
        problems, the exact solution is guaranteed to be returned.
    )pbdoc", py::arg("dists"), py::arg("duration_seconds") = DEFAULT_TIME_LIMIT);

    m.def("is_valid_tour", &is_valid_tour, R"pbdoc(
        Determine whether a tour is a valid TSP tour.

        Determines whether the given solution is valid.
    )pbdoc", py::arg("n"), py::arg("tour"));

    m.def("compute_cost", &compute_cost_wrapper, R"pbdoc(
        Compute the cost of a TSP tour.

        Computes the cost of a TSP tour.
    )pbdoc", py::arg("tour"), py::arg("dists"));

    m.def("greedy_nearest_neighbor", &greedy_nearest_neighbor, R"pbdoc(
        Solve TSP using greedy nearest neighbor heuristic.

        Solve TSP using the greedy nearest neighbor heuristic. This is a very
        rough approximation to the exact solution. The preferred way is to
        use `find_tour`.
    )pbdoc", py::arg("dists"));

    m.def("solve_tsp_exact", &solve_tsp_exact, R"pbdoc(
        Solve TSP exactly using dynamic programming. Should not be used for
        n much more than ~20.

        Solves TSP optimally using the bottom-up Held-Karp's algorithm in
        O(2^n * n^2) time. This is tractable for small n but quickly becomes
        untractable for n even of medium size.
    )pbdoc", py::arg("dists"));

    m.def("score_tour", &score_tour, R"pbdoc(
        Return the score of a tour, where 1 is better and 0 is worst.

        Returns a double in the range 0..1, where 0 denotes a solution that
        is worse than or equivalent to the naive solution and 1 denotes a
        solution that is equivalent to the optimal solution
    )pbdoc", py::arg("tour"), py::arg("dists"), py::arg("opt_cost"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
