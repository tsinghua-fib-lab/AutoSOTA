#include <random>
#include <vector>
#include <chrono>
#include "cuda_check.cuh"
#include <cuda_bf16.h>
#include <CLI/CLI.hpp>
#include "dispatch.h"


template<class scalar_t>
std::vector<scalar_t> random_vector(std::size_t n_elements, int seed) {
    std::vector<scalar_t> out(n_elements);
    std::uniform_real_distribution<float> dist(-2.0, 2.0);
    std::mt19937 rng(seed);
    for(size_t i = 0; i < n_elements; ++i) {
        out[i] = (scalar_t)dist(rng);
    }
    return out;
}

template<class scalar_t>
struct Benchmark {
    Benchmark() = default;
    Benchmark(::Shape s) :
        Shape(s),
        FragLen(s.F),
        Queries(s.F * s.W * s.Hq * s.S * s.E),
        Locations(s.F * s.W * s.S),
        Keys(s.F),
        Values(s.F)
    {
    }
    ::Shape Shape;
    std::vector<int> FragLen;
    std::vector<scalar_t> Queries;
    std::vector<int> Locations;
    std::vector<std::vector<scalar_t>> Keys;
    std::vector<std::vector<scalar_t>> Values;

    [[nodiscard]] size_t output_size() const {
        return (size_t)Shape.W * Shape.Hq * Shape.S * Shape.Ev;
    }
};

template<class scalar_t>
Benchmark<scalar_t> make_test(const Shape& shape, int prefix, int worker_seq, int sep) {
    Benchmark<scalar_t> test_case{shape};
    int frag_lengths[] = {prefix, sep, worker_seq, worker_seq};
    test_case.FragLen.assign(frag_lengths, frag_lengths + 4);
    for(auto& v : test_case.Locations) {
        v = 20'000;     // not the real distribution; means that everything attends to everything
    }

    test_case.Queries = random_vector<scalar_t>(shape.F * shape.W * shape.Hq * shape.S * shape.E, 42);
    std::vector<const scalar_t*> k_ptr;
    std::vector<const scalar_t*> v_ptr;
    for(int i = 0; i < shape.F; ++i) {
        test_case.Keys[i] = random_vector<scalar_t>(shape.Hkv * frag_lengths[i] * shape.E, 97 + 13 * i);
        test_case.Values[i] = random_vector<scalar_t>(shape.Hkv * frag_lengths[i] * shape.Ev, 3451 + 17 * i);
        k_ptr.push_back(test_case.Keys[i].data());
        v_ptr.push_back(test_case.Values[i].data());
    }

    return test_case;
}

template<class scalar_t>
void run_benchmarks(const Shape& shape, int prefix_length, bool profile, const std::vector<std::string>& kernels) {
    Benchmark<scalar_t> test = make_test<scalar_t>(shape, prefix_length, 20, 5);

    std::vector<const scalar_t*> k_ptr;
    std::vector<const scalar_t*> v_ptr;
    for(int i = 0; i < test.Shape.F; ++i) {
        k_ptr.push_back(test.Keys[i].data());
        v_ptr.push_back(test.Values[i].data());
    }

    // GPU version
    int* d_locations;
    int* d_frag_lengths;
    scalar_t* d_queries;
    scalar_t* d_keys;
    scalar_t* d_values;
    scalar_t** d_keys_ptr;
    scalar_t** d_values_ptr;
    CUDA_CHECK_THROW(cudaMalloc(&d_locations, test.Locations.size()*sizeof(int)));
    CUDA_CHECK_THROW(cudaMemcpy(d_locations, test.Locations.data(), test.Locations.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMalloc(&d_frag_lengths, test.FragLen.size()*sizeof(int)));
    CUDA_CHECK_THROW(cudaMemcpy(d_frag_lengths, test.FragLen.data(), test.FragLen.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMalloc(&d_queries, test.Queries.size()*sizeof(scalar_t)));
    CUDA_CHECK_THROW(cudaMemcpy(d_queries, test.Queries.data(), test.Queries.size()*sizeof(scalar_t), cudaMemcpyHostToDevice));

    {
        int tk = 0;
        for (const auto &k: test.Keys) {
            tk += k.size();
        }
        CUDA_CHECK_THROW(cudaMalloc(&d_keys, tk * sizeof(scalar_t)));
        std::vector<const scalar_t *> k_ptr_src;

        tk = 0;
        for (auto &k: test.Keys) {
            k_ptr_src.push_back(d_keys + tk);
            CUDA_CHECK_THROW(cudaMemcpy(d_keys + tk, k.data(), k.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));
            tk += k.size();
        }
        CUDA_CHECK_THROW(cudaMalloc(&d_keys_ptr, k_ptr_src.size() * sizeof(void *)));
        CUDA_CHECK_THROW(
                cudaMemcpy(d_keys_ptr, k_ptr_src.data(), k_ptr_src.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }

    {
        int tv = 0;
        for (const auto &v: test.Values) {
            tv += v.size();
        }
        CUDA_CHECK_THROW(cudaMalloc(&d_values, tv * sizeof(scalar_t)));
        std::vector<const scalar_t *> v_ptr_src;
        tv = 0;
        for (auto &v: test.Values) {
            v_ptr_src.push_back(d_values + tv);
            CUDA_CHECK_THROW(cudaMemcpy(d_values + tv, v.data(), v.size() * sizeof(scalar_t), cudaMemcpyHostToDevice));
            tv += v.size();
        }
        CUDA_CHECK_THROW(cudaMalloc(&d_values_ptr, v_ptr_src.size() * sizeof(void *)));
        CUDA_CHECK_THROW(
                cudaMemcpy(d_values_ptr, v_ptr_src.data(), v_ptr_src.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }


    // OK, finally, all the data is there. Allocate output
    scalar_t* d_output;
    CUDA_CHECK_THROW(cudaMalloc(&d_output, test.output_size() * sizeof(scalar_t)));
    int* thrash;
    CUDA_CHECK_THROW(cudaMalloc(&thrash, 1024*1024*1024));

    for(const auto& kernel_id : kernels) {
        try {
            // if we're not running with the profiler, do warm-up
            if (!profile) {
                auto err = hogwild_attention_gpu_dispatch(
                        d_output, 1.f / sqrtf(128), d_locations, d_queries, d_frag_lengths,
                        (const scalar_t **) d_keys_ptr, (const scalar_t **) d_values_ptr, test.Shape,
                        kernel_id);
                CUDA_CHECK_THROW(err);
                CUDA_CHECK_THROW(cudaDeviceSynchronize());
            }
            auto start = std::chrono::steady_clock::now();
            long duration = 0;
            long wall_time = 0;
            int repeat = 0;
            do {
                CUDA_CHECK_THROW(cudaMemset(thrash, 0, 1024 * 1024 * 1024));   // clear L2 cache
                CUDA_CHECK_THROW(cudaDeviceSynchronize());
                auto start_kernel = std::chrono::steady_clock::now();
                auto err = hogwild_attention_gpu_dispatch(
                        d_output, 1.f / sqrtf(128), d_locations, d_queries, d_frag_lengths,
                        (const scalar_t **) d_keys_ptr, (const scalar_t **) d_values_ptr, test.Shape,
                        kernel_id);
                CUDA_CHECK_THROW(err);
                CUDA_CHECK_THROW(cudaDeviceSynchronize());
                duration += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start_kernel).count();
                wall_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start).count();
                ++repeat;
            } while (wall_time < 1'000'000 && !profile);
            printf("Kernel %s took %ld µs [%d repeats]\n", kernel_id.c_str(), duration / repeat, repeat);
        } catch (std::exception& e) {
            fprintf(stderr, "Error during benchmark of kernel %s\n", kernel_id.c_str());
            throw;
        }
    }
}

int main(int argc, const char** argv) {
    CLI::App app;

    bool profile{false};
    int prefix_length{4096};
    std::string kernels = "all";
    std::string dtype = "bf16";
    app.add_flag("--profile", profile, "Enable profiling mode");
    app.add_option("--prefill", prefix_length, "Prefix length");
    app.add_option("--kernel", kernels, "Which kernel version to profile");
    app.add_option("--dtype", dtype, "bf16|fp16|fp32");

    CLI11_PARSE(app, argc, argv);

    std::vector<std::string> all_kernels = {};
    if(kernels == "all") {
        all_kernels.assign(get_all_versions().begin(), get_all_versions().end());
    } else {
        all_kernels.push_back(kernels);
    }

    Shape shape{4, 2, 40, 8, 128, 128, 1};
    if(dtype == "bf16") {
        run_benchmarks<nv_bfloat16>(shape, prefix_length, profile, all_kernels);
    } else if(dtype == "fp16") {
        run_benchmarks<half>(shape, prefix_length, profile, all_kernels);
    } else if(dtype == "fp32") {
        run_benchmarks<float>(shape, prefix_length, profile, all_kernels);
    } else {
        fprintf(stderr, "Invalid data type %s\n", dtype.c_str());
        std::exit(1);
    }
}