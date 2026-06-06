#include <random>
#include "hogwild_cpu.h"
#include <vector>
#include <chrono>
#include <fstream>
#include <cassert>
#include "cuda_check.cuh"
#include "dispatch.h"
#include <CLI/CLI.hpp>
#include <cuda_bf16.h>


template<class scalar_t>
std::vector<scalar_t> random_vector(std::size_t n_elements, int seed) {
    std::vector<scalar_t> out(n_elements);
    std::uniform_real_distribution<float> dist(-2.0, 2.0);
    std::mt19937 rng(seed);
    for(int i = 0; i < n_elements; ++i) {
        out[i] = (scalar_t)dist(rng);
    }
    return out;
}

template<class scalar_t>
struct TestCase {
    TestCase() = default;
    TestCase(struct Shape s) :
        Shape(s),
        FragLen(s.F),
        Queries(s.F * s.W * s.Hq * s.S * s.E),
        Locations(s.F * s.W * s.S),
        Keys(s.F),
        Values(s.F),
        Expected(s.W * s.Hq * s.S * s.Ev)
    {
        if constexpr(std::is_same_v<scalar_t, nv_bfloat16>) {
            rel = 0.05;
        } else if constexpr(std::is_same_v<scalar_t, half>) {
            rel = 1e-3;
        } else {
            rel = 1e-6;
        }
    }
    ::Shape Shape;
    std::vector<int> FragLen;
    std::vector<scalar_t> Queries;
    std::vector<int> Locations;
    std::vector<std::vector<scalar_t>> Keys;
    std::vector<std::vector<scalar_t>> Values;
    std::vector<float> Expected;
    std::vector<float> QK;
    std::vector<float> Scores;
    float rel = 1e-6;
};

template<class T>
void read_binary(T* dst, std::fstream& src, size_t n) {
    if(!src.is_open()) {
        exit(2);
    }
    static_assert(std::is_trivially_copyable_v<T>, "Cannot read binary representation of non-trivial type");
    src.read(reinterpret_cast<char*>(dst), n * sizeof(T));
}

template<class T>
void read_binary(std::vector<T>& dst, std::fstream& src) {
    if(!src.is_open()) {
        exit(2);
    }
    static_assert(std::is_trivially_copyable_v<T>, "Cannot read binary representation of non-trivial type");
    src.read(reinterpret_cast<char*>(dst.data()), dst.size() * sizeof(T));
}

template<class scalar_t>
TestCase<scalar_t> load_test(const char* file_name) {
    int header[256];
    std::fstream src(file_name, std::fstream::in | std::fstream::binary);
    read_binary(header, src, 256);
    int F = header[1];
    int W = header[2];
    int Hq = header[3];
    int Hkv = header[4];
    int E = header[5];
    int Ev = header[6];
    int S = header[7];
    int dtype = header[8];
    TestCase<scalar_t> test_case{Shape{F, W, Hq, Hkv, E, Ev, S}};
    if constexpr(std::is_same_v<scalar_t, nv_bfloat16>) {
        assert(dtype == 2);
    } else if constexpr(std::is_same_v<scalar_t, half>) {
        assert(dtype == 1);
    } else if constexpr(std::is_same_v<scalar_t, float>) {
        assert(dtype == 0);
    }
    read_binary(test_case.FragLen, src);
    int total_length = std::accumulate(test_case.FragLen.begin(), test_case.FragLen.end(), 0);
    read_binary(test_case.Queries, src);
    read_binary(test_case.Locations, src);
    for(int f = 0; f < F; ++f) {
        test_case.Keys[f].resize(Hkv * test_case.FragLen[f] * E);
        read_binary(test_case.Keys[f], src);
    }
    for(int f = 0; f < F; ++f) {
        test_case.Values[f].resize(Hkv * test_case.FragLen[f] * Ev);
        read_binary(test_case.Values[f], src);
    }
    read_binary(test_case.Expected, src);
    test_case.QK.resize(W * S * Hq * total_length);
    read_binary(test_case.QK, src);
    test_case.Scores.resize(W * S * Hq * total_length);
    read_binary(test_case.Scores, src);
    return test_case;
}

template<class scalar_t>
TestCase<scalar_t> make_test() {
    Shape shape{4, 2, 40, 8, 128, 128, 2};
    TestCase<scalar_t> test_case{shape};
    int locations[] = {10, 11, 10, 11,      // common cache
                       3, 4, 3, 4,          // separator
                       0, 1, 5, 6,        // W1 cache
                       5, 6, 0, 1       // W2 cache
    };
    int frag_lengths[] = {500, 3, 20, 20};
    for(int i = 0; i < 16; ++i) {
        locations[i] += 500;
    }
    test_case.FragLen.assign(frag_lengths, frag_lengths + 4);
    test_case.Locations.assign(locations, locations + 16);

    test_case.Queries = random_vector<scalar_t>(shape.F * shape.W * shape.Hq * shape.S * shape.E, 42);
    std::vector<const scalar_t*> k_ptr;
    std::vector<const scalar_t*> v_ptr;
    for(int i = 0; i < shape.F; ++i) {
        test_case.Keys[i] = random_vector<scalar_t>(shape.Hkv * frag_lengths[i] * shape.E, 97 + 13 * i);
        test_case.Values[i] = random_vector<scalar_t>(shape.Hkv * frag_lengths[i] * shape.Ev, 3451 + 17 * i);
        k_ptr.push_back(test_case.Keys[i].data());
        v_ptr.push_back(test_case.Values[i].data());
    }

    hogwild_attention_cpu(test_case.Expected.data(), nullptr, nullptr, 1.f / sqrtf(128),
                          test_case.Locations.data(), test_case.Queries.data(),
                          test_case.FragLen.data(), k_ptr.data(), v_ptr.data(), shape);

    return test_case;
}

template<class scalar_t>
void run_tests(const std::string& source, const std::vector<std::string>& all_kernels) {
    TestCase<scalar_t> test = source.empty() ? make_test<scalar_t>(): load_test<scalar_t>(source.c_str());

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
    CUDA_CHECK_THROW(cudaMalloc(&d_output, test.Expected.size() * sizeof(scalar_t)));

    // run stuff, finally
    int kv_total = std::accumulate(test.FragLen.begin(), test.FragLen.end(), 0);
    printf("input shape: %d x %d x %d x %d - kv: %d\n", test.Shape.W, test.Shape.Hq, test.Shape.S, test.Shape.Ev, kv_total);
    for(const auto& kernel_id : all_kernels) {
        CUDA_CHECK_THROW(cudaMemset(d_output, 0, test.Expected.size() * sizeof(scalar_t)));
        auto error = hogwild_attention_gpu_dispatch(d_output, 1.f / sqrtf(128), d_locations, d_queries, d_frag_lengths,
                                       (const scalar_t **) d_keys_ptr, (const scalar_t **) d_values_ptr, test.Shape,
                                       kernel_id);
        if(error != cudaSuccess) {
            printf("%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
            continue;
        }
        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        std::vector<scalar_t> back(test.Expected.size());
        CUDA_CHECK_THROW(cudaMemcpy(back.data(), d_output, back.size() * sizeof(scalar_t), cudaMemcpyDeviceToHost));
        printf("RS %s %zu\n", kernel_id.c_str(), back.size());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        float *h_ptr = test.Expected.data();
        scalar_t *d_ptr = back.data();
        int errors = 0;
        for (int w = 0; w < test.Shape.W; ++w) {
            for (int h = 0; h < test.Shape.Hq; ++h) {
                for (int s = 0; s < test.Shape.S; ++s) {
                    for (int e = 0; e < test.Shape.Ev; ++e) {
                        if (fabsf(*h_ptr - (float) *d_ptr) > (1e-5 + test.rel * fabsf(*h_ptr)) || std::isnan(*h_ptr) ||
                            std::isnan((float) *d_ptr)) {
                            printf("[%d %d %d %d] %f ≠ %f [err: %f; max: %f]\n", w, h, s, e, *h_ptr, (float) *d_ptr,
                                   fabsf(*h_ptr - (float) *d_ptr), (1e-5 + test.rel * fabsf(*h_ptr)));
                            if (++errors > 20)
                                exit(1);
                        }
                        ++h_ptr;
                        ++d_ptr;
                    }
                }
            }
        }
    }
}

int main(int argc, const char** argv) {
    CLI::App app;

    std::string source;
    std::string kernel = "v26";
    std::string dtype = "fp32";
    app.add_option("source", source, ".bin file with inputs and expected outputs.");
    app.add_option("--kernel", kernel, "Which kernel version to profile");
    app.add_option("--dtype", dtype, "bf16|fp16|fp32. Ignored if a .bin file is provided.");
    CLI11_PARSE(app, argc, argv);

    std::vector<std::string> all_kernels = {};
    if(kernel == "all") {
        all_kernels.assign(get_all_versions().begin(), get_all_versions().end());
    } else {
        all_kernels.push_back(kernel);
    }

    if(!source.empty()) {
        int header[9];
        std::fstream src(source, std::fstream::in | std::fstream::binary);
        read_binary(header, src, 9);
        switch(header[8]) {
            case 0:
                dtype = "fp32";
                break;
            case 1:
                dtype = "fp16";
                break;
            case 2:
                dtype = "bf16";
                break;
        }
    }

    fprintf(stdout, "Testing kernels in dtype %s\n", dtype.c_str());
    if(dtype == "bf16") {
        run_tests<nv_bfloat16>(source, all_kernels);
    } else if(dtype == "fp16") {
        run_tests<half>(source, all_kernels);
    } else if(dtype == "fp32") {
        run_tests<float>(source, all_kernels);
    } else {
        fprintf(stderr, "Invalid data type %s\n", dtype.c_str());
        std::exit(1);
    }

    /*
    if(argc == 2) {
        // CPU test
        std::vector<float> cp(test.Expected.size());
        std::vector<float> as(test.Scores.size());
        std::vector<float> qk(test.QK.size());
        hogwild_attention_cpu(cp.data(), as.data(), qk.data(), 1.f / sqrtf(128),
                              test.Locations.data(), test.Queries.data(),
                              test.FragLen.data(), k_ptr.data(), v_ptr.data(), test.Shape);

        // check qk
        for(int i = 0; i < test.Scores.size(); ++i) {
            //printf("%f %f\n", qk[i], test.QK[i]);
            if((std::fabs(qk[i] - test.QK[i]) > (1e-3 + test.rel * fabs(test.QK[i])) || std::isnan(qk[i])) && !std::isinf(test.QK[i])) {
                printf("[qk@%d] %f %f\n", i, qk[i], test.QK[i]);
                exit(1);
            }
        }

        // check attention scores
        for(int i = 0; i < test.Scores.size(); ++i) {
            if(std::fabs(as[i] - test.Scores[i]) > (1e-5 + test.rel * fabs(test.Scores[i])) || std::isnan(as[i])) {
                printf("[att@%d] %f %f\n", i, as[i], test.Scores[i]);
                exit(1);
            }
        }
    }*/
}