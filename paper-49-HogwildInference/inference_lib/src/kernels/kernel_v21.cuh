// move reduction back into a separate kernel.
// in terms of pure kernel times, this is faster
#include "common.h"
#include "vec.cuh"
#include "cuda_check.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline_primitives.h>

namespace cg = cooperative_groups;

namespace v21
{
constexpr const int SubWarpSize = 8;
constexpr const int WarpSize = 32;

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void hogwild_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const int* fragment_lengths,
        const scalar_t* const* key_fragments,
        const scalar_t* const* value_fragments,
        Shape shape) {
    // Input:   keys: [Hkv, fragment_lengths[i], E] for i in [F]
    //          values: [Hkv, fragment_lengths[i], Ev] for i in [F]
    //          fragment_lengths: [F]
    //          queries: [F, W, Hq, S, E]
    //          locations [F, W, S]
    // Scratch: workspace [W, Hq, S, Ev] (in float32, iff scalar_t != float32) + [W, Hq, S] BlockResult
    // Output:  [W, Hq, S, Ev]
    // attention mask: s attends to l iff locations[b, s] >= l (i.e., shifted causal masking)

    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S;
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    ptrdiff_t q_stride = E * S * Hq * W;
    extern __shared__ float scratch[];

    // adjust scale so we can use base-2 exponent later on
    float l2scale = scale / std::log(2.f);

    int hkv = blockIdx.x;
    int w = blockIdx.y % W;
    int s = blockIdx.y / W;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int hq = hkv * GQA;
    ptrdiff_t q_offset = ((w * Hq + hq) * S + s) * E;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);   // vectors per head per thread
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);  // vectors per head per thread

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using full_fvec_t = GenericVector<float, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>;
    qk_cache_t q_cache[GQA];

    // combine values
    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    // determine maximum and online logsumexp
    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

        for (int f = 0; f < shape.F; ++f) {
            int q_loc = locations[(f * W + w) * S + s];
            int L = fragment_lengths[f];
            int maxL = std::min(L, q_loc + 1);

            for (int gqa = 0; gqa < GQA; ++gqa) {
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    full_vec_t qv = full_vec_t::load(queries + f * q_stride + q_offset + gqa * S * E + e);
                    for (int j = 0; j < VecSize; ++j) {
                        q_cache[gqa][ee * VecSize + j] = qv[j];
                    }
                }
            }

            const scalar_t* value_fragment = value_fragments[f];
            const scalar_t* key_fragment = key_fragments[f];

            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;
                ptrdiff_t k_offset = (hkv * L + l) * E;
                ptrdiff_t v_offset = (hkv * L + l) * Ev;
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x,
                                            key_fragment + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x,
                                            value_fragment + v_offset + e, sizeof(full_vec_t));
                }
            };

            int stage = 0;
            ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
            __pipeline_commit();
            ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
            __pipeline_commit();

            for (int ll = split; ll < maxL; ll += StepSize) {
                int l = ll + sub_warp.meta_group_rank() * splits;
                qk_cache_t keys;
                v_cache_t vals;
                __pipeline_wait_prior(1);
                if (l >= maxL) continue;
                unsigned mask = __activemask();

                for (int ee = 0; ee < VPH_k; ++ee) {
                    full_vec_t tmp = keys_lookahead[(stage * VPH_k + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) {
                        keys[ee * VecSize + j] = (float)tmp[j];
                    }
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    full_vec_t tmp = vals_lookahead[(stage * VPH_v + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) {
                        vals[ee * VecSize + j] = (float)tmp[j];
                    }
                }

                ldg_sts((stage + 2) % 2, l + 2 * StepSize);
                stage = (stage + 1) % 2;
                __pipeline_commit();

                float qk[GQA] = {};
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    for (int ee = 0; ee < VPH_k; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            qk[gqa] += q_cache[gqa][ee * VecSize + j] * keys[ee * VecSize + j];
                        }
                    }
                }

                // important: By having the warp shuffles together like this in a separate loop,
                // the compiler ends up generating better sequenced assembly, where we first initiate a
                // bunch of shuffles and only then do the addition, hiding the latency much better
                // than in the single-loop version.
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0100, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0010, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0001, 8);
                }

                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    if (qk[gqa] > maximum[gqa]) {
                        float rescale = std::exp2f(l2scale * (maximum[gqa] - qk[gqa]));
                        for (int j = 0; j < v_cache_t::size; ++j) {
                            v_cache[gqa][j] *= rescale;
                        }
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));

                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                        }
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    // Each sub-warp stores its local softmax statistics to shared memory
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        // combine split-k results
        if (sub_warp.thread_rank() == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + 2 * sub_warp.meta_group_rank() + 2 * WarpSize * gqa);
        }
    }

    __syncthreads();

    // Reduce stats over the entire block
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        float r_max = maximum[gqa];
        float l_max = maximum[gqa];
        float r_lse = 0;
        if (warp.thread_rank() < SubWarpMetaSize) {
            stats_t data = stats_t::load(scratch + 2 * warp.thread_rank() + 2 * WarpSize * gqa);
            r_max = data[0];
            r_lse = data[1];
        }

        maximum[gqa] = cg::reduce(warp, r_max, cg::greater<float>{});
        r_lse *= std::exp2f(l2scale * (r_max - maximum[gqa]));
        lse[gqa] = cg::reduce(warp, r_lse, cg::plus<float>{});

        // Note: It *is* possible that no thread in this warp had any valid position (due to causal masking),
        // which would lead to division by zero -> 0 * inf = NaN here.
        if (lse[gqa] != 0) {
            float rescale = std::exp2f(l2scale * (l_max - maximum[gqa])) / lse[gqa];
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[gqa][j] *= rescale;
            }
        }

        if (threadIdx.x == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        }

        // now reduce value across subwarp within a warp
        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_cache[gqa][ee * VecSize + j];
                static_assert(SubWarpSize == 8);
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_cache[gqa][ee * VecSize + j] = v;
            }
        }
    }

    __syncthreads();

    // we've reduced within one warp, so now only one subwarp per warp has
    // to write global results
    if (sub_warp.meta_group_rank() % (WarpSize / SubWarpSize) == 0) {
        #pragma unroll
        for (int gqa = 0; gqa < GQA; ++gqa) {
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    full_fvec_t store;
                    for (int j = 0; j < VecSize; ++j) {
                        store[j] = v_cache[gqa][ee * VecSize + j];
                    }
                    store.store(scratch + e + Ev * sub_warp.meta_group_rank() / (WarpSize / SubWarpSize) + gqa * 256 / WarpSize * Ev);
                }
            }
    }
    __syncthreads();

    int gqa = warp.meta_group_rank();
    if (gqa >= GQA) return;
    int h = hkv * GQA + gqa;
    int res_base = ((w * Hq + h) * S + s);
    int res_inc = W * Hq * S;
    int res_idx = res_base + split * res_inc;
    float* global_accumulator = reinterpret_cast<float*>(workspace);
    float* lse_target = global_accumulator + W * Hq * S * Ev * splits;

    stats_t data = stats_t::load(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
    float own_lse = data[1];
    float own_max = data[0];
    own_lse = std::log2(own_lse) + l2scale * own_max;

    for (int e = vec_t::size * warp.thread_rank(); e < Ev; e += vec_t::size * warp.size()) {
        // merge the local results
        fvec_t res = fvec_t::zeros();
        for (int j = 0; j < SubWarpMetaSize / (WarpSize / SubWarpSize); ++j) {
            fvec_t sv = fvec_t::load(scratch + e + Ev * j + gqa * 256 / WarpSize * Ev);
            for (int jj = 0; jj < vec_t::size; ++jj) {
                res[jj] += sv[jj];
            }
        }
        res.store(global_accumulator + res_idx * Ev + e);
    }

    lse_target[res_idx] = own_lse;
}


template<int Ev, class scalar_t>
__global__ __launch_bounds__(32) void hogwild_attention_reduce_kernel(
        scalar_t* out, const float* v_buffer, const float* lse_buffer, int splits, Shape shape) {
    int h = blockIdx.x;
    int w = blockIdx.y % shape.W;
    int s = blockIdx.y / shape.W;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    using v_cache_t = GenericVector<float, Ev / warp.size()>;
    v_cache_t v_cache = v_cache_t::zeros();

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;

    float own_lse = std::numeric_limits<float>::lowest();

    for (int split = 0; split < splits; ++split) {
        int res_idx = ((split * shape.W + w) * shape.Hq + h) * shape.S + s;
        const float* split_res = v_buffer + res_idx * Ev;
        float res_lse = lse_buffer[res_idx];
        if (res_lse == std::numeric_limits<float>::lowest()) {
            continue;
        }
        float max = std::max(own_lse, res_lse);
        float sa = std::exp2f(own_lse - max);
        float sb = std::exp2f(res_lse - max);
        float rescaler_a = sa / (sa + sb);
        float rescaler_b = sb / (sa + sb);
        #pragma unroll
        for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
            int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
            fvec_t sv = fvec_t::load_lu(split_res + e);
            for (int jj = 0; jj < fvec_t::size; ++jj) {
                float old = v_cache[ee + jj];
                float upd = old * rescaler_a + sv[jj] * rescaler_b;
                v_cache[ee + jj] = upd;
            }
        }
        own_lse = std::log2(sa + sb) + max;
    }

    for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
        int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
        vec_t st = vec_t::zeros();
        for (int jj = 0; jj < fvec_t::size; ++jj) {
            st[jj] = (scalar_t)v_cache[ee + jj];
        }
        st.store(out + ((w * shape.Hq + h) * shape.S + s) * Ev + e);
    }
}

template<class scalar_t>
cudaError_t hogwild_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const int* fragment_lengths,
                           const scalar_t** key_fragments,
                           const scalar_t** value_fragments,
                           Shape shape) {
    int problem_size = shape.Hkv * shape.W * shape.S;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // Note: The current kernel will **not** work if there is only one split!
    int splits = max(2, sms / problem_size);

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)shape.W * (unsigned)shape.S, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = shape.Ev * sizeof(float) * block_dim.x / 32 * (shape.Hq / shape.Hkv);
    smem += 2 * sizeof(float) * (shape.Hq / shape.Hkv);
    smem = std::max(smem, 2 * (shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));
    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    std::size_t required_workspace = shape.W * shape.Hq * shape.S * splits;  // [W, Hq, S, K]
    size_t alloc = required_workspace * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_workspace) {
        if (workspace)
            CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, alloc));
        CUDA_RETURN_ON_ERROR(cudaMemset(workspace, 0, alloc));
        workspace_size = required_workspace;
    }

    if (shape.E == 128 && shape.Ev == 128) {
        if(shape.Hq == shape.Hkv * 5) {
            CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(hogwild_attention_gpu_kernel21<128, 128, 5, scalar_t>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            hogwild_attention_gpu_kernel21<128, 128, 5><<<grid_dim, block_dim, smem>>>(
                    out, workspace, scale, locations, queries, fragment_lengths, key_fragments, value_fragments, shape);
        } else if(shape.Hq == shape.Hkv * 4) {
            CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(hogwild_attention_gpu_kernel21<128, 128, 4, scalar_t>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            hogwild_attention_gpu_kernel21<128, 128, 4><<<grid_dim, block_dim, smem>>>(
                    out, workspace, scale, locations, queries, fragment_lengths, key_fragments, value_fragments, shape);
        } else if(shape.Hq == shape.Hkv * 8) {
            CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(hogwild_attention_gpu_kernel21<128, 128, 8, scalar_t>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            hogwild_attention_gpu_kernel21<128, 128, 8><<<grid_dim, block_dim, smem>>>(
                    out, workspace, scale, locations, queries, fragment_lengths, key_fragments, value_fragments, shape);
        } else {
            printf("Unsupported GQA\n");
            return cudaError_t::cudaErrorNotYetImplemented;
        }

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.W * (unsigned)shape.S, 1};
        hogwild_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, (float*)workspace, (float*)workspace + splits * shape.W * shape.Hq * shape.S * shape.Ev,
                splits, shape);
    } else {
        printf("Unsupported head dimension\n");
        return cudaError_t::cudaErrorNotYetImplemented;
    }
    return cudaGetLastError();
}

}  // namespace v21
