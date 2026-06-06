#include <ATen/Tensor.h>
#include <ATen/ops/empty.h>
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/library.h>
#include <vector>

#include "kernels/kernel_v21.cuh"
#include "rope.cuh"

template<class scalar_t>
const scalar_t* torch_get_pointer(const at::Tensor& tensor) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return tensor.const_data_ptr<float>();
    } else if constexpr (std::is_same_v<scalar_t, half>) {
        return reinterpret_cast<const half*>(tensor.const_data_ptr<at::Half>());
    } else if constexpr (std::is_same_v<scalar_t, nv_bfloat16>) {
        return reinterpret_cast<const nv_bfloat16*>(tensor.const_data_ptr<at::BFloat16>());
    } else {
        return nullptr;
    }
}

template<class scalar_t>
scalar_t* torch_get_pointer(at::Tensor& tensor) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return tensor.data_ptr<float>();
    } else if constexpr (std::is_same_v<scalar_t, half>) {
        return reinterpret_cast<half*>(tensor.data_ptr<at::Half>());
    } else if constexpr (std::is_same_v<scalar_t, nv_bfloat16>) {
        return reinterpret_cast<nv_bfloat16*>(tensor.data_ptr<at::BFloat16>());
    } else {
        return nullptr;
    }
}

template<class scalar_t>
void hogwild_attention_tpl(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const std::vector<at::Tensor>& key_fragments,
        const std::vector<at::Tensor>& value_fragments)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // extract pointers and sizes
    int W = out.size(0);
    int Hq = out.size(1);
    int S = out.size(2);
    int Ev = out.size(3);
    TORCH_CHECK(out.is_contiguous());
    scalar_t* out_ptr = torch_get_pointer<scalar_t>(out);
    // Input:   keys: [Hkv, fragment_lengths[i], E] for i in [F]
    //          values: [Hkv, fragment_lengths[i], Ev] for i in [F]

    int F = locations.size(0);
    TORCH_CHECK_EQ(locations.size(1), W);
    TORCH_CHECK_EQ(locations.size(2), S);
    TORCH_CHECK(locations.is_contiguous());
    const int* loc_ptr = locations.const_data_ptr<int>();

    int E = queries.size(4);
    TORCH_CHECK_EQ(queries.size(0), F);
    TORCH_CHECK_EQ(queries.size(1), W);
    TORCH_CHECK_EQ(queries.size(2), Hq);
    TORCH_CHECK_EQ(queries.size(3), S);
    TORCH_CHECK(queries.is_contiguous());
    const scalar_t* query_ptr = torch_get_pointer<scalar_t>(queries);

    TORCH_CHECK_EQ(fragment_lengths.size(0), F);
    TORCH_CHECK(fragment_lengths.is_contiguous());
    const int* fl_ptr = fragment_lengths.const_data_ptr<int>();

    // check key and value fragments
    TORCH_CHECK_EQ(key_fragments.size(), F);
    TORCH_CHECK_EQ(value_fragments.size(), F);
    // Make exactly one cached memory allocation to store the pointers in
    // NOTE: This is neither thread safe, nor will this memory ever be released again.
    static const scalar_t** frag_ptrs = nullptr;
    if(frag_ptrs == nullptr) {
        C10_CUDA_CHECK(cudaMalloc(&frag_ptrs, sizeof(void *) * 1024));
    }

    std::vector<const scalar_t*> frag_ptrs_host(2*F);
    bool has_batch_dim = key_fragments[0].dim() == 4;
    int fo = has_batch_dim ? 1 : 0;
    int Hkv = key_fragments[0].size(fo);
    for(int f = 0; f < F; ++f) {
        TORCH_CHECK_EQ(key_fragments[f].size(fo + 0), Hkv);
        TORCH_CHECK_EQ(value_fragments[f].size(fo + 0), Hkv);
        int fl = key_fragments[f].size(fo + 1);
        TORCH_CHECK_EQ(value_fragments[f].size(fo + 1), fl);
        TORCH_CHECK_EQ(key_fragments[f].size(fo + 2), E);
        TORCH_CHECK_EQ(value_fragments[f].size(fo + 2), Ev);

        TORCH_CHECK(key_fragments[f].is_contiguous());
        TORCH_CHECK(value_fragments[f].is_contiguous());

        frag_ptrs_host[f] = torch_get_pointer<scalar_t>(key_fragments[f]);
        frag_ptrs_host[F + f] = torch_get_pointer<scalar_t>(value_fragments[f]);
    }

    C10_CUDA_CHECK(cudaMemcpyAsync(frag_ptrs, frag_ptrs_host.data(), 2*sizeof(void*)*F, cudaMemcpyHostToDevice));

    // finally, launch
    Shape shape = {F, W, Hq, Hkv, E, Ev, S};
    C10_CUDA_CHECK(v21::hogwild_attention_gpu(out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
                          frag_ptrs, frag_ptrs + F, shape));
}

template<class scalar_t>
void hogwild_rope_tpl(
        at::Tensor& out, const at::Tensor& queries, const at::Tensor& cosines,
        const at::Tensor& sines)
{
    // extract pointers and sizes
    int F = out.size(0);
    int W = out.size(1);
    int Hq = out.size(2);
    int S = out.size(3);
    int E = out.size(4);
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(queries.is_contiguous());
    TORCH_CHECK(cosines.is_contiguous());
    TORCH_CHECK(sines.is_contiguous());

    TORCH_CHECK_EQ(queries.size(0), W);
    TORCH_CHECK_EQ(queries.size(1), Hq);
    TORCH_CHECK_EQ(queries.size(2), S);
    TORCH_CHECK_EQ(queries.size(3), E);

    TORCH_CHECK_EQ(cosines.size(0), F);
    TORCH_CHECK_EQ(cosines.size(1), W);
    TORCH_CHECK_EQ(cosines.size(2), S);
    TORCH_CHECK_EQ(cosines.size(3), E);

    TORCH_CHECK_EQ(sines.size(0), F);
    TORCH_CHECK_EQ(sines.size(1), W);
    TORCH_CHECK_EQ(sines.size(2), S);
    TORCH_CHECK_EQ(sines.size(3), E);

    rope_gpu(torch_get_pointer<scalar_t>(out), torch_get_pointer<scalar_t>(queries),
             torch_get_pointer<float>(cosines), torch_get_pointer<float>(sines),
                     F, W, Hq, S, E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hogwild_attention(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const std::vector<at::Tensor>& key_fragments,
        const std::vector<at::Tensor>& value_fragments)
{
    if(out.dtype() == at::kHalf) {
        hogwild_attention_tpl<half>(out, scale, locations, queries, fragment_lengths, key_fragments, value_fragments);
    } else if (out.dtype() == at::kFloat) {
        hogwild_attention_tpl<float>(out, scale, locations, queries, fragment_lengths, key_fragments, value_fragments);
    } else if (out.dtype() == at::kBFloat16) {
        hogwild_attention_tpl<nv_bfloat16>(out, scale, locations, queries, fragment_lengths, key_fragments, value_fragments);
    }
}

void hogwild_rope(
        at::Tensor& out, const at::Tensor& queries, const at::Tensor& cosines, const at::Tensor& sines)
{
    if(out.dtype() == at::kHalf) {
        hogwild_rope_tpl<half>(out, queries, cosines, sines);
    } else if (out.dtype() == at::kFloat) {
        hogwild_rope_tpl<float>(out, queries, cosines, sines);
    } else if (out.dtype() == at::kBFloat16) {
        hogwild_rope_tpl<nv_bfloat16>(out, queries, cosines, sines);
    }
}

void hogwild_fused(
        at::Tensor& out, at::Tensor& rotated_queries, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const std::vector<at::Tensor>& key_fragments,
        const std::vector<at::Tensor>& value_fragments,
        const at::Tensor& cosines, const at::Tensor& sines)
{
    std::vector<at::Tensor> key_fragments_contiguous;
    std::vector<at::Tensor> val_fragments_contiguous;
    key_fragments_contiguous.reserve(key_fragments.size());
    val_fragments_contiguous.reserve(key_fragments.size());
    for(int i = 0; i < key_fragments.size(); ++i) {
        key_fragments_contiguous.push_back(key_fragments[i].contiguous());
        val_fragments_contiguous.push_back(value_fragments[i].contiguous());
    }
    hogwild_rope(rotated_queries, queries, cosines, sines);
    hogwild_attention(out, scale, locations, rotated_queries, fragment_lengths, key_fragments_contiguous, val_fragments_contiguous);
}

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit_libhogatt(void) {
    static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "libhogatt", /* name of module */
            NULL,            /* module documentation, may be NULL */
            -1,              /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
            NULL,            /* methods */
    };
    return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY(libhogatt, m) {
    std::vector<at::Tag> tags;
    tags.push_back(at::Tag::needs_fixed_stride_order);
    m.def("hogwild_sdpa(Tensor(a!) output, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor[] key_fragments, Tensor[] value_fragments) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("hogwild_rope(Tensor(a!) output, Tensor queries, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("hogwild_fused(Tensor(a!) output, Tensor(b!) rq, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor[] key_fragments, Tensor[] value_fragments, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
}

TORCH_LIBRARY_IMPL(libhogatt, CUDA, m) {
    m.impl("hogwild_sdpa", hogwild_attention);
    m.impl("hogwild_rope", hogwild_rope);
    m.impl("hogwild_fused", hogwild_fused);
}
