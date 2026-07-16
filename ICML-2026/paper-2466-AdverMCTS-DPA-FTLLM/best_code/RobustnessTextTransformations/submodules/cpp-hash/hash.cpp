#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <openssl/sha.h>
#include <iostream>
#include <string.h>
#include <string>
#include "consts.h"

#define CHECK_SIZE(x, y) TORCH_CHECK(x.size() == y.size(0), #y " must have the same number of batches than there are seeds.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Forward declaration
void hash_tokens_cuda(BYTE* seeds, torch::Tensor output);
void levenshtein_cuda(torch::Tensor scores, torch::Tensor output, float gamma);

torch::Tensor hash_single_token(std::vector<std::string> seeds, int token) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor results = torch::ones(seeds.size(), options); 
    int n = 0;
    for (auto seed : seeds)
    {
        // Setup seed
        unsigned char ibuf[24];
        SHA1((unsigned char *)seed.c_str(), seed.length(), ibuf);

        // Hash
        *(uint32_t *)(ibuf+20) = (uint32_t) token;
        unsigned char obuf[20];
        SHA1(ibuf, 24, obuf);
        uint32_t value = *(uint32_t *)(obuf+16);
        float norm_value = float(value) / MAX_UINT32;
        results.index_put_({n++}, norm_value);
    }

    return results;
}
torch::Tensor hash_tokens(std::vector<std::string> seeds, torch::Tensor target)
{
    CHECK_SIZE(seeds, target);

    if (!target.device().is_cuda())
    {
        // CPU version
        int vocab_size = target.size(1);
        int batch_size = target.size(0);

        int n = 0;
        for (auto seed : seeds)
        {
            // Setup seed
            unsigned char ibuf[24];
            SHA1((unsigned char *)seed.c_str(), seed.length(), ibuf);
            for (int i = 0; i < vocab_size; i++)
            {
                *(uint32_t *)(ibuf+20) = (uint32_t) i;
                unsigned char obuf[20];
                SHA1(ibuf, 24, obuf);
                //for (int k = 0; k < 20; k++) { printf("%02x ", obuf[k]); } printf("\n");
                uint32_t value = *(uint32_t *)(obuf+16);
                float norm_value = float(value) / MAX_UINT32;
                target.index_put_({n,i}, norm_value);
            }
            n++;
        }
    }
    else
    {
        CHECK_CONTIGUOUS(target);

        // Hash seeds
        unsigned char c_seeds[seeds.size()*SHA1_BLOCK_SIZE];
        int i = 0;
        for (auto seed : seeds)
        {
            SHA1((unsigned char *)seed.c_str(), seed.length(), c_seeds + SHA1_BLOCK_SIZE*i++);
        }

        // Hash tokens
        const at::cuda::OptionalCUDAGuard device_guard(device_of(target));
        hash_tokens_cuda(c_seeds, target);
    }
    return target;
}
torch::Tensor levenshtein(torch::Tensor scores, torch::Tensor output, float gamma)
{
    if (!output.device().is_cuda())
    {
        int key_len = scores.size(0);
        int seq_len = output.size(1)-1;
        int i,j;
        for (i=1; i<=seq_len; i++)
        {
            for(j=1; j<=seq_len; j++)
            {
                float cost= scores[(i -1)%key_len][j-1].item<float>();
                float val = output[i-1][j].item<float>() + gamma;
                if (output[i][j-1].item<float>() + gamma < val) val = output[i][j-1].item<float>() + gamma;
                if (output[i-1][j-1].item<float>() + cost < val) val = output[i-1][j-1].item<float>() + cost;
                output.index_put_({i,j}, val);
            }
        }
    }
    else
    {
        CHECK_CONTIGUOUS(output);
        CHECK_CONTIGUOUS(scores);
        const at::cuda::OptionalCUDAGuard device_guard(device_of(output)); 
        levenshtein_cuda(scores, output, gamma);
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_index_hash", &hash_tokens, "Index Range Hash");
    m.def("index_hash", &hash_single_token, "Single Index Hash");
    m.def("levenshtein", &levenshtein, "Levenshtein");
}

