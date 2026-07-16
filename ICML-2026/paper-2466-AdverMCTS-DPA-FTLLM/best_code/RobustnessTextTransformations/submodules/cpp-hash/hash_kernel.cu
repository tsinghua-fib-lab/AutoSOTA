#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include "consts.h"

/*
 * sha1.cu CUDA Implementation of SHA1 Hashing       
 * SOURCE: https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha1.cu
 *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

 
/**************************** DATA TYPES ****************************/
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[5];
	WORD k[4];
} CUDA_SHA1_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define SWAP32(x) (((x>>24)&0xff) | ((x<<8)&0xff0000) | ((x>>8)&0xff00) | ((x<<24)&0xff000000))

/*********************** FUNCTION DEFINITIONS ***********************/
__device__  __forceinline__ void cuda_sha1_transform(CUDA_SHA1_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, i, j, t, m[80];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) + (data[j + 1] << 16) + (data[j + 2] << 8) + (data[j + 3]);
	for ( ; i < 80; ++i) {
		m[i] = (m[i - 3] ^ m[i - 8] ^ m[i - 14] ^ m[i - 16]);
		m[i] = (m[i] << 1) | (m[i] >> 31);
	}

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];

	for (i = 0; i < 20; ++i) {
		t = ROTLEFT(a, 5) + ((b & c) ^ (~b & d)) + e + ctx->k[0] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}
	for ( ; i < 40; ++i) {
		t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[1] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}
	for ( ; i < 60; ++i) {
		t = ROTLEFT(a, 5) + ((b & c) ^ (b & d) ^ (c & d))  + e + ctx->k[2] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}
	for ( ; i < 80; ++i) {
		t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[3] + m[i];
		e = d;
		d = c;
		c = ROTLEFT(b, 30);
		b = a;
		a = t;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
}

__device__ void cuda_sha1_init(CUDA_SHA1_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x67452301;
	ctx->state[1] = 0xEFCDAB89;
	ctx->state[2] = 0x98BADCFE;
	ctx->state[3] = 0x10325476;
	ctx->state[4] = 0xc3d2e1f0;
	ctx->k[0] = 0x5a827999;
	ctx->k[1] = 0x6ed9eba1;
	ctx->k[2] = 0x8f1bbcdc;
	ctx->k[3] = 0xca62c1d6;
}

__device__ void cuda_sha1_update(CUDA_SHA1_CTX *ctx, const BYTE data[], size_t len)
{
	size_t i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			cuda_sha1_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void cuda_sha1_final(CUDA_SHA1_CTX *ctx, float* hash)
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		cuda_sha1_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	cuda_sha1_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and MD uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
    *hash = float(SWAP32(ctx->state[4])) / MAX_UINT32;
}

__global__ void kernel_sha1_hash(BYTE* seeds, torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> output, WORD n_tokens)
{
    // Index setup
    WORD batch_idx = blockIdx.y;
	WORD token_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (token_idx >= n_tokens) return;
    //printf("Working on %d : %d\n", batch_idx, token_idx);

    // Hash seed
	BYTE* seed = seeds + batch_idx * SHA1_BLOCK_SIZE;
	CUDA_SHA1_CTX ctx;
	cuda_sha1_init(&ctx);
    //printf("Init %d\n", token_idx);
	cuda_sha1_update(&ctx, seed, SHA1_BLOCK_SIZE);
    //printf("Update seed %d\n", token_idx);

    // Add index
    cuda_sha1_update(&ctx, (BYTE*) &token_idx, 4); 
    //printf("Update idx %d\n", token_idx);

    // Compute
    float val = 0;
	cuda_sha1_final(&ctx, &val);
    //printf("Finished hash %d : %f\n", token_idx, val);
    output[batch_idx][token_idx] = val;
    //printf("Final %d\n", token_idx);
}

void hash_tokens_cuda(BYTE* seeds, torch::Tensor output)
{
    const WORD batch_size = output.size(0);
    const WORD n_tokens = output.size(1);
	BYTE *cuda_seeds;
	cudaMalloc(&cuda_seeds, SHA1_BLOCK_SIZE * batch_size);
	cudaMemcpy(cuda_seeds, seeds, SHA1_BLOCK_SIZE * batch_size, cudaMemcpyHostToDevice);

	WORD threads = 1024;
	dim3 blocks((n_tokens + threads - 1) / threads, batch_size);
 
    //printf("About to start\n");
	kernel_sha1_hash << < blocks, threads >> > (cuda_seeds, output.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(), n_tokens);

	// cudaDeviceSynchronize();
	// cudaError_t error = cudaGetLastError();
	// if (error != cudaSuccess) {
	// 	printf("Error cuda sha1 hash: %s \n", cudaGetErrorString(error));
	// }

	cudaFree(cuda_seeds);
}

__global__ void kernel_levenshtein(torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> scores, torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> output, WORD key_len, WORD seq_len, WORD output_len, float gamma)
{
    // Index setup
	WORD offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset >= output_len) return;
    //printf("Working on %d : %d\n", batch_idx, token_idx);

    WORD i,j;
    for (i=1; i<= seq_len; i++) 
    {
        for (j=1; j<=seq_len; j++)
        {
            float cost = scores[(offset + i -1)%key_len][j-1];
            output[offset][i][j] = output[offset][i-1][j] + gamma;
            if (output[offset][i][j-1] + gamma < output[offset][i][j]) output[offset][i][j] = output[offset][i][j-1] + gamma;
            if (output[offset][i-1][j-1] + cost < output[offset][i][j]) output[offset][i][j] = output[offset][i-1][j-1] + cost;
        }
    }
}

void levenshtein_cuda(torch::Tensor scores, torch::Tensor output, float gamma)
{
    const WORD key_len = scores.size(0);
    const WORD output_len = output.size(0);
    const WORD seq_len = output.size(1)-1;
    //BYTE *cuda_seeds;
	//cudaMalloc(&cuda_seeds, SHA1_BLOCK_SIZE * batch_size);
	//cudaMemcpy(cuda_seeds, seeds, SHA1_BLOCK_SIZE * batch_size, cudaMemcpyHostToDevice);

	WORD threads = 1024;
	WORD block = (output_len + threads - 1) / threads;
 
    //printf("About to start\n");
	kernel_levenshtein << < block, threads >> > (scores.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(), output.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(), key_len, seq_len, output_len, gamma);

	// cudaDeviceSynchronize();
	// cudaError_t error = cudaGetLastError();
	// if (error != cudaSuccess) {
	// 	printf("Error cuda sha1 hash: %s \n", cudaGetErrorString(error));
	// }
}

