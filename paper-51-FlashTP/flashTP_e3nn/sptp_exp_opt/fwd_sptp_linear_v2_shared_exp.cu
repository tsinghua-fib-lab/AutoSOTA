#include "sptp_exp_opt.hpp"
#include <cmath>

#define WARPSIZE 32 
#define MAX_IN1_IR_CNT 512 // not just len(i_in1) but if u is larger than 32 need multiple IR_CNT 
#define MAX_NUM_PATH 512 // also need to account for u > 32
#define MAX_U_FIBER_CNT 5265 // up to L_max =5
#define MAX_U_CG_VAL_CNT 256 // up to L_max =5

__constant__ int in1_idxing[MAX_IN1_IR_CNT];
__constant__ int in1_ival[MAX_IN1_IR_CNT];
__constant__ int in1_related_path_idx[MAX_IN1_IR_CNT];

__constant__ ushort2 path_array1[MAX_NUM_PATH];
__constant__ uchar4 path_array2[MAX_NUM_PATH];
__constant__ ushort2 per_path_fiber_start[MAX_NUM_PATH];
__constant__ float path_weight[MAX_NUM_PATH];
__constant__ int per_path_weight_pos[MAX_NUM_PATH];

// __constant__ uchar4 fiber_array[MAX_U_FIBER_CNT];
__constant__ float unique_cg_val[MAX_U_CG_VAL_CNT];

template <typename scalar_t>
__global__ void fwd_sptp_lienar_kernel_v2_shared_exp(
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    const float* __restrict__ weight,
    float* __restrict__ out,

    const int* __restrict__ per_edge_src,
    const int* __restrict__ per_edge_dst,
    const u_char* __restrict__ fiber_data,
    // const float* __restrict__ unique_cg_val_data,

    const size_t batch_size,
    const size_t out_size,
    const size_t weight_size,
    const size_t in1_size,
    const size_t in2_size,
    const size_t perwarp_in2_size,
    const size_t path_cnt,
    const size_t max_ir_dim,
    const size_t fiber_array_size
    // const size_t unique_cg_val_size
    )
    {
    extern __shared__ scalar_t shmem[];
    // Input dL_dO => batch, ir, mul order
    // 2D grid, 2D block
    // grid (path, batch), block (mul(same path), batch)
    // intra-warp (u parallel) x , inter-warp (batch) y 
    const int target_in1 = blockIdx.y;
    const int global_t_batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int shmem_warp_pos_start = threadIdx.y*blockDim.x;

    u_char* fiber_array_uchar = (u_char*) (shmem + blockDim.y * (blockDim.x * (max_ir_dim*3)) + blockDim.y * perwarp_in2_size);
    const int thread_idx = threadIdx.y* blockDim.x + threadIdx.x;
    for(int fiber_idx = thread_idx; fiber_idx < fiber_array_size; fiber_idx+=blockDim.x*blockDim.y) {
        fiber_array_uchar[fiber_idx] = fiber_data[fiber_idx];
    }
    uchar4* fiber_array = (uchar4*) fiber_array_uchar;
    __syncthreads();


    if(global_t_batch_idx >= batch_size) return;

    const int src_idx = per_edge_src[global_t_batch_idx];
    const int dst_idx = per_edge_dst[global_t_batch_idx];

    scalar_t* my_batch_shmem_start = shmem + threadIdx.y * (blockDim.x * (max_ir_dim*3) );

    scalar_t* my_shmem_in1 = my_batch_shmem_start + threadIdx.x*max_ir_dim; 
    scalar_t* my_shmem_uvuv = my_batch_shmem_start +  blockDim.x*(max_ir_dim) + threadIdx.x*max_ir_dim;
    scalar_t* shmem_scratch = my_batch_shmem_start + blockDim.x*(max_ir_dim*2);
    
    scalar_t* shmem_in2 = shmem + blockDim.y * (blockDim.x * (max_ir_dim*3)) + threadIdx.y * perwarp_in2_size;
    
    const int in1_start = in1_idxing[target_in1];
    const int in1_end = in1_idxing[target_in1+1];
    const int i_val = in1_ival[target_in1];
    const int path_idx_start = in1_related_path_idx[target_in1];
    const int path_idx_end = in1_related_path_idx[target_in1+1];

    // using reg_dL_din1 for dummy => need to initialize ...
    unsigned long long in1_idx = src_idx*in1_size + in1_start+threadIdx.x;
    for(int shmem_idx = threadIdx.x; in1_idx < src_idx*in1_size + in1_end; shmem_idx+=WARPSIZE, in1_idx+=WARPSIZE) {
        shmem_scratch[shmem_idx] = in1[in1_idx];
    }
    __syncwarp();
    for(int i =0, shmem_idx = threadIdx.x*i_val; i<i_val; i++, shmem_idx++){
        my_shmem_in1[i] = shmem_scratch[shmem_idx];
    }

    // load all in2 from main mem
    for(int shmem_idx = threadIdx.x, in2_idx = global_t_batch_idx*in2_size + threadIdx.x; shmem_idx < in2_size; in2_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
        shmem_in2[shmem_idx] = in2[in2_idx];
    }
    __syncwarp();

    // for(int fiber_idx = thread_idx; fiber_idx < unique_cg_val_size; fiber_idx+=blockDim.x*blockDim.y) {
    //     unique_cg_val[fiber_idx] = unique_cg_val_data[fiber_idx];
    // }

    // for path_chunk
    // path idx == k idx
    // path index == 
    const unsigned long long g_t_dldo_start = dst_idx*out_size;
    const unsigned long long g_t_w_start = global_t_batch_idx*weight_size;

    for(int path_idx=path_idx_start; path_idx < path_idx_end; path_idx++){
        const ushort2 path_info1 = path_array1[path_idx]; // k_start, k_end
        const uchar4 path_info2 = path_array2[path_idx]; // k_val, j_start, j_val, j_end

        for(int i=0; i<max_ir_dim;i++){
            my_shmem_uvuv[i] = 0.0;
        }
        
        // stall due to global memory access (better if it is load to shared memory and accessed)
        // possible optimization point with gather scatter      
        // odd number of k_val (2n+1) no bank conflict
        
        // Loading Weight from global memory is a major memory bottleneck
        const int weight_pos = g_t_w_start + per_path_weight_pos[path_idx]+threadIdx.x;
        float reg_w_path_norm = weight[weight_pos] * path_weight[path_idx];
        
        const ushort2 fiber_idx_info = per_path_fiber_start[path_idx];
        // for nnz in the fiber
        // uchar4 fiber;
        for(ushort fiber_idx = fiber_idx_info.x; fiber_idx < fiber_idx_info.y; fiber_idx++){
            // mult k with all w => dL_duvuv
            uchar4 fiber = fiber_array[fiber_idx]; // i, j, k, cg idx
            my_shmem_uvuv[fiber.z] += my_shmem_in1[fiber.x] * shmem_in2[path_info2.y+fiber.y] * unique_cg_val[fiber.w];
        }
        // store out first in shared mem
        for(int i =0, shmem_idx = threadIdx.x*path_info2.x; i<path_info2.x; i++, shmem_idx++){
            shmem_scratch[shmem_idx] = my_shmem_uvuv[i] * reg_w_path_norm;
        }
        __syncwarp();

        // atomicadd
        unsigned long long out_idx = g_t_dldo_start+path_info1.x + threadIdx.x; 
        for(int shmem_idx = threadIdx.x; out_idx < g_t_dldo_start+ path_info1.y; out_idx+=WARPSIZE, shmem_idx+=WARPSIZE) {
            atomicAdd(out+out_idx, shmem_scratch[shmem_idx]);
        }
    }
}

void fwd_sptp_linear_cuda_v2_shared_exp(
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor out,
    
    torch::Tensor per_edge_src,
    torch::Tensor per_edge_dst,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
    ){

    // TODO: not transposed (z, mul, ir)
    const auto batch_size = in2.size(0);
    const auto in1_size = in1.size(1);
    const auto in2_size = in2.size(1);
    const auto out_size = out.size(1);
    const auto weight_size = weight.size(1);
    const auto batch_block = (int) std::ceil((float)batch_size/(float)per_block_batch);
    dim3 grid(batch_block, path_cnt);
    dim3 block(WARPSIZE, per_block_batch);
    int perwarp_in2_size = in2_size;
    if (in2_size%2 ==0) perwarp_in2_size = in2_size+1;

    // setup constant memory 
    cudaMemcpyToSymbol(in1_idxing, t_in1_idxing.data<int>(), at::numel(t_in1_idxing)*sizeof(int)); // int , MAX_IN1_IR_CNT
    cudaMemcpyToSymbol(in1_ival, t_in1_ival.data<int>(),  at::numel(t_in1_ival)*sizeof(int)); // int , MAX_IN1_IR_CNT
    cudaMemcpyToSymbol(in1_related_path_idx, t_in1_related_path_idx.data<int>(), at::numel(t_in1_related_path_idx)*sizeof(int)); // int  , MAX_IN1_IR_CNT
    
    cudaMemcpyToSymbol(path_array1, t_path_array1.data<u_short>(), at::numel(t_path_array1)*sizeof(u_short) ); // ushort2, MAX_NUM_PATH
    cudaMemcpyToSymbol(path_array2, t_path_array2.data<u_char>(), at::numel(t_path_array2)*sizeof(u_char)); // uchar4, MAX_NUM_PAT
    cudaMemcpyToSymbol(per_path_fiber_start, t_per_path_fiber_start.data<u_short>(), at::numel(t_per_path_fiber_start)*sizeof(u_short)); // ushort2, MAX_NUM_PATH
    cudaMemcpyToSymbol(path_weight, t_path_weight.data<float>(), at::numel(t_path_weight)*sizeof(float)); // float, MAX_NUM_PATH
    cudaMemcpyToSymbol(per_path_weight_pos, t_per_path_weight_pos.data<int>(), at::numel(t_per_path_weight_pos)*sizeof(int)); // int , MAX_NUM_PATH

    // cudaMemcpyToSymbol(fiber_array, t_fiber_array.data<u_char>(), at::numel(t_fiber_array)*sizeof(u_char)); // u_char4), MAX_U_FIBER_CNT 
    cudaMemcpyToSymbol(unique_cg_val, t_unique_cg_val.data<float>(), at::numel(t_unique_cg_val)*sizeof(float) ); // float , MAX_U_CG_VAL_CNT

    const int shared_memory_bytes = sizeof(float) * per_block_batch * (WARPSIZE * (max_ir_dim*3) + perwarp_in2_size) + at::numel(t_fiber_array); // + at::numel(t_unique_cg_val)*sizeof(float);

    // int carveout = 100;
    // CHECK_CUDA_ERROR(cudaFuncSetAttribute(
    //     sptp_all_forward_kernel_v1<float>,
    //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        fwd_sptp_lienar_kernel_v2_shared_exp<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));

    fwd_sptp_lienar_kernel_v2_shared_exp<float><<<grid, block, shared_memory_bytes>>>(
        in1.data<float>(),
        in2.data<float>(),
        weight.data<float>(),
        out.data<float>(),

        per_edge_src.data<int>(),
        per_edge_dst.data<int>(),
        t_fiber_array.data<u_char>(),
        // t_unique_cg_val.data<float>(),

        batch_size,
        out_size,
        weight_size,
        in1_size,
        in2_size,
        perwarp_in2_size,
        path_cnt,
        max_ir_dim,
        at::numel(t_fiber_array)
        // at::numel(t_unique_cg_val)
        );
}
    