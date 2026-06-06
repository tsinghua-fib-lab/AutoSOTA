#ifndef __CUDACC__
    #define __CUDACC__
    #define __NVCC__
#endif
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
namespace cg = cooperative_groups;

#include <c10/cuda/CUDAException.h>
#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include "binning.h"

 __global__ void duplicate_with_keys_kernel(
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 3,torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
    int TileSizeX,
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
    )
{
    int view_id = blockIdx.y;
    

    if (blockIdx.x * blockDim.x + threadIdx.x < prefix_sum.size(1))
    {
        int point_id = depth_sorted_pointid[view_id][blockIdx.x * blockDim.x + threadIdx.x];
        int end = prefix_sum[view_id][blockIdx.x * blockDim.x + threadIdx.x];

        //int end = prefix_sum[view_id][point_id+1];
        int l = LU[view_id][0][point_id];
        int u = LU[view_id][1][point_id];
        int r = RD[view_id][0][point_id];
        int d = RD[view_id][1][point_id];
        int count = 0;
        if ((r - l) * (d - u) < 32)
        {
            for (int i = u; i < d; i++)
            {
                for (int j = l; j < r; j++)
                {
                    int tile_id = i * TileSizeX + j;
                    table_tileId[view_id][end - 1 - count] = tile_id + 1;// tile_id 0 means invalid!
                    table_pointId[view_id][end - 1 - count] = point_id;
                    count++;
                }
            }
        }
    }
}

 __global__ void large_points_duplicate_with_keys_kernel(
     const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> LU,//viewnum,2,pointnum
     const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> RD,//viewnum,2,pointnum
     const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> prefix_sum,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> depth_sorted_pointid,//viewnum,pointnum
     const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> large_index,//tasknum,2
     int TileSizeX,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_tileId,
     torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> table_pointId
 )
 {
     auto block = cg::this_thread_block();
     auto warp = cg::tiled_partition<32>(block);
     int task_index = warp.meta_group_size() * block.group_index().x + warp.meta_group_rank();

     if (task_index < large_index.size(0))
     {
         int view_id = large_index[task_index][0];
         int point_index = large_index[task_index][1];
         int point_id = depth_sorted_pointid[view_id][point_index];
         int end = prefix_sum[view_id][point_index];

         //int end = prefix_sum[view_id][point_id+1];
         int l = LU[view_id][0][point_id];
         int u = LU[view_id][1][point_id];
         int width = RD[view_id][0][point_id]-l;
         int height = RD[view_id][1][point_id]-u;
         for (int i = warp.thread_rank(); i < width * height; i+=warp.num_threads())
         {
             int col = l + (i % width);
             int row = u + (i / width);
             int tile_id = row * TileSizeX + col;
             table_tileId[view_id][end - 1 - i] = tile_id + 1;// tile_id 0 means invalid!
             table_pointId[view_id][end - 1 - i] = point_id;
         }
     }
 }

std::vector<at::Tensor> duplicateWithKeys(at::Tensor LU, at::Tensor RD, at::Tensor prefix_sum, at::Tensor depth_sorted_pointid,
    at::Tensor large_index,int64_t allocate_size, int64_t TilesSizeX)
{
    at::DeviceGuard guard(LU.device());
    int64_t view_num = LU.sizes()[0];
    int64_t points_num = LU.sizes()[2];

    std::vector<int64_t> output_shape{ view_num, allocate_size };

    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_tileId = torch::zeros(output_shape, opt);
    opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(LU.device()).requires_grad(false);
    auto table_pointId= torch::zeros(output_shape, opt);

    dim3 Block3d(std::ceil(points_num/1024.0f), view_num, 1);
    

    duplicate_with_keys_kernel<<<Block3d ,1024>>>(
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        TilesSizeX,
        table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    
    int large_points_num = large_index.size(0);
    if (large_points_num > 0) {
    int blocksnum = std::ceil((large_points_num * 32) / 1024.0f);
    large_points_duplicate_with_keys_kernel << <blocksnum, 1024 >> > (
        LU.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        RD.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        prefix_sum.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        depth_sorted_pointid.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        large_index.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        TilesSizeX,
        table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        table_pointId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    }

    return { table_tileId ,table_pointId };
    
}

__global__ void tile_range_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2,torch::RestrictPtrTraits> table_tileId,//viewnum,pointnum
    int table_length,
    int max_tileId,
    torch::PackedTensorAccessor32 < int32_t, 2, torch::RestrictPtrTraits> tile_range
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // NOTE ABOUT EMPTY TILE GAPS:
    // This kernel does not produce a fully dense monotonic range table for every tile ID.
    // It only writes boundary markers at transitions between occupied tiles, so empty tiles
    // in gaps between occupied tiles can have malformed (start, end) pairs.
    //
    // At a transition from occupied tile A to occupied tile B (B > A+1), only
    // tile_range[A+1] is written into the gap; the rest remain -1. This leaves:
    //   - tile A+1:       start = valid, end = -1    -> malformed
    //   - tiles A+2..B-2: start = -1,   end = -1    -> both -1
    //   - tile B-1:       start = -1,   end = valid  -> malformed
    //
    // Concrete example:
    // - table_tileId = [2, 2, 6, 6]
    // - total tile count = 9 (valid tile IDs: 1..9, tile 0 is padding/invalid)
    // - resulting tile_range = [-1, -1, 0, 2, -1, -1, 2, 4, -1, -1, 4]
    // Per-tile interpretation:
    // - tile 1: start = -1, end = 0   -> invalid
    // - tile 2: start = 0,  end = 2   -> valid
    // - tile 3: start = 2,  end = -1  -> invalid
    // - tile 4: start = -1, end = -1  -> invalid
    // - tile 5: start = -1, end = 2   -> invalid
    // - tile 6: start = 2,  end = 4   -> valid
    // - tile 7: start = 4,  end = -1  -> invalid
    // - tile 8: start = -1, end = -1  -> invalid
    // - tile 9: start = -1, end = 4   -> invalid
    //
    // The main raster path in raster.cu is unaffected: the inner loop does not execute
    // for malformed pairs. Any other consumer must guard with (end >= start >= 0);
    // tiles passing this check are either occupied (end > start) or single-slot gap
    // tiles where start == end (count = 0).


    // head
    if (index == 0)
    {
        int tile_id=table_tileId[view_id][index];
        tile_range[view_id][tile_id] = index;
    }
    
    //tail
    if (index == table_length - 1)
    {
        tile_range[view_id][max_tileId + 1] = table_length;
        // The fix at tail. See: tile_range_debug.py for details
        int cur_tile = table_tileId[view_id][index];
        tile_range[view_id][cur_tile + 1] = table_length;
    }
    
    if (index < table_length-1)
    {
        int cur_tile = table_tileId[view_id][index];
        int next_tile= table_tileId[view_id][index+1];
        if (cur_tile!=next_tile)
        {
            if (cur_tile + 1 < next_tile)
            {
                tile_range[view_id][cur_tile + 1] = index + 1;
            }
            tile_range[view_id][next_tile] = index + 1;
        }
    }
}

at::Tensor tileRange(at::Tensor table_tileId, int64_t table_length, int64_t max_tileId)
{
    at::DeviceGuard guard(table_tileId.device());

    int64_t view_num = table_tileId.sizes()[0];
    std::vector<int64_t> output_shape{ view_num,max_tileId + 1 + 1 };//+1 for tail
    //printf("\ntensor shape in tileRange:%ld,%ld\n", view_num, max_tileId+1-1);
    auto opt = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(table_tileId.device()).requires_grad(false);
    auto out = torch::ones(output_shape, opt)*-1;

    dim3 Block3d(std::ceil(table_length / 1024.0f), view_num, 1);

    tile_range_kernel<<<Block3d, 1024 >>>
        (table_tileId.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), table_length, max_tileId, out.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;

    return out;
}

__global__ void create_ROI_AABB_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_ndc,        //viewnum,4,pointnum
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> tensor_eigen_val,  //viewnum,2,pointnum
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> tensor_eigen_vec,  //viewnum,2,2,pointnum
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> tensor_opacity,  //viewnum,pointnum
    int img_h,int img_w,int img_tile_h,int img_tile_w,int tilesize,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_left_up,
    torch::PackedTensorAccessor32 < int32_t, 3, torch::RestrictPtrTraits> tensor_right_down
)
{
    int view_id = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < tensor_ndc.size(2))
    {
        float4 ndc{ tensor_ndc[view_id][0][index],tensor_ndc[view_id][1][index],
            tensor_ndc[view_id][2][index] ,tensor_ndc[view_id][3][index] };
        bool bVisible = !((ndc.x < -1.3f) || (ndc.x > 1.3f) || (ndc.y < -1.3f) || (ndc.y > 1.3f) || (ndc.z > 1.0f) || (ndc.z < 0.0f));
        if (bVisible)
        {
            float opacity = max(tensor_opacity[view_id][index], 1.0f / 255);
            float coefficient = 2 * log(255 * opacity);
            float axis_length[2]{ 0,0 };
            axis_length[0] = sqrt(coefficient * tensor_eigen_val[view_id][0][index]);
            axis_length[1] = sqrt(coefficient * tensor_eigen_val[view_id][1][index]);
            float2 axis_dir[2];
            axis_dir[0].x = tensor_eigen_vec[view_id][0][0][index];
            axis_dir[0].y = tensor_eigen_vec[view_id][0][1][index];
            axis_dir[1].x = tensor_eigen_vec[view_id][1][0][index];
            axis_dir[1].y = tensor_eigen_vec[view_id][1][1][index];
            float2 axis[2];
            axis[0].x = axis_dir[0].x * axis_length[0];
            axis[0].y = axis_dir[0].y * axis_length[0];
            axis[1].x = axis_dir[1].x * axis_length[1];
            axis[1].y = axis_dir[1].y * axis_length[1];

            float2 screen_uv{ ndc.x * 0.5f + 0.5f,ndc.y * 0.5f + 0.5f };
            float2 coord{ screen_uv.x * img_w - 0.5f,screen_uv.y * img_h - 0.5f };
            float min_x = coord.x - abs(axis[0].x) - abs(axis[1].x);
            float max_x = coord.x + abs(axis[0].x) + abs(axis[1].x);
            float min_y = coord.y - abs(axis[0].y) - abs(axis[1].y);
            float max_y = coord.y + abs(axis[0].y) + abs(axis[1].y);
            int2 left_up{ min_x / tilesize,min_y / tilesize };
            int2 right_down{ ceil(max_x / tilesize),ceil(max_y / tilesize) };
            tensor_left_up[view_id][0][index] = min(max(left_up.x,0), img_tile_w);
            tensor_left_up[view_id][1][index] = min(max(left_up.y,0),img_tile_h);
            tensor_right_down[view_id][0][index] = min(max(right_down.x,0), img_tile_w);
            tensor_right_down[view_id][1][index] = min(max(right_down.y,0), img_tile_h);
        }
        else
        {
            tensor_left_up[view_id][0][index] = 0;
            tensor_left_up[view_id][1][index] = 0;
            tensor_right_down[view_id][0][index] = 0;
            tensor_right_down[view_id][1][index] = 0;
        }
    }
}

std::vector<at::Tensor> create_ROI_AABB(at::Tensor ndc, at::Tensor eigen_val, at::Tensor eigen_vec, at::Tensor opacity,
    int64_t height,int64_t width, int64_t tilesize)
{
    at::DeviceGuard guard(ndc.device());

    int views_num = ndc.size(0);
    int points_num = ndc.size(2);
    at::Tensor left_up = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));
    at::Tensor right_down = torch::empty({ views_num,2,points_num }, ndc.options().dtype(torch::kInt32));

    dim3 Block3d(std::ceil(points_num / 256.0f), views_num, 1);
    create_ROI_AABB_kernel<<<Block3d,256>>>(ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        eigen_val.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        eigen_vec.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        height, width,ceil(height/(float)tilesize), ceil(width / (float)tilesize), tilesize,
        left_up.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        right_down.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>()
        );
    CUDA_CHECK_ERRORS;
    return { left_up ,right_down };
}
