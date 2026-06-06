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
#include "transform.h"


template <typename scalar_t,bool TRNASPOSE=true>
__global__ void jacobian_rayspace_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> translated_position,    //[batch,4,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> proj_matrix,    //[batch,2] 
    const int output_h,const int output_w,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian         //[batch,3,3,point_num]
    )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;
    if (batch_id < translated_position.size(0) && index < translated_position.size(2))
    {
        float focalx = proj_matrix[batch_id][0][0]*output_w*0.5;
        float focaly = proj_matrix[batch_id][1][1]*output_h*0.5;

        float reciprocal_tz = 1.0f/max(translated_position[batch_id][2][index],1e-2f);//near plane 0.01
        float square_reciprocal_tz = reciprocal_tz * reciprocal_tz;

        jacobian[batch_id][0][0][index] = focalx * reciprocal_tz;
        jacobian[batch_id][1][1][index] = focaly * reciprocal_tz;
        if (TRNASPOSE)
        {
            jacobian[batch_id][0][2][index] = -focalx * translated_position[batch_id][0][index] * square_reciprocal_tz;
            jacobian[batch_id][1][2][index] = -focaly * translated_position[batch_id][1][index] * square_reciprocal_tz;
        }
        else
        {
            jacobian[batch_id][2][0][index] = -focalx * translated_position[batch_id][0][index] * square_reciprocal_tz;
            jacobian[batch_id][2][1][index] = -focaly * translated_position[batch_id][1][index] * square_reciprocal_tz;
        }
    }
}

at::Tensor jacobianRayspace(
    at::Tensor translated_position, //N,4,P
    at::Tensor proj_matrix, //N,2
    int64_t output_h,int64_t output_w,
    bool bTranspose
)
{
    int N = translated_position.size(0);
    int P = translated_position.size(2);
    at::Tensor jacobian_matrix = torch::zeros({N,3,3,P}, translated_position.options());

    int threadsnum = 256;
    dim3 Block3d(std::ceil(P/(float)threadsnum), N, 1);
    if (bTranspose)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(translated_position.scalar_type(), __FUNCTION__, [&] {jacobian_rayspace_kernel<scalar_t,true > << <Block3d, threadsnum >> > (
            translated_position.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            proj_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            output_h,output_w,
            jacobian_matrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });
    }
    else
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(translated_position.scalar_type(), __FUNCTION__, [&] {jacobian_rayspace_kernel<scalar_t, false > << <Block3d, threadsnum >> > (
            translated_position.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            proj_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            output_h,output_w,
            jacobian_matrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });
    }

    CUDA_CHECK_ERRORS;
    return jacobian_matrix;

}

__global__ void create_transform_matrix_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> quaternion,    //[3,point_num]  
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> scale,    //[4,point_num] 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> transform         //[3,3,point_num]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index < quaternion.size(1))
    {
        float r = quaternion[0][index];
        float x = quaternion[1][index];
        float y = quaternion[2][index];
        float z = quaternion[3][index];

        float scale_x = scale[0][index];
        float scale_y = scale[1][index];
        float scale_z = scale[2][index];

        transform[0][0][index] = (1 - 2 * (y * y + z * z))*scale_x;
        transform[0][1][index] = 2 * (x * y + r * z) * scale_x;
        transform[0][2][index] = 2 * (x * z - r * y) * scale_x;

        transform[1][0][index] = 2 * (x * y - r * z) * scale_y;
        transform[1][1][index] = (1 - 2 * (x * x + z * z)) * scale_y;
        transform[1][2][index] = 2 * (y * z + r * x) * scale_y;

        transform[2][0][index] = 2 * (x * z + r * y) * scale_z;
        transform[2][1][index] = 2 * (y * z - r * x) * scale_z;
        transform[2][2][index] = (1 - 2 * (x * x + y * y)) * scale_z;
    }
}

at::Tensor createTransformMatrix_forward(at::Tensor quaternion, at::Tensor scale)
{
    int P = quaternion.size(1);
    at::Tensor transform_matrix = torch::empty({ 3,3,P }, scale.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);
    create_transform_matrix_forward_kernel << <blocknum, threadsnum >> > (
        quaternion.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
    return transform_matrix;
}

__global__ void create_transform_matrix_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> quaternion,    //[3,point_num]  
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> scale,    //[4,point_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_transform,         //[3,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_quaternion,    //[4,point_num]  
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_scale    //[3,point_num] 

)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index < quaternion.size(1))
    {
        float r = quaternion[0][index];
        float x = quaternion[1][index];
        float y = quaternion[2][index];
        float z = quaternion[3][index];

        float dt[9];
        dt[0 * 3 + 0] = grad_transform[0][0][index];
        dt[0 * 3 + 1] = grad_transform[0][1][index];
        dt[0 * 3 + 2] = grad_transform[0][2][index];

        dt[1 * 3 + 0] = grad_transform[1][0][index];
        dt[1 * 3 + 1] = grad_transform[1][1][index];
        dt[1 * 3 + 2] = grad_transform[1][2][index];

        dt[2 * 3 + 0] = grad_transform[2][0][index];
        dt[2 * 3 + 1] = grad_transform[2][1][index];
        dt[2 * 3 + 2] = grad_transform[2][2][index];

        {
            float grad_scale_x = 0;
            grad_scale_x += (1 - 2 * (y * y + z * z)) * dt[0 * 3 + 0];
            grad_scale_x += 2 * (x * y + r * z) * dt[0 * 3 + 1];
            grad_scale_x += 2 * (x * z - r * y) * dt[0 * 3 + 2];
            grad_scale[0][index] = grad_scale_x;
        }

        {
            float grad_scale_y = 0;
            grad_scale_y += 2 * (x * y - r * z) * dt[1 * 3 + 0];
            grad_scale_y += (1 - 2 * (x * x + z * z)) * dt[1 * 3 + 1];
            grad_scale_y += 2 * (y * z + r * x) * dt[1 * 3 + 2];
            grad_scale[1][index] = grad_scale_y;
        }

        {
            float grad_scale_z = 0;
            grad_scale_z += 2 * (x * z + r * y) * dt[2 * 3 + 0];
            grad_scale_z += 2 * (y * z - r * x) * dt[2 * 3 + 1];
            grad_scale_z += (1 - 2 * (x * x + y * y)) * dt[2 * 3 + 2];
            grad_scale[2][index] = grad_scale_z;
        }

        {
            dt[0 * 3 + 0] *= scale[0][index];
            dt[0 * 3 + 1] *= scale[0][index];
            dt[0 * 3 + 2] *= scale[0][index];

            dt[1 * 3 + 0] *= scale[1][index];
            dt[1 * 3 + 1] *= scale[1][index];
            dt[1 * 3 + 2] *= scale[1][index];

            dt[2 * 3 + 0] *= scale[2][index];
            dt[2 * 3 + 1] *= scale[2][index];
            dt[2 * 3 + 2] *= scale[2][index];

            grad_quaternion[0][index] = 2 * z * (dt[0*3+1] - dt[1*3+0]) + 2 * y * (dt[2*3+0] - dt[0*3+2]) + 2 * x * (dt[1*3+2] - dt[2*3+1]);
            grad_quaternion[1][index] = 2 * y * (dt[1*3+0] + dt[0*3+1]) + 2 * z * (dt[2*3+0] + dt[0*3+2]) + 2 * r * (dt[1*3+2] - dt[2*3+1]) - 4 * x * (dt[2*3+2] + dt[1*3+1]);
            grad_quaternion[2][index] = 2 * x * (dt[1*3+0] + dt[0*3+1]) + 2 * r * (dt[2*3+0] - dt[0*3+2]) + 2 * z * (dt[1*3+2] + dt[2*3+1]) - 4 * y * (dt[2*3+2] + dt[0*3+0]);
            grad_quaternion[3][index] = 2 * r * (dt[0*3+1] - dt[1*3+0]) + 2 * x * (dt[2*3+0] + dt[0*3+2]) + 2 * y * (dt[1*3+2] + dt[2*3+1]) - 4 * z * (dt[1*3+1] + dt[0*3+0]);
        }




    }
}


std::vector<at::Tensor> createTransformMatrix_backward(at::Tensor transform_matrix_grad, at::Tensor quaternion, at::Tensor scale)
{
    //todo
    int P = quaternion.size(1);
    at::Tensor grad_quaternion = torch::empty({ 4,P }, transform_matrix_grad.options());
    at::Tensor grad_scale = torch::empty({ 3,P }, transform_matrix_grad.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);
    create_transform_matrix_backward_kernel << <blocknum, threadsnum >> > (
        quaternion.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        transform_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_quaternion.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_scale.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;


    return { grad_quaternion,grad_scale };
}


__global__ void world2ndc_backword_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_project_matrix,    //[batch,4,4]  
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc_position,    //[batch,4,point_num] 
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> repc_hom_w_tensor,         //[batch,1,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_ndc_pos,    //[batch,4,point_num]  
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_position    //[4,point_num] 

)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int batch_id = 0; batch_id < ndc_position.size(0); batch_id++)
    {
        if (batch_id < ndc_position.size(0) && index < ndc_position.size(2))
        {
            float repc_hom_w = repc_hom_w_tensor[batch_id][0][index];

            float mul1 = ndc_position[batch_id][0][index] * repc_hom_w;
            float mul2 = ndc_position[batch_id][1][index] * repc_hom_w;
            float mul3 = ndc_position[batch_id][2][index] * repc_hom_w;

            float grad_x = (view_project_matrix[batch_id][0][0] * repc_hom_w - view_project_matrix[batch_id][0][3] * mul1) * grad_ndc_pos[batch_id][0][index]
                + (view_project_matrix[batch_id][0][1] * repc_hom_w - view_project_matrix[batch_id][0][3] * mul2) * grad_ndc_pos[batch_id][1][index]
                + (view_project_matrix[batch_id][0][2] * repc_hom_w - view_project_matrix[batch_id][0][3] * mul3) * grad_ndc_pos[batch_id][2][index];

            float grad_y = (view_project_matrix[batch_id][1][0] * repc_hom_w - view_project_matrix[batch_id][1][3] * mul1) * grad_ndc_pos[batch_id][0][index]
                + (view_project_matrix[batch_id][1][1] * repc_hom_w - view_project_matrix[batch_id][1][3] * mul2) * grad_ndc_pos[batch_id][1][index]
                + (view_project_matrix[batch_id][1][2] * repc_hom_w - view_project_matrix[batch_id][1][3] * mul3) * grad_ndc_pos[batch_id][2][index];

            float grad_z = (view_project_matrix[batch_id][2][0] * repc_hom_w - view_project_matrix[batch_id][2][3] * mul1) * grad_ndc_pos[batch_id][0][index]
                + (view_project_matrix[batch_id][2][1] * repc_hom_w - view_project_matrix[batch_id][2][3] * mul2) * grad_ndc_pos[batch_id][1][index]
                + (view_project_matrix[batch_id][2][2] * repc_hom_w - view_project_matrix[batch_id][2][3] * mul3) * grad_ndc_pos[batch_id][2][index];

            grad_position[0][index] = grad_x;
            grad_position[1][index] = grad_y;
            grad_position[2][index] = grad_z;
            grad_position[3][index] = 0;
        }
    }
}

at::Tensor world2ndc_backword(at::Tensor view_project_matrix, at::Tensor ndc_position, at::Tensor repc_hom_w, at::Tensor grad_ndcpos)
{


    int N = grad_ndcpos.size(0);
    int P = grad_ndcpos.size(2);
    at::Tensor d_position = torch::empty({ 4,P }, grad_ndcpos.options());

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);

    world2ndc_backword_kernel << <blocknum, threadsnum >> > (
        view_project_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        ndc_position.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        repc_hom_w.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_ndcpos.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_position.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
    return d_position;
}
template <typename scalar_t,int ROW,int COL>
__device__ void load_matrix(scalar_t(* __restrict__ dest)[ROW][COL], const torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> source)
{
    
    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            (*dest)[i][j] = source[i][j];
        }
    }
}

template <typename scalar_t, int ROW, int COL>
__device__ void load_matrix_batch(scalar_t(*__restrict__ dest)[ROW][COL], const torch::TensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, int32_t> source,int index)
{

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            (*dest)[i][j] = source[i][j][index];
        }
    }
}

template <typename scalar_t, int ROW, int COL>
__device__ void load_matrix_batch(scalar_t(*__restrict__ dest)[ROW][COL], const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> source, int index)
{

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            (*dest)[i][j] = source[i][j][index];
        }
    }
}

template <typename scalar_t, int ROW, int COL>
__device__ void save_matrix(const scalar_t(*__restrict__ source)[ROW][COL], torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> dest)
{

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            dest[i][j]=(*source)[i][j];
        }
    }
}

template <typename scalar_t, int ROW, int COL>
__device__ void save_matrix_batch(const scalar_t(*__restrict__ source)[ROW][COL], torch::TensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, int32_t> dest,int index)
{

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            dest[i][j][index] = (*source)[i][j];
        }
    }
}

template <typename scalar_t, int ROW, int COL>
__device__ void save_matrix_batch(const scalar_t(*__restrict__ source)[ROW][COL], torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dest, int index)
{

    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            dest[i][j][index] = (*source)[i][j];
        }
    }
}

template <typename scalar_t, int M, int N,int K, bool A_trans =false,bool B_trans =false>
__device__ void matmul(scalar_t(* __restrict__ A)[A_trans?M:K], scalar_t(* __restrict__ B)[B_trans?K:N], scalar_t(* __restrict__ output)[N])
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            scalar_t temp = 0.0;
            for (int k = 0; k < K; k++)
            {
                if(A_trans==false && B_trans==false)
                    temp+=A[i][k] * B[k][j];
                else if (A_trans == true && B_trans == false)
                    temp += A[k][i] * B[k][j];
                else if (A_trans == false && B_trans == true)
                    temp += A[i][k] * B[j][k];
                else if (A_trans == true && B_trans == true)
                    temp += A[k][i] * B[j][k];
            }
            output[i][j] = temp;
        }
    }
}

template <typename scalar_t, int M, int N>
__device__ void matmul_AtA(scalar_t(*__restrict__ A)[N], scalar_t(*__restrict__ output)[N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            scalar_t temp = 0.0;
            for (int k = 0; k < M; k++)
            {
                temp += A[k][i] * A[k][j];
            }
            output[i][j] = temp;
        }
    }
}

template <typename scalar_t>
__global__ void create_cov2d_forward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian_matrix,    //[batch,3,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> view_matrix,    //[batch,4,4] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> world_transform_matrix,    //[3,3,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cov2d         //[batch,2,2,point_num]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    __shared__ scalar_t view[3][3];
    if (threadIdx.x < 9 && batch_id < view_matrix.size(0))
    {
        int row = threadIdx.x / 3;
        int col = threadIdx.x % 3;
        view[row][col] = view_matrix[batch_id][row][col];
    }
    __syncthreads();

    if (batch_id < view_matrix.size(0) && index < world_transform_matrix.size(2))
    {
        // world_transform_matrix @ view_matrix
        scalar_t T[3][3];
        scalar_t temp0[3][3];
        load_matrix_batch<scalar_t, 3, 3>(&T, world_transform_matrix, index);
        matmul<scalar_t, 3, 3, 3>(T, view, temp0);//world_transform_matrix@view_matrix

        scalar_t J[3][2];
        scalar_t temp1[3][2];
        load_matrix_batch<scalar_t, 3, 2>(&J, jacobian_matrix[batch_id],index);
        matmul<scalar_t, 3, 2, 3>(temp0, J, temp1);//(world_transform_matrix@view_matrix)@jacobian_matrix

        scalar_t result[2][2];
        matmul_AtA<scalar_t, 3, 2>(temp1, result);//A.trans@A

        //low-pass filter
        result[0][0] += 0.3f;
        result[1][1] += 0.3f;

        save_matrix_batch<scalar_t, 2, 2>(&result, cov2d[batch_id],index);
    }
}


at::Tensor createCov2dDirectly_forward(
    at::Tensor J, //N,3,3,P
    at::Tensor view_matrix, //N,4,4
    at::Tensor transform_matrix //3,3,P
)
{
    int N = view_matrix.size(0);
    int P = transform_matrix.size(2);
    assert(J.size(0) == N);
    assert(J.size(3) == P);
    at::Tensor cov2d = torch::empty({ N,2,2,P }, transform_matrix.options());

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(transform_matrix.scalar_type(), __FUNCTION__, [&] {
        create_cov2d_forward<scalar_t> << <Block3d, threadsnum >> > (
            J.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            view_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            transform_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            cov2d.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });

    /*create_cov2d_forward<float> << <Block3d, threadsnum >> > (
        J.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        cov2d.packed_accessor32<float, 4, torch::RestrictPtrTraits>());*/

    CUDA_CHECK_ERRORS;
    return cov2d;

}

template <typename scalar_t>
__global__ void create_cov2d_backward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cov2d_grad,    //[batch,2,2,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian_matrix,    //[batch,3,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> view_matrix,    //[batch,4,4] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> world_transform_matrix,    //[3,3,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> transform_matrix_grad         //[3,3,point_num]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t view[3][3];
    scalar_t dL_dTrans_sum[3][3] = {0};
    for (int batch_id = 0; batch_id < view_matrix.size(0); batch_id++)
    {
        if (threadIdx.x < 9 )
        {
            int row = threadIdx.x / 3;
            int col = threadIdx.x % 3;
            view[row][col] = view_matrix[batch_id][row][col];
        }
        __syncthreads();

        if ( index < world_transform_matrix.size(2))
        {
            scalar_t view_rayspace_transform[3][2];
            scalar_t rayspace_transform[3][2];
            load_matrix_batch<scalar_t, 3, 2>(&rayspace_transform, jacobian_matrix[batch_id],index);
            matmul<scalar_t, 3, 2, 3>(view, rayspace_transform, view_rayspace_transform);
            scalar_t world_transform[3][3];
            load_matrix_batch<scalar_t, 3, 3>(&world_transform, world_transform_matrix,index);

            scalar_t T[3][2];
            matmul<scalar_t, 3, 2, 3>(world_transform, view_rayspace_transform, T);

            // cov2d_grad is symmetric.Gradient calculation can be simplified.
            // dL/dT=2 * T@cov2d_grad
            scalar_t dL_dCov2d[2][2];
            scalar_t dL_dT[3][2];
            load_matrix_batch<scalar_t, 2, 2>(&dL_dCov2d, cov2d_grad[batch_id],index);
            matmul<scalar_t, 3, 2, 2>(T, dL_dCov2d, dL_dT);
            dL_dT[0][0] *= 2; dL_dT[0][1] *= 2;
            dL_dT[1][0] *= 2; dL_dT[1][1] *= 2;
            dL_dT[2][0] *= 2; dL_dT[2][1] *= 2;

            //dL/dtransform = dL_dT@view_rayspace_transform.transpose()
            scalar_t dL_dTrans[3][3];
            matmul<scalar_t, 3, 3, 2, false, true>(dL_dT, view_rayspace_transform, dL_dTrans);
            dL_dTrans_sum[0][0] += dL_dTrans[0][0];
            dL_dTrans_sum[0][1] += dL_dTrans[0][1];
            dL_dTrans_sum[0][2] += dL_dTrans[0][2];
            dL_dTrans_sum[1][0] += dL_dTrans[1][0];
            dL_dTrans_sum[1][1] += dL_dTrans[1][1];
            dL_dTrans_sum[1][2] += dL_dTrans[1][2];
            dL_dTrans_sum[2][0] += dL_dTrans[2][0];
            dL_dTrans_sum[2][1] += dL_dTrans[2][1];
            dL_dTrans_sum[2][2] += dL_dTrans[2][2];

        }
        __syncthreads();
    }

    if (index < world_transform_matrix.size(2))
    {
        save_matrix_batch<scalar_t, 3, 3>(&dL_dTrans_sum, transform_matrix_grad,index);
    }
}

at::Tensor createCov2dDirectly_backward(
    at::Tensor cov2d_grad, //N,2,2,P
    at::Tensor J, //N,3,3,P
    at::Tensor view_matrix, //N,1,4,4
    at::Tensor transform_matrix //3,3,P
)
{
    int N = view_matrix.size(0);
    int P = transform_matrix.size(2);
    assert(cov2d_grad.size(0) == N);
    assert(cov2d_grad.size(3) == P);
    at::Tensor transform_matrix_grad = torch::empty({ 3,3,P }, cov2d_grad.options());

    int threadsnum = 1024;
    int blocknum=std::ceil(P / (float)threadsnum);


    create_cov2d_backward<float> << <blocknum, threadsnum >> > (
        cov2d_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        J.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
    return transform_matrix_grad;

}



// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

template <typename scalar_t,int degree>
__global__ void sh2rgb_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_base,    //[1,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_rest,    //[(deg + 1) ** 2-1,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dirs,    //[batch,3,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rgb         //[batch,3,point_num]
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (batch_id < rgb.size(0) && index < rgb.size(2))
    {
        float3 result;
        result.x = SH_C0 * SH_base[0][0][index];
        result.y = SH_C0 * SH_base[0][1][index];
        result.z = SH_C0 * SH_base[0][2][index];
        if (degree > 0)
        {
            float x = dirs[batch_id][0][index];
            float y = dirs[batch_id][1][index];
            float z = dirs[batch_id][2][index];
            result.x = result.x - SH_C1 * y * SH_rest[0][0][index] + SH_C1 * z * SH_rest[1][0][index] - SH_C1 * x * SH_rest[2][0][index];
            result.y = result.y - SH_C1 * y * SH_rest[0][1][index] + SH_C1 * z * SH_rest[1][1][index] - SH_C1 * x * SH_rest[2][1][index];
            result.z = result.z - SH_C1 * y * SH_rest[0][2][index] + SH_C1 * z * SH_rest[1][2][index] - SH_C1 * x * SH_rest[2][2][index];

            if (degree > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                result.x = result.x + 
                    SH_C2[0] * xy * SH_rest[3][0][index] +
                    SH_C2[1] * yz * SH_rest[4][0][index] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][0][index] +
                    SH_C2[3] * xz * SH_rest[6][0][index] +
                    SH_C2[4] * (xx - yy) * SH_rest[7][0][index];
                result.y = result.y +
                    SH_C2[0] * xy * SH_rest[3][1][index] +
                    SH_C2[1] * yz * SH_rest[4][1][index] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][1][index] +
                    SH_C2[3] * xz * SH_rest[6][1][index] +
                    SH_C2[4] * (xx - yy) * SH_rest[7][1][index];
                result.z = result.z +
                    SH_C2[0] * xy * SH_rest[3][2][index] +
                    SH_C2[1] * yz * SH_rest[4][2][index] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * SH_rest[5][2][index] +
                    SH_C2[3] * xz * SH_rest[6][2][index] +
                    SH_C2[4] * (xx - yy) * SH_rest[7][2][index];

                if (degree > 2)
                {
                    result.x = result.x +
                        SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][0][index] +
                        SH_C3[1] * xy * z * SH_rest[9][0][index] +
                        SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][0][index] +
                        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][0][index] +
                        SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][0][index] +
                        SH_C3[5] * z * (xx - yy) * SH_rest[13][0][index] +
                        SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][0][index];
                    result.y = result.y +
                        SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][1][index] +
                        SH_C3[1] * xy * z * SH_rest[9][1][index] +
                        SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][1][index] +
                        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][1][index] +
                        SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][1][index] +
                        SH_C3[5] * z * (xx - yy) * SH_rest[13][1][index] +
                        SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][1][index];
                    result.z = result.z +
                        SH_C3[0] * y * (3.0f * xx - yy) * SH_rest[8][2][index] +
                        SH_C3[1] * xy * z * SH_rest[9][2][index] +
                        SH_C3[2] * y * (4.0f * zz - xx - yy) * SH_rest[10][2][index] +
                        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH_rest[11][2][index] +
                        SH_C3[4] * x * (4.0f * zz - xx - yy) * SH_rest[12][2][index] +
                        SH_C3[5] * z * (xx - yy) * SH_rest[13][2][index] +
                        SH_C3[6] * x * (xx - 3.0f * yy) * SH_rest[14][2][index];
                }
            }

        }
        result.x += 0.5f;
        result.y += 0.5f;
        result.z += 0.5f;
        rgb[batch_id][0][index] = result.x;
        rgb[batch_id][1][index] = result.y;
        rgb[batch_id][2][index] = result.z;
    }
}

at::Tensor sh2rgb_forward(int64_t degree, at::Tensor sh_base, at::Tensor sh_rest, at::Tensor dir)
{
    int N = dir.size(0);
    int P = dir.size(2);
    at::Tensor rgb = torch::empty({ N,3,P }, sh_base.options());

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    switch (degree)
    {
    case 0:
        sh2rgb_forward_kernel<float, 0> << <Block3d, threadsnum >> > (
            sh_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 1:
        sh2rgb_forward_kernel<float, 1> << <Block3d, threadsnum >> > (
            sh_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 2:
        sh2rgb_forward_kernel<float, 2> << <Block3d, threadsnum >> > (
            sh_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 3:
        sh2rgb_forward_kernel<float, 3> << <Block3d, threadsnum >> > (
            sh_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    default:
        ;
    }

    

    CUDA_CHECK_ERRORS;
    return rgb;
}



template <typename scalar_t, int degree>
__global__ void sh2rgb_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dirs,    //[batch,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_base,    //[1,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_rest,    //[(deg + 1) ** 2-1,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rgb_grad,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_base_grad,   //[1,3,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_rest_grad,   //[(deg + 1) ** 2-1,3,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dir_grad//[batch,3,point_num] 
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float dRGBdx[3]{ 0,0,0 };
    float dRGBdy[3]{ 0,0,0 };
    float dRGBdz[3]{ 0,0,0 };

    for (int batch_id = 0; batch_id < rgb_grad.size(0); batch_id++)
    {
        if ( index < rgb_grad.size(2))
        {
            float3 dL_dRGB{ rgb_grad[batch_id][0][index], rgb_grad[batch_id][1][index], rgb_grad[batch_id][2][index] };

            float dRGBdsh0 = SH_C0;
            SH_base_grad[0][0][index] = dRGBdsh0 * dL_dRGB.x;
            SH_base_grad[0][1][index] = dRGBdsh0 * dL_dRGB.y;
            SH_base_grad[0][2][index] = dRGBdsh0 * dL_dRGB.z;

            if (degree > 0)
            {
                float x = dirs[batch_id][0][index];
                float y = dirs[batch_id][1][index];
                float z = dirs[batch_id][2][index];

                float dRGBdsh1 = -SH_C1 * y;
                float dRGBdsh2 = SH_C1 * z;
                float dRGBdsh3 = -SH_C1 * x;
                SH_rest_grad[0][0][index] = dRGBdsh1 * dL_dRGB.x;
                SH_rest_grad[1][0][index] = dRGBdsh2 * dL_dRGB.x;
                SH_rest_grad[2][0][index] = dRGBdsh3 * dL_dRGB.x;
                SH_rest_grad[0][1][index] = dRGBdsh1 * dL_dRGB.y;
                SH_rest_grad[1][1][index] = dRGBdsh2 * dL_dRGB.y;
                SH_rest_grad[2][1][index] = dRGBdsh3 * dL_dRGB.y;
                SH_rest_grad[0][2][index] = dRGBdsh1 * dL_dRGB.z;
                SH_rest_grad[1][2][index] = dRGBdsh2 * dL_dRGB.z;
                SH_rest_grad[2][2][index] = dRGBdsh3 * dL_dRGB.z;

                dRGBdx[0] += -SH_C1 * SH_rest[2][0][index];
                dRGBdx[1] += -SH_C1 * SH_rest[2][1][index];
                dRGBdx[2] += -SH_C1 * SH_rest[2][2][index];
                dRGBdy[0] += -SH_C1 * SH_rest[0][0][index];
                dRGBdy[1] += -SH_C1 * SH_rest[0][1][index];
                dRGBdy[2] += -SH_C1 * SH_rest[0][2][index];
                dRGBdz[0] += SH_C1 * SH_rest[1][0][index];
                dRGBdz[1] += SH_C1 * SH_rest[1][1][index];
                dRGBdz[2] += SH_C1 * SH_rest[1][2][index];

                if (degree > 1)
                {
                    float xx = x * x, yy = y * y, zz = z * z;
                    float xy = x * y, yz = y * z, xz = x * z;

                    float dRGBdsh4 = SH_C2[0] * xy;
                    float dRGBdsh5 = SH_C2[1] * yz;
                    float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
                    float dRGBdsh7 = SH_C2[3] * xz;
                    float dRGBdsh8 = SH_C2[4] * (xx - yy);

                    SH_rest_grad[3][0][index] = dRGBdsh4 * dL_dRGB.x;
                    SH_rest_grad[4][0][index] = dRGBdsh5 * dL_dRGB.x;
                    SH_rest_grad[5][0][index] = dRGBdsh6 * dL_dRGB.x;
                    SH_rest_grad[6][0][index] = dRGBdsh7 * dL_dRGB.x;
                    SH_rest_grad[7][0][index] = dRGBdsh8 * dL_dRGB.x;
                    SH_rest_grad[3][1][index] = dRGBdsh4 * dL_dRGB.y;
                    SH_rest_grad[4][1][index] = dRGBdsh5 * dL_dRGB.y;
                    SH_rest_grad[5][1][index] = dRGBdsh6 * dL_dRGB.y;
                    SH_rest_grad[6][1][index] = dRGBdsh7 * dL_dRGB.y;
                    SH_rest_grad[7][1][index] = dRGBdsh8 * dL_dRGB.y;
                    SH_rest_grad[3][2][index] = dRGBdsh4 * dL_dRGB.z;
                    SH_rest_grad[4][2][index] = dRGBdsh5 * dL_dRGB.z;
                    SH_rest_grad[5][2][index] = dRGBdsh6 * dL_dRGB.z;
                    SH_rest_grad[6][2][index] = dRGBdsh7 * dL_dRGB.z;
                    SH_rest_grad[7][2][index] = dRGBdsh8 * dL_dRGB.z;

                    dRGBdx[0] += SH_C2[0] * y * SH_rest[3][0][index] + SH_C2[2] * 2.f * -x * SH_rest[5][0][index] + SH_C2[3] * z * SH_rest[6][0][index] + SH_C2[4] * 2.f * x * SH_rest[7][0][index];
                    dRGBdx[1] += SH_C2[0] * y * SH_rest[3][1][index] + SH_C2[2] * 2.f * -x * SH_rest[5][1][index] + SH_C2[3] * z * SH_rest[6][1][index] + SH_C2[4] * 2.f * x * SH_rest[7][1][index];
                    dRGBdx[2] += SH_C2[0] * y * SH_rest[3][2][index] + SH_C2[2] * 2.f * -x * SH_rest[5][2][index] + SH_C2[3] * z * SH_rest[6][2][index] + SH_C2[4] * 2.f * x * SH_rest[7][2][index];
                    
                    dRGBdy[0] += SH_C2[0] * x * SH_rest[3][0][index] + SH_C2[1] * z * SH_rest[4][0][index] + SH_C2[2] * 2.f * -y * SH_rest[5][0][index] + SH_C2[4] * 2.f * -y * SH_rest[7][0][index];
                    dRGBdy[1] += SH_C2[0] * x * SH_rest[3][1][index] + SH_C2[1] * z * SH_rest[4][1][index] + SH_C2[2] * 2.f * -y * SH_rest[5][1][index] + SH_C2[4] * 2.f * -y * SH_rest[7][1][index];
                    dRGBdy[2] += SH_C2[0] * x * SH_rest[3][2][index] + SH_C2[1] * z * SH_rest[4][2][index] + SH_C2[2] * 2.f * -y * SH_rest[5][2][index] + SH_C2[4] * 2.f * -y * SH_rest[7][2][index];
                    
                    dRGBdz[0] += SH_C2[1] * y * SH_rest[4][0][index] + SH_C2[2] * 2.f * 2.f * z * SH_rest[5][0][index] + SH_C2[3] * x * SH_rest[6][0][index];
                    dRGBdz[1] += SH_C2[1] * y * SH_rest[4][1][index] + SH_C2[2] * 2.f * 2.f * z * SH_rest[5][1][index] + SH_C2[3] * x * SH_rest[6][1][index];
                    dRGBdz[2] += SH_C2[1] * y * SH_rest[4][2][index] + SH_C2[2] * 2.f * 2.f * z * SH_rest[5][2][index] + SH_C2[3] * x * SH_rest[6][2][index];

                    if (degree > 2)
                    {
                        float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                        float dRGBdsh10 = SH_C3[1] * xy * z;
                        float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                        float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                        float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                        SH_rest_grad[8][0][index] = dRGBdsh9 * dL_dRGB.x;
                        SH_rest_grad[9][0][index] = dRGBdsh10 * dL_dRGB.x;
                        SH_rest_grad[10][0][index] = dRGBdsh11 * dL_dRGB.x;
                        SH_rest_grad[11][0][index] = dRGBdsh12 * dL_dRGB.x;
                        SH_rest_grad[12][0][index] = dRGBdsh13 * dL_dRGB.x;
                        SH_rest_grad[13][0][index] = dRGBdsh14 * dL_dRGB.x;
                        SH_rest_grad[14][0][index] = dRGBdsh15 * dL_dRGB.x;
                        SH_rest_grad[8][1][index] = dRGBdsh9 * dL_dRGB.y;
                        SH_rest_grad[9][1][index] = dRGBdsh10 * dL_dRGB.y;
                        SH_rest_grad[10][1][index] = dRGBdsh11 * dL_dRGB.y;
                        SH_rest_grad[11][1][index] = dRGBdsh12 * dL_dRGB.y;
                        SH_rest_grad[12][1][index] = dRGBdsh13 * dL_dRGB.y;
                        SH_rest_grad[13][1][index] = dRGBdsh14 * dL_dRGB.y;
                        SH_rest_grad[14][1][index] = dRGBdsh15 * dL_dRGB.y;
                        SH_rest_grad[8][2][index] = dRGBdsh9 * dL_dRGB.z;
                        SH_rest_grad[9][2][index] = dRGBdsh10 * dL_dRGB.z;
                        SH_rest_grad[10][2][index] = dRGBdsh11 * dL_dRGB.z;
                        SH_rest_grad[11][2][index] = dRGBdsh12 * dL_dRGB.z;
                        SH_rest_grad[12][2][index] = dRGBdsh13 * dL_dRGB.z;
                        SH_rest_grad[13][2][index] = dRGBdsh14 * dL_dRGB.z;
                        SH_rest_grad[14][2][index] = dRGBdsh15 * dL_dRGB.z;

                        dRGBdx[0] += (
                            SH_C3[0] * SH_rest[8][0][index] * 3.f * 2.f * xy +
                            SH_C3[1] * SH_rest[9][0][index] * yz +
                            SH_C3[2] * SH_rest[10][0][index] * -2.f * xy +
                            SH_C3[3] * SH_rest[11][0][index] * -3.f * 2.f * xz +
                            SH_C3[4] * SH_rest[12][0][index] * (-3.f * xx + 4.f * zz - yy) +
                            SH_C3[5] * SH_rest[13][0][index] * 2.f * xz +
                            SH_C3[6] * SH_rest[14][0][index] * 3.f * (xx - yy));
                        dRGBdx[1] += (
                            SH_C3[0] * SH_rest[8][1][index] * 3.f * 2.f * xy +
                            SH_C3[1] * SH_rest[9][1][index] * yz +
                            SH_C3[2] * SH_rest[10][1][index] * -2.f * xy +
                            SH_C3[3] * SH_rest[11][1][index] * -3.f * 2.f * xz +
                            SH_C3[4] * SH_rest[12][1][index] * (-3.f * xx + 4.f * zz - yy) +
                            SH_C3[5] * SH_rest[13][1][index] * 2.f * xz +
                            SH_C3[6] * SH_rest[14][1][index] * 3.f * (xx - yy));
                        dRGBdx[2] += (
                            SH_C3[0] * SH_rest[8][2][index] * 3.f * 2.f * xy +
                            SH_C3[1] * SH_rest[9][2][index] * yz +
                            SH_C3[2] * SH_rest[10][2][index] * -2.f * xy +
                            SH_C3[3] * SH_rest[11][2][index] * -3.f * 2.f * xz +
                            SH_C3[4] * SH_rest[12][2][index] * (-3.f * xx + 4.f * zz - yy) +
                            SH_C3[5] * SH_rest[13][2][index] * 2.f * xz +
                            SH_C3[6] * SH_rest[14][2][index] * 3.f * (xx - yy));
                        dRGBdy[0] += (
                            SH_C3[0] * SH_rest[8][0][index] * 3.f * (xx - yy) +
                            SH_C3[1] * SH_rest[9][0][index] * xz +
                            SH_C3[2] * SH_rest[10][0][index] * (-3.f * yy + 4.f * zz - xx) +
                            SH_C3[3] * SH_rest[11][0][index] * -3.f * 2.f * yz +
                            SH_C3[4] * SH_rest[12][0][index] * -2.f * xy +
                            SH_C3[5] * SH_rest[13][0][index] * -2.f * yz +
                            SH_C3[6] * SH_rest[14][0][index] * -3.f * 2.f * xy);
                        dRGBdy[1] += (
                            SH_C3[0] * SH_rest[8][1][index] * 3.f * (xx - yy) +
                            SH_C3[1] * SH_rest[9][1][index] * xz +
                            SH_C3[2] * SH_rest[10][1][index] * (-3.f * yy + 4.f * zz - xx) +
                            SH_C3[3] * SH_rest[11][1][index] * -3.f * 2.f * yz +
                            SH_C3[4] * SH_rest[12][1][index] * -2.f * xy +
                            SH_C3[5] * SH_rest[13][1][index] * -2.f * yz +
                            SH_C3[6] * SH_rest[14][1][index] * -3.f * 2.f * xy);
                        dRGBdy[2] += (
                            SH_C3[0] * SH_rest[8][2][index] * 3.f * (xx - yy) +
                            SH_C3[1] * SH_rest[9][2][index] * xz +
                            SH_C3[2] * SH_rest[10][2][index] * (-3.f * yy + 4.f * zz - xx) +
                            SH_C3[3] * SH_rest[11][2][index] * -3.f * 2.f * yz +
                            SH_C3[4] * SH_rest[12][2][index] * -2.f * xy +
                            SH_C3[5] * SH_rest[13][2][index] * -2.f * yz +
                            SH_C3[6] * SH_rest[14][2][index] * -3.f * 2.f * xy);
                        dRGBdz[0] += (
                            SH_C3[1] * SH_rest[9][0][index] * xy +
                            SH_C3[2] * SH_rest[10][0][index] * 4.f * 2.f * yz +
                            SH_C3[3] * SH_rest[11][0][index] * 3.f * (2.f * zz - xx - yy) +
                            SH_C3[4] * SH_rest[12][0][index] * 4.f * 2.f * xz +
                            SH_C3[5] * SH_rest[13][0][index] * (xx - yy));
                        dRGBdz[1] += (
                            SH_C3[1] * SH_rest[9][1][index] * xy +
                            SH_C3[2] * SH_rest[10][1][index] * 4.f * 2.f * yz +
                            SH_C3[3] * SH_rest[11][1][index] * 3.f * (2.f * zz - xx - yy) +
                            SH_C3[4] * SH_rest[12][1][index] * 4.f * 2.f * xz +
                            SH_C3[5] * SH_rest[13][1][index] * (xx - yy));
                        dRGBdz[2] += (
                            SH_C3[1] * SH_rest[9][2][index] * xy +
                            SH_C3[2] * SH_rest[10][2][index] * 4.f * 2.f * yz +
                            SH_C3[3] * SH_rest[11][2][index] * 3.f * (2.f * zz - xx - yy) +
                            SH_C3[4] * SH_rest[12][2][index] * 4.f * 2.f * xz +
                            SH_C3[5] * SH_rest[13][2][index] * (xx - yy));
                    }
                }

            }


            dir_grad[batch_id][0][index] = dRGBdx[0] * rgb_grad[batch_id][0][index] + dRGBdx[1] * rgb_grad[batch_id][1][index] + dRGBdx[2] * rgb_grad[batch_id][2][index];
            dir_grad[batch_id][1][index] = dRGBdy[0] * rgb_grad[batch_id][0][index] + dRGBdy[1] * rgb_grad[batch_id][1][index] + dRGBdy[2] * rgb_grad[batch_id][2][index];
            dir_grad[batch_id][2][index] = dRGBdz[0] * rgb_grad[batch_id][0][index] + dRGBdz[1] * rgb_grad[batch_id][1][index] + dRGBdz[2] * rgb_grad[batch_id][2][index];
        }

        
    }

}

std::vector<at::Tensor> sh2rgb_backward(int64_t degree, at::Tensor rgb_grad, int64_t sh_rest_dim, at::Tensor dir, at::Tensor SH_base, at::Tensor SH_rest)
{
    int N = rgb_grad.size(0);
    int P = rgb_grad.size(2);
    int C = rgb_grad.size(1);

    at::Tensor sh_grad = torch::empty({ 1 ,C,P }, rgb_grad.options());
    at::Tensor sh_rest_grad = torch::zeros({ sh_rest_dim ,C,P }, rgb_grad.options());
    at::Tensor dir_grad = torch::empty_like(dir);

    int threadsnum = 256;
    int blocknum=std::ceil(P / (float)threadsnum);

    switch (degree)
    {
    case 0:
        sh2rgb_backward_kernel<float, 0> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 1:
        sh2rgb_backward_kernel<float, 1> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 2:
        sh2rgb_backward_kernel<float, 2> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    case 3:
        sh2rgb_backward_kernel<float, 3> << <blocknum, threadsnum >> > (
            dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_base.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            SH_rest.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rgb_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            sh_rest_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dir_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
        break;
    default:
        ;
    }



    CUDA_CHECK_ERRORS;
    return { sh_grad,sh_rest_grad,dir_grad };
}


template <typename scalar_t>
__global__ void eigh_and_inv_2x2matrix_kernel_forward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,    //[batch,2,2,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> val,   //[batch,2,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> vec,   //[batch,2,2,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> inv   //[batch,2,2,point_num] 
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (batch_id < input.size(0) && index < input.size(3))
    {
        float input_matrix[2][2] = { {input[batch_id][0][0][index],input[batch_id][0][1][index]},{input[batch_id][1][0][index],input[batch_id][1][1][index]}};
        float det = input_matrix[0][0] * input_matrix[1][1] - input_matrix[0][1] * input_matrix[1][0];
        //a*c-b*b  ->  (a-b)(c-b)+b*(a+c-2*b)
        float det1 = (input_matrix[0][0] - input_matrix[0][1]) * (input_matrix[1][1] - input_matrix[0][1]) + input_matrix[0][1] * (input_matrix[0][0] + input_matrix[1][1] - 2 * input_matrix[0][1]);
        det=(abs(det) < abs(1e-5f * input_matrix[0][1] * input_matrix[1][0])) ? det1 : det;
        
        
        float temp0 = input_matrix[0][0] + input_matrix[1][1];
        float temp1 = sqrt((input_matrix[0][0] - input_matrix[1][1]) * (input_matrix[0][0] - input_matrix[1][1])
            + 4 * input_matrix[0][1] * input_matrix[0][1]);
        temp1 = max(temp1, 1e-9f);

        float eig_value0 = 0.5 * (temp0 - temp1);
        float eig_value1= 0.5 * (temp0 + temp1);

        val[batch_id][0][index] = eig_value0;
        val[batch_id][1][index] = eig_value1;

        float vec_0[2];
        float vec_1[2];
        if (abs(eig_value0 - input_matrix[0][0]) > abs(eig_value0 - input_matrix[1][1]))
        {
            vec_0[0] = -input_matrix[0][1]; vec_0[1] = input_matrix[0][0] - eig_value0;
            vec_1[0] = eig_value1 - input_matrix[1][1]; vec_1[1] = input_matrix[0][1];
        }
        else
        {
            vec_0[0] = input_matrix[1][1] - eig_value0; vec_0[1] = -input_matrix[0][1];
            vec_1[0] = input_matrix[0][1]; vec_1[1] = eig_value1 - input_matrix[0][0];
        }
        float length0_rec = 1.0f / sqrt(vec_0[0] * vec_0[0] + vec_0[1] * vec_0[1]);
        float length1_rec = 1.0f / sqrt(vec_1[0] * vec_1[0] + vec_1[1] * vec_1[1]);
        vec[batch_id][0][0][index] = vec_0[0] * length0_rec; vec[batch_id][0][1][index] = vec_1[0] * length1_rec;
        vec[batch_id][1][0][index] = vec_0[1] * length0_rec; vec[batch_id][1][1][index] = vec_1[1] * length1_rec;
        
        det = (abs(det) < 1e-9f ? 1e-9 : det);
        float det_recip = 1 / det;
        inv[batch_id][0][1][index] = -input_matrix[0][1] * det_recip;
        inv[batch_id][1][0][index] = -input_matrix[1][0] * det_recip;
        inv[batch_id][0][0][index] = input_matrix[1][1] * det_recip;
        inv[batch_id][1][1][index] = input_matrix[0][0] * det_recip;

    }

}


template <typename scalar_t>
__global__ void inv_2x2matrix_kernel_backward(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> Invmatrix,    //[batch,2,2,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dInvmatrix,    //[batch,2,2,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dMatrix   //[batch,2,2,point_num] 
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (batch_id < Invmatrix.size(0) && index < Invmatrix.size(3))
    {
        scalar_t inv_matrix[2][2];
        scalar_t dl_dinvmatrix[2][2];
        scalar_t temp[2][2];
        scalar_t dl_dmatrix[2][2];

        load_matrix_batch<scalar_t, 2, 2>(&inv_matrix, Invmatrix[batch_id],index);
        load_matrix_batch<scalar_t, 2, 2>(&dl_dinvmatrix, dL_dInvmatrix[batch_id],index);

        matmul<scalar_t, 2, 2, 2>(inv_matrix, dl_dinvmatrix, temp);
        matmul<scalar_t, 2, 2, 2>(temp, inv_matrix, dl_dmatrix);

        dl_dmatrix[0][0] = -dl_dmatrix[0][0]; dl_dmatrix[0][1] = -dl_dmatrix[0][1];
        dl_dmatrix[1][0] = -dl_dmatrix[1][0]; dl_dmatrix[1][1] = -dl_dmatrix[1][1];

        save_matrix_batch<scalar_t, 2, 2>(&dl_dmatrix, dL_dMatrix[batch_id],index);
    }

}

std::vector<at::Tensor> eigh_and_inv_2x2matrix_forward(at::Tensor input)
{
    int N = input.size(0);
    int P = input.size(3);
    at::Tensor vec = torch::empty({ N,2,2,P }, input.options().requires_grad(false));
    at::Tensor val = torch::empty({ N,2,P }, input.options().requires_grad(false));
    at::Tensor inv = torch::empty({ N,2,2,P }, input.options());

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), __FUNCTION__, [&] {eigh_and_inv_2x2matrix_kernel_forward<scalar_t > << <Block3d, threadsnum >> > (
        input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        val.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        vec.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        inv.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });
    CUDA_CHECK_ERRORS;
    return { val,vec,inv };
}

at::Tensor inv_2x2matrix_backward(at::Tensor inv_matrix,at::Tensor dL_dInvMatrix)
{
    int N = inv_matrix.size(0);
    int P = inv_matrix.size(3);
    at::Tensor dL_dMatrix = torch::empty_like(dL_dInvMatrix);

    int threadsnum = 1024;
    dim3 Block3d(std::ceil(P / (float)threadsnum), N, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inv_matrix.scalar_type(), __FUNCTION__, [&] {inv_2x2matrix_kernel_backward<scalar_t > << <Block3d, threadsnum >> > (
        inv_matrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        dL_dInvMatrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        dL_dMatrix.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()); });
    CUDA_CHECK_ERRORS;
    return dL_dMatrix;

}
