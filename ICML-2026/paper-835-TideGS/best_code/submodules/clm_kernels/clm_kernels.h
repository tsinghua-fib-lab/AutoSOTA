#pragma once
#include <torch/extension.h>

void SetSignal(torch::Tensor& signal_tensor, int microbatch_idx, int signal);

void ComputeCntH(
    torch::Tensor &bitmap,
    torch::Tensor &tmp_buffer,
    int grid_size,
    int blk_size
);

void ExtractFFS(
    torch::Tensor &input,
    torch::Tensor &output
);

void ScatterToBit(
    torch::Tensor& bitmap,
    torch::Tensor& filter,
    int bit
);

void SendSHS2GpuStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    int grid_size,
    int block_size);

void SendSHS2CpuGradBufferStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    bool accum,
    int grid_size,
    int block_size);

void SendSHS2GpuStreamRetentionCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& param_indices_from_host,
    torch::Tensor& param_indices_from_rtnt,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);

void SendSHS2CpuGradBufferStreamRetentionCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& grad_indices_to_host,
    torch::Tensor& grad_indices_to_rtnt,
    bool accum,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);