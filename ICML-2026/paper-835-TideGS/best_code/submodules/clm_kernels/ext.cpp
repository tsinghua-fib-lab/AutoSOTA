#include <torch/extension.h>
#include "clm_kernels.h"
#include "ssim.h"
#include "adam.h"
#include "compute_sh_bwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_signal", &SetSignal);
  m.def("compute_cnt_h", &ComputeCntH);
  m.def("extract_ffs", &ExtractFFS);
  m.def("scatter_to_bit", &ScatterToBit);
  m.def("send_shs2gpu_stream", &SendSHS2GpuStreamCUDA);
  m.def("send_shs2cpu_grad_buffer_stream", &SendSHS2CpuGradBufferStreamCUDA);
  m.def("send_shs2gpu_stream_retention", &SendSHS2GpuStreamRetentionCUDA);
  m.def("send_shs2cpu_grad_buffer_stream_retention", &SendSHS2CpuGradBufferStreamRetentionCUDA);

  // Add SSIM functions
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);

  m.def("selective_adam_update", &selective_adam_update);

  m.def("compute_sh_bwd_inplace", &compute_sh_bwd_inplace_tensor);
}