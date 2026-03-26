#include <torch/extension.h>
#include "sptp_exp_opt.hpp"
#include <iostream>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sptp_linear_fwd_v2_shared_exp", &sptp_linear_fwd_v2_shared_exp, "sptp_linear_fwd_v2_shared_exp");
    m.def("sptp_linear_bwd_v1_shared_exp", &sptp_linear_bwd_v1_shared_exp, "sptp_linear_bwd_v1_shared_exp");
    m.def("sptp_linear_bwd_bwd_v2_shared_exp", &sptp_linear_bwd_bwd_v2_shared_exp, "sptp_linear_bwd_bwd_v2_shared_exp");
}