#pragma once

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

int chemtrain_register_openequivariance_xla_ffi(
    const PJRT_Api* api,
    const char* platform_name);

#ifdef __cplusplus
}
#endif