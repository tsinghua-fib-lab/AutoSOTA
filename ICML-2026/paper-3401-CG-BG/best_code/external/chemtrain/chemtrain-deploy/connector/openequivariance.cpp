#include "connector/openequivariance.h"

#include <cstring>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"

// Exported by the patched OpenEquivariance source.
extern "C" void* oeq_conv_forward_handler();
extern "C" void* oeq_conv_backward_handler();
extern "C" void* oeq_conv_double_backward_handler();

namespace {

const PJRT_FFI* FindFfiExtension(const PJRT_Api* api) {
  if (api == nullptr) return nullptr;

  for (PJRT_Extension_Base* ext = api->extension_start;
       ext != nullptr;
       ext = ext->next) {
    if (ext->type == PJRT_Extension_Type_FFI) {
      return reinterpret_cast<const PJRT_FFI*>(ext);
    }
  }

  return nullptr;
}

PJRT_Error* RegisterOne(
    const PJRT_FFI* ffi,
    const char* target_name,
    void* handler,
    const char* platform_name) {
  PJRT_FFI_Register_Handler_Args args;
  std::memset(&args, 0, sizeof(args));

  args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  args.target_name = target_name;
  args.target_name_size = std::strlen(target_name);
  args.handler = handler;
  args.platform_name = platform_name;
  args.platform_name_size = std::strlen(platform_name);

  args.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(
      PJRT_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);

  return ffi->register_handler(&args);
}

int RegisterOrFail(
    const PJRT_FFI* ffi,
    const char* target_name,
    void* handler,
    const char* platform_name) {
  PJRT_Error* err = RegisterOne(ffi, target_name, handler, platform_name);
  if (err != nullptr) {
    // In production, convert PJRT_Error to your existing Status/StatusOr path.
    // Do not ignore this: failing here means the StableHLO will fail later.
    return 1;
  }
  return 0;
}

}  // namespace

extern "C" int chemtrain_register_openequivariance_xla_ffi(
    const PJRT_Api* api,
    const char* platform_name) {
  const PJRT_FFI* ffi = FindFfiExtension(api);
  if (ffi == nullptr || ffi->register_handler == nullptr) {
    return 1;
  }

  int rc = 0;

  rc |= RegisterOrFail(
      ffi, "conv_forward", oeq_conv_forward_handler(), platform_name);
  rc |= RegisterOrFail(
      ffi, "conv_backward", oeq_conv_backward_handler(), platform_name);
  rc |= RegisterOrFail(
      ffi, "conv_double_backward", oeq_conv_double_backward_handler(), platform_name);

  return rc;
}
