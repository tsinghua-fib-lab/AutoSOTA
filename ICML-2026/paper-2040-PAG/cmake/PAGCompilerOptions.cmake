include(CheckCXXCompilerFlag)

function(pag_require_cxx_flag flag variable)
  string(REGEX REPLACE "[^A-Za-z0-9]" "_" flag_id "${flag}")
  check_cxx_compiler_flag("${flag}" "${variable}_${flag_id}")
  if(NOT ${variable}_${flag_id})
    message(FATAL_ERROR "The current C++ compiler does not support required flag ${flag}")
  endif()
endfunction()

function(pag_add_common_options target)
  target_compile_features(${target} PUBLIC cxx_std_17)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_compile_options(${target} PRIVATE ${PAG_OPTIMIZATION_LEVEL})

  if(PAG_NATIVE_ARCH)
    pag_require_cxx_flag("-march=native" PAG_HAS_FLAG)
    target_compile_options(${target} PRIVATE -march=native)
  endif()

  if(PAG_ENABLE_WARNINGS)
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
  endif()
endfunction()

function(pag_enable_avx512 target)
  if(NOT PAG_USE_AVX512)
    message(FATAL_ERROR
            "PAG_USE_AVX512=OFF is not available yet: this release currently "
            "contains only the AVX-512 backend. Keep PAG_USE_AVX512=ON or add "
            "a scalar/AVX2 backend before disabling it.")
  endif()

  set(PAG_AVX512_FLAGS
      -mavx512f
      -mavx512dq
      -mavx512cd
      -mavx512bw
      -mavx512vl
      -mavx512vnni)
  foreach(flag IN LISTS PAG_AVX512_FLAGS)
    pag_require_cxx_flag("${flag}" PAG_HAS_AVX512_FLAG)
  endforeach()

  target_compile_options(${target} PRIVATE ${PAG_AVX512_FLAGS})
  target_compile_definitions(${target} PUBLIC PAG_USE_AVX512=1)
endfunction()
