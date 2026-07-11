# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@xla//xla/tsl/platform:build_config.bzl", "tf_proto_library")

cc_library(
    name = "openequivariance",
    srcs = [
        "openequivariance.cpp",
    ],
    hdrs = [
        "openequivariance.h",
    ],
    deps = [
        "@openequivariance_src//:oeq_xla_ffi_handlers",

        # Existing XLA/PJRT dependencies.
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",

        # Depending on your XLA revision, this may need adjustment.
        # Query:
        #   bazel query '@xla//xla/pjrt/c:*'
    ],
    alwayslink = True,
)

cc_binary(
    name = "libconnector.so",
    srcs = [
        "libconnector.cpp",
        "libconnector.h",
    ],
    linkopts = [
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    tags = ["manual"],
    deps = [
        ":runner",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/profiler/lib:nvtx_utils",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt/cpu:cpu_client",
    ],
)

cc_library(
    name = "runner",
    srcs = [
        "libconnector.h",
        "runner.cpp",
        "runner.h",
    ],
    deps = [
        ":buffer",
        ":compiler",
        ":communication",
        ":domain",
        ":graph_builder",
        ":model_proto_cc",
        ":utils",
        ":xla_call_module_loader",
        ":openequivariance",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

cc_library(
    name = "communication",
    srcs = ["communication.cpp"],
    hdrs = [
        "communication.h",
        "libconnector.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/backends/gpu:ffi",
        "@xla//xla/ffi:ffi",
        "@xla//xla/ffi:type_registry",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_ffi_extension_hdrs",
        "@xla//xla/stream_executor:device_memory",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:memory_allocation",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
    ]
)

cc_library(
    name = "buffer",
    srcs = [
        "buffer.cpp",
        "buffer.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

cc_library(
    name = "graph_builder",
    srcs = ["graph_builder.cpp"],
    hdrs = ["graph_builder.h"],
    deps = [
        ":buffer",
        ":utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

cc_library(
    name = "domain",
    srcs = ["domain.cpp"],
    hdrs = ["domain.h"],
    deps = [
        ":buffer",
        ":utils",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal_util",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cpp"],
    hdrs = ["compiler.h"],
    deps = [
        ":utils",
        ":xla_call_module_loader",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/mlir/framework/ir:xla_framework",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "xla_call_module_loader",
    srcs = ["xla_call_module_loader.cpp"],
    hdrs = ["xla_call_module_loader.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/import:passes",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
)

proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
)

cc_proto_library(
    name = "model_proto_cc",
    deps = [":model_proto"],
)
