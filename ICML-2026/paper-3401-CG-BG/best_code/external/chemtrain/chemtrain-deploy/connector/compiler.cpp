/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "connector/compiler.h"
#include "connector/utils.h"
#include "connector/xla_call_module_loader.h"

#include "xla/xla_data.pb.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/framework/ir/xla_framework.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

#include "absl/types/span.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"

#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"


namespace jcn {

    Compiler::Compiler(const std::string& mlir_module)
        : mlir_module(mlir_module) {

        // We add some dialects to interpret the MLIR module from JAX

        mlir::DialectRegistry registry;

        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::ml_program::MLProgramDialect>();
        registry.insert<mlir::shape::ShapeDialect>();
        mlir::func::registerAllExtensions(registry);
        mlir::mhlo::registerAllMhloDialects(registry);
        mlir::sdy::registerAllDialects(registry);
        mlir::stablehlo::registerAllDialects(registry);

        context.appendDialectRegistry(registry);

    }

    void Compiler::compile(
        const int n_atoms,
        std::vector<std::vector<int64_t>> graph_shapes,
        std::vector<xla::PrimitiveType> graph_types
        ) {

        Logger logger = Logger::getlogger();

        // For shape refinement, we have to provide the shapes of the input tensors
        xla::Shape position_shape = xla::ShapeUtil::MakeShape(xla::F32, absl::Span<const int64_t>{n_atoms, 3});
        xla::Shape species_shape = xla::ShapeUtil::MakeShape(xla::S32, absl::Span<const int64_t>{n_atoms});
        xla::Shape num_shape = xla::ShapeUtil::MakeShape(xla::S32, absl::Span<const int64_t>{});
        xla::Shape newton_flag = xla::ShapeUtil::MakeShape(xla::PRED, absl::Span<const int64_t>{});

        std::vector<xla::Shape> inputShapes = {
            position_shape, species_shape, num_shape, num_shape, newton_flag};

        // For different graphs, the input shapes can vary
        int input_args = 5;
        for (size_t i = 0; i < graph_shapes.size(); i++) {
            xla::Shape neighbor_shape = xla::ShapeUtil::MakeShape(
                graph_types[i], graph_shapes[i]);
            inputShapes.push_back(neighbor_shape);
            input_args++;
        }

        if (logger.log(LogLevel::INFO)) {
            logger.log(LogLevel::INFO, "Input shapes for the XLA computation:");
            for (const auto& shape : inputShapes) {
                logger.log(
                    LogLevel::INFO,
                    "Input: " +  xla::PrimitiveType_Name(shape.element_type()) + ":" + shape.ToString()
                );
            }
        }

        std::vector<std::string> disabled_checks = {};
        std::vector<std::string> platforms = {"cuda"};

        std::unique_ptr<XlaCallModuleLoader> module_loader = XlaCallModuleLoader::Create(
            &context, 10, mlir_module, disabled_checks, platforms, input_args,
            false, false).value();

        // We now follow the steps as in the XLACallModuleLoader from tensorflow
        absl::Status status;

        status = module_loader->ValidateXlaCallModuleInvariants();
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to validate dialect: " + std::string(status.message())
            );
        }

        status = module_loader->SetPlatformIndex("cuda");
        if (!status.ok()) {
            throw std::runtime_error(
              "Failed to set platform index: " + std::string(status.message())
            );
        }

        status = module_loader->RefineDynamicShapes(inputShapes);
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to refine dynamic shapes: " + std::string(status.message())
            );
        }

        status = module_loader->ValidateStaticShapes();
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to validate static shapes: " + std::string(status.message())
            );
        }

        status = module_loader->PrepareStablehloForLowering();
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to lower to XLA: " + std::string(status.message())
            );
        }

        auto res = module_loader->ToXlaComputation();
        if (!res.ok()) {
            throw std::runtime_error(
                "Failed to convert the module to XLA computation: " + std::string(status.message())
            );
        }

        xla::XlaComputation computation = std::move(res).value();

        absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> status_or_ref = 
            xla::ConvertHloToStablehlo(export_context, &computation.proto());
        if (!status_or_ref.ok()) {
            throw std::runtime_error(
                "Failed to convert the computation to MLIR: " + std::string(status.message())
            );
        }

        module_ref = std::move(status_or_ref).value();

    }


} // namespace jcn
