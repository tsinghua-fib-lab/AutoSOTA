// compression_backend.cuh
// Stateless compression backend using nvcomp ANS.
// Compressed data is managed by Python (PyTorch buffers), backend only handles
// compression/decompression operations with lazy DecompressionConfig caching.
#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <unordered_map>
#include <string>

#define CUDA_CHECK(cond)                                                 \
    do {                                                                 \
        cudaError_t err = cond;                                          \
        if (err != cudaSuccess) {                                        \
            throw std::runtime_error(                                    \
                std::string("CUDA error: ") + cudaGetErrorString(err) +  \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                \
    } while (false)

class CompressionBackend {
private:
    int chunk_size_;
    
    // Per-device resources (streams, events, managers are device-specific)
    struct DeviceResources {
        cudaStream_t stream = nullptr;
        cudaEvent_t event = nullptr;
        nvcomp::ANSManager* manager = nullptr;
        std::unordered_map<std::string, nvcomp::DecompressionConfig> decomp_configs;
        
        // Default constructor
        DeviceResources() = default;
        
        // Move constructor - transfer ownership of resources
        DeviceResources(DeviceResources&& other) noexcept
            : stream(other.stream)
            , event(other.event)
            , manager(other.manager)
            , decomp_configs(std::move(other.decomp_configs))
        {
            other.stream = nullptr;
            other.event = nullptr;
            other.manager = nullptr;
        }
        
        // Move assignment - transfer ownership of resources
        DeviceResources& operator=(DeviceResources&& other) noexcept {
            if (this != &other) {
                // Clean up existing resources
                cleanup_();
                
                // Transfer ownership
                stream = other.stream;
                event = other.event;
                manager = other.manager;
                decomp_configs = std::move(other.decomp_configs);
                
                // Nullify source
                other.stream = nullptr;
                other.event = nullptr;
                other.manager = nullptr;
            }
            return *this;
        }
        
        // Delete copy constructor and copy assignment
        DeviceResources(const DeviceResources&) = delete;
        DeviceResources& operator=(const DeviceResources&) = delete;
        
        ~DeviceResources() {
            cleanup_();
        }
        
    private:
        void cleanup_() {
            decomp_configs.clear();
            if (manager) {
                delete manager;
                manager = nullptr;
            }
            if (event) {
                cudaEventDestroy(event);
                event = nullptr;
            }
            if (stream) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
    };
    
    std::unordered_map<int, DeviceResources> device_resources_;

    // Lazily create per-device resources
    DeviceResources& ensure_device_resources_(int device_id) {
        auto it = device_resources_.find(device_id);
        if (it == device_resources_.end()) {
            c10::cuda::CUDAGuard device_guard(device_id);
            
            DeviceResources res;
            CUDA_CHECK(cudaStreamCreate(&res.stream));
            CUDA_CHECK(cudaEventCreate(&res.event));
            res.manager = new nvcomp::ANSManager(
                chunk_size_,
                nvcompBatchedANSCompressDefaultOpts,
                nvcompBatchedANSDecompressDefaultOpts,
                res.stream
            );
            
            it = device_resources_.emplace(device_id, std::move(res)).first;
        }
        return it->second;
    }

public:
    CompressionBackend(int chunk_size = 1 << 18)
        : chunk_size_(chunk_size) {}

    ~CompressionBackend() {
        clear_cache();
    }

    // Compress input tensor, returns compressed data tensor.
    // Caller (Python) is responsible for storing the returned tensor.
    // Device is inferred from input tensor.
    torch::Tensor compress(torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
        TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8 tensor");
        TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

        int device_id = input.device().index();
        c10::cuda::CUDAGuard device_guard(device_id);
        DeviceResources& res = ensure_device_resources_(device_id);
        
        size_t original_size = input.numel();

        nvcomp::CompressionConfig comp_config =
            res.manager->configure_compression(original_size);

        torch::Tensor temp_output = torch::empty(
            {static_cast<int64_t>(comp_config.max_compressed_buffer_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input.device())
        );

        res.manager->compress(
            input.data_ptr<uint8_t>(),
            temp_output.data_ptr<uint8_t>(),
            comp_config
        );

        size_t compressed_size = res.manager->get_compressed_output_size(
            temp_output.data_ptr<uint8_t>()
        );

        torch::Tensor compressed = torch::empty(
            {static_cast<int64_t>(compressed_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input.device())
        );

        CUDA_CHECK(cudaMemcpyAsync(
            compressed.data_ptr<uint8_t>(),
            temp_output.data_ptr<uint8_t>(),
            compressed_size,
            cudaMemcpyDeviceToDevice,
            res.stream
        ));

        CUDA_CHECK(cudaStreamSynchronize(res.stream));

        return compressed;
    }

    // Decompress with lazy config caching.
    // On first call per key, reads header from compressed data to build DecompressionConfig.
    // Subsequent calls use cached config for zero-overhead decompression.
    // Device is inferred from compressed tensor.
    void decompress(
        const std::string& key,
        torch::Tensor compressed,
        torch::Tensor output_buffer
    ) {
        TORCH_CHECK(compressed.is_cuda(), "Compressed tensor must be on CUDA device");
        TORCH_CHECK(compressed.dtype() == torch::kUInt8, "Compressed tensor must be uint8");
        TORCH_CHECK(compressed.is_contiguous(), "Compressed tensor must be contiguous");
        TORCH_CHECK(output_buffer.is_cuda(), "Output buffer must be on CUDA device");
        TORCH_CHECK(output_buffer.dtype() == torch::kUInt8, "Output buffer must be uint8");
        TORCH_CHECK(output_buffer.is_contiguous(), "Output buffer must be contiguous");
        TORCH_CHECK(compressed.device() == output_buffer.device(), 
                    "Compressed and output tensors must be on the same device");

        int device_id = compressed.device().index();
        c10::cuda::CUDAGuard device_guard(device_id);
        DeviceResources& res = ensure_device_resources_(device_id);

        // Lazy config initialization: build from compressed header on first call
        auto it = res.decomp_configs.find(key);
        if (it == res.decomp_configs.end()) {
            // configure_decompression reads header from compressed data
            auto config = res.manager->configure_decompression(
                compressed.data_ptr<uint8_t>()
            );
            it = res.decomp_configs.emplace(key, config).first;
        }

        res.manager->decompress(
            output_buffer.data_ptr<uint8_t>(),
            compressed.data_ptr<uint8_t>(),
            it->second
        );

        // Record event on our stream after decompression
        CUDA_CHECK(cudaEventRecord(res.event, res.stream));

        // Make PyTorch's current stream wait on our event
        // This does NOT block the CPU - it's GPU-side synchronization only
        cudaStream_t torch_stream = c10::cuda::getCurrentCUDAStream(device_id).stream();
        CUDA_CHECK(cudaStreamWaitEvent(torch_stream, res.event, 0));
    }

    // Release all device resources. They will be lazily recreated on next use.
    void clear_cache() {
        for (auto& [device_id, res] : device_resources_) {
            c10::cuda::CUDAGuard device_guard(device_id);
            if (res.stream) {
                CUDA_CHECK(cudaStreamSynchronize(res.stream));
            }
            res.decomp_configs.clear();
            if (res.manager) {
                delete res.manager;
                res.manager = nullptr;
            }
        }
        device_resources_.clear();
    }

    // Clear cache for a specific device only.
    void clear_cache(int device_id) {
        auto it = device_resources_.find(device_id);
        if (it != device_resources_.end()) {
            c10::cuda::CUDAGuard device_guard(device_id);
            if (it->second.stream) {
                CUDA_CHECK(cudaStreamSynchronize(it->second.stream));
            }
            device_resources_.erase(it);
        }
    }

    // Block CPU until all device streams complete.
    void synchronize() {
        for (auto& [device_id, res] : device_resources_) {
            c10::cuda::CUDAGuard device_guard(device_id);
            CUDA_CHECK(cudaStreamSynchronize(res.stream));
        }
    }

    // Synchronize a specific device only.
    void synchronize(int device_id) {
        auto it = device_resources_.find(device_id);
        if (it != device_resources_.end()) {
            c10::cuda::CUDAGuard device_guard(device_id);
            CUDA_CHECK(cudaStreamSynchronize(it->second.stream));
        }
    }

    int get_chunk_size() const {
        return chunk_size_;
    }
};