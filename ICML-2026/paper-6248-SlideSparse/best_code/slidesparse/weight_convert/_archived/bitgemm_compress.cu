/* BitNet 权重量化压缩核心实现（基于 cuSPARSELt）。
nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared bitgemm_compress.cu -lcuda -lcusparseLt -gencode=arch=compute_80,code=compute_80 -o libbitnet_compress.so

负责构建结构化稀疏矩阵乘计划、查询压缩尺寸并执行权重压缩。
注意：为与推理内核保持一致，这里的 matmul 拓扑采用“稀疏在左 W（不转置）× 稠密在右 A（转置）→ 输出行主序视图 C(N×M)”。
 */

#include <cusparseLt.h>         // cuSPARSELt API，用于结构化稀疏矩阵乘
#include <cuda_runtime_api.h>   // CUDA 运行时 API，提供流和内存管理
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

// --------------------------- 调试工具函数区 ---------------------------
// 主要负责读取环境变量、初始化调试日志流，并在需要时写入详细信息。

// 调试日志默认开关 (可通过修改此常量或设置环境变量覆盖)。
static constexpr bool kDebugDefaultEnabled = true;
// 强制使用的算法 ID（-1时使用默认算法，>=0使用指定算法）。
static constexpr int kForcedAlgConfigId = 6;

// 判断是否启用 BitNet 压缩模块的调试输出。
static bool bitnet_debug_enabled() {
    static bool flag = []() {
        const char *env = std::getenv("BITNET_CUSPARSE_DEBUG");
        if (!env) {
            return kDebugDefaultEnabled;
        }
        return std::strcmp(env, "0") != 0;  // 仅当设置且不为 "0" 才开启
    }();
    return flag;
}

// 获取全局日志输出流，懒加载并与 Python 侧使用的日志路径保持一致。
static std::ofstream &bitnet_debug_stream() {
    static std::ofstream stream;
    static bool initialized = false;
    if (!initialized) {
        const char *env = std::getenv("BITNET_DEBUG_LOG_PATH");
        const char *path = env ? env : "covert_debug.log";
        stream.open(path, std::ios::out | std::ios::app);
        initialized = true;
    }
    return stream;
}

// 写入一条调试日志（线程安全，并根据开关决定是否输出）。
static void bitnet_debug_write(const std::string &message) {
    if (!bitnet_debug_enabled()) {
        return;
    }
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ofstream &stream = bitnet_debug_stream();
    if (!stream.is_open()) {
        return;
    }
    stream << "[bitnet-compress-debug] " << message << std::endl;
    stream.flush();
}

// 便捷宏：包装流式写法并附带调试开关。
#define DEBUG_LOG(msg)                                                          \
    do {                                                                        \
        if (bitnet_debug_enabled()) {                                           \
            std::ostringstream _bitnet_debug_oss;                               \
            _bitnet_debug_oss << msg;                                           \
            bitnet_debug_write(_bitnet_debug_oss.str());                        \
        }                                                                       \
    } while (0)

static void debug_dump_pointer_info(const char *name, const void *ptr) {
    if (!bitnet_debug_enabled()) {
        return;
    }
    {
        std::ostringstream oss;
        oss << std::hex << std::showbase << "指针 " << name << "=" << ptr;
        bitnet_debug_write(oss.str());
    }
    if (!ptr) {
        return;
    }
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaPointerGetAttributes(" << name << ") 失败: "
            << cudaGetErrorString(err) << " (" << err << ")";
        bitnet_debug_write(oss.str());
        return;
    }
    std::ostringstream oss;
    oss << name << " 属性: type=" << attr.type << " device=" << attr.device;
    bitnet_debug_write(oss.str());
}

// 包装 cuSPARSELt 调用，自动在失败时抛出异常并打印位置。
#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            std::cerr << "cuSPARSELt call failed at line " << __LINE__         \
                      << ", status=" << status << std::endl;                  \
            std::cerr << "function: " << #func << std::endl;                 \
            throw std::runtime_error("cuSPARSELt call failed");               \
        }                                                                      \
    }

// 包装 CUDA 调用，确保任何错误都会立即暴露。
#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA call failed at line " << __LINE__ << ": "      \
                      << cudaGetErrorString(status) << " (" << status << ")"  \
                      << std::endl;                                           \
            throw std::runtime_error("CUDA call failed");                     \
        }                                                                      \
    }

static cusparseLtHandle_t g_cusparselt_handle;  // 全局 cuSPARSELt 句柄
static bool g_handle_initialized = false;       // 标记句柄是否已初始化
static std::mutex g_handle_mutex;               // 保护句柄初始化的互斥量
static std::mutex g_cusparselt_call_mutex;      // 串行化 cuSPARSELt 调用，避免线程竞争

// 缓存键：使用 (M, N, K) 唯一标识一次矩阵乘计划。
struct MatmulPlanKey {
    int m;
    int n;
    int k;

    bool operator<(const MatmulPlanKey &other) const {
        return std::tie(m, n, k) < std::tie(other.m, other.n, other.k);
    }
};

// 缓存的矩阵乘计划上下文，包含所有 cuSPARSELt 所需描述符与计划对象。
// 注意：为与推理路径保持一致，采用“稀疏在左 W、不转置；稠密在右 A，转置；输出使用行主序视图 C(N×M)”
struct MatmulPlanContext {
    cusparseLtMatDescriptor_t matW_left{};   // 左操作数：结构化稀疏权重 W (N×K, int8, RowMajor)
    cusparseLtMatDescriptor_t matA_right{};  // 右操作数：稠密激活 A (M×K, int8, RowMajor)
    cusparseLtMatDescriptor_t matR{};        // 输出：行主序视图 R (N×M, int32, RowMajor)
    cusparseLtMatmulDescriptor_t matmul{};
    cusparseLtMatmulAlgSelection_t alg_sel{};
    cusparseLtMatmulPlan_t plan{};
    bool initialized = false;

    MatmulPlanContext() = default;
};

static std::map<MatmulPlanKey, MatmulPlanContext> g_matmul_plan_cache;  // (M,N,K)->计划 上的缓存
static std::mutex g_matmul_plan_mutex;  // 保护计划缓存访问
static bool g_plan_cleanup_registered = false;  // 确保退出时只注册一次清理函数

// 销毁单个计划上下文，释放所有 cuSPARSELt 资源。
static void destroy_plan(MatmulPlanContext &ctx) {
    if (!ctx.initialized) {
        return;
    }
    cusparseLtMatmulPlanDestroy(&ctx.plan);
    cusparseLtMatmulAlgSelectionDestroy(&ctx.alg_sel);
    cusparseLtMatDescriptorDestroy(&ctx.matA_right);
    cusparseLtMatDescriptorDestroy(&ctx.matW_left);
    cusparseLtMatDescriptorDestroy(&ctx.matR);
    ctx.initialized = false;
}

// 遍历缓存并销毁所有计划，供程序退出或异常清理时调用。
static void cleanup_all_plans() {
    std::lock_guard<std::mutex> lock(g_matmul_plan_mutex);
    for (auto &kv : g_matmul_plan_cache) {
        destroy_plan(kv.second);
    }
    g_matmul_plan_cache.clear();
}

// 通过 std::atexit 注册一次全局清理函数。
static void register_plan_cleanup_once() {
    if (!g_plan_cleanup_registered) {
        std::atexit([]() { cleanup_all_plans(); });
        g_plan_cleanup_registered = true;
    }
}

// 获取（或新建）全局 cuSPARSELt 句柄，确保线程安全。
static cusparseLtHandle_t get_cusparselt_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_initialized) {
        // 首次访问时完成 cuSPARSELt 句柄初始化，并注册退出清理逻辑。
        cusparseStatus_t status = cusparseLtInit(&g_cusparselt_handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize cuSPARSELt handle, status="
                      << status << std::endl;
            throw std::runtime_error("cuSPARSELt handle creation failed");
        }
        g_handle_initialized = true;
        DEBUG_LOG("已初始化全局 cuSPARSELt 句柄");
        std::atexit([]() {
            // 进程退出时释放全局句柄，防止资源泄漏。
            if (g_handle_initialized) {
                cusparseLtDestroy(&g_cusparselt_handle);
                g_handle_initialized = false;
                DEBUG_LOG("全局 cuSPARSELt 句柄在退出时被销毁");
            }
        });
    }
    return g_cusparselt_handle;
}

// 按给定维度 (M, N, K) 获取矩阵乘计划；若缓存中不存在则创建并缓存。
static MatmulPlanContext &get_or_create_plan(cusparseLtHandle_t handle, int M,
                                             int N, int K) {
    MatmulPlanKey key{M, N, K};

    std::lock_guard<std::mutex> lock(g_matmul_plan_mutex);
    auto iter = g_matmul_plan_cache.find(key);
    if (iter != g_matmul_plan_cache.end()) {
        DEBUG_LOG("命中计划缓存: M=" << M << " N=" << N << " K=" << K);
        return iter->second;
    }

    register_plan_cleanup_once();

    auto insert_ret = g_matmul_plan_cache.try_emplace(key);
    MatmulPlanContext &ctx = insert_ret.first->second;

    if (!insert_ret.second) {
        return ctx;
    }

    DEBUG_LOG("创建新的压缩计划 (稀疏在左): M=" << M << " N=" << N << " K=" << K);

    auto order_row = CUSPARSE_ORDER_ROW;              // 行主序存储（W,A,R）
    auto order_col = CUSPARSE_ORDER_COL;              // 列主序存储
    auto opW = CUSPARSE_OPERATION_NON_TRANSPOSE;      // W 不转置（触发稀疏核族）
    auto opA = CUSPARSE_OPERATION_TRANSPOSE;          // A 在乘法中转置（W*Aᵀ）

    int ldW = K;  // 行主序下 W 的 leading dimension
    int ldA = K;  // 行主序下 A 的 leading dimension
    int ldR = M;  // 行主序下 R 的 leading dimension
    unsigned alignment = 16;  // 数据对齐要求
    DEBUG_LOG("描述符配置: ldW=" << ldW << " ldA=" << ldA << " ldR=" << ldR
                                 << " alignment=" << alignment);

    bool matW_ready = false;
    bool matA_ready = false;
    bool matR_ready = false;
    bool alg_sel_ready = false;
    bool plan_ready = false;

    try {
        // Step 1: 初始化结构化稀疏矩阵 W 的描述符（2:4 稀疏约束，左操作数）
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            &handle, &ctx.matW_left, N, K, ldW, alignment, CUDA_R_8I,
            order_row, CUSPARSELT_SPARSITY_50_PERCENT));
        matW_ready = true;
        DEBUG_LOG("matW_left 描述符初始化完成 (结构化稀疏)");

        // Step 2: 初始化稠密矩阵 A 的描述符（右操作数）
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &ctx.matA_right, M, K,
                                                     ldA, alignment, CUDA_R_8I,
                                                     order_row));
        matA_ready = true;
        DEBUG_LOG("matA_right 描述符初始化完成");

        // Step 3: 初始化输出矩阵 R 的描述符（行主序、int32 累加精度）
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &ctx.matR, N, M,
                                                     ldR, alignment, CUDA_R_32I, 
                                                     order_row));
        matR_ready = true;
        DEBUG_LOG("matR 描述符初始化完成 (RowMajor的N×M)");

        // Step 4: 构建矩阵乘描述符，关联 W/A/R 及计算类型（opW, opA）
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            &handle, &ctx.matmul, opW, opA, &ctx.matW_left, &ctx.matA_right,
            &ctx.matR, &ctx.matR, CUSPARSE_COMPUTE_32I));
        DEBUG_LOG("Matmul 描述符初始化完成 (W * Aᵀ)");

        // Step 5: 初始化算法选择结构
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
            &handle, &ctx.alg_sel, &ctx.matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        alg_sel_ready = true;
        DEBUG_LOG("算法选择初始化完成");

        // Step 6: 如果环境变量指定了强制算法 ID，则应用该设置
        if (kForcedAlgConfigId >= 0) {

        int forced_alg_id = kForcedAlgConfigId;
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                        &handle, &ctx.alg_sel,
                        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                        &forced_alg_id, sizeof(forced_alg_id)));
        DEBUG_LOG("强制使用算法 ID=" << kForcedAlgConfigId);
        }

        // Step 7: 基于描述符和算法创建可执行的矩阵乘计划
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &ctx.plan, &ctx.matmul,
                                                &ctx.alg_sel));
        plan_ready = true;
        DEBUG_LOG("Matmul 计划初始化完成");

        ctx.initialized = true;
        DEBUG_LOG("计划缓存插入成功，总数量=" << g_matmul_plan_cache.size());
        return ctx;
    }
    catch (...) {
        // 若过程中任一步骤失败，按相反顺序释放已创建的资源并移除缓存条目。
        if (plan_ready) {
            cusparseLtMatmulPlanDestroy(&ctx.plan);
        }
        if (alg_sel_ready) {
            cusparseLtMatmulAlgSelectionDestroy(&ctx.alg_sel);
        }
        if (matW_ready) {
            cusparseLtMatDescriptorDestroy(&ctx.matW_left);
        }
        if (matA_ready) {
            cusparseLtMatDescriptorDestroy(&ctx.matA_right);
        }
        if (matR_ready) {
            cusparseLtMatDescriptorDestroy(&ctx.matR);
        }
        g_matmul_plan_cache.erase(insert_ret.first);
        DEBUG_LOG("创建压缩计划失败，已清理并移除缓存条目");
        throw;
    }
}

}  // namespace

// C 接口：根据 (M, N, K) 查询压缩结果与临时缓冲区所需的字节数。
// 供 Python 端提前分配内存，避免 GPU 侧再次申请。
extern "C" void bitlinear_get_compress_sizes(int M, int N, int K,
                                              size_t *compressed_size,
                                              size_t *temp_buffer_size) {
    if (!compressed_size || !temp_buffer_size) {
        throw std::runtime_error(
            "bitlinear_get_compress_sizes: output pointer cannot be null");
    }

    // Step 1: 获取全局 cuSPARSELt 句柄与缓存计划
    cusparseLtHandle_t handle = get_cusparselt_handle();

    std::lock_guard<std::mutex> call_lock(g_cusparselt_call_mutex);
    MatmulPlanContext &ctx = get_or_create_plan(handle, M, N, K);

    // Step 2: 直接调用 cuSPARSELt API 获取字节数
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &ctx.plan,
                                                 compressed_size,
                                                 temp_buffer_size));

    DEBUG_LOG("查询压缩尺寸: M=" << M << " N=" << N
                                << " K=" << K
                                << " 压缩后字节数=" << *compressed_size
                                << " 临时缓冲字节数=" << *temp_buffer_size);
}
// C 接口：将二维 int8 权重压缩为 cuSPARSELt 所需的结构化稀疏格式。
// 调用流程概览：
//   Step 1. 取得计划上下文、获取尺寸估计
//   Step 2. 准备临时缓冲区与有效性检查标记
//   Step 3. 在 cuSPARSELt 中设置稀疏矩阵指针，构建临时计划
//   Step 4. 校验 2:4 稀疏性
//   Step 5. 执行压缩并同步结果
//   Step 6. 清理临时资源并返回
extern "C" void bitlinear_compress_weight(int8_t *input_weight,
                                          void *compressed_weight,
                                          void *temp_buffer, int M, int N,
                                          int K, cudaStream_t stream) {
    if (!input_weight || !compressed_weight) {
        throw std::runtime_error(
            "bitlinear_compress_weight: input/output pointer cannot be null");
    }
    // Step 1: 获取全局句柄与计划上下文
    cusparseLtHandle_t handle = get_cusparselt_handle();

    std::lock_guard<std::mutex> call_lock(g_cusparselt_call_mutex);
    MatmulPlanContext &ctx = get_or_create_plan(handle, M, N, K);

    size_t estimated_compressed_size = 0;
    size_t estimated_temp_buffer_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &ctx.plan,
                                                 &estimated_compressed_size,
                                                 &estimated_temp_buffer_size));
    DEBUG_LOG("通用尺寸估计 (Python 使用): 压缩后字节数="
              << estimated_compressed_size << " 临时缓冲字节数="
              << estimated_temp_buffer_size);

    size_t compressed_size = 0;
    size_t temp_buffer_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &ctx.plan,
                                                 &compressed_size,
                                                 &temp_buffer_size));
    DEBUG_LOG("压缩请求: 压缩结果需要=" << compressed_size
              << " 字节, 临时缓冲需要=" << temp_buffer_size << " 字节");

    void *temp_ptr = temp_buffer;  // 允许调用者复用已有缓冲区
    bool temp_allocated = false;   // 若调用方未提供，则内部自动分配
    if (temp_buffer_size > 0 && temp_ptr == nullptr) {
        CHECK_CUDA(cudaMalloc(&temp_ptr, temp_buffer_size));
        temp_allocated = true;
        DEBUG_LOG("自动分配临时缓冲区: 大小=" << temp_buffer_size
                  << " 指针=" << temp_ptr);
        debug_dump_pointer_info("临时缓冲区", temp_ptr);
    }

    int *d_valid = nullptr;  // 用于存储 cuSPARSELt 稀疏性校验结果
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_valid), sizeof(int)));
    DEBUG_LOG("为稀疏性检查分配 d_valid，指针=" << d_valid);

    void *sparse_mat_ptr = static_cast<void *>(input_weight);
    bool attribute_set = false;

    // 辅助 lambda：重置稀疏矩阵指针到 null（抛出错误版本）。指向 matmul 上的“左稀疏 W”
    auto reset_sparse_ptr_throw = [&]() {
        if (!attribute_set) {
            return;
        }
        void *null_ptr = nullptr;
        CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
            &handle, &ctx.matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &null_ptr,
            sizeof(null_ptr)));
        attribute_set = false;
    };

    // 辅助 lambda：重置稀疏矩阵指针到 null（吞异常版本）。指向 matmul 上的“左稀疏 W”
    auto reset_sparse_ptr_nothrow = [&]() {
        if (!attribute_set) {
            return;
        }
        void *null_ptr = nullptr;
        cusparseLtMatmulDescSetAttribute(
            &handle, &ctx.matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &null_ptr,
            sizeof(null_ptr));
        attribute_set = false;
    };

    try {
    // Step 2: 将原始稀疏矩阵指针写入 cuSPARSELt 描述符（挂载到 Matmul 的稀疏左操作数 W）
        CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
            &handle, &ctx.matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
            &sparse_mat_ptr, sizeof(sparse_mat_ptr)));
        attribute_set = true;
        DEBUG_LOG("设置稀疏矩阵原始指针=" << sparse_mat_ptr);
        debug_dump_pointer_info("输入权重", input_weight);

        // Step 3: 直接复用缓存计划与算法配置，无需重新初始化临时对象
        DEBUG_LOG("复用缓存计划执行压缩: plan=" << &ctx.plan
                  << " alg_sel=" << &ctx.alg_sel);
        size_t actual_compressed_size = 0;
        size_t actual_temp_buffer_size = 0;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
            &handle, &ctx.plan, &actual_compressed_size,
            &actual_temp_buffer_size));
        DEBUG_LOG("复用计划的精确尺寸需求: 压缩后字节数="
                  << actual_compressed_size << " 临时缓冲字节数="
                  << actual_temp_buffer_size);
        if (actual_compressed_size > estimated_compressed_size) {
            std::cerr << "[bitnet-compress-debug][FATAL] 精确压缩尺寸 > 通用尺寸估计: "
                      << "estimated=" << estimated_compressed_size
                      << " actual=" << actual_compressed_size << std::endl;
        }
        compressed_size = actual_compressed_size;
        temp_buffer_size = actual_temp_buffer_size;

    // Step 4: 执行 2:4 稀疏性校验（针对左稀疏 W），结果写入 d_valid
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &ctx.matmul,
                                                 input_weight, d_valid, stream));
        DEBUG_LOG("启动 2:4 稀疏性校验");

        int is_valid = 0;
        CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        if (stream) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        DEBUG_LOG("稀疏性校验完成，结果=" << is_valid);

        CHECK_CUDA(cudaFree(d_valid));
        d_valid = nullptr;
        DEBUG_LOG("释放 d_valid 内存");

        if (is_valid != 0) {
            // 校验失败：释放资源并抛出错误
            reset_sparse_ptr_throw();
            if (temp_allocated && temp_ptr) {
                CHECK_CUDA(cudaFree(temp_ptr));
            }
            throw std::runtime_error(
                "bitlinear_compress_weight: matrix violates 2:4 pattern");
        }
        DEBUG_LOG("矩阵通过 2:4 稀疏性校验");

    // Step 5: 提交压缩任务（将 W 压缩为 cuSPARSELt 期望的结构化格式），并在指定 CUDA 流上同步
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &ctx.plan,
                                               input_weight, compressed_weight,
                                               temp_ptr, stream));
        DEBUG_LOG("提交 cuSPARSELt 压缩任务");
        if (stream) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        DEBUG_LOG("压缩任务同步完成");

        // Step 6: 联合清理临时资源，并打印调试信息
        reset_sparse_ptr_throw();

        debug_dump_pointer_info("压缩后权重", compressed_weight);
    } catch (...) {
        // 捕获任意异常：同步当前流、回收资源并重新抛出
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
        reset_sparse_ptr_nothrow();
        if (d_valid) {
            cudaFree(d_valid);
        }
        if (temp_allocated && temp_ptr) {
            cudaFree(temp_ptr);
        }
        DEBUG_LOG("压缩过程中发生异常，资源已尝试回收");
        throw;
    }

    if (temp_allocated && temp_ptr) {
        CHECK_CUDA(cudaFree(temp_ptr));
        temp_ptr = nullptr;
        DEBUG_LOG("释放自动分配的临时缓冲区");
    }

    // 最终安全检查：确认压缩过程没有遗留 CUDA 错误。
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        throw std::runtime_error(
            "bitlinear_compress_weight: CUDA error detected after compress");
    }
    DEBUG_LOG("权重压缩操作成功完成");
}
