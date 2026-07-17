/*
nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared bitgemm_sparse.cu -lcuda -lcusparseLt -gencode=arch=compute_80,code=compute_80 -o libbitnet_sparse.so
 * BitNet CUDA GEMM 推理内核 - cuSPARSELt 优化版 稀疏W[N,K]在左，激活A[M,K]在右
 * 注意！这里得到的GEMM结果是转置后的行主序结果 R[N,M]，而不是推理预期的 R[M,N]，需要外部转置
 * 
 * 本文件实现了 BitNet 模型中核心的矩阵乘法 (GEMM) 操作，专门为推理过程设计。
 * 它利用 NVIDIA 的 cuSPARSELt 库来加速经过 2:4 结构化稀疏压缩的权重矩阵
 * 与输入激活矩阵之间的乘法。
 * 
 * 主要特点:
 * 1. 高性能: 使用 cuSPARSELt 库，该库为支持稀疏张量核心 (Sparse Tensor Core)
 *    的现代 NVIDIA GPU (如 Ampere 架构及以后) 提供了高度优化的稀疏-稠密矩阵乘法。
 * 2. 仅推理: 此内核只包含执行乘法所必需的逻辑，权重的稀疏化和压缩
 *    假定已在离线阶段完成。
 * 3. 计划缓存: 为了最小化 cuSPARSELt API 的调用开销，内核会为每个
 *    唯一的矩阵维度 (M, N, K) 创建并缓存一个执行计划 (MatmulPlan)。
 *    这使得重复执行相同形状的乘法时几乎没有初始化成本。
 * 4. 动态工作区管理: 内核能查询并返回 cuSPARSELt 所需的临时工作区
 *    (workspace) 大小，由调用方 (Python) 负责分配和传入。
 * 5. 线程安全: 使用互斥锁 (mutex) 保护全局资源（句柄和缓存），确保在
 *    多线程环境中安全调用。
 */

#include <cusparseLt.h>       // cuSPARSELt 库主头文件，提供结构化稀疏矩阵运算功能
#include <cuda_runtime_api.h> // CUDA 运行时 API，用于设备内存管理等
#include <stdint.h>           // 标准整数类型，如 int8_t, int32_t

#include <cstdlib>            // C 标准库，提供 getenv
#include <cstring>            // C 字符串处理，提供 strcmp
#include <exception>          // C++ 异常处理
#include <iomanip>            // I/O 操纵符，用于格式化输出
#include <iostream>           // C++ 标准输入输出流
#include <map>                // C++ 标准库，用于实现计划缓存
#include <mutex>              // C++ 标准库，用于线程同步
#include <stdexcept>          // C++ 标准异常类
#include <tuple>              // C++ 标准库，用于组合 map 的键
#include <utility>            // C++ 标准库，提供 std::move 等

// 匿名命名空间：限定后续定义的变量和函数仅在本文件内可见，避免与其他文件产生链接冲突。
namespace {

// 控制是否在计划创建阶段执行算法搜索。
static constexpr bool kEnableMatmulSearch = false;
// 调试日志默认开关 (可通过修改此常量或设置环境变量覆盖)。
static constexpr bool kDebugDefaultEnabled = false;
// 可调算法搜索迭代次数（>0 时将覆盖库默认设置，默认为5）。
static constexpr int kMatmulSearchIterations = 20;
// 强制使用的算法 ID（-1自动搜索，>=0 时跳过搜索，直接使用指定算法）。
static constexpr int kForcedAlgConfigId = 6;

// === 调试工具 ===

/**
 * @brief 检查是否启用了调试日志。
 *
 * 通过读取环境变量 `BITNET_CUSPARSE_DEBUG` 来决定。
 * 默认行为是关闭调试，只有当环境变量被设置为非 "0" 的值时才开启。
 * 这种设计允许在不重新编译代码的情况下，在运行时动态控制日志输出。
 *
 * @return 如果应打印调试信息，则返回 true，否则返回 false。
 */
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

/**
 * @brief 调试日志宏。
 *
 * 这是一个方便的宏，只有在 `bitnet_debug_enabled()` 返回 true 时，
 * 才会将消息打印到标准输出。这避免了在生产环境中产生不必要的日志输出，
 * 同时也避免了在代码中到处写 `if (bitnet_debug_enabled())` 的判断。
 */
#define DEBUG_LOG(msg)                                                         \
    do {                                                                       \
        if (bitnet_debug_enabled()) {                                          \
            std::cout << "[bitnet-infer-debug] " << msg << std::endl;         \
        }                                                                      \
    } while (0)

/**
 * @brief 打印有关 GPU 内存使用情况的调试信息。
 * @param stage 描述当前所处阶段的字符串，例如 "调用前" 或 "调用后"。
 */
static void debug_dump_mem_info(const char *stage) {
    if (!bitnet_debug_enabled()) {
        return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::cout << "[bitnet-infer-debug] " << stage
                  << " cudaMemGetInfo failed: " << cudaGetErrorString(err)
                  << " (" << err << ")" << std::endl;
        return;
    }
    std::cout << "[bitnet-infer-debug] " << stage << " free=" << free_bytes
              << " total=" << total_bytes << std::endl;
}

/**
 * @brief 同步设备并报告任何错误。
 *
 * `cudaDeviceSynchronize()` 会阻塞主机线程，直到设备上所有先前提交的
 * CUDA 任务（包括内核执行、内存拷贝等）都完成。
 * 这在调试时非常有用，可以确保在检查点处，所有异步操作都已结束，
 * 从而可以准确地检查其结果或捕获之前发生的错误。
 *
 * @param stage 描述当前所处阶段的字符串。
 */
static void debug_sync_device(const char *stage) {
    if (!bitnet_debug_enabled()) {
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "[bitnet-infer-debug] cudaDeviceSynchronize(" << stage
                  << ") failed: " << cudaGetErrorString(err) << " (" << err
                  << ")" << std::endl;
    } else {
        std::cout << "[bitnet-infer-debug] cudaDeviceSynchronize(" << stage
                  << ") success" << std::endl;
    }
}

/**
 * @brief 检查并报告最近一次的异步 CUDA 错误。
 *
 * `cudaPeekAtLastError()` 会返回由任何线程、任何流上的任何 CUDA 调用
 * 产生的最后一个错误，但不会清除该错误状态。这对于在不阻塞执行
 * (不像 `cudaDeviceSynchronize`) 的情况下快速检查是否有问题发生很有用。
 *
 * @param stage 描述当前所处阶段的字符串。
 */
static void debug_report_pending_error(const char *stage) {
    if (!bitnet_debug_enabled()) {
        return;
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cout << "[bitnet-infer-debug] pending CUDA error at " << stage
                  << ": " << cudaGetErrorString(err) << " (" << err << ")"
                  << std::endl;
    }
}

// === API 错误检查宏 ===

/**
 * @brief 检查 cuSPARSELt API 调用的返回状态。
 *
 * 如果调用返回的 `cusparseStatus_t` 不是 `CUSPARSE_STATUS_SUCCESS`，
 * 这个宏会打印详细的错误信息（包括文件名、行号、函数名和错误码），
 * 然后抛出一个 `std::runtime_error` 异常，中断程序执行。
 * 这种做法确保了 API 的任何失败都会被立即捕获，防止程序在错误状态下继续运行。
 */
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

/**
 * @brief 检查 CUDA 运行时 API 调用的返回状态。
 *
 * 类似于 `CHECK_CUSPARSE`，但用于检查 `cudaError_t` 类型的返回值。
 * 如果调用失败，它会使用 `cudaGetErrorString` 将错误码转换为可读的
 * 错误信息，并抛出异常。
 */
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


// === 全局资源管理 ===
// 为了避免每次调用都创建和销毁 cuSPARSELt 句柄和矩阵乘法计划，
// 我们使用全局变量来持有这些资源，并在程序生命周期内复用它们。
// 这对于性能至关重要，因为这些对象的创建开销很大。

// 全局 cuSPARSELt 句柄。它是所有 cuSPARSELt API 调用的入口点。
static cusparseLtHandle_t g_cusparselt_handle;
// 标记句柄是否已初始化。
static bool g_handle_initialized = false;
// 用于保护句柄初始化的互斥锁，确保多线程安全。
static std::mutex g_handle_mutex;
// 用于串行化对 cuSPARSELt API 的调用，特别是在操作共享计划时，提供额外的保护。
static std::mutex g_cusparselt_call_mutex;

/**
 * @brief 用于缓存矩阵乘法计划的键。
 *
 * 矩阵乘法计划与矩阵的维度 (M, N, K) 紧密相关。
 * 我们使用这个结构体作为 std::map 的键，来唯一标识一个乘法配置。
 */
struct MatmulPlanKey {
    int m;
    int n;
    int k;

    // 重载小于操作符，使得 MatmulPlanKey 可以被用作 std::map 的键。
    bool operator<(const MatmulPlanKey &other) const {
        return std::tie(m, n, k) < std::tie(other.m, other.n, other.k);
    }
};

/**
 * @brief 存储一个完整的矩阵乘法计划及其相关资源。
 *
 * 这个结构体封装了执行一次特定维度 (M, N, K) 的稀疏矩阵乘法所需的所有
 * cuSPARSELt 对象。缓存这个结构体可以避免在每次调用时重复创建这些对象。
 */
struct MatmulPlanContext {
    cusparseLtMatDescriptor_t matW_left{};      // 左操作数：结构化稀疏权重矩阵 W 的描述符
    cusparseLtMatDescriptor_t matA_right{};     // 右操作数：稠密输入激活矩阵 A 的描述符
    cusparseLtMatDescriptor_t matR{};           // 输出矩阵 R 的描述符（行主序[N,M]，和期望的形状不同）
    cusparseLtMatmulDescriptor_t matmul{};      // 矩阵乘法操作本身的描述符
    cusparseLtMatmulAlgSelection_t alg_sel{};   // 所选算法的描述符
    cusparseLtMatmulPlan_t plan{};              // 最终的执行计划
    size_t workspace_size = 0;                  // 此计划需要的临时工作区大小 (字节)
    bool initialized = false;                   // 标记此上下文是否已成功初始化
    bool search_completed = false;              // 标记是否已完成算法搜索
};

// 全局的计划缓存，从 MatmulPlanKey 映射到 MatmulPlanContext。
static std::map<MatmulPlanKey, MatmulPlanContext> g_matmul_plan_cache;
// 保护计划缓存的互斥锁，确保线程安全地访问和修改缓存。
static std::mutex g_matmul_plan_mutex;
// 标记是否已注册了程序退出时的清理函数。
static bool g_plan_cleanup_registered = false;

/**
 * @brief 销毁一个 MatmulPlanContext 中的所有 cuSPARSELt 对象。
 *
 * 这个函数负责释放与一个特定计划相关的所有 GPU 资源。
 * 它必须在程序退出前被调用，以防止内存泄漏。
 *
 * @param ctx 要销毁的计划上下文。
 */
static void destroy_plan(MatmulPlanContext &ctx) {
    if (!ctx.initialized) {
        return;
    }
    // 按照创建的反向顺序销毁对象
    cusparseLtMatmulPlanDestroy(&ctx.plan);
    cusparseLtMatmulAlgSelectionDestroy(&ctx.alg_sel);
    cusparseLtMatDescriptorDestroy(&ctx.matA_right);
    cusparseLtMatDescriptorDestroy(&ctx.matW_left);
    cusparseLtMatDescriptorDestroy(&ctx.matR);
    ctx.workspace_size = 0;
    ctx.initialized = false;
}

/**
 * @brief 清理所有已缓存的矩阵乘法计划。
 *
 * 这个函数会遍历全局计划缓存，并销毁其中的每一个计划。
 * 它将在程序退出时被自动调用。
 */
static void cleanup_all_plans() {
    std::lock_guard<std::mutex> lock(g_matmul_plan_mutex);
    for (auto &kv : g_matmul_plan_cache) {
        destroy_plan(kv.second);
    }
    g_matmul_plan_cache.clear();
}

/**
 * @brief 注册一个在程序退出时执行的清理函数。
 *
 * 使用 `std::atexit` 来确保 `cleanup_all_plans` 函数会在 `main` 函数
 * 结束后被调用。这是一种可靠的资源回收机制。
 * `g_plan_cleanup_registered` 标志确保这个注册操作只执行一次。
 */
static void register_plan_cleanup_once() {
    if (!g_plan_cleanup_registered) {
        std::atexit([]() { cleanup_all_plans(); });
        g_plan_cleanup_registered = true;
    }
}

/**
 * @brief 获取全局的 cuSPARSELt 句柄 (线程安全的单例模式)。
 *
 * 第一次调用此函数时，它会初始化一个全局的 `cusparseLtHandle_t`，
 * 并注册一个清理函数以便在程序退出时销毁该句柄。
 * 后续的调用将直接返回已创建的句柄。
 *
 * @return 对全局 cuSPARSELt 句柄的引用。
 */
static cusparseLtHandle_t get_cusparselt_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_initialized) {
        cusparseStatus_t status = cusparseLtInit(&g_cusparselt_handle);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize cuSPARSELt handle, status="
                      << status << std::endl;
            throw std::runtime_error("cuSPARSELt handle creation failed");
        }
        g_handle_initialized = true;
        std::atexit([]() {
            if (g_handle_initialized) {
                cusparseLtDestroy(&g_cusparselt_handle);
                g_handle_initialized = false;
            }
        });
    }
    return g_cusparselt_handle;
}

/**
 * @brief 获取或创建一个用于特定维度 (M, N, K) 的矩阵乘法计划。
 *
 * 这是此内核的核心优化所在。函数首先检查全局缓存中是否已存在
 * 对应维度的计划。如果存在，则直接返回；如果不存在，则执行以下步骤：
 * 1. 创建所有必需的描述符 (matW_left, matA_right, matR)。
 * 2. 配置矩阵乘法描述符 (matmul)。
 * 3. 选择一个合适的算法 (alg_sel)。
 * 4. 基于以上所有信息，创建一个最终的执行计划 (plan)。
 * 5. 查询该计划所需的临时工作区大小。
 * 6. 将新创建的计划存入缓存，以备将来使用。
 *
 * @param handle 全局 cuSPARSELt 句柄。
 * @param M W=[N,K]
 * @param N A=[M,K]
 * @param K R=[N,M]
 * @return 对缓存中 MatmulPlanContext 的引用。
 */
static MatmulPlanContext &get_or_create_plan(
    cusparseLtHandle_t handle, int M, int N, int K,
    const int8_t *sample_activation = nullptr, const void *sample_weight = nullptr,
    int32_t *sample_output = nullptr, void *workspace = nullptr,
    cudaStream_t stream = nullptr, bool allow_search = false) {
    // 计划缓存的键由矩阵维度 (M, N, K) 唯一确定。
    MatmulPlanKey key{M, N, K};

    MatmulPlanContext *ctx_ptr = nullptr;
    bool need_search = false;

    {
        std::unique_lock<std::mutex> lock(g_matmul_plan_mutex);
        // 1. 先尝试在缓存中查找现有计划。
        auto iter = g_matmul_plan_cache.find(key);
        if (iter != g_matmul_plan_cache.end()) {
            ctx_ptr = &iter->second;
            DEBUG_LOG("plan cache hit: M=" << M << " N=" << N << " K=" << K);
        } else {
            // 2. 缓存未命中时，准备创建新的计划上下文。
            register_plan_cleanup_once();

            auto insert_ret = g_matmul_plan_cache.try_emplace(key);
            MatmulPlanContext &ctx = insert_ret.first->second;

            // --- 以下标志用于确保异常发生时能够安全地销毁已创建的资源 ---
            bool matW_ready = false;
            bool matA_ready = false;
            bool matR_ready = false;
            bool alg_sel_ready = false;
            bool plan_ready = false;
            bool force_algorithm = false;

            DEBUG_LOG("creating new plan: M=" << M << " N=" << N << " K=" << K);

            // --- 计划的基础配置 ---
            auto order_row = CUSPARSE_ORDER_ROW;              // 行主序布局（W,A,R）
            auto order_col = CUSPARSE_ORDER_COL;              // 列主序布局
            auto opW = CUSPARSE_OPERATION_NON_TRANSPOSE;      // 左操作数 W 不转置
            auto opA = CUSPARSE_OPERATION_TRANSPOSE;          // 右操作数 A 在乘法前转置
            int ldW = K;                                      // leading dimension
            int ldA = K;                                      // leading dimension
            int ldR = M;                                      // leading dimension
            unsigned alignment = 16;                          // 内存对齐要求

            try {
                // 3. 初始化稀疏权重矩阵 W 的描述符 (结构化稀疏 2:4，左操作数)。
                DEBUG_LOG("正在调用 cusparseLtStructuredDescriptorInit (matW_left)...");
                CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                    &handle, &ctx.matW_left, N, K, ldW, alignment, CUDA_R_8I,
                    order_row, CUSPARSELT_SPARSITY_50_PERCENT));
                DEBUG_LOG("cusparseLtStructuredDescriptorInit (matW_left) 调用成功");
                matW_ready = true;

                // 6. 初始化输入矩阵 A 的描述符 (稠密，右操作数)。
                DEBUG_LOG("正在调用 cusparseLtDenseDescriptorInit (matA_right)...");
                CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
                    &handle, &ctx.matA_right, M, K, ldA, alignment, CUDA_R_8I,
                    order_row));
                DEBUG_LOG("cusparseLtDenseDescriptorInit (matA_right) 调用成功");
                matA_ready = true;

                // 7. 初始化输出矩阵 R 的描述符 (行主序视图, int32 accumulation)。
                DEBUG_LOG("正在调用 cusparseLtDenseDescriptorInit (matR)...");
                CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
                    &handle, &ctx.matR, N, M, ldR, alignment, CUDA_R_32I,
                    order_row));
                DEBUG_LOG("cusparseLtDenseDescriptorInit (matR) 调用成功");
                matR_ready = true;

                // 8. 构建矩阵乘法描述符，将 W/A/R 描述符与 opW/opA 组合起来。
                DEBUG_LOG("正在调用 cusparseLtMatmulDescriptorInit...");
                CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
                    &handle, &ctx.matmul, opW, opA, &ctx.matW_left,
                    &ctx.matA_right, &ctx.matR, &ctx.matR,
                    CUSPARSE_COMPUTE_32I));
                DEBUG_LOG("cusparseLtMatmulDescriptorInit 调用成功");

                // 9. 初始化算法选择结构，默认让库给出一个可行算法。
                DEBUG_LOG("正在初始化调用 cusparseLtMatmulAlgSelectionInit...");
                CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
                    &handle, &ctx.alg_sel, &ctx.matmul,
                    CUSPARSELT_MATMUL_ALG_DEFAULT));
                DEBUG_LOG("cusparseLtMatmulAlgSelectionInit 调用成功");
                alg_sel_ready = true;

                int search_iterations = kMatmulSearchIterations;
                if (search_iterations > 0) {
                    DEBUG_LOG("setting matmul search iterations to "
                              << search_iterations);
                    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                        &handle, &ctx.alg_sel,
                        CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
                        &search_iterations, sizeof(search_iterations)));
                }

                int forced_alg_id = kForcedAlgConfigId;
                if (forced_alg_id >= 0) {
                    DEBUG_LOG("forcing matmul algorithm id to "
                              << forced_alg_id);
                    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                        &handle, &ctx.alg_sel,
                        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                        &forced_alg_id, sizeof(forced_alg_id)));
                    force_algorithm = true;
                }

                // 10. 基于描述符和算法选择创建最终的执行计划。
                DEBUG_LOG("正在调用 cusparseLtMatmulPlanInit...");
                CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
                    &handle, &ctx.plan, &ctx.matmul, &ctx.alg_sel));
                DEBUG_LOG("cusparseLtMatmulPlanInit 调用成功");
                plan_ready = true;

                // 11. 查询执行计划是否需要额外的 workspace，并记录下来。
                ctx.workspace_size = 0;
                cusparseStatus_t ws_status = cusparseLtMatmulGetWorkspace(
                    &handle, &ctx.plan, &ctx.workspace_size);
                if (ws_status != CUSPARSE_STATUS_SUCCESS) {
                    if (ws_status == CUSPARSE_STATUS_NOT_SUPPORTED) {
                        ctx.workspace_size = 0;
                    } else {
                        DEBUG_LOG("正在调用 CHECK_CUSPARSE 以处理 cusparseLtMatmulGetWorkspace 返回码...");
                        CHECK_CUSPARSE(ws_status);
                        DEBUG_LOG("CHECK_CUSPARSE 已成功处理 cusparseLtMatmulGetWorkspace 返回码");
                    }
                }

                ctx.initialized = true;
                ctx.search_completed = force_algorithm;
                DEBUG_LOG("plan created, workspace=" << ctx.workspace_size);
            } catch (...) {
                // 如果任一步骤失败，需要回滚已创建的所有资源，并移除缓存条目。
                if (plan_ready) {
                    cusparseLtMatmulPlanDestroy(&ctx.plan);
                }
                if (alg_sel_ready) {
                    cusparseLtMatmulAlgSelectionDestroy(&ctx.alg_sel);
                }
                if (matR_ready) {
                    cusparseLtMatDescriptorDestroy(&ctx.matR);
                }
                if (matA_ready) {
                    cusparseLtMatDescriptorDestroy(&ctx.matA_right);
                }
                if (matW_ready) {
                    cusparseLtMatDescriptorDestroy(&ctx.matW_left);
                }
                ctx.workspace_size = 0;
                ctx.initialized = false;
                g_matmul_plan_cache.erase(insert_ret.first);
                DEBUG_LOG("plan creation failed; entry removed from cache");
                throw;
            }

            ctx_ptr = &ctx;
        }

        bool search_allowed = allow_search && kEnableMatmulSearch &&
                               (kForcedAlgConfigId < 0);
        if (search_allowed && ctx_ptr && !ctx_ptr->search_completed) {
            // 11. 如果需要执行算法搜索，必须确保传入了有效的示例数据指针。
            if (!sample_activation || !sample_weight || !sample_output) {
                throw std::invalid_argument(
                    "matmul search requires valid activation, weight, and output pointers");
            }
            if (ctx_ptr->workspace_size > 0 && workspace == nullptr) {
                throw std::invalid_argument(
                    "workspace pointer must be provided before matmul search runs");
            }
            // 在锁释放后执行搜索，避免长时间持有缓存互斥量。
            need_search = true;
        }
    }

    if (need_search && ctx_ptr) {
        // 12. 使用示例数据触发一次算法搜索，得到针对该维度的最快计划。
        float alpha = 1.0f; // 乘积项的缩放因子
        float beta = 0.0f;  // 原始输出矩阵的贡献因子 (0 表示不累加)
        cudaStream_t streams[1] = {stream};
        int num_streams = (stream != nullptr) ? 1 : 0;
        void *search_workspace = (ctx_ptr->workspace_size > 0) ? workspace : nullptr;
        size_t workspace_before = ctx_ptr->workspace_size;
        
        DEBUG_LOG("running cusparseLtMatmulSearch for plan (workspace_before="
                  << workspace_before << ")");
        CHECK_CUSPARSE(cusparseLtMatmulSearch(
            &handle, &ctx_ptr->plan, &alpha, sample_weight, sample_activation,
            &beta, sample_output, sample_output, search_workspace,
            num_streams ? streams : nullptr, num_streams));

        int tuned_alg_id = -1;
        int alg_space = 0;
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(
            &handle, &ctx_ptr->alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &tuned_alg_id, sizeof(tuned_alg_id)));
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(
            &handle, &ctx_ptr->alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
            &alg_space, sizeof(alg_space)));
        DEBUG_LOG("matmul search tuned algorithm id=" << tuned_alg_id
                                                       << ", candidate_count="
                                                       << alg_space);

        // 基于搜索到的算法选择创建最终的执行计划。
        DEBUG_LOG("正在调用搜索后的 cusparseLtMatmulPlanInit...");
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
                &handle, &ctx_ptr->plan, &ctx_ptr->matmul, &ctx_ptr->alg_sel));
        DEBUG_LOG("cusparseLtMatmulPlanInit 调用成功");

        size_t workspace_after = workspace_before;
        cusparseStatus_t ws_status = cusparseLtMatmulGetWorkspace(
            &handle, &ctx_ptr->plan, &workspace_after);
        if (ws_status != CUSPARSE_STATUS_SUCCESS) {
            if (ws_status == CUSPARSE_STATUS_NOT_SUPPORTED) {
                workspace_after = 0;
            } else {
                CHECK_CUSPARSE(ws_status);
            }
        }

        {
            // 13. 标记计划已经完成算法搜索，后续直接复用。
            std::lock_guard<std::mutex> lock(g_matmul_plan_mutex);
            ctx_ptr->workspace_size = workspace_after;
            ctx_ptr->search_completed = true;
        }
        DEBUG_LOG("matmul search completed (workspace_before="
                  << workspace_before << ", workspace_after="
                  << workspace_after << ")");
    } else if (ctx_ptr && kForcedAlgConfigId >= 0) {
        DEBUG_LOG("matmul search skipped; using forced algorithm id "
                  << kForcedAlgConfigId);
    }

    return *ctx_ptr;
}

}  // namespace

// === C 语言外部接口 ===
// extern "C" 确保了这些函数在被编译后，其符号名不会被 C++ 的 name mangling
// 改变，从而可以被 Python 的 ctypes 等工具通过原始函数名找到。

/**
 * @brief 查询执行特定维度 GEMM 所需的临时工作区大小。
 *
 * 这个函数允许调用方 (Python) 在执行实际的矩阵乘法之前，提前知道
 * 需要分配多大的 GPU 内存作为临时缓冲区。
 *
 * @param M W=[N,K] (Row)
 * @param N A=[M,K] (Row)
 * @param K R=[N,M] (Row)
 * @param workspace_size [out] 用于接收所需工作区大小 (字节) 的指针。
 */
extern "C" void bitlinear_get_workspace_size(int M, int N, int K,
                                              size_t *workspace_size) {
    if (!workspace_size) {
        throw std::invalid_argument("workspace_size pointer must not be null");
    }

    cusparseLtHandle_t handle = get_cusparselt_handle();
    MatmulPlanContext &ctx = get_or_create_plan(handle, M, N, K);

    *workspace_size = ctx.workspace_size;
}

/**
 * @brief 执行 int8 输入与 int8 压缩权重的结构化稀疏矩阵乘法。
 *
 * 这是本内核的核心功能函数。它接收输入激活、预压缩的权重，并计算出int32 精度的输出结果。
 *
 * N/T乘法，Row/Row/Row主序
 * 计算公式: 逻辑上输出 R = W * Aᵀ（全部为行主序，因为这样才能使用cuSPARSElt最快的kernel）
 * 输出的R形状为[N,M]，需要在后面使用RowMajor读出为正确形状，或者显式转置
 * alpha=1.0, beta=0.0（均为float，cusparseLt特别要求这两个系数不能是int类型）
 *
 * @param input_activation  [in] 指向输入激活矩阵 (A) 的 GPU 指针。维度 (M, K)，int8，行主序。
 * @param compressed_weight [in] 指向预压缩权重矩阵 (W) 的 GPU 指针。原始维度 (N, K)，int8，压缩前为行主序。
 * @param output_result     [out] 指向输出结果矩阵 (R) 的 GPU 指针。维度 (N, M)，int32，行主序。
 * @param M W=[N,K] (Row)
 * @param N A=[M,K] (Row)
 * @param K R=[N,M] (Row)
 * @param workspace         [in] 指向临时工作区的 GPU 指针。其大小应通过 `bitlinear_get_workspace_size` 查询。
 * @param stream            [in] 用于执行此操作的 CUDA 流。
 */
extern "C" void bitlinear_int8_GEMM_sparse(const int8_t *input_activation,
                                     const void *compressed_weight,
                                     int32_t *output_result,
                                     int M, int N, int K, void *workspace,
                                     cudaStream_t stream) {
    // --- 输入参数校验 ---
    if (!input_activation || !compressed_weight || !output_result) {
        throw std::invalid_argument(
            "input_activation, compressed_weight, and output_result must be non-null");
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("matrix dimensions must be positive");
    }

    // 1. 获取全局句柄和对应维度的计划
    cusparseLtHandle_t handle = get_cusparselt_handle();
    DEBUG_LOG("bitlinear_int8_GEMM_sparse launch (sparse-left) M=" << M << " N=" << N
                                                            << " K=" << K
                                                            << " stream="
                                                            << stream);

    debug_report_pending_error("before bitlinear_int8_GEMM_sparse");

    std::lock_guard<std::mutex> call_lock(g_cusparselt_call_mutex);
    MatmulPlanContext &ctx = get_or_create_plan(handle, M, N, K,
                                                input_activation,
                                                compressed_weight,
                                                output_result, workspace,
                                                stream, true);

    // 2. 校验工作区
    if (ctx.workspace_size > 0 && workspace == nullptr) {
        throw std::invalid_argument(
            "workspace pointer must be provided for this (M,N,K) combination");
    }

    // 3. 设置标量参数(必须是float)
    float alpha = 1.0f; // 乘积项的缩放因子
    float beta = 0.0f;  // 原始输出矩阵的贡献因子 (0 表示不累加)

    // 4. 执行矩阵乘法，注意W在左，A在右
    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &ctx.plan, &alpha,
                                    compressed_weight, input_activation, &beta,
                                    output_result, output_result, workspace,
                                    stream ? &stream : nullptr, stream ? 1 : 0));

    // 在调试模式下，可以取消注释下面的行来强制同步并检查错误，但这会影响性能。
    //debug_sync_device("bitlinear_int8_GEMM_sparse");
    //debug_report_pending_error("after bitlinear_int8_GEMM_sparse");

    // 5. 检查异步错误
    //    由于 CUDA 操作是异步的，`cusparseLtMatmul` 会立即返回。
    //    `cudaGetLastError` 可以捕获在之前提交到流中的任何操作（包括这个 matmul）
    //    所产生的错误。虽然它不能保证 matmul 已经完成，但如果之前的 API 调用
    //    有参数错误等问题，这里通常能检测到。
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "bitlinear_int8_GEMM_sparse: CUDA error detected: "
                  << cudaGetErrorString(cuda_error) << std::endl;
        throw std::runtime_error("CUDA error detected during matmul");
    }

}
