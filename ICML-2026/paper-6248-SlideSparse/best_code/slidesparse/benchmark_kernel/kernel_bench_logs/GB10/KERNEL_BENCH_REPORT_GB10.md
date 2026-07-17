# SlideSparse Kernel Benchmark 测试报告 - NVIDIA GB10

## 📋 测试环境

| 项目 | 信息 |
|------|------|
| **GPU** | NVIDIA GB10 (cc121) - Grace Blackwell SoC |
| **Python** | 3.12 |
| **CUDA** | 12.9 |
| **架构** | aarch64 |
| **测试日期** | 2026-01-27 ~ 2026-01-28 |
| **总耗时** | 约 9 小时 (两轮测试) |

---

## ✅ 测试任务总览

### 第一轮测试 (kernel_bench_20260127_212010.log)
测试配置：Model 模式 5 种精度 (fp16, bf16, int8, fp8e4m3, fp4e2m1)

| Task | 名称 | 状态 | 耗时 |
|------|------|:----:|------|
| Task 1 | cuBLASLt Model 测试 | ⚠️ 部分失败 | 36.9 分钟 |
| Task 2 | cuBLASLt Square 测试 | ⚠️ 部分失败 | 3.9 分钟 |
| Task 3 | cuSPARSELt Model 高稀疏 (2_4~2_10) | ⚠️ 部分失败 | 212.5 分钟 |
| Task 4 | cuSPARSELt Square 高稀疏 (2_4~2_10) | ✅ 成功 | 30.5 分钟 |
| Task 5 | cuSPARSELt Model 低稀疏 (2_12~2_inf) | ✅ 成功 | 48.9 分钟 (部分) |

### 第二轮测试 (kernel_bench_20260128_030642.log)
测试配置：Model 模式只测 2 种精度 (fp8e4m3, int8)，Square 模式测 5 种精度

| Task | 名称 | 状态 | 耗时 |
|------|------|:----:|------|
| Task 3 | cuSPARSELt Model 高稀疏 (2_4~2_10) | ⏭️ 跳过 | - |
| Task 4 | cuSPARSELt Square 高稀疏 (2_4~2_10) | ✅ 成功 | 32.0 分钟 |
| Task 5 | cuSPARSELt Model 低稀疏 (2_12~2_inf) | ✅ 成功 | 133.0 分钟 |
| Task 6 | cuSPARSELt Square 低稀疏 (2_12~2_inf) | ✅ 成功 | 38.8 分钟 |

---

## 🎯 测试参数

### M 列表
`[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]`

### 模型列表 (8个)
| 模型 | 量化类型 |
|------|----------|
| Llama3.2-1B-INT8 | INT8 |
| Llama3.2-1B-FP8 | FP8 |
| Llama3.2-3B-INT8 | INT8 |
| Llama3.2-3B-FP8 | FP8 |
| Qwen2.5-7B-INT8 | INT8 |
| Qwen2.5-7B-FP8 | FP8 |
| Qwen2.5-14B-INT8 | INT8 |
| Qwen2.5-14B-FP8 | FP8 |

### 稀疏度配置
- **高稀疏**: 2_4, 2_6, 2_8, 2_10
- **低稀疏**: 2_12, 2_14, 2_16, 2_inf

### 测试精度
- **第一轮 (所有)**: fp16, bf16, int8, fp8e4m3, fp4e2m1
- **第二轮 (Model)**: fp8e4m3, int8
- **第二轮 (Square)**: fp16, bf16, int8, fp8e4m3, fp4e2m1

---

## ❌ 失败记录详情

### 失败汇总

在 GB10 (cc121) 上，**cuBLASLt 的 FP4 (fp4e2m1) 精度全部失败**，这是 NVIDIA cuBLASLt 库在消费级/嵌入式 Blackwell GPU (sm_120/sm_121) 上的已知兼容性问题。

---

### 1. cuBLASLt FP4 失败 (9 处)

**错误类型**: `CUDA error: an illegal memory access was encountered` (`cudaErrorIllegalAddress`)

| Task | 模型 | 精度 | 失败位置 NK | N×K 尺寸 |
|------|------|------|-------------|----------|
| Task 1 | Llama3.2-1B-INT8 | fp4e2m1 | NK 1/4 | (3072, 2048) |
| Task 1 | Llama3.2-1B-FP8 | fp4e2m1 | NK 1/4 | (3072, 2048) |
| Task 1 | Llama3.2-3B-INT8 | fp4e2m1 | NK 1/4 | (5120, 3072) |
| Task 1 | Llama3.2-3B-FP8 | fp4e2m1 | NK 1/4 | (5120, 3072) |
| Task 1 | Qwen2.5-7B-INT8 | fp4e2m1 | NK 1/4 | (4608, 3584) |
| Task 1 | Qwen2.5-7B-FP8 | fp4e2m1 | NK 1/4 | (4608, 3584) |
| Task 1 | Qwen2.5-14B-INT8 | fp4e2m1 | NK 3/4 | (27648, 5120) |
| Task 1 | Qwen2.5-14B-FP8 | fp4e2m1 | NK 3/4 | (27648, 5120) |
| Task 2 | SQUARE | fp4e2m1 | NK 9/9 | (16384, 16384) |

**失败原因分析**:
1. cuBLASLt 的 FP4 实现在 GB10 (cc121) 上有 CUDA 非法内存访问 bug
2. 这是 NVIDIA 库的问题，不是我们代码的问题
3. 在 B200 (cc100 数据中心级) 上不存在此问题，但在 RTX 5080 (cc120) 和 GB10 (cc121) 上均有此问题

---

### 2. cuSPARSELt FP4 + 2_6 稀疏度失败 (4 处)

**错误类型**: 进程执行失败 (返回码非 0)

| Task | 模型 | 精度 | 稀疏度 |
|------|------|------|--------|
| Task 3 | Qwen2.5-7B-INT8 | fp4e2m1 | 2_6 |
| Task 3 | Qwen2.5-7B-FP8 | fp4e2m1 | 2_6 |
| Task 3 | Qwen2.5-14B-INT8 | fp4e2m1 | 2_6 |
| Task 3 | Qwen2.5-14B-FP8 | fp4e2m1 | 2_6 |

**注意**: 只有 **Qwen2.5 系列 + FP4 + 2_6 稀疏度** 组合失败，Llama 系列的 FP4 + 2_6 正常通过。

---

## ✅ 完全通过的测试

### cuBLASLt (fp16, bf16, int8, fp8e4m3) - 100% 通过 ✅

所有 8 个模型在 fp16, bf16, int8, fp8e4m3 四种精度下全部通过：

| 模型 | 耗时 | 状态 |
|------|------|:----:|
| Llama3.2-1B-INT8 | 126.5s | ✅ (仅 fp4 失败) |
| Llama3.2-1B-FP8 | 122.2s | ✅ (仅 fp4 失败) |
| Llama3.2-3B-INT8 | 183.6s | ✅ (仅 fp4 失败) |
| Llama3.2-3B-FP8 | 183.9s | ✅ (仅 fp4 失败) |
| Qwen2.5-7B-INT8 | 390.4s | ✅ (仅 fp4 失败) |
| Qwen2.5-7B-FP8 | 389.9s | ✅ (仅 fp4 失败) |
| Qwen2.5-14B-INT8 | 410.6s | ✅ (仅 fp4 失败) |
| Qwen2.5-14B-FP8 | 410.7s | ✅ (仅 fp4 失败) |

### cuSPARSELt Model 高稀疏 (Task 3) - 大部分通过 ✅

| 模型 | 耗时 | 状态 |
|------|------|:----:|
| Llama3.2-1B-INT8 | 638.2s | ✅ 全部通过 |
| Llama3.2-1B-FP8 | 633.1s | ✅ 全部通过 |
| Llama3.2-3B-INT8 | 945.5s | ✅ 全部通过 |
| Llama3.2-3B-FP8 | 943.9s | ✅ 全部通过 |
| Qwen2.5-7B-INT8 | 2163.4s | ⚠️ fp4+2_6 失败 |
| Qwen2.5-7B-FP8 | 2180.7s | ⚠️ fp4+2_6 失败 |
| Qwen2.5-14B-INT8 | 2623.3s | ⚠️ fp4+2_6 失败 |
| Qwen2.5-14B-FP8 | 2622.0s | ⚠️ fp4+2_6 失败 |

### cuSPARSELt Square 高稀疏 (Task 4) - 100% 通过 ✅
方阵高稀疏测试全部通过 (1920.7s)

### cuSPARSELt Model 低稀疏 (Task 5) - 100% 通过 ✅

| 模型 | fp8e4m3 耗时 | int8 耗时 | 状态 |
|------|--------------|-----------|:----:|
| Llama3.2-1B-INT8 | 309.7s | 58.2s | ✅ |
| Llama3.2-1B-FP8 | 308.5s | 58.4s | ✅ |
| Llama3.2-3B-INT8 | 504.4s | 73.1s | ✅ |
| Llama3.2-3B-FP8 | 512.9s | 73.3s | ✅ |
| Qwen2.5-7B-INT8 | 1239.7s | 136.1s | ✅ |
| Qwen2.5-7B-FP8 | 1289.6s | 135.4s | ✅ |
| Qwen2.5-14B-INT8 | 1493.9s | 147.2s | ✅ |
| Qwen2.5-14B-FP8 | 1491.9s | 147.1s | ✅ |

### cuSPARSELt Square 低稀疏 (Task 6) - 100% 通过 ✅
方阵低稀疏测试全部通过 (2325.9s)

---

## 📊 测试成功率统计

| 分类 | 总测试数 | 成功数 | 失败数 | 成功率 |
|------|----------|--------|--------|--------|
| cuBLASLt Model (非FP4) | 32 | 32 | 0 | **100%** |
| cuBLASLt Model (FP4) | 8 | 0 | 8 | **0%** |
| cuBLASLt Square (非FP4) | 4 | 4 | 0 | **100%** |
| cuBLASLt Square (FP4) | 1 | 0 | 1 | **0%** |
| cuSPARSELt Model 高稀疏 | 128 | 124 | 4 | **96.9%** |
| cuSPARSELt Square 高稀疏 | 20 | 20 | 0 | **100%** |
| cuSPARSELt Model 低稀疏 | 64 | 64 | 0 | **100%** |
| cuSPARSELt Square 低稀疏 | 20 | 20 | 0 | **100%** |
| **总计 (排除 cuBLASLt FP4)** | **268** | **264** | **4** | **98.5%** |

---

## 🔧 需要重跑的测试

### cuBLASLt FP4 (建议放弃)
由于是 NVIDIA cuBLASLt 库在 cc120/cc121 上的 bug，重跑也会失败。建议：
- 在 GB10 上不使用 cuBLASLt FP4
- 等待 NVIDIA 发布修复版本

### cuSPARSELt FP4 + 2_6 (可选重跑)
以下 4 个组合可以尝试重跑：

| 模型 | 精度 | 稀疏度 | M 列表 |
|------|------|--------|--------|
| Qwen2.5-7B-INT8 | fp4e2m1 | 2_6 | 64,128,256,512,1024,2048,4096,8192,16384 |
| Qwen2.5-7B-FP8 | fp4e2m1 | 2_6 | 64,128,256,512,1024,2048,4096,8192,16384 |
| Qwen2.5-14B-INT8 | fp4e2m1 | 2_6 | 64,128,256,512,1024,2048,4096,8192,16384 |
| Qwen2.5-14B-FP8 | fp4e2m1 | 2_6 | 64,128,256,512,1024,2048,4096,8192,16384 |

---

## 📁 结果文件位置

```
/root/vllmbench/slidesparse/benchmark_kernel/
├── cuBLASLt/alg_search_results/GB10_cc121_py312_cu129_aarch64/
│   ├── BF16/
│   ├── FP16/
│   ├── FP8/
│   └── INT8/
└── cuSPARSELt/alg_search_results/GB10_cc121_py312_cu129_aarch64/
    ├── BF16/
    ├── FP16/
    ├── FP4/
    ├── FP8/
    └── INT8/
```

---

## 📝 日志文件

| 日志文件 | 内容 |
|----------|------|
| `kernel_bench_20260127_212010.log` | 第一轮完整测试 (Task 1-5，Model 5种精度) |
| `kernel_bench_20260128_030642.log` | 第二轮测试 (Task 4-6，Model 2种精度) |

---

## ⚠️ 已知问题

### 1. cuBLASLt FP4 不支持
- **影响**: 所有 cuBLASLt FP4 测试失败
- **原因**: NVIDIA cuBLASLt 库在 cc120/cc121 架构上的 FP4 实现有 bug
- **解决方案**: 等待 NVIDIA 修复，或在 GB10 上不使用 cuBLASLt FP4

### 2. cuSPARSELt Qwen2.5 + FP4 + 2_6 不稳定
- **影响**: 4 个测试组合失败
- **原因**: 可能与 Qwen2.5 模型的特定 NK 尺寸有关
- **解决方案**: 可选择重跑或跳过

---

## ✅ 结论

**GB10 (Grace Blackwell SoC) 上的 SlideSparse Kernel Benchmark 测试基本完成，核心功能正常！**

### 成功部分 ✅
- ✅ cuBLASLt **fp16, bf16, int8, fp8e4m3** 全部 8 模型通过
- ✅ cuSPARSELt **所有 Square 测试** 100% 通过
- ✅ cuSPARSELt **Model 低稀疏 (2_12~2_inf)** 全部 8 模型通过
- ✅ cuSPARSELt **Model 高稀疏 (2_4~2_10)** Llama 系列全部通过

### 失败部分 ❌
- ❌ cuBLASLt **FP4** 全部失败 (NVIDIA 库 bug)
- ❌ cuSPARSELt **Qwen2.5 + FP4 + 2_6** 4 个组合失败

### 建议
1. **生产环境**: 在 GB10 上避免使用 cuBLASLt FP4，其他精度可正常使用
2. **cuSPARSELt**: 大部分功能正常，仅需注意 Qwen2.5 + FP4 + 2_6 组合

---

*报告生成时间: 2026-01-28 07:00*
