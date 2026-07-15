# APPFL MPI Memory Profiling and Optimization [Experimental]

This directory contains tools for memory profiling and optimization of APPFL's MPI-based federated learning components using memray.

## Quick Start

1. **Install memray**:
   ```bash
   pip install memray
   ```

2. **Run memory profiling experiments**:

   **MPI MNIST experiment** (recommended - lightweight CNN model):
   ```bash
   cd examples
   chmod +x memory_profiling_mpi/run_mpi_mnist_experiment.sh
   ./memory_profiling_mpi/run_mpi_mnist_experiment.sh
   ```

   **MPI CIFAR-10 experiment** (ResNet18 with real CIFAR-10 data):
   ```bash
   cd examples
   chmod +x memory_profiling_mpi/run_mpi_cifar_experiment.sh
   ./memory_profiling_mpi/run_mpi_cifar_experiment.sh
   ```

   **MPI ResNet experiment** (alternative - ResNet with dummy data for training focus):
   ```bash
   cd examples
   chmod +x memory_profiling_mpi/run_mpi_resnet_experiment.sh
   ./memory_profiling_mpi/run_mpi_resnet_experiment.sh
   ```

3. **Or run experiments manually**:
   ```bash
   cd examples

   # MNIST experiment (CNN model)
   # Original version (without MPI memory optimizations)
   mpiexec -n 3 python memory_profiling_mpi/run_mpi_memray.py \
       --server_config ./resources/configs/mnist/server_fedavg.yaml \
       --client_config ./resources/configs/mnist/client_1.yaml

   # Optimized version (with MPI memory optimizations)
   mpiexec -n 3 python memory_profiling_mpi/run_mpi_memray.py \
       --server_config ./resources/configs/mnist/server_fedavg.yaml \
       --client_config ./resources/configs/mnist/client_1.yaml \
       --use_optimized_version

   # CIFAR-10 experiment (ResNet18 model)
   mpiexec -n 3 python memory_profiling_mpi/run_mpi_memray.py \
       --server_config ./resources/configs/cifar10/server_fedavg.yaml \
       --client_config ./resources/configs/cifar10/client_1.yaml \
       --use_optimized_version
   ```

4. **View results**: Each experiment run creates a timestamped subdirectory with all results:
   - MNIST: `./memory_profiles/mpi_mnist_20240326_143022/`
   - CIFAR-10: `./memory_profiles/mpi_cifar_20240326_143022/`
   - ResNet: `./memory_profiles/mpi_resnet_20240326_143022/`

## Files Overview

- `run_mpi_memray.py` - Memory profiling wrapper for MPI federated learning
- `run_mpi_mnist_experiment.sh` - **MNIST MPI memory profiling experiment (recommended for beginners)**
- `run_mpi_cifar_experiment.sh` - **CIFAR-10 MPI memory profiling experiment (ResNet18 + real data)**
- `run_mpi_resnet_experiment.sh` - ResNet MPI memory profiling experiment (dummy data focus)
- `analyze_mpi_profiles.py` - Basic analysis script for MPI memory profiles (deprecated - use generate_comprehensive_results.py)
- `result_profile_analysis.py` - Advanced result analysis with statistical visualizations
- `generate_comprehensive_results.py` - **NEW: Comprehensive results generator similar to gRPC memory profiling**

## MPI Memory Optimizations Implemented

APPFL now includes built-in MPI memory optimizations that can be enabled with the `--use_optimized_version` flag. These optimizations are integrated directly into the MPI communicator classes.

### Components Optimized

#### 1. MPIServerCommunicator
- **Efficient Serialization**: Memory-optimized model serialization using context managers
- **Strategic Cleanup**: Immediate cleanup of request buffers and response data
- **Garbage Collection**: Periodic GC during message processing
- **CPU-first Loading**: Reduces memory pressure during model operations

#### 2. MPIClientCommunicator
- **Optimized Model Transfer**: Memory-efficient model serialization before sending
- **Resource Cleanup**: Automatic cleanup of communication buffers
- **Memory-aware Deserialization**: CPU-first model loading with immediate cleanup

#### 3. MPI Serializer
- **Context Managers**: Memory-efficient BytesIO handling
- **CPU Loading**: Models loaded to CPU first to reduce memory pressure
- **Automatic GC**: Strategic garbage collection after operations

### Configuration

Memory optimizations are controlled by `optimize_memory` flags:

```yaml
# Enable in server configuration
server_configs:
  optimize_memory: true

# Or pass to communicator constructors
# MPIServerCommunicator(..., optimize_memory=True)
# MPIClientCommunicator(..., optimize_memory=True)
```

Or automatically enabled with `--use_optimized_version` flag in profiling scripts.

### Expected Benefits

- **Reduced Peak Memory Usage**: 15-30% reduction in memory footprint during MPI communication
- **Better Large Model Handling**: Efficient tensor operations for models like ResNet
- **Improved MPI Performance**: Optimized serialization for large parameter transfers
- **Faster Resource Cleanup**: Strategic cleanup reduces memory leaks

### Viewing Memory Profiles

#### Comprehensive Results Generation (Recommended)
**NEW:** Use the comprehensive results generator for complete analysis similar to gRPC memory profiling:

```bash
# Generate complete analysis with flamegraphs, plots, and detailed reports
python memory_profiling_mpi/generate_comprehensive_results.py ./memory_profiles/mpi_cifar_20240326_143022

# With custom output directory
python memory_profiling_mpi/generate_comprehensive_results.py ./memory_profiles/mpi_cifar_20240326_143022 --output-dir ./custom_analysis
```

**Automatically generates:**
- 🔥 **Interactive flamegraphs** for each rank and version
- 📊 **Comparison plots** showing optimization impact
- 📈 **Statistical analysis** with percentage improvements
- 📋 **Summary reports** for each profile
- 🎯 **Role-based analysis** (server vs client memory usage)
- 📝 **Comprehensive text report** with recommendations

#### Advanced Statistical Analysis
Use the statistical analysis script for detailed visualizations and reports:

```bash
# Run advanced statistical analysis with plots and reports (use specific timestamped directory)
python memory_profiling_mpi/result_profile_analysis.py --profiles-dir ./memory_profiles/mpi_cifar_20240326_143022

# Run analysis without plots (for headless environments)
python memory_profiling_mpi/result_profile_analysis.py --profiles-dir ./memory_profiles/mpi_resnet_20240326_144530 --no-plots

# Analyze with custom output directory
python memory_profiling_mpi/result_profile_analysis.py --profiles-dir ./memory_profiles/mpi_cifar_20240326_143022 --output-dir ./custom_analysis
```

#### Manual Analysis
Use memray directly to analyze the generated profiles (note: use `python -m memray` instead of just `memray`):

```bash
# Generate flamegraph for server (rank 0) - use specific timestamped directory
python -m memray flamegraph memory_profiles/mpi_cifar_20240326_143022/mpi_rank_0_optimized_memory_profile.bin

# View memory statistics for client (rank 1)
python -m memray stats memory_profiles/mpi_cifar_20240326_143022/mpi_rank_1_optimized_memory_profile.bin

# Compare original vs optimized
python -m memray stats memory_profiles/mpi_cifar_20240326_143022/mpi_rank_0_original_memory_profile.bin
python -m memray stats memory_profiles/mpi_cifar_20240326_143022/mpi_rank_0_optimized_memory_profile.bin
```

#### Generated Analysis Files
The result analysis script generates:
- **Comprehensive plots**: Memory usage comparisons, allocation patterns, optimization impact
- **Summary report**: Detailed text report with optimization metrics and improvements
- **Interactive visualizations**: Bar charts, scatter plots, and comparison graphs

#### Output Directory Structure
Each experiment run creates a timestamped directory:
```
memory_profiles/
├── mpi_mnist_20240326_143022/          # MNIST experiment (CNN model)
│   ├── mpi_rank_0_original_memory_profile.bin
│   ├── mpi_rank_0_optimized_memory_profile.bin
│   ├── mpi_rank_1_original_memory_profile.bin
│   ├── mpi_rank_1_optimized_memory_profile.bin
│   ├── mpi_rank_2_original_memory_profile.bin
│   ├── mpi_rank_2_optimized_memory_profile.bin
│   ├── *.html                          # Generated flamegraphs
│   └── analysis/                       # Comprehensive analysis results
│       ├── flamegraphs/                # Interactive flamegraphs by rank
│       ├── summaries/                  # Per-profile summary reports
│       ├── *.png                       # Comparison and visualization plots
│       ├── *.csv                       # Statistical data export
│       └── mpi_memory_optimization_report.txt
├── mpi_cifar_20240326_143500/          # CIFAR-10 experiment (ResNet18)
│   └── ... (similar structure)
└── mpi_resnet_20240326_144530/         # ResNet dummy experiment
    └── ... (similar structure)
```

## Configuration Handling

The MPI memory profiling script follows the same pattern as `examples/mpi/run_mpi.py`:

- **Server Configuration**: Uses the config file specified via `--server_config` parameter
- **Client Configuration**: Uses base config file specified via `--client_config` parameter
  - Base client config is customized for each MPI rank by automatically updating:
    - `client_id` → `f"Client{rank}"`
    - `dataset_kwargs.client_id` → `rank - 1` (0-indexed)
    - `dataset_kwargs.num_clients` → total number of clients
    - `dataset_kwargs.visualization` → `True` only for rank 1, `False` for others
- **Configuration Loading**: Follows standard APPFL pattern:
  1. Load and customize client config file, create ClientAgent
  2. Get additional configuration from server via `get_configuration()`
  3. Load server config into existing ClientAgent via `load_config()`
  4. Get and load initial global model via `load_parameters()`
- **Memory Optimization**: Applied to both server and client configurations when `--use_optimized_version` is enabled

Example config structures:
```
# MNIST experiment
resources/configs/mnist/
├── server_fedavg.yaml      # Server configuration (CNN model, FedAvg aggregator)
└── client_1.yaml           # Base client configuration (MNIST dataset, CPU device)

# CIFAR-10 experiment
resources/configs/cifar10/
├── server_fedavg.yaml      # Server configuration (ResNet18 model, FedAvg aggregator)
└── client_1.yaml           # Base client configuration (CIFAR-10 dataset, CUDA device)

# ResNet dummy experiment (original)
memory_profiling/configs/
├── server_resnet_dummy.yaml      # Server configuration (ResNet18, dummy data)
└── client_1_resnet_dummy.yaml    # Base client configuration (dummy data focus)
```

## Key Differences from gRPC Memory Profiling

- **MPI-specific optimizations**: Focused on MPI message passing and serialization
- **Multi-process profiling**: Each MPI rank generates its own memory profile
- **Rank-based analysis**: Separate profiles for server (rank 0) and clients (ranks 1+)
- **Client config auto-discovery**: Automatically finds and uses client-specific configurations
- **Batched communication**: Optimizations for MPI batched client scenarios

## Troubleshooting

### Common Issues

**1. Flamegraph generation fails:**
```bash
# Check if profile files are valid
python -m memray stats ./memory_profiles/mpi_resnet_20250826_110512/mpi_rank_0_original_memory_profile.bin

# Check file sizes (empty files = 0 bytes indicate profiling failed)
ls -la ./memory_profiles/mpi_resnet_20250826_110512/*.bin
```

**2. Empty or corrupted profile files:**
- **Cause**: MPI processes may have crashed or been killed before completing
- **Solution**: Check MPI experiment logs, reduce timeout, or run with fewer processes
- **Debug**: Use `diagnose_profiles()` function in experiment scripts

**3. Memory profiling fails:**
- **Cause**: Insufficient memory, MPI communication errors, or config issues
- **Solution**:
  ```bash
  # Test basic MPI functionality first
  mpiexec -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"

  # Test memray functionality
  python -c "import memray; print('Memray working')"

  # Run with minimal processes
  mpiexec -n 2 python memory_profiling_mpi/run_mpi_memray.py --server_config ... --client_config ...
  ```

**4. Dependencies:**
- Ensure MPI is properly installed: `pip install mpi4py`
- Check that memray supports your system: `python -m memray --help`
- Verify configs exist: `ls examples/memory_profiling/configs/`
- For large models, increase system memory limits if needed

**5. Configuration issues:**
- Verify server and client config files are valid YAML
- Check that dataset paths and model configs are correct
- Ensure memory optimization flags are compatible with your setup
