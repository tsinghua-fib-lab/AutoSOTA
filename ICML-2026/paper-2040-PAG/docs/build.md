# Build and Packaging

## Default Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The default build creates:

```text
build/libpag_core.a
build/PAG
```

## CMake Options

```bash
-DPAG_BUILD_CLI=ON
-DPAG_BUILD_EXAMPLES=OFF
-DPAG_BUILD_PYTHON=OFF
-DPAG_USE_AVX512=ON
-DPAG_NATIVE_ARCH=OFF
-DPAG_ENABLE_WARNINGS=OFF
-DPAG_OPTIMIZATION_LEVEL=-O3
```

`PAG_USE_AVX512=ON` is required in this release. `PAG_NATIVE_ARCH=OFF` keeps builds less tied to one local machine; enable it only for local CPU-specific tuning.

## C++ Install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/pag-install
cmake --build build -j$(nproc)
cmake --build build --target install
```

Installed CMake consumers can use:

```cmake
find_package(PAG CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE PAG::pag_core)
```

## Python Install

```bash
python -m pip install .
```

The Python build uses `scikit-build-core` and `pybind11`. It builds the extension module and installs the `pag` package.

To build the Python module directly through CMake:

```bash
python -m pip install pybind11
cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release \
  -DPAG_BUILD_CLI=OFF -DPAG_BUILD_PYTHON=ON \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build_python -j$(nproc)
```
