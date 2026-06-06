#!/usr/bin/env bash

# Pinet small benchmarks
for id in "dc3_simple_1" "dc3_nonconvex_1"; do
    for config in "benchmark_small_autotune"; do
        for seed in 0 1 2 3 4; do
            python -m src.benchmarks.QP.run_QP \
            --id "$id" \
            --config "$config" \
            --seed "$seed"
        done
    done
done

# Pinet large benchmarks
for id in "dc3_simple_2" "dc3_nonconvex_2"; do
    for config in "benchmark_large_autotune"; do
        for seed in 0 1 2 3 4; do
            python -m src.benchmarks.QP.run_QP \
            --id "$id" \
            --config "$config" \
            --seed "$seed"
        done
    done
done

# JAXopt small benchmarks
for id in "dc3_simple_1" "dc3_nonconvex_1"; do
    for config in "benchmark_jaxopt_small"; do
        for seed in 0 1 2 3 4; do
            python -m src.benchmarks.QP.run_QP \
            --id "$id" \
            --config "$config" \
            --seed "$seed"
        done
    done
done

# JAXopt large benchmarks
for id in "dc3_simple_2" "dc3_nonconvex_2"; do
    for config in "benchmark_jaxopt_large"; do
        for seed in 0 1 2 3 4; do
            python -m src.benchmarks.QP.run_QP \
            --id "$id" \
            --config "$config" \
            --seed "$seed"
        done
    done
done

# cvxpylayers small non-convex
for seed in 0 1 2 3 4; do
    python -m src.benchmarks.QP.run_QP \
    --id dc3_nonconvex_1 \
    --config benchmark_cvxpy \
    --seed "$seed"
