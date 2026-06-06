// Copyright (c) 2023-2024, Aalto University, developed by Erik Schultheis
// Copyright (c) 2025 Institute of Science and Technology Austria (ISTA).
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <type_traits>
#include <vector_types.h>

namespace detail
{
enum class TransferMode {
    DEFAULT,
    LDG,
    LU
};

template<TransferMode Mode>
struct Transfer;

template<>
struct Transfer<TransferMode::DEFAULT> {
    template<class T>
    __host__ __device__ static void call(T* dst, const T* src) {
        *dst = *src;
    }
};

template<>
struct Transfer<TransferMode::LDG> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldg(src);
    }
};

template<>
struct Transfer<TransferMode::LU> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldlu(src);
    }
};


/*!
 * \brief Copies `NBytes` from `src` to `dst`, using `CopyType` to perform memory access.
 * \details
 * This means that pointers need to be aligned according to `CopyType`'s requirements,
 * and copies are (most likely) be performed using vectorized access according to
 * `CopyType`.
 * The ranges `[src, src+NBytes)` and `[dst, dst + NBytes)` must be non-overlapping.
 *
 * This function is used to implement `memcpy_aligned`, and generally not intended to
 * be used directly.
 */
template<class CopyType, int NBytes, TransferMode Mode, class TrueType>
__host__ __device__ void memcpy_as(TrueType* __restrict__ dst, const TrueType* __restrict__ src) {
    static_assert(NBytes % sizeof(TrueType) == 0, "Number of bytes must be a multiple of the true type size");
    static_assert(NBytes % sizeof(CopyType) == 0, "Number of bytes must be a multiple of the copy type size");

    // in order to do simple byte-level copying, the underlying type must be trivially copyable (i.e., compatible
    // with memcpy)
    static_assert(std::is_trivially_copyable_v<TrueType>, "TrueType must be trivially copyable");
    const auto* read_address = reinterpret_cast<const CopyType*>(src);
    auto* write_address = reinterpret_cast<CopyType*>(dst);
    #pragma unroll
    for (int i = 0; i < NBytes; i += sizeof(CopyType)) {
        Transfer<Mode>::call(write_address, read_address);
        ++read_address;
        ++write_address;
    }
}

/*!
 * \brief Assume an array of objects of `size` bytes each, what is the alignment
 * of an individual element of that array.
 * \details Assume that the array itself starts at a 16-byte aligned address,
 * what is the worst-case alignment of any object. E.g., for objects of 4 bytes,
 * alignment is 4, for 6 bytes it is 2, etc.
 */
constexpr __host__ __device__ std::size_t alignment_from_size(std::size_t size) {
    for (int i = 2; i <= 16; i *= 2) {
        if ((size % i) != 0) {
            return i / 2;
        }
    }
    return 16;
}
}  // namespace detail

/*!
 * \brief Synchronous copy from `src` to `dst` using the widest memory loads
 * possible. The number of elements to copy has to be a compile-time constant.
 * \details The size of the memory load is chosen based on the _total_ amount
 * of bytes being transferred, not on the alignment requirement of a single
 * element. For example, when copying 2 ints, a single 8-byte load is used, but when
 * copying 3 ints, three separate 4-byte loads are needed.
 * \sa detail::alignment_from_size
 * \tparam Count Number of elements to copy.
 * \tparam T Type of the elements to copy. Needs to be trivially copyable.
 */
template<std::size_t Count, detail::TransferMode Mode, class T>
__host__ __device__ void memcpy_aligned(T* dst, const T* src, std::integral_constant<std::size_t, Count> = {}) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    constexpr const int NBytes = sizeof(T) * Count;
    using detail::memcpy_as;

    // ideally, we'd just use a simple memcpy, like below, but that does
    // not always generate vectorized loads
    // std::memcpy(values, __builtin_assume_aligned(address, bytes), bytes);

    if constexpr (NBytes % sizeof(int4) == 0) {
        memcpy_as<int4, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(int2) == 0) {
        memcpy_as<int2, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(int1) == 0) {
        memcpy_as<int1, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(short1) == 0) {
        memcpy_as<short1, NBytes, Mode>(dst, src);
    } else {
        memcpy_as<char1, NBytes, Mode>(dst, src);
    }
}

/*!
 * \brief Helper type that implements a SIMD-like vector type.
 * \details Contrary to nvidia's `float4` type, which allows access to its components only
 * through `.xyzw`, this provides a much more natural interface. It also generalizes to other
 * data types, and allows an arbitrary number of elements.
 * \tparam ElementType Type of a single element inside the vector. Needs to be trivial (e.g., memcpy-able)
 * \tparam ElementCount How many elements in this vector. To get full benefits from vectorized load instructions,
 * the total size of this vector needs to be a multiple of 16 bytes.
 */
template<class ElementType, std::size_t ElementCount>
class alignas(detail::alignment_from_size(sizeof(ElementType) * ElementCount)) GenericVector {
    static_assert(std::is_trivial_v<ElementType>, "Only trivial types are supported");

public:
    GenericVector() = default;

    constexpr static __host__ __device__ GenericVector constant(ElementType value) {
        GenericVector result;
        for (int k = 0; k < size; ++k) {
            result.values[k] = value;
        }
        return result;
    }

    constexpr static __host__ __device__ GenericVector zeros() {
        return constant(0.f);
    }

    constexpr static __host__ __device__ GenericVector ones() {
        return constant(1.f);
    }

    template<class U>
    constexpr static __host__ __device__ GenericVector from(GenericVector<U, ElementCount> other) {
        GenericVector<ElementType, ElementCount> result;
        for (int i = 0; i < ElementCount; ++i) {
            result[i] = static_cast<ElementType>(other[i]);
        }
        return result;
    }

    constexpr __host__ __device__ ElementType& operator[](int index) {
        return values[index];
    }

    constexpr __host__ __device__ const ElementType& operator[](int index) const {
        return values[index];
    }

    static constexpr const std::size_t size = ElementCount;
    static constexpr const std::size_t bytes = ElementCount * sizeof(ElementType);

    static __host__ __device__ GenericVector load(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::DEFAULT>(result.values, address);
        return result;
    }

    static __device__ GenericVector load_ldg(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LDG>(result.values, address);
        return result;
    }

    static __device__ GenericVector load_lu(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LU>(result.values, address);
        return result;
    }

    __host__ __device__ void store(ElementType* dst) {
        memcpy_aligned<size, detail::TransferMode::DEFAULT>(dst, values);
    }

private:
    ElementType values[size];
};
