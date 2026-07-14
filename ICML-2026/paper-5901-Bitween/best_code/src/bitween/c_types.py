import random

# Mapping of type names to their corresponding format specifiers.
format_specifiers = {
    "double": "%lf",
    "float": "%f",
    "int": "%d",
    "unsigned int": "%u",
    "long": "%ld",
    "unsigned": "%u",  # Assuming unsigned int
    "unsigned long": "%lu",
    "long long": "%lld",
    "unsigned long long": "%llu",
    "size_t": "%zu",  # Use "%u" or "%lu" for older C versions without C99 support
    "uint8_t": "%hhu",
    "uint16_t": "%hu",
    "uint32_t": "%u",  # Assuming uint32_t maps to unsigned int
    "uint64_t": "%llu",  # Assuming uint64_t maps to unsigned long long
    "int8_t": "%hhd",
    "int16_t": "%hd",
    "int32_t": "%d",  # Assuming int32_t maps to int
    "int64_t": "%lld",  # Assuming int64_t maps to long long
    "long double": "%Lf",
    "float32_t": "%f",  # Assuming float32_t maps to float
    "float64_t": "%lf",  # Assuming float64_t maps to double
    "float128_t": "%Lf",  # Assuming float128_t maps to long double
    "double32_t": "%f",  # Assuming this is a non-standard type that maps to float
    "double64_t": "%lf",  # Assuming double64_t maps to double
    "double128_t": "%Lf",  # Assuming double128_t maps to long double
}

int_types = {
    "int",
    "unsigned int",
    "long",
    "unsigned",
    "unsigned long",
    "long long",
    "unsigned long long",
    "size_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
}

int_upper_bound_civl = 10

float_types = {
    "float",
    "double",
    "long double",
    "float32_t",
    "float64_t",
    "float128_t",
    "double32_t",
    "double64_t",
    "double128_t",
}

float_upper_bound_civl = 10.0

random_functions = {
    # 32-bit signed
    "int": (random.randint, (0, 300)),
    # IEEE 754 single-precision
    "float": (random.uniform, (-5, 5)),
    # IEEE 754 double-precision
    "double": (random.uniform, (-5, 5)),
    # 64-bit signed
    "long": (random.randint, (0, 300)),
    # Assuming unsigned int, 32-bit
    "unsigned": (random.randint, (0, 300)),
    # 32-bit unsigned
    "unsigned int": (random.randint, (0, 300)),
    # 64-bit unsigned
    "unsigned long": (random.randint, (0, 300)),
    # 64-bit signed
    "long long": (random.randint, (-9223372036854775808, 9223372036854775807)),
    # 64-bit unsigned
    "unsigned long long": (random.randint, (0, 18446744073709551615)),
    # IEEE 754 extended double-precision (commonly 80-bit on x86-64, but up to 128-bit on some systems)
    "long double": (random.uniform, (-1.1e4932, 1.1e4932)),
    # Assuming size_t is 64-bit unsigned
    "size_t": (random.randint, (0, 18446744073709551615)),
    "uint8_t": (random.randint, (0, 255)),
    "uint16_t": (random.randint, (0, 65535)),
    "uint32_t": (random.randint, (0, 4294967295)),
    "uint64_t": (random.randint, (0, 18446744073709551615)),
    "int8_t": (random.randint, (-128, 127)),
    "int16_t": (random.randint, (-32768, 32767)),
    "int32_t": (random.randint, (-2147483648, 2147483647)),
    "int64_t": (random.randint, (-9223372036854775808, 9223372036854775807)),
    "float32_t": (random.uniform, (-3.4e38, 3.4e38)),  # IEEE 754 single-precision
    "float64_t": (random.uniform, (-1.7e308, 1.7e308)),  # IEEE 754 double-precision
    # Commonly treated as 'long double'
    "float128_t": (random.uniform, (-1.1e4932, 1.1e4932)),
    # Assuming a 32-bit double type, non-standard, treated as float
    "double32_t": (random.uniform, (-3.4e38, 3.4e38)),
    # Double precision
    "double64_t": (random.uniform, (-1.7e308, 1.7e308)),
    # Treated as 'long double'
    "double128_t": (random.uniform, (-1.1e4932, 1.1e4932)),
}

random_functions_default = {
    # 32-bit signed
    "int": (random.randint, (-2147483648, 2147483647)),
    # IEEE 754 single-precision
    "float": (random.uniform, (-3.4e38, 3.4e38)),
    # IEEE 754 double-precision
    "double": (random.uniform, (-1.7e308, 1.7e308)),
    # 64-bit signed
    "long": (random.randint, (-9223372036854775808, 9223372036854775807)),
    # Assuming unsigned int, 32-bit
    "unsigned": (random.randint, (0, 4294967295)),
    # 32-bit unsigned
    "unsigned int": (random.randint, (0, 4294967295)),
    # 64-bit unsigned
    "unsigned long": (random.randint, (0, 18446744073709551615)),
    # 64-bit signed
    "long long": (random.randint, (-9223372036854775808, 9223372036854775807)),
    # 64-bit unsigned
    "unsigned long long": (random.randint, (0, 18446744073709551615)),
    # IEEE 754 extended double-precision (commonly 80-bit on x86-64, but up to 128-bit on some systems)
    "long double": (random.uniform, (-1.1e4932, 1.1e4932)),
    # Assuming size_t is 64-bit unsigned
    "size_t": (random.randint, (0, 18446744073709551615)),
    "uint8_t": (random.randint, (0, 255)),
    "uint16_t": (random.randint, (0, 65535)),
    "uint32_t": (random.randint, (0, 4294967295)),
    "uint64_t": (random.randint, (0, 18446744073709551615)),
    "int8_t": (random.randint, (-128, 127)),
    "int16_t": (random.randint, (-32768, 32767)),
    "int32_t": (random.randint, (-2147483648, 2147483647)),
    "int64_t": (random.randint, (-9223372036854775808, 9223372036854775807)),
    "float32_t": (random.uniform, (-3.4e38, 3.4e38)),  # IEEE 754 single-precision
    "float64_t": (random.uniform, (-1.7e308, 1.7e308)),  # IEEE 754 double-precision
    # Commonly treated as 'long double'
    "float128_t": (random.uniform, (-1.1e4932, 1.1e4932)),
    # Assuming a 32-bit double type, non-standard, treated as float
    "double32_t": (random.uniform, (-3.4e38, 3.4e38)),
    # Double precision
    "double64_t": (random.uniform, (-1.7e308, 1.7e308)),
    # Treated as 'long double'
    "double128_t": (random.uniform, (-1.1e4932, 1.1e4932)),
}

# Common functions from math.h and their corresponding return types.
math_functions = {
    "acos": "double",
    "acosf": "float",
    "acosh": "double",
    "acoshf": "float",
    "acoshl": "long double",
    "acosl": "long double",
    "asin": "double",
    "asinf": "float",
    "asinh": "double",
    "asinhf": "float",
    "asinhl": "long double",
    "asinl": "long double",
    "atan": "double",
    "atan2": "double",
    "atan2f": "float",
    "atan2l": "long double",
    "atanf": "float",
    "atanh": "double",
    "atanhf": "float",
    "atanhl": "long double",
    "atanl": "long double",
    "cbrt": "double",
    "cbrtf": "float",
    "cbrtl": "long double",
    "ceil": "double",
    "ceilf": "float",
    "ceill": "long double",
    "copysign": "double",
    "copysignf": "float",
    "copysignl": "long double",
    "cos": "double",
    "cosf": "float",
    "cosh": "double",
    "coshf": "float",
    "coshl": "long double",
    "cosl": "long double",
    "erf": "double",
    "erfc": "double",
    "erfcf": "float",
    "erfcl": "long double",
    "erff": "float",
    "erfl": "long double",
    "exp": "double",
    "exp2": "double",
    "exp2f": "float",
    "exp2l": "long double",
    "expf": "float",
    "expl": "long double",
    "expm1": "double",
    "expm1f": "float",
    "expm1l": "long double",
    "fabs": "double",
    "fabsf": "float",
    "fabsl": "long double",
    "fdim": "double",
    "fdimf": "float",
    "fdiml": "long double",
    "floor": "double",
    "floorf": "float",
    "floorl": "long double",
    "fma": "double",
    "fmaf": "float",
    "fmal": "long double",
    "fmax": "double",
    "fmaxf": "float",
    "fmaxl": "long double",
    "fmin": "double",
    "fminf": "float",
    "fminl": "long double",
    "fmod": "double",
    "fmodf": "float",
    "fmodl": "long double",
    "frexp": "double",
    "frexpf": "float",
    "frexpl": "long double",
    "hypot": "double",
    "hypotf": "float",
    "hypotl": "long double",
    "ilogb": "int",
    "ilogbf": "int",
    "ilogbl": "int",
    "ldexp": "double",
    "ldexpf": "float",
    "ldexpl": "long double",
    "lgamma": "double",
    "lgammaf": "float",
    "lgammal": "long double",
    "llrint": "long long",
    "llrintf": "long long",
    "llrintl": "long long",
    "llround": "long long",
    "llroundf": "long long",
    "llroundl": "long long",
    "log": "double",
    "log10": "double",
    "log10f": "float",
    "log10l": "long double",
    "log1p": "double",
    "log1pf": "float",
    "log1pl": "long double",
    "log2": "double",
    "log2f": "float",
    "log2l": "long double",
    "logb": "double",
    "logbf": "float",
    "logbl": "long double",
    "logf": "float",
    "logl": "long double",
    "lrint": "long",
    "lrintf": "long",
    "lrintl": "long",
    "lround": "long",
    "lroundf": "long",
    "lroundl": "long",
    "modf": "double",
    "modff": "float",
    "modfl": "long double",
    "nan": "double",
    "nanf": "float",
    "nanl": "long double",
    "nearbyint": "double",
    "nearbyintf": "float",
    "nearbyintl": "long double",
    "nextafter": "double",
    "nextafterf": "float",
    "nextafterl": "long double",
    "nexttoward": "double",
    "nexttowardf": "float",
    "nexttowardl": "long double",
    "pow": "double",
    "powf": "float",
    "powl": "long double",
    "remainder": "double",
    "remainderf": "float",
    "remainderl": "long double",
    "remquo": "double",
    "remquof": "float",
    "remquol": "long double",
    "rint": "double",
    "rintf": "float",
    "rintl": "long double",
    "round": "double",
    "roundf": "float",
    "roundl": "long double",
    "scalbln": "double",
    "scalblnf": "float",
    "scalblnl": "long double",
    "scalbn": "double",
    "scalbnf": "float",
    "scalbnl": "long double",
    "sin": "double",
    "sinf": "float",
    "sinh": "double",
    "sinhf": "float",
    "sinhl": "long double",
    "sinl": "long double",
    "sqrt": "double",
    "sqrtf": "float",
    "sqrtl": "long double",
    "tan": "double",
    "tanf": "float",
    "tanh": "double",
    "tanhf": "float",
    "tanhl": "long double",
    "tanl": "long double",
    "tgamma": "double",
    "tgammaf": "float",
    "tgammal": "long double",
    "trunc": "double",
    "truncf": "float",
    "truncl": "long double",
    "y0": "double",
    "y1": "double",
    "yn": "double",
}
