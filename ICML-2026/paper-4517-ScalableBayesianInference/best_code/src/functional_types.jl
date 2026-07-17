# ------------------------------------------------------------------------------
# Functional Type Classification
#
# Classifies linear functionals into categories for ordering purposes.
# Uses multiple dispatch to detect functional types automatically.
# ------------------------------------------------------------------------------

using FunctionalGPs

export FunctionalCategory, EVALUATION, DERIVATIVE, INTEGRAL, FACE_INTEGRAL, OTHER
export functional_category, get_coordinates, output_length

"""
    FunctionalCategory

Categories of linear functionals for ordering purposes.
"""
@enum FunctionalCategory begin
    INTEGRAL           # Pure integrals / cell integrals (coarsest)
    FACE_INTEGRAL      # Mixed eval⊗integral (middle tier)
    EVALUATION         # Point evaluations
    DERIVATIVE         # Derivatives (finest)
    OTHER
end

"""
    functional_category(L)

Determine the category of a linear functional using multiple dispatch.
Returns a `FunctionalCategory` enum value.

# Supported types
- `EvaluationFunctional` → `EVALUATION`
- `VectorizedLebesgueIntegral` → `INTEGRAL`
- Composed functionals with `PartialDerivative` → `DERIVATIVE`
- Everything else → `OTHER`
"""
functional_category(::EvaluationFunctional) = EVALUATION
functional_category(::VectorizedLebesgueIntegral) = INTEGRAL

# Composed functionals (e.g., L_eval ∘ PartialDerivative)
function functional_category(L::LinFctlLinFuncOpConcat)
    # Check if any of the operators is a PartialDerivative
    for op in L.linfuncops
        if op isa PartialDerivative
            return DERIVATIVE
        end
    end
    # Delegate to inner functional (e.g., for tensor products without derivatives)
    return functional_category(L.linfctl)
end

# Tensor product functionals (e.g., Eval ⊗ Integral for face integrals)
function functional_category(L::TensorProductFunctional)
    cats = [functional_category(f) for f in L.factors]
    has_eval = EVALUATION in cats
    has_integral = INTEGRAL in cats

    if has_eval && has_integral
        return FACE_INTEGRAL
    elseif all(c -> c == INTEGRAL, cats)
        return INTEGRAL  # pure integral (cell integral in 2D)
    else
        return EVALUATION  # all evals (point eval on product grid)
    end
end

# Sum of functionals (e.g., L_uxx + L_uyy for Laplacian)
function functional_category(L::FunctionalGPs.SumLinearFunctional)
    # Category is the "finest" (highest) among summands
    cats = [functional_category(f) for f in L.summands]
    # DERIVATIVE > EVALUATION > FACE_INTEGRAL > INTEGRAL
    if DERIVATIVE in cats
        return DERIVATIVE
    elseif EVALUATION in cats
        return EVALUATION
    elseif FACE_INTEGRAL in cats
        return FACE_INTEGRAL
    elseif INTEGRAL in cats
        return INTEGRAL
    else
        return OTHER
    end
end

# Fallback for unknown types
functional_category(::Any) = OTHER

# ------------------------------------------------------------------------------
# Coordinate Extraction
# ------------------------------------------------------------------------------

"""
    get_coordinates(L) -> Matrix{Float64}

Extract spatial coordinates from a linear functional for use in ordering.
Returns a matrix of shape `(d, n)` where `d` is the spatial dimension and `n` is the number of outputs.
"""
function get_coordinates(L::EvaluationFunctional)
    X = L.X
    return _extract_coordinates(X)
end

# Handle different types of point collections
function _extract_coordinates(X::AbstractVector)
    first_x = first(X)
    d = first_x isa Number ? 1 : length(first_x)
    n = length(X)
    coords = Matrix{Float64}(undef, d, n)
    for (i, x) in enumerate(X)
        coords[:, i] .= x
    end
    return coords
end

function _extract_coordinates(X::AbstractRange)
    # 1D range of points
    return reshape(collect(Float64, X), 1, length(X))
end

function _extract_coordinates(X::FactorizedGrid)
    # 2D (or higher) factorized grid - iterate over Cartesian product
    ranges = X.ranges
    d = length(ranges)
    n = prod(length.(ranges))
    coords = Matrix{Float64}(undef, d, n)

    # Iterate in column-major order (matching how FunctionalGPs linearizes)
    idx = 1
    for pt in Iterators.product(ranges...)
        for dim in 1:d
            coords[dim, idx] = pt[dim]
        end
        idx += 1
    end
    return coords
end

function get_coordinates(L::VectorizedLebesgueIntegral)
    # Use interval midpoints as representative coordinates
    domains = L.domains
    return _extract_domain_coordinates(domains)
end

function _extract_domain_coordinates(domains::AbstractVector)
    if domains[1] isa Interval
        # 1D intervals
        return reshape(midpoint.(domains), 1, length(domains))
    else
        # Multi-dimensional domains - use centroid
        n = length(domains)
        d = length(domains[1])
        coords = Matrix{Float64}(undef, d, n)
        for (i, dom) in enumerate(domains)
            for dim in 1:d
                coords[dim, i] = midpoint(dom[dim])
            end
        end
        return coords
    end
end

function _extract_domain_coordinates(domains::Base.Iterators.ProductIterator)
    # 2D (or higher) product of interval sets
    axes = domains.iterators
    d = length(axes)
    n = prod(length.(axes))
    coords = Matrix{Float64}(undef, d, n)

    idx = 1
    for cell in domains
        for dim in 1:d
            coords[dim, idx] = midpoint(cell[dim])
        end
        idx += 1
    end
    return coords
end

function _extract_domain_coordinates(domains::FactorizedBoxDomains)
    # 2D (or higher) factorized box domains
    interval_vecs = domains.interval_vecs
    d = length(interval_vecs)
    n = prod(length.(interval_vecs))
    coords = Matrix{Float64}(undef, d, n)

    idx = 1
    for cell in Iterators.product(interval_vecs...)
        for dim in 1:d
            coords[dim, idx] = midpoint(cell[dim])
        end
        idx += 1
    end
    return coords
end

# Composed functionals: coordinates come from inner functional
get_coordinates(L::LinFctlLinFuncOpConcat) = get_coordinates(L.linfctl)

# Sum of functionals: coordinates come from first summand (assumed same for all)
get_coordinates(L::FunctionalGPs.SumLinearFunctional) = get_coordinates(first(L.summands))

# Tensor product functionals: build coordinates from factor combinations
function get_coordinates(L::TensorProductFunctional)
    factors = L.factors
    d = length(factors)

    # Get 1D coordinates for each factor
    factor_coords = [_get_factor_coords(f) for f in factors]

    # Build Cartesian product (column-major order)
    n = prod(length.(factor_coords))
    coords = Matrix{Float64}(undef, d, n)

    idx = 1
    for pt in Iterators.product(factor_coords...)
        for dim in 1:d
            coords[dim, idx] = pt[dim]
        end
        idx += 1
    end
    return coords
end

# Helper to get 1D coordinates from a factor
function _get_factor_coords(L::EvaluationFunctional)
    X = L.X
    if X isa AbstractRange
        return collect(Float64, X)
    else
        return Float64[x isa Number ? x : x[1] for x in X]
    end
end

_get_factor_coords(L::VectorizedLebesgueIntegral) = Float64[midpoint(d) for d in L.domains]

# Fallback: try to extract from common patterns
function get_coordinates(L)
    if hasproperty(L, :X)
        X = L.X
        return reshape(collect(X), 1, length(X))
    elseif hasproperty(L, :domains)
        return reshape(midpoint.(L.domains), 1, length(L.domains))
    else
        error("Cannot extract coordinates from functional of type $(typeof(L)). " *
              "Please implement `get_coordinates` for this type.")
    end
end

# ------------------------------------------------------------------------------
# Output Length Extraction
# ------------------------------------------------------------------------------

"""
    output_length(L) -> Int

Get the number of outputs from a linear functional.
"""
output_length(L::EvaluationFunctional) = _collection_length(L.X)
output_length(L::VectorizedLebesgueIntegral) = _collection_length(L.domains)

_collection_length(X::AbstractVector) = length(X)
_collection_length(X::AbstractRange) = length(X)
_collection_length(X::FactorizedGrid) = prod(length.(X.ranges))
_collection_length(X::FactorizedBoxDomains) = prod(length.(X.interval_vecs))
_collection_length(X::Base.Iterators.ProductIterator) = prod(length.(X.iterators))
output_length(L::LinFctlLinFuncOpConcat) = output_length(L.linfctl)
output_length(L::TensorProductFunctional) = prod(FunctionalGPs.output_shape(L))
output_length(L::FunctionalGPs.SumLinearFunctional) = output_length(first(L.summands))

function output_length(L)
    # Try common patterns
    if hasproperty(L, :X)
        return length(L.X)
    elseif hasproperty(L, :domains)
        return length(L.domains)
    else
        # Try calling output_shape if available
        return prod(FunctionalGPs.output_shape(L))
    end
end
