using Flux
using Functors

"""
    Flux.trainable(layer::DisjunctiveProjectionLayer)

Projection layers currently have no trainable parameters.

The neural network backbone is trainable; the projection layer is a
deterministic differentiable map.
"""
Flux.trainable(layer::DisjunctiveProjectionLayer) = NamedTuple()
Functors.children(layer::DisjunctiveProjectionLayer) = NamedTuple()

Functors.functor(::Type{<:DisjunctiveProjectionLayer}, layer) = (NamedTuple(), _ -> layer)