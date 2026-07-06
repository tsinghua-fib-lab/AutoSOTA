using Random
using StatsBase
import Random.rand

struct AliasSampler
    alias::Array{Int64,1}
    ap::Array{Float64,1}
end

function AliasSampler(w::AbstractWeights)
	n = length(w)
    ap = Vector{Float64}(undef, n)
    alias = Vector{Int}(undef, n)
    StatsBase.make_alias_table!(w, sum(w), ap, alias)
    return AliasSampler(alias, ap)
end

function rand(self::AliasSampler)
    n = length(self.alias)
    s = Random.RangeGenerator(1:n)
    j = rand(s);
    return (rand() < self.ap[j] ? j : self.alias[j])
end