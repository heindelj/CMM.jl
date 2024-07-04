
abstract type AbstractForceFieldStorage end
abstract type AbstractForceField end

struct ForceFieldResults
    energies::Dict{Symbol, Float64}
    grads::Union{Vector{MVector{3, Float64}}, Nothing}
end
ForceFieldResults() = ForceFieldResults(Dict{Symbol, Float64}(:Total => 0.0), nothing)
ForceFieldResults(natoms::Int) = ForceFieldResults(Dict{Symbol, Float64}(:Total => 0.0), [@MVector zeros(3) for _ in 1:natoms])
