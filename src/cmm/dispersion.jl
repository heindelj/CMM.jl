include("../dispersion.jl")

function dispersion_energy!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::AbstractForceField)
    dispersion_energy = total_two_and_three_body_dispersion_energy(coords, labels, fragment_indices, ff.params)
    
    ff.results.energies[:Dispersion] = dispersion_energy
    ff.results.energies[:Total] += dispersion_energy
    return dispersion_energy
end