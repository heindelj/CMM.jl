import CMM.inc_gamma

# This file is meant to include implementations of empirical potentials
# which can be used as alternative parameterizations of an existing model.
# For instance, we use the TTTR potential as the short-range model for
# ion-ion interactions. This means we switch between two parameterizations
# of the same energy types. That is, we go from an empirical potential at
# short range which represents just the pairwise interaction to a many-body
# model at medium and long-range so that we can represent arbitrary systems
# and configurations.

function TTTR_potential(
	coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol, Float64}
)

	x = b * R
    λ1 = inc_gamma(x, 1)
    λ4 = inc_gamma(x, 4)
    λ6 = inc_gamma(x, 6)

    return A * exp(-x) - (λ1 * C1 / R + λ4 * C4 / R^4 + λ6 * C6 / R^6)
end