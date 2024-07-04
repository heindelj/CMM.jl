
function get_one_body_properties!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::AbstractForceField)

    update_model_multipoles_and_local_axes!(
        coords, labels, fragment_indices,
        ff.params, ff.storage.multipoles, ff.storage.local_axes,
        ff.results.grads, ff.storage.one_body_charge_grads
    )
    one_body_energy = get_coupled_morse_and_bend_energy_and_grads!(coords, labels, fragment_indices, ff.params, ff.storage.deformation_grads)
    get_geometry_dependent_atomic_hardness!(coords, labels, fragment_indices, ff.storage.η_fq, ff.params)

    ff.results.energies[:Distortion] = one_body_energy
    ff.results.energies[:Total] += one_body_energy
    return one_body_energy
end