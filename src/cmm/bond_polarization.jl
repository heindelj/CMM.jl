
function get_bond_polarization_energy(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM)

    @assert ff.results.grads === nothing "Implement the bond polarization gradients you dummy!"

    # Deformation energy correction for Badger rule
    one_body_energy = get_coupled_morse_and_bend_energy_and_grads!(
        coords, labels,
        fragment_indices,
        ff.params,
        nothing
    )
    one_body_energy_with_response = get_field_dependent_morse_and_bend_energy_and_grads!(
        coords, labels, fragment_indices, ff.params,
        ff.storage.E_field_core + ff.storage.E_field_shell,
        ff.storage.E_field_induced, ff.storage.applied_field,
        ff.storage.deformation_grads,
        ff.storage.E_field_gradients_core + ff.storage.E_field_gradients_shell,
        ff.storage.E_field_gradients_induced
    )

    bond_pol_energy = (one_body_energy_with_response - one_body_energy)
    ff.results.energies[:bond_polarization] = bond_pol_energy
    ff.results.energies[:Total] += bond_pol_energy

    return bond_pol_energy
end