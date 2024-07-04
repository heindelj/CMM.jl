import CMM.CMM_FF

function permanent_electrostatics!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    num_fragments = count(>(0), length.(fragment_indices))

    get_all_core_shell_electrostatic_quantities!(
        coords,
        labels,
        fragment_indices,
        ff.storage.multipoles,
        ff.storage.ϕ_core,
        ff.storage.ϕ_shell,
        ff.storage.E_field_core,
        ff.storage.E_field_shell,
        ff.storage.E_field_gradients_core,
        ff.storage.E_field_gradients_shell,
        ff.storage.E_field_gradient_gradients_core,
        ff.storage.E_field_gradient_gradients_shell,
        ff.params,        
        ff.storage.shell_damping_type,
        ff.storage.overlap_damping_type
    )

    coulomb_energy = 0.0
    for i in eachindex(ff.storage.multipoles)
        coulomb_energy += ff.storage.multipoles[i].Z * ff.storage.ϕ_core[i]
        coulomb_energy += ff.storage.multipoles[i].q_shell * ff.storage.ϕ_shell[i]
        coulomb_energy -= ff.storage.multipoles[i].μ ⋅ ff.storage.E_field_shell[i]
        coulomb_energy -= ff.storage.multipoles[i].Q ⋅ ff.storage.E_field_gradients_shell[i] / 3.0
    end
    coulomb_energy *= 0.5

    # Deformation energy correction for Badger rule
    μ_1 = ff.params[:OH_dip_deriv_1]
    μ_2 = ff.params[:OH_dip_deriv_2]
    one_body_energy = get_coupled_morse_and_bend_energy_and_grads!(
        coords, labels,
        fragment_indices,
        ff.params,
        nothing
    )
    one_body_energy_with_response = get_field_dependent_morse_and_bend_energy_and_grads!(
        coords, labels, fragment_indices, ff.params,
        ff.storage.E_field_core,
        μ_1, μ_2, 0.0, 0.0,
        ff.storage.Δq_ct, 0.0, 0.0, false,
        ff.storage.electrostatic_grads,
        ff.storage.E_field_gradients_shell
    )
    bond_pol_energy = (one_body_energy_with_response - one_body_energy)
    ff.results.energies[:bond_pol_perm] = bond_pol_energy
    coulomb_energy += bond_pol_energy

    if haskey(ff.results.energies, :Electrostatics)
        ff.results.energies[:Electrostatics] += coulomb_energy
    else
        ff.results.energies[:Electrostatics] = coulomb_energy
    end
    ff.results.energies[:Total] += coulomb_energy
    return coulomb_energy
end

# HERE: Finish the implementation of the electrostatics stuff.
# Need to:
# (1): Make the CMM object that hangs onto all the storage
# (2): Write a new version of the build function which will build the
# appropriate CMM object with all the distances, damping exponents and so on.
# (3): Then, actually call the electrostatics function and make sure
# we get the right energies compared to reference.

#function permanent_electrostatics!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
#    num_fragments = count(>(0), length.(fragment_indices))
#
#    get_all_core_shell_electrostatic_quantities!(
#        coords,
#        labels,
#        fragment_indices,
#        ff.storage.multipoles,
#        ff.storage.ϕ_core,
#        ff.storage.ϕ_shell,
#        ff.storage.E_field_core,
#        ff.storage.E_field_shell,
#        ff.storage.E_field_gradients_core,
#        ff.storage.E_field_gradients_shell,
#        ff.storage.E_field_gradient_gradients_core,
#        ff.storage.E_field_gradient_gradients_shell,
#        ff.params,        
#        ff.storage.shell_damping_type,
#        ff.storage.overlap_damping_type
#    )
#
#    coulomb_energy = 0.0
#    for i in eachindex(ff.storage.multipoles)
#        coulomb_energy += ff.storage.multipoles[i].Z * ff.storage.ϕ_core[i]
#        coulomb_energy += ff.storage.multipoles[i].q_shell * ff.storage.ϕ_shell[i]
#        coulomb_energy -= ff.storage.multipoles[i].μ ⋅ ff.storage.E_field_shell[i]
#        coulomb_energy -= ff.storage.multipoles[i].Q ⋅ ff.storage.E_field_gradients_shell[i] / 3.0
#    end
#    coulomb_energy *= 0.5
#
#    # Deformation energy correction for Badger rule
#    μ_1 = ff.params[:OH_dip_deriv_1]
#    μ_2 = ff.params[:OH_dip_deriv_2]
#    one_body_energy = get_coupled_morse_and_bend_energy_and_grads!(
#        coords, labels,
#        fragment_indices,
#        ff.params,
#        nothing
#    )
#    one_body_energy_with_response = get_field_dependent_morse_and_bend_energy_and_grads!(
#        coords, labels, fragment_indices, ff.params,
#        ff.storage.E_field_core,
#        μ_1, μ_2, 0.0, 0.0,
#        ff.storage.Δq_ct, 0.0, 0.0, false,
#        ff.storage.electrostatic_grads,
#        ff.storage.E_field_gradients_shell
#    )
#    bond_pol_energy = (one_body_energy_with_response - one_body_energy)
#    ff.results.energies[:bond_pol_perm] = bond_pol_energy
#    coulomb_energy += bond_pol_energy
#
#    if haskey(ff.results.energies, :Electrostatics)
#        ff.results.energies[:Electrostatics] += coulomb_energy
#    else
#        ff.results.energies[:Electrostatics] = coulomb_energy
#    end
#    ff.results.energies[:Total] += coulomb_energy
#    return coulomb_energy
#end

function permanent_undamped_electrostatics!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    num_fragments = count(>(0), length.(fragment_indices))
    num_polarizable_sites = length(labels) - count(==("O"), labels)

    get_all_undamped_electrostatic_quantities!(
        coords,
        ff.storage.multipoles,
        fragment_indices,
        ff.storage.ϕ_core,
        ff.storage.E_field_core,
        ff.storage.E_field_gradients_core
    )

    coulomb_energy = 0.0
    for i in eachindex(ff.storage.multipoles)
        coulomb_energy += (ff.storage.multipoles[i].q_shell + ff.storage.multipoles[i].Z) * ff.storage.ϕ_core[i]
        coulomb_energy -= ff.storage.multipoles[i].μ ⋅ ff.storage.E_field_core[i]
        coulomb_energy -= ff.storage.multipoles[i].Q ⋅ ff.storage.E_field_gradients_core[i] / 3.0
    end
    coulomb_energy *= 0.5
    ff.results.energies[:Electrostatics_no_cp] = coulomb_energy
    ff.results.energies[:Total] += coulomb_energy
    return coulomb_energy
end