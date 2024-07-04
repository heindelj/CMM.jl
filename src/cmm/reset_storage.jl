import CMM.CMM_FF

function reset_storage!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    @views ff.storage.ϕ_core[:] -= ff.storage.ϕ_core
    @views ff.storage.ϕ_shell[:] -= ff.storage.ϕ_shell
    @views ff.storage.ϕ_induced[:] -= ff.storage.ϕ_induced
    @views ff.storage.ϕ_exch_pol[:] -= ff.storage.ϕ_exch_pol
    @views ff.storage.ϕ_dispersion[:] -= ff.storage.ϕ_dispersion
    @views ff.storage.ϕ_repulsion[:] -= ff.storage.ϕ_repulsion
    @views ff.storage.ϕ_donor[:] -= ff.storage.ϕ_donor
    @views ff.storage.ϕ_acceptor[:] -= ff.storage.ϕ_acceptor
    @views ff.storage.E_field_core[:] -= ff.storage.E_field_core
    @views ff.storage.E_field_shell[:] -= ff.storage.E_field_shell
    @views ff.storage.E_field_induced[:] -= ff.storage.E_field_induced
    @views ff.storage.E_field_exch_pol[:] -= ff.storage.E_field_exch_pol
    @views ff.storage.E_field_dispersion[:] -= ff.storage.E_field_dispersion
    @views ff.storage.E_field_repulsion[:] -= ff.storage.E_field_repulsion
    @views ff.storage.E_field_donor[:] -= ff.storage.E_field_donor
    @views ff.storage.E_field_acceptor[:] -= ff.storage.E_field_acceptor
    @views ff.storage.E_field_gradients_core[:] -= ff.storage.E_field_gradients_core
    @views ff.storage.E_field_gradients_shell[:] -= ff.storage.E_field_gradients_shell
    @views ff.storage.E_field_gradients_induced[:] -= ff.storage.E_field_gradients_induced
    @views ff.storage.E_field_gradients_exch_pol[:] -= ff.storage.E_field_gradients_exch_pol
    @views ff.storage.E_field_gradients_dispersion[:] -= ff.storage.E_field_gradients_dispersion
    @views ff.storage.E_field_gradients_repulsion[:] -= ff.storage.E_field_gradients_repulsion
    @views ff.storage.E_field_gradients_donor[:] -= ff.storage.E_field_gradients_donor
    @views ff.storage.E_field_gradients_acceptor[:] -= ff.storage.E_field_gradients_acceptor
    @views ff.storage.E_field_gradient_gradients_core[:] -= ff.storage.E_field_gradient_gradients_core
    @views ff.storage.E_field_gradient_gradients_shell[:] -= ff.storage.E_field_gradient_gradients_shell
    @views ff.storage.E_field_gradient_gradients_induced[:] -= ff.storage.E_field_gradient_gradients_induced
    @views ff.storage.E_field_gradient_gradients_exch_pol[:] -= ff.storage.E_field_gradient_gradients_exch_pol
    @views ff.storage.E_field_gradient_gradients_dispersion[:] -= ff.storage.E_field_gradient_gradients_dispersion
    @views ff.storage.E_field_gradient_gradients_repulsion[:] -= ff.storage.E_field_gradient_gradients_repulsion
    @views ff.storage.E_field_gradient_gradients_donor[:] -= ff.storage.E_field_gradient_gradients_donor
    @views ff.storage.E_field_gradient_gradients_acceptor[:] -= ff.storage.E_field_gradient_gradients_acceptor
    @views ff.storage.Δq_ct[:] -= ff.storage.Δq_ct
    @views ff.storage.b_vec[:] -= ff.storage.b_vec
    @views ff.storage.solution_vector[:] -= ff.storage.solution_vector
    @views ff.storage.polarization_matrix[:, :] -= ff.storage.polarization_matrix
    if ff.results.grads !== nothing && length(ff.results.grads) > 0
        @views ff.storage.torques[:] -= ff.storage.torques
        @views ff.storage.one_body_charge_grads[:] -= ff.storage.one_body_charge_grads
        @views ff.results.grads[:] -= ff.results.grads
        @views ff.storage.deformation_grads[:] -= ff.storage.deformation_grads
        @views ff.storage.pauli_grads[:] -= ff.storage.pauli_grads
        @views ff.storage.dispersion_grads[:] -= ff.storage.dispersion_grads
        @views ff.storage.electrostatic_grads[:] -= ff.storage.electrostatic_grads
        @views ff.storage.polarization_grads[:] -= ff.storage.polarization_grads
        @views ff.storage.charge_transfer_grads[:] -= ff.storage.charge_transfer_grads
    end
    for key in keys(ff.results.energies)
        ff.results.energies[key] = 0.0
    end
    return 0.0
end