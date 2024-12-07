import CMM.CMM_FF

function short_range_energy!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    num_fragments = ff.storage.num_fragments

    dispersion_charges = [sqrt(abs(ff.params[Symbol(labels[i], :_C6_disp)])) for i in eachindex(labels)]
    multipolar_dispersion!(
        coords, labels, fragment_indices, ff.params,
        ff.storage.multipoles,
        dispersion_charges,
        ff.storage.ϕ_dispersion,
        ff.storage.E_field_dispersion,
        ff.storage.E_field_gradients_dispersion,
        SlaterOverlapDamping()
    )

    # NOTE: I am working with the convention that all dispersion coefficients
    # are positive and therefore, the energy calculated in this way will be
    # negative when I subtract the dispersion coefficient times the potential.
    # The potential itself (barring any massive anisotropies) must be positive
    # since it is just sqrt(C_6,j)/r_ij^6.
    # Therefore, we also invert the signs of everything else so that the field
    # a dipole interacts with and the force due to a charge have the same parity.
    dispersion_energy = 0.0
    for i in eachindex(ff.storage.multipoles)
        #K_i_μ = ff.params[Symbol(labels[i], :_K_dispersion_μ)]
        #K_i_Q = ff.params[Symbol(labels[i], :_K_dispersion_Q)]
        dispersion_energy -= dispersion_charges[i] * ff.storage.ϕ_dispersion[i]
        #dispersion_energy += K_i_μ * ff.storage.multipoles[i].μ ⋅ ff.storage.E_field_dispersion[i]
        #dispersion_energy += K_i_Q * ff.storage.multipoles[i].Q ⋅ ff.storage.E_field_gradients_dispersion[i] / 3.0
    end
    dispersion_energy *= 0.5

    if ff.storage.include_charge_transfer
        get_explicit_charge_transfer!(
            coords, labels, fragment_indices,
            ff.params, ff.storage.Δq_ct
        )
        
        ct_donor_charges = [ff.params[Symbol(labels[i], :_q_ct_donor)] for i in eachindex(labels)]
        ct_acceptor_charges = [ff.params[Symbol(labels[i], :_q_ct_acceptor)] for i in eachindex(labels)]
        multipolar_charge_transfer!(
            coords, labels, fragment_indices, ff.params,
            ff.storage.multipoles, ff.storage.E_field_core,
            ct_donor_charges, ct_acceptor_charges,
            ff.storage.ϕ_donor, ff.storage.ϕ_acceptor,
            ff.storage.E_field_donor, ff.storage.E_field_acceptor,
            ff.storage.E_field_gradients_donor, ff.storage.E_field_gradients_acceptor,
            SlaterOverlapDamping()
        )
        
        direct_ct_energy = 0.0
        for i in eachindex(ff.storage.multipoles)
            K_i_donor_μ = ff.params[Symbol(labels[i], :_K_ct_donor_μ)]
            K_i_donor_Q = ff.params[Symbol(labels[i], :_K_ct_donor_Q)]
            direct_ct_energy += ct_acceptor_charges[i] * ff.storage.ϕ_acceptor[i]
            direct_ct_energy += ct_donor_charges[i] * ff.storage.ϕ_donor[i]
            direct_ct_energy -= K_i_donor_μ * ff.storage.multipoles[i].μ ⋅ ff.storage.E_field_donor[i]
            direct_ct_energy -= K_i_donor_Q * ff.storage.multipoles[i].Q ⋅ ff.storage.E_field_gradients_donor[i] / 3.0
        end
        direct_ct_energy *= 0.5
        ff.results.energies[:CT_direct] = direct_ct_energy
    end

    ff.results.energies[:Dispersion] = dispersion_energy
    # ^^^ The actual charge transfer energy also has a component
    # associated with how it affects polarization. This cannot be
    # computed without doing a CT-free calculation, so we only report
    # the direct part of CT which is easy to compute.

    repulsion_charges = [ff.params[Symbol(labels[i], :_q_repulsion)] for i in eachindex(labels)]
    # TODO: This can be stored ahead of time ^^^
    update_repulsion_charges_with_charge_flux!(coords, labels, fragment_indices, repulsion_charges, ff.params, nothing)

    display(repulsion_charges)

    multipolar_pauli_repulsion!(
        coords, labels, fragment_indices, ff.params,
        ff.storage.multipoles,
        repulsion_charges,
        ff.storage.ϕ_repulsion,
        ff.storage.E_field_repulsion,
        ff.storage.E_field_gradients_repulsion,
        SlaterOverlapDamping()
    )
    exchange_energy = 0.0
    for i in eachindex(ff.storage.multipoles)
        K_i_μ = ff.params[Symbol(labels[i], :_K_repulsion_μ)]
        K_i_Q = ff.params[Symbol(labels[i], :_K_repulsion_Q)]
        exchange_energy += repulsion_charges[i] * ff.storage.ϕ_repulsion[i]
        exchange_energy -= K_i_μ * ff.storage.multipoles[i].μ ⋅ ff.storage.E_field_repulsion[i]
        exchange_energy -= K_i_Q * ff.storage.multipoles[i].Q ⋅ ff.storage.E_field_gradients_repulsion[i] / 3.0
    end
    exchange_energy *= 0.5
    
    ff.results.energies[:Pauli] = exchange_energy

    if ff.storage.include_charge_transfer == false
        ff.results.energies[:Total] += dispersion_energy + exchange_energy
        return exchange_energy + dispersion_energy
    end

    ff.results.energies[:Total] += direct_ct_energy + dispersion_energy + exchange_energy
    return exchange_energy + direct_ct_energy + dispersion_energy
end

function short_range_energy_and_gradients!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    short_range_energy = short_range_energy!(coords, labels, fragment_indices, ff)

    repulsion_charges = [ff.params[Symbol(labels[i], :_q_repulsion)] for i in eachindex(labels)]
    # TODO: This can be stored ahead of time ^^^
    #update_repulsion_charges_with_charge_flux!(coords, labels, fragment_indices, repulsion_charges, ff.params, nothing)

    for i in eachindex(coords)
        K_μ = ff.params[Symbol(labels[i], :_K_repulsion_μ)]
        K_Q = ff.params[Symbol(labels[i], :_K_repulsion_Q)]
        
        ### charges ###
        ff.storage.pauli_grads[i] -= repulsion_charges[i] * ff.storage.E_field_repulsion[i]
        
        ### dipoles ###
        ff.storage.pauli_grads[i] -= K_μ * ff.storage.E_field_gradients_repulsion[i] * ff.storage.multipoles[i].μ

        ### quadrupoles ###
        #@views ff.storage.pauli_grads[i][1] -= ff.storage.E_field_gradient_gradients_repulsion[i][:, :, 1] ⋅ ff.storage.multipoles[i].Q  #/ 3.0
        #@views ff.storage.pauli_grads[i][2] -= ff.storage.E_field_gradient_gradients_repulsion[i][:, :, 2] ⋅ ff.storage.multipoles[i].Q  #/ 3.0
        #@views ff.storage.pauli_grads[i][3] -= ff.storage.E_field_gradient_gradients_repulsion[i][:, :, 3] ⋅ ff.storage.multipoles[i].Q  #/ 3.0
    end

    # JOE! Look here if you are confused about gradients for this term!!
    # TODO: Also need to implement the variable repulsion charge gradients (completely analogous to charge flux)
    # TODO: Get torque gradients for repulsion dipoles!
    # TODO: Will have to get gradients for overlap terms here!

    return short_range_energy
end

function short_range_energy_isotropic!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    num_fragments = ff.storage.num_fragments
    @views Δq_ct = ff.storage.b_vec[length(labels)+1:length(labels)+num_fragments]
    exchange_energy, dispersion_energy, cp_energy, direct_ct_energy, fractional_charge_energy = total_short_range_energy!(coords, labels, fragment_indices, ff.params, Δq_ct, ff.storage.C6, ff.storage.include_charge_transfer)
    ff.results.energies[:Pauli] = exchange_energy
    ff.results.energies[:Dispersion] = dispersion_energy
    ff.results.energies[:Electrostatics] = cp_energy
    ff.results.energies[:CT_direct] = direct_ct_energy #+ fractional_charge_energy
    #ff.results.energies[:CT_fractional_charge] = fractional_charge_energy
    ff.results.energies[:Total] += exchange_energy + dispersion_energy + cp_energy + direct_ct_energy #+ fractional_charge_energy
    # ^^^ The actual charge transfer energy also has a component
    # associated with how it affects polarization. This cannot be
    # computed without doing a CT-free calculation, so we only report
    # the direct part of CT which is easy to compute.
    return exchange_energy + dispersion_energy + cp_energy + direct_ct_energy #+ fractional_charge_energy
end

function charge_transfer_constraint_gradients!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::AbstractForceField)
    num_fragments = ff.storage.num_fragments
    @views λ = ff.storage.solution_vector[length(labels)+1:length(labels)+num_fragments]
    evaluate_Δq_grads!(
        coords, labels, fragment_indices,
        ff.params, λ, ff.storage.charge_transfer_grads
    )
    return 0.0
end