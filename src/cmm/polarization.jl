import CMM.CMM_FF

function polarization_energy!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::AbstractForceField)
    num_fragments = ff.storage.num_fragments
    natoms = length(labels)

    #get_model_inverse_polarizabilities!(
    #    ff.storage.α_inv, labels, ff.storage.local_axes, ff.params
    #)

    get_model_inverse_polarizabilities_with_ion_ion_damping!(
        ff.storage.α_inv, labels, ff.storage.E_field_core,
        ff.storage.local_axes, ff.params
    )

    ### Fill quadrupole polarizabilities and rotate to global frame ###
    # The allocation required here is temporary and in the future will be
    # be done ahead of time.
    #α_quad = [@MMatrix zeros(6,6) for _ in eachindex(labels)]
    #get_quadrupole_polarizabilities!(
    #    α_quad, labels, ff.storage.local_axes, ff.params
    #)
    # Get the quad polarization energy by multiplying with the
    # Mandel form of the field gradient at each atom.
    #for i in eachindex(labels)
    #    E_fg = ff.storage.E_field_gradients_shell[i]
    #    E_fg_mandel = [E_fg[1],E_fg[5],E_fg[9],E_fg[8],E_fg[7],E_fg[4]]
    #    display(α_quad[i] * E_fg_mandel ⋅ E_fg_mandel / 6.0)
    #    # HERE: Currently not getting proper rotational invariance
    #    # so there is something wrong with the implementation but not sure what.
    #end

    ### Fill b_vec with appropriate potential and field ###
    # negate ϕ to get correct fluctuating charge energy
    for i in eachindex(labels)
        @views ff.storage.b_vec[i] = -(ff.storage.ϕ_core[i])
    end

    # Fill in fragment specific charges due to charge transfer
    for i_frag in eachindex(fragment_indices)
        @views ff.storage.b_vec[natoms+i_frag] = sum(ff.storage.Δq_ct[fragment_indices[i_frag]])
    end

    # Now add the field at each polarizable site to the b_vec
    @views E_field_damped = reshape(ff.storage.b_vec[(natoms+num_fragments+1):end], (3, :))
    for i in eachindex(labels)
        @views E_field_damped[:, i] = (
            ff.storage.E_field_core[i] + ff.storage.applied_field
        )
    end

    # Apply external field if present
    if norm(ff.storage.applied_field) > 1e-10
        for i in eachindex(coords)
            ff.storage.b_vec[i] += coords[i] ⋅ ff.storage.applied_field
        end
    end

    # Option to solve polarization equations by SCF or by direct inversion.
    # Obviously this needs to be an option a user can actually set but while
    # implementing, I'll have it like this. 6/25/24
    solve_by_direct_inversion = true
    if solve_by_direct_inversion
        ### Multipolar Polarization Interactions ###
        # charge - charge interactions
        @views add_charge_charge_interactions_to_polarization_matrix!(
            ff.storage.polarization_matrix[1:(natoms+num_fragments), 1:(natoms+num_fragments)],
            coords,
            labels,
            fragment_indices,
            ff.storage.η_fq,
            ff.params
        )
        # dipole - dipole interactions
        @views add_dipole_dipole_interactions_to_polarization_tensor!(
            ff.storage.polarization_matrix[(natoms+num_fragments+1):end, (natoms+num_fragments+1):end],
            coords,
            labels,
            fragment_indices,
            ff.storage.α_inv,
            ff.params
        )
        # charge - dipole interactions
        @views add_charge_dipole_interactions_to_polarization_tensor!(
            ff.storage.polarization_matrix[1:natoms, (natoms+num_fragments+1):end],
            ff.storage.polarization_matrix[(natoms+num_fragments+1):end, 1:natoms],
            coords,
            labels,
            fragment_indices,
            ff.params
        )

        # Solve the polarization equations to get charge and dipole rearrangements
        @views ff.storage.solution_vector[:] = ff.storage.polarization_matrix \ ff.storage.b_vec
        polarization_energy = get_polarization_energy(ff.storage.solution_vector, ff.storage.polarization_matrix, ff.storage.b_vec)
        
        @views δq = ff.storage.solution_vector[1:natoms]
        @views induced_dipoles = reshape(ff.storage.solution_vector[(natoms+num_fragments+1):end], (3, :))
        for i in eachindex(ff.storage.induced_multipoles)
            ff.storage.induced_multipoles[i].q_shell = δq[i]
            @views ff.storage.induced_multipoles[i].μ = induced_dipoles[:, i]
        end

        # Get induced electrostatics quantities for bond polarization and
        # gradients if needed
        get_all_induced_electrostatic_quantities!(
            coords,
            labels,
            fragment_indices,
            ff.storage.induced_multipoles,
            ff.storage.ϕ_induced,
            ff.storage.E_field_induced,
            ff.storage.E_field_gradients_induced,
            ff.params,
            SlaterPolarizationDamping()
        )

    else
        # TODO: This is just unreachable code right now. Will eventually switch to
        # iterative solution (probably above a certain system size). 6/27/24

        # Solve by iteration

        # The algorithm will basically be to do Jacobi method to fill solution vector
        # Then from that fill the potential, field, and so on.
        # Then check that we satisfy the charge constraints to within tolerance
        # and satisfy the convergence of the induced potential and field or dipoles
        # or whatever.
        # Will have to do a swap of solution_vector_1 and 2 on each iteration
        # Then will fill induced_multipoles array with the appropriate components
        # of the mixture of solution_vectors.

        max_iterations = 200
        constraint_tolerance = 1e-8
        charge_tolerance = 1e-8
        dip_tolerance = 1e-8

        mean_constraint_error = 1.0
        mean_charge_error = 1.0
        mean_dipole_error = 1.0
        induced_multipoles_2 = copy(ff.storage.induced_multipoles)
        λ_lagrange = zeros(length(fragment_indices))

        λ_mix = 0.7
        for i in 1:max_iterations
            @views ff.storage.ϕ_induced[:] -= ff.storage.ϕ_induced
            @views ff.storage.E_field_induced[:] -= ff.storage.E_field_induced
            # Get updated potentials and fields from new
            # induced multipoles
            get_all_induced_electrostatic_quantities!(
                coords,
                labels,
                fragment_indices,
                ff.storage.induced_multipoles,
                ff.storage.ϕ_induced,
                ff.storage.E_field_induced,
                ff.storage.E_field_gradients_induced,
                ff.params,
                SlaterPolarizationDamping()
            )



            # Compute change in induced charges, dipoles, and lagrange multipliers
            # from previous and current step.
            
        end
    end

    if ff.storage.include_exch_pol
        exch_pol_charges = [ff.params[Symbol(labels[i], :_q_exch_pol)] for i in eachindex(labels)]
        multipolar_exchange_polarization!(
            coords, labels, fragment_indices, ff.params,
            exch_pol_charges,
            ff.storage.ϕ_exch_pol,
            ff.storage.E_field_exch_pol,
            SlaterOverlapDamping()
        )
        short_range_polarization_energy = 0.0
        for i in eachindex(ff.storage.multipoles)
            q_exch_pol_i = ff.params[Symbol(labels[i], :_q_exch_pol)]
            short_range_polarization_energy += q_exch_pol_i * ff.storage.ϕ_exch_pol[i]

            # NOTE(JOE): Using dipoles in the exchange polarization has never
            # been all that useful. My suspicion is that it could be more useful
            # if we had an adequate definition of exchange polarization but
            # currently we just can't tell what's going on since exch. pol. cannot
            # be separated from the total polarization energy. It's commented out
            # because it isn't used but I'm just leaving it there cause eventually
            # I think this will become useful. -Joe 6/17/24

            #K_i_μ = ff.params[Symbol(labels[i], :_K_exch_pol_μ)]
            #μ_exch_pol_i = K_i_μ * ff.storage.multipoles[i].μ
            #if is_ion(labels[i])
            #    μ_exch_pol_i = K_i_μ * normalize(ff.storage.E_field_induced[i])
            #end
            #short_range_polarization_energy -= μ_exch_pol_i ⋅ ff.storage.E_field_exch_pol[i]
        end
        short_range_polarization_energy *= 0.5
        polarization_energy += short_range_polarization_energy
    end

    # Deformation energy correction for Badger rule
    μ_1 = ff.params[:OH_dip_deriv_1]
    μ_2 = ff.params[:OH_dip_deriv_2]
    ct_slope1 = ff.params[:OH_dip_deriv_slope_ct_1]
    ct_slope2 = ff.params[:OH_dip_deriv_slope_ct_2]
    one_body_energy = get_field_dependent_morse_and_bend_energy_and_grads!(
        coords, labels, fragment_indices, ff.params,
        ff.storage.E_field_core,
        μ_1, μ_2, 0.0, 0.0,
        ff.storage.Δq_ct, 0.0, 0.0, false,
        nothing, nothing
        #ff.storage.electrostatic_grads,
        #ff.storage.E_field_gradients_core + ff.storage.E_field_gradients_shell
    )
    one_body_energy_with_response = get_field_dependent_morse_and_bend_energy_and_grads!(
        coords, labels, fragment_indices, ff.params,
        ff.storage.E_field_core + ff.storage.E_field_induced,
        μ_1, μ_2, 0.0, 0.0,
        ff.storage.Δq_ct, ct_slope1, ct_slope2,
        ff.storage.include_charge_transfer,
        nothing, nothing
        #ff.storage.polarization_grads, ff.storage.E_field_gradients_induced
    )
    bond_pol_energy = (one_body_energy_with_response - one_body_energy)
    ff.results.energies[:bond_pol_induced] = bond_pol_energy
    polarization_energy += bond_pol_energy
    
    # NOTE: When charge transfer is included in the polarization calculation
    # then the polarization energy here includes part of the CT energy.
    # The two can only be separated by doing a calculation without CT
    # and taking the difference.
    if ff.storage.include_exch_pol
        ff.results.energies[:ShortRangePolarization] = short_range_polarization_energy
    end
    ff.results.energies[:Polarization] = polarization_energy
    ff.results.energies[:Total] += polarization_energy
    return polarization_energy
end

function polarization_energy_and_electrostatic_gradients!(coords::AbstractVector{MVector{3, Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, ff::CMM_FF)
    num_fragments = ff.storage.num_fragments
    # set up polarization matrix and solve system of equations
    #polarization_energy = polarization_energy!(coords, labels, fragment_indices, ff)
    #exchange_polarization_energy_gradients!(coords, labels, fragment_indices, ff.params, ff.storage.polarization_grads)

    #get_all_damped_electrostatic_quantities!(
    #    coords,
    #    labels,
    #    fragment_indices,
    #    ff.storage.induced_multipoles,
    #    ff.storage.ϕ_induced,
    #    ff.storage.E_field_induced,
    #    ff.storage.E_field_gradients_induced,
    #    ff.params,
    #    ff.storage.damping_type
    #)

    get_core_shell_electrostatic_and_polarization_gradients!(
        coords, labels, fragment_indices,
        ff.storage.multipoles, ff.storage.induced_multipoles,
        ff.storage.E_field_core, ff.storage.E_field_shell,
        ff.storage.E_field_induced,
        ff.storage.E_field_gradients_core, ff.storage.E_field_gradients_shell,
        ff.storage.E_field_gradients_induced,
        ff.storage.E_field_gradient_gradients_core, ff.storage.E_field_gradient_gradients_shell,
        ff.storage.E_field_gradient_gradients_induced,
        ff.storage.electrostatic_grads, ff.storage.polarization_grads, ff.params,
        ff.storage.shell_damping_type, ff.storage.overlap_damping_type
    )

    #get_mutual_polarization_gradients!(
    #    coords, labels, fragment_indices,
    #    ff.storage.induced_multipoles,
    #    ff.storage.polarization_grads, ff.params
    #)

    #get_variable_polarizability_gradients!(
    #    coords, labels, fragment_indices,
    #    ff.storage.induced_multipoles,
    #    ff.storage.polarization_grads, ff.params
    #)

    ### get all torques ###
    for i in eachindex(ff.storage.torques)
        #ff.storage.torques[i] += cross(ff.storage.multipoles[i].μ, ff.storage.E_field_shell[i])
        #torque_on_quadrupole!(ff.storage.torques[i], ff.storage.multipoles[i].Q, ff.storage.E_field_gradients_shell[i])
        # CHANGE ABOVE TO FIELD_GRADS_OVERLAP ONCE YOU GET DAMPERD QUAD_QUAD INTERACTIONS WORKING
    end

    #for i in eachindex(coords)
    #    torque_to_force!(coords, ff.storage.local_axes[i], ff.storage.electrostatic_grads, ff.storage.torques)
    #end

    #for i in eachindex(ff.storage.torques)
    #    ff.storage.torques[i] = cross(ff.storage.multipoles[i].μ, ff.storage.E_field_induced[i])
    #    torque_on_quadrupole!(ff.storage.torques[i], ff.storage.multipoles[i].Q, ff.storage.E_field_gradients_induced[i])
    #end

    #for k in eachindex(labels)
    #    get_dipole_polarizability_derivatives!(ff.storage.α_inv[k], ff.storage.dα[1], ff.storage.dα[2], ff.storage.dα[3])
    #    ff.storage.torques[k][1] -= 0.5 * ff.storage.induced_multipoles[k].μ' * (ff.storage.α_inv[k] * ff.storage.dα[1] * ff.storage.α_inv[k]) * ff.storage.induced_multipoles[k].μ
    #    ff.storage.torques[k][2] -= 0.5 * ff.storage.induced_multipoles[k].μ' * (ff.storage.α_inv[k] * ff.storage.dα[2] * ff.storage.α_inv[k]) * ff.storage.induced_multipoles[k].μ
    #    ff.storage.torques[k][3] -= 0.5 * ff.storage.induced_multipoles[k].μ' * (ff.storage.α_inv[k] * ff.storage.dα[3] * ff.storage.α_inv[k]) * ff.storage.induced_multipoles[k].μ
    #end

    #for i in eachindex(coords)
    #    torque_to_force!(coords, ff.storage.local_axes[i], ff.storage.polarization_grads, ff.storage.torques)
    #end

    ### variable charge gradients ###
    #add_variable_charge_gradients!(ff.storage.electrostatic_grads, ff.storage.one_body_charge_grads, ff.storage.ϕ_shell, fragment_indices)
    #add_variable_charge_gradients!(ff.storage.polarization_grads, ff.storage.one_body_charge_grads, ff.storage.ϕ_induced, fragment_indices)
    return 0.0
    #return polarization_energy
end