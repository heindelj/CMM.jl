
"""
Forms the dipole-dipole interaction tensor. The full interaction tensor is
a 3Nx3N tensor which when multiplied by the induced dipole moments gives
the electric field at each atom. See Eq. 5 of https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.1c00537
for the equation. So, finding the induced dipoles consists of solving the
system of equations Tμ=E.
"""
function add_dipole_dipole_interactions_to_polarization_tensor!(
    T::AbstractMatrix{Float64},
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    α_inv::AbstractVector{MMatrix{3, 3, Float64, 9}},
    params::Dict{Symbol, Float64}
)
    natoms = length(labels)
    num_polarizable_sites = natoms
    num_fragments = count(>(0), length.(fragment_indices))
    @assert size(T) == (3 * num_polarizable_sites, 3 * num_polarizable_sites) "Provided interaction tensor is not 3Nx3N where N is number of polarizable sites."

    a = get_a(params, SlaterOverlapDamping())
    for i_frag in eachindex(fragment_indices)
        for i in fragment_indices[i_frag]
            is_ion_i = is_ion(labels[i])
            b_i = abs(params[Symbol(labels[i], :_b_elec)])
            for j_frag in eachindex(fragment_indices)
                if i_frag < j_frag
                    for j in fragment_indices[j_frag]
                        is_ion_j = is_ion(labels[j])
                        b_j = abs(params[Symbol(labels[j], :_b_elec)])
                        if i != j
                            r_vec = coords[j] - coords[i]
                            r_ij = norm(r_vec)
                            b_ij = sqrt(b_i * b_j)
                            br = b_ij * r_ij
                            u_overlap = br

                            λ3 = get_λ3(u_overlap, a, SlaterPolarizationDamping())
                            λ5 = get_λ5(u_overlap, a, SlaterPolarizationDamping())
                            
                            @views T[(3*(i-1)+1):3*i, (3*(j-1)+1):3*j] = (
                                -(λ5 * 3 * r_vec * r_vec' / r_ij^2 - λ3 * diagm(ones(3))) / r_ij^3
                            )
                            @views T[(3*(j-1)+1):3*j, (3*(i-1)+1):3*i] = (
                                -(λ5 * 3 * r_vec * r_vec' / r_ij^2 - λ3 * diagm(ones(3))) / r_ij^3
                            )
                        end
                    end
                end
            end
        end
    end

    for i_frag in eachindex(fragment_indices)
        for i in fragment_indices[i_frag]
            @views T[3*(i-1)+1:3*i, 3*(i-1)+1:3*i] = α_inv[i]
        end
    end
end

"""
Adds damped charge-dipole interaction tensors to the total T tensor.
"""
function add_charge_dipole_interactions_to_polarization_tensor!(
    T_upper_right::AbstractMatrix{Float64},
    T_lower_left::AbstractMatrix{Float64},
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol, Float64}
)

    natoms = length(labels)
    num_polarizable_sites = natoms
    num_fragments = count(>(0), length.(fragment_indices))

    @assert size(T_upper_right) == (num_polarizable_sites, 3 * num_polarizable_sites) "Provided upper right interaction tensor is not 3NxN where N is the number of polarizable sites."
    @assert size(T_lower_left) == (3 * num_polarizable_sites, num_polarizable_sites) "Provided lower left interaction tensor is not Nx3N where N is the number of polarizable sites."
    a = get_a(params, SlaterOverlapDamping())
    for i_frag in eachindex(fragment_indices)
        for i in fragment_indices[i_frag]
            is_ion_i = is_ion(labels[i])
            b_i = abs(params[Symbol(labels[i], :_b_elec)])
            for j_frag in eachindex(fragment_indices)
                if i_frag < j_frag
                    for j in fragment_indices[j_frag]
                        is_ion_j = is_ion(labels[j])
                        b_j = abs(params[Symbol(labels[j], :_b_elec)])
                        if i != j
                            r_vec = coords[j] - coords[i]
                            r_ij = norm(r_vec)
                            b_ij = sqrt(b_i * b_j)
                            br = b_ij * r_ij
                            u_overlap = br

                            λ3 = get_λ3(u_overlap, a, SlaterPolarizationDamping())

                            # charge-dipole i->j #
                            @views T_upper_right[i, (3*(j-1)+1):3*j] = (
                                λ3 * -r_vec / r_ij^3
                            )
                            # charge-dipole j->i #
                            @views T_upper_right[j, (3*(i-1)+1):3*i] = (
                                λ3 * r_vec / r_ij^3
                            )

                            # dipole-charge i->j #
                            @views T_lower_left[(3*(i-1)+1):3*i, j] = (
                                λ3 * r_vec / r_ij^3
                            )
                            # dipole-charge j->i #
                            @views T_lower_left[(3*(j-1)+1):3*j, i] = (
                                λ3 * -r_vec / r_ij^3
                            )
                        end
                    end
                end
            end
        end
    end
end

"""
Takes a matrix block and fills out the elements with charge-charge interactions
damped by a slater Tang-Toennies like term. Currently, the damping is hard-coded,
but in the future the damping will be determined by a function
which can be provided by the caller.
"""
function add_charge_charge_interactions_to_polarization_matrix!(
    T::AbstractMatrix{Float64},
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::Vector{Vector{Int}},
    η_fq::AbstractVector{Float64},
    params::Dict{Symbol, Float64}
)
    natoms = length(labels)
    num_fragments = count(>(0), length.(fragment_indices))
    num_polarizable_sites = natoms
    @assert length(T) == (num_polarizable_sites + num_fragments) * (num_polarizable_sites + num_fragments)

    a = 1.0
    for i_frag in eachindex(fragment_indices)
        for i in fragment_indices[i_frag]
            # fill diagonal with hardness parameters and add charge neutrality
            # constraints for each fragment
            T[i, i] = 2.0 * η_fq[i]
            T[num_polarizable_sites+i_frag, i] = 1.0
            T[i, num_polarizable_sites+i_frag] = 1.0
            b_i = abs(params[Symbol(labels[i], :_b_elec)])
            for j_frag in eachindex(fragment_indices)
                if i_frag < j_frag
                    for j in fragment_indices[j_frag]
                        b_j = abs(params[Symbol(labels[j], :_b_elec)])
                        if i != j
                            r_ij = norm(coords[j] - coords[i])
                            b_ij = sqrt(b_i * b_j)
                            br = b_ij * r_ij
                            u_overlap = br
                            
                            # Shell i to Shell j #
                            λ1 = get_λ1(u_overlap, a, SlaterPolarizationDamping())
                            T[i, j] = λ1 / r_ij
                            T[j, i] = λ1 / r_ij
                        end
                    end
                end
            end
        end
    end
end

"""
Computes the polarization gradients which are the same as the
regular electrostatic gradients except we include damping.
Also, we accumulate the induced potentials and fields for use
in the variable charge gradients and torque calculation respectively.
"""
function polarization_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    permanent_multipoles::AbstractVector{Multipole1},
    induced_multipoles::AbstractVector{Multipole1},
    ϕ::AbstractVector{Float64},
    E_field::AbstractVector{MVector{3, Float64}},
    grads::AbstractVector{MVector{3, Float64}},
    params::Dict{Symbol, Float64},
)

    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                q_i = permanent_multipoles[i].q
                q_i_ind = induced_multipoles[i].q
                @views μ_i = permanent_multipoles[i].μ
                @views μ_i_ind = induced_multipoles[i].μ
                b_i = params[Symbol(labels[i], :_b_exch)]
                for j in fragment_indices[j_frag]
                    q_j = permanent_multipoles[j].q
                    q_j_ind = induced_multipoles[j].q
                    @views μ_j = permanent_multipoles[j].μ
                    @views μ_j_ind = induced_multipoles[j].μ
                    b_j = params[Symbol(labels[j], :_b_exch)]
                    b_ij = sqrt(b_i * b_j)
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    x = slater_damping_value(r_ij, b_ij)
                    # induced charge damping gradient
                    a_damp_δq = inc_gamma(x, 1)
                    a_damp_δq_gradient = inc_gamma_derivative(x, 1) * slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    # induced charge induced dipole damping gradient
                    a_damp_δq_μ_ind = inc_gamma(x, 2)
                    a_damp_δq_μ_ind_gradient = inc_gamma_derivative(x, 2) * slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    # induced dipole damping gradient
                    a_damp_μ_ind = inc_gamma(x, 3)
                    a_damp_μ_ind_gradient = inc_gamma_derivative(x, 3) * slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    
                    # electric fields
                    E_field_qi     = get_electric_field_charge(q_i, -r_ij_vec)
                    E_field_qi_ind = get_electric_field_charge(q_i_ind, -r_ij_vec)
                    E_field_qj     = get_electric_field_charge(q_j, r_ij_vec)
                    E_field_qj_ind = get_electric_field_charge(q_j_ind, r_ij_vec)
                    E_field_μi     = get_electric_field_dipole(μ_i, -r_ij_vec)
                    E_field_μi_ind = get_electric_field_dipole(μ_i_ind, -r_ij_vec)
                    E_field_μj     = get_electric_field_dipole(μ_j, r_ij_vec)
                    E_field_μj_ind = get_electric_field_dipole(μ_j_ind, r_ij_vec)

                    # add induced electric potentials to total potential
                    ϕ[i] += get_electric_potential_charge(q_j_ind, r_ij)
                    ϕ[j] += get_electric_potential_charge(q_i_ind, r_ij)
                    ϕ[i] += get_electric_potential_dipole(μ_j_ind, -r_ij_vec)
                    ϕ[j] += get_electric_potential_dipole(μ_i_ind,  r_ij_vec)

                    # add induced fields to total field
                    E_field[i] += E_field_qj_ind
                    E_field[i] += E_field_μj_ind
                    E_field[j] += E_field_qi_ind
                    E_field[j] += E_field_μi_ind
                    
                    ### gradient of charge-charge interaction ###
                    grads[i] -= q_i_ind * E_field_qj
                    grads[j] += q_i_ind * E_field_qj
                    grads[j] -= q_j_ind * E_field_qi
                    grads[i] += q_j_ind * E_field_qi
                    grads[i] -= a_damp_δq * q_i_ind * E_field_qj_ind - q_i_ind * q_j_ind * a_damp_δq_gradient / r_ij
                    grads[j] += a_damp_δq * q_i_ind * E_field_qj_ind - q_i_ind * q_j_ind * a_damp_δq_gradient / r_ij

                    ### gradient of charge-dipole interaction ###
                    grads[i] -= q_i_ind * E_field_μj
                    grads[j] += q_i_ind * E_field_μj
                    grads[j] -= q_j_ind * E_field_μi
                    grads[i] += q_j_ind * E_field_μi

                    grads[i] -= q_i * E_field_μj_ind
                    grads[j] += q_i * E_field_μj_ind
                    grads[j] -= q_j * E_field_μi_ind
                    grads[i] += q_j * E_field_μi_ind

                    grads[i] -= a_damp_δq_μ_ind * q_i_ind * E_field_μj_ind + μ_j_ind ⋅ E_field_qi_ind * a_damp_δq_μ_ind_gradient
                    grads[j] += a_damp_δq_μ_ind * q_i_ind * E_field_μj_ind + μ_j_ind ⋅ E_field_qi_ind * a_damp_δq_μ_ind_gradient
                    grads[j] -= a_damp_δq_μ_ind * q_j_ind * E_field_μi_ind - μ_i_ind ⋅ E_field_qj_ind * a_damp_δq_μ_ind_gradient
                    grads[i] += a_damp_δq_μ_ind * q_j_ind * E_field_μi_ind - μ_i_ind ⋅ E_field_qj_ind * a_damp_δq_μ_ind_gradient

                    ### induced_dipole_i dipole_j ###
                    μ_i_mag = norm(μ_i_ind)
                    μ_j_mag = norm(μ_j)
                    if μ_i_mag > 0.0 && μ_j_mag > 0.0
                        ei = normalize(μ_i_ind)
                        ej = normalize(μ_j)
                        r_ij_hat = normalize(r_ij_vec)
                        ci = ei ⋅ r_ij_hat
                        cj = ej ⋅ r_ij_hat
                        cij = ei ⋅ ej
                        dip_dip_grad = 3.0 * (
                            (cij - 5.0 * ci * cj) * r_ij_hat + cj * ei + ci * ej
                        ) * μ_i_mag * μ_j_mag / r_ij^4
                        grads[i] += dip_dip_grad
                        grads[j] -= dip_dip_grad
                    end

                    ### dipole_i induced_dipole_j ###
                    μ_i_mag = norm(μ_i)
                    μ_j_mag = norm(μ_j_ind)
                    if μ_i_mag > 0.0 && μ_j_mag > 0.0
                        ei = normalize(μ_i)
                        ej = normalize(μ_j_ind)
                        r_ij_hat = normalize(r_ij_vec)
                        ci = ei ⋅ r_ij_hat
                        cj = ej ⋅ r_ij_hat
                        cij = ei ⋅ ej
                        dip_dip_grad = 3.0 * (
                            (cij - 5.0 * ci * cj) * r_ij_hat + cj * ei + ci * ej
                        ) * μ_i_mag * μ_j_mag / r_ij^4
                        grads[i] += dip_dip_grad
                        grads[j] -= dip_dip_grad
                    end

                    ### induced dipole-induced dipole interaction ###
                    μ_i_mag = norm(μ_i_ind)
                    μ_j_mag = norm(μ_j_ind)
                    if μ_i_mag > 0.0 && μ_j_mag > 0.0
                        ei = normalize(μ_i_ind)
                        ej = normalize(μ_j_ind)
                        r_ij_hat = normalize(r_ij_vec)
                        ci = ei ⋅ r_ij_hat
                        cj = ej ⋅ r_ij_hat
                        cij = ei ⋅ ej
                        dip_dip_grad = 3.0 * (
                            (cij - 5.0 * ci * cj) * r_ij_hat + cj * ei + ci * ej
                        ) * μ_i_mag * μ_j_mag / r_ij^4
                        grads[i] -= a_damp_μ_ind * dip_dip_grad - μ_i_ind ⋅ E_field_μj_ind * a_damp_μ_ind_gradient
                        grads[j] += a_damp_μ_ind * dip_dip_grad - μ_i_ind ⋅ E_field_μj_ind * a_damp_μ_ind_gradient
                    end
                end
            end
        end
    end
end

function polarization_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    permanent_multipoles::AbstractVector{Multipole2},
    induced_multipoles::AbstractVector{Multipole1},
    ϕ::AbstractVector{Float64},
    E_field::AbstractVector{MVector{3, Float64}},
    E_field_gradients::AbstractVector{MMatrix{3, 3, Float64, 9}},
    grads::AbstractVector{MVector{3, Float64}},
    params::Dict{Symbol, Float64}
)

    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                q_i = permanent_multipoles[i].q
                q_i_ind = induced_multipoles[i].q
                @views μ_i = permanent_multipoles[i].μ
                @views μ_i_ind = induced_multipoles[i].μ
                b_i = params[Symbol(labels[i], :_b_exch)]
                for j in fragment_indices[j_frag]
                    q_j = permanent_multipoles[j].q
                    q_j_ind = induced_multipoles[j].q
                    @views μ_j = permanent_multipoles[j].μ
                    @views μ_j_ind = induced_multipoles[j].μ
                    b_j = params[Symbol(labels[j], :_b_exch)]
                    b_ij = sqrt(b_i * b_j)
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    x = slater_damping_value(r_ij, b_ij)
                    # induced charge damping gradient
                    a_damp_δq = inc_gamma(x, 1)
                    a_damp_δq_gradient = inc_gamma_derivative(x, 1) * slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    # induced charge induced dipole damping gradient
                    a_damp_δq_μ_ind = inc_gamma(x, 2)
                    a_damp_δq_μ_ind_gradient = inc_gamma_derivative(x, 2) * slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    # induced dipole damping gradient
                    a_damp_μ_ind = inc_gamma(x, 3)
                    a_damp_μ_ind_gradient = inc_gamma_derivative(x, 3) * slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    
                    # electric fields
                    E_field_qi     = get_electric_field_charge(q_i, -r_ij_vec)
                    E_field_qi_ind = get_electric_field_charge(q_i_ind, -r_ij_vec)
                    E_field_qj     = get_electric_field_charge(q_j, r_ij_vec)
                    E_field_qj_ind = get_electric_field_charge(q_j_ind, r_ij_vec)
                    E_field_μi     = get_electric_field_dipole(μ_i, -r_ij_vec)
                    E_field_μi_ind = get_electric_field_dipole(μ_i_ind, -r_ij_vec)
                    E_field_μj     = get_electric_field_dipole(μ_j, r_ij_vec)
                    E_field_μj_ind = get_electric_field_dipole(μ_j_ind, r_ij_vec)

                    # add induced electric potentials to total potential
                    ϕ[i] += get_electric_potential_charge(q_j_ind, r_ij)
                    ϕ[j] += get_electric_potential_charge(q_i_ind, r_ij)
                    ϕ[i] += get_electric_potential_dipole(μ_j_ind, -r_ij_vec)
                    ϕ[j] += get_electric_potential_dipole(μ_i_ind,  r_ij_vec)

                    # add induced fields to total field
                    E_field[i] += E_field_qj_ind
                    E_field[i] += E_field_μj_ind
                    E_field[j] += E_field_qi_ind
                    E_field[j] += E_field_μi_ind
                    
                    ### gradient of charge-charge interaction ###
                    grads[i] -= q_i_ind * E_field_qj
                    grads[j] += q_i_ind * E_field_qj
                    grads[j] -= q_j_ind * E_field_qi
                    grads[i] += q_j_ind * E_field_qi
                    grads[i] -= a_damp_δq * q_i_ind * E_field_qj_ind - q_i_ind * q_j_ind * a_damp_δq_gradient / r_ij
                    grads[j] += a_damp_δq * q_i_ind * E_field_qj_ind - q_i_ind * q_j_ind * a_damp_δq_gradient / r_ij

                    ### gradient of charge-dipole interaction ###
                    grads[i] -= q_i_ind * E_field_μj
                    grads[j] += q_i_ind * E_field_μj
                    grads[j] -= q_j_ind * E_field_μi
                    grads[i] += q_j_ind * E_field_μi

                    grads[i] -= q_i * E_field_μj_ind
                    grads[j] += q_i * E_field_μj_ind
                    grads[j] -= q_j * E_field_μi_ind
                    grads[i] += q_j * E_field_μi_ind

                    grads[i] -= a_damp_δq_μ_ind * q_i_ind * E_field_μj_ind + μ_j_ind ⋅ E_field_qi_ind * a_damp_δq_μ_ind_gradient
                    grads[j] += a_damp_δq_μ_ind * q_i_ind * E_field_μj_ind + μ_j_ind ⋅ E_field_qi_ind * a_damp_δq_μ_ind_gradient
                    grads[j] -= a_damp_δq_μ_ind * q_j_ind * E_field_μi_ind - μ_i_ind ⋅ E_field_qj_ind * a_damp_δq_μ_ind_gradient
                    grads[i] += a_damp_δq_μ_ind * q_j_ind * E_field_μi_ind - μ_i_ind ⋅ E_field_qj_ind * a_damp_δq_μ_ind_gradient

                    ### induced_dipole_i dipole_j ###
                    μ_i_mag = norm(μ_i_ind)
                    μ_j_mag = norm(μ_j)
                    if μ_i_mag > 0.0 && μ_j_mag > 0.0
                        ei = normalize(μ_i_ind)
                        ej = normalize(μ_j)
                        r_ij_hat = normalize(r_ij_vec)
                        ci = ei ⋅ r_ij_hat
                        cj = ej ⋅ r_ij_hat
                        cij = ei ⋅ ej
                        dip_dip_grad = 3.0 * (
                            (cij - 5.0 * ci * cj) * r_ij_hat + cj * ei + ci * ej
                        ) * μ_i_mag * μ_j_mag / r_ij^4
                        grads[i] += dip_dip_grad
                        grads[j] -= dip_dip_grad
                    end

                    ### dipole_i induced_dipole_j ###
                    μ_i_mag = norm(μ_i)
                    μ_j_mag = norm(μ_j_ind)
                    if μ_i_mag > 0.0 && μ_j_mag > 0.0
                        ei = normalize(μ_i)
                        ej = normalize(μ_j_ind)
                        r_ij_hat = normalize(r_ij_vec)
                        ci = ei ⋅ r_ij_hat
                        cj = ej ⋅ r_ij_hat
                        cij = ei ⋅ ej
                        dip_dip_grad = 3.0 * (
                            (cij - 5.0 * ci * cj) * r_ij_hat + cj * ei + ci * ej
                        ) * μ_i_mag * μ_j_mag / r_ij^4
                        grads[i] += dip_dip_grad
                        grads[j] -= dip_dip_grad
                    end

                    ### induced dipole-induced dipole interaction ###
                    μ_i_mag = norm(μ_i_ind)
                    μ_j_mag = norm(μ_j_ind)
                    if μ_i_mag > 0.0 && μ_j_mag > 0.0
                        ei = normalize(μ_i_ind)
                        ej = normalize(μ_j_ind)
                        r_ij_hat = normalize(r_ij_vec)
                        ci = ei ⋅ r_ij_hat
                        cj = ej ⋅ r_ij_hat
                        cij = ei ⋅ ej
                        dip_dip_grad = 3.0 * (
                            (cij - 5.0 * ci * cj) * r_ij_hat + cj * ei + ci * ej
                        ) * μ_i_mag * μ_j_mag / r_ij^4
                        grads[i] -= a_damp_μ_ind * dip_dip_grad - μ_i_ind ⋅ E_field_μj_ind * a_damp_μ_ind_gradient
                        grads[j] += a_damp_μ_ind * dip_dip_grad - μ_i_ind ⋅ E_field_μj_ind * a_damp_μ_ind_gradient
                    end
                end
            end
        end
    end
end