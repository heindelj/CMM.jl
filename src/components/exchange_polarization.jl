
"""
Simulates the decrease in energy of polarization due to the exchange interaction.
See the awesome paper: https://pubs.acs.org/doi/pdf/10.1021/j100383a027 (Tang and Toennies)
for a discussion. This function implements Eq. (73), using the slater version of a born-mayer model.

Note that in practice, this seems to give stabilizing energies for water and then
makes very small corrections for ions. Sometimes they are repulsive.
So, basically, it seems to compensate for whatever is missing from the dipole
polarization.
"""
function exchange_polarization_energy(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
)

    exchange_polarization_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    b_i = params[Symbol(labels[i], :_b_elec)]
                    a_exch_pol_i = params[Symbol(labels[i], :_a_exch_pol)]
                    for j in fragment_indices[j_frag]
                        a_exch_pol_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_exch_pol, params)
                        if !has_pairwise
                            a_exch_pol_j = params[Symbol(labels[j], :_a_exch_pol)]
                            a_exch_pol_ij = a_exch_pol_i * a_exch_pol_j
                        end
                        # There is an optional pairwise squared overlap contribution
                        # We don't check if we found the pairwise parameter since it will be zero by default
                        a_exch_pol_sq_ij, _ = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_exch_pol_sq, params)
                        
                        b_j =  params[Symbol(labels[j], :_b_elec)]
                        b_ij = sqrt(b_i * b_j)

                        r_ij = norm(coords[i] - coords[j])
                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        
                        exchange_polarization_energy += (
                            a_exch_pol_ij * slater_overlap +
                            a_exch_pol_sq_ij * slater_overlap^2
                        )
                    end
                end
            end
        end
    end
    return exchange_polarization_energy
end

function exchange_polarization_energy_gradients!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    grads::AbstractVector{MVector{3, Float64}}
)
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    α_i = get_mean_polarizability(labels[i], params)
                    b_i = params[Symbol(labels[i], :_b_exch)]
                    a_exch_pol_i = params[Symbol(labels[i], :_a_exch_pol)]
                    for j in fragment_indices[j_frag]
                        α_j = get_mean_polarizability(labels[j], params)
                        a_exch_pol_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_exch_pol, params)
                        if !has_pairwise
                            a_exch_pol_j = params[Symbol(labels[j], :_a_exch_pol)]
                            a_exch_pol_ij = a_exch_pol_i * a_exch_pol_j
                        end
                        
                        b_j  = params[Symbol(labels[j], :_b_exch)]
                        b_ij = sqrt(b_i * b_j)

                        r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(coords[j] - coords[i])
                        x = slater_damping_value(r_ij, b_ij)
                        x_gradient_i = slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)

                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        slater_overlap_gradient_j = exp(-b_ij * r_ij) * (
                            (2 / 3 * b_ij^2 * r_ij + b_ij) +
                            -b_ij * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        ) * r_ij_vec / r_ij
                        grads[i] -= a_exch_pol_ij * (
                            inc_gamma(x, 3) * (0.5 * (α_i + α_j)) / (r_ij^3) * slater_overlap_gradient_j -
                            (0.5 * (α_i + α_j)) / (r_ij^3) * slater_overlap * inc_gamma_derivative(x, 3) * x_gradient_i +
                            inc_gamma(x, 3) * (-1.5 * (α_i + α_j)) / (r_ij^4) * slater_overlap * r_ij_vec / r_ij
                        )
                        grads[j] += a_exch_pol_ij * (
                            inc_gamma(x, 3) * (0.5 * (α_i + α_j)) / (r_ij^3) * slater_overlap_gradient_j -
                            (0.5 * (α_i + α_j)) / (r_ij^3) * slater_overlap * inc_gamma_derivative(x, 3) * x_gradient_i +
                            inc_gamma(x, 3) * (-1.5 * (α_i + α_j)) / (r_ij^4) * slater_overlap * r_ij_vec / r_ij
                        )
                    end
                end
            end
        end
    end
end