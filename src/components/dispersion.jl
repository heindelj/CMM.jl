
function total_c6_slater_dispersion_energy(coords::Vector{MVector{3,Float64}}, labels::AbstractVector{String}, fragment_indices::Vector{Vector{Int}}, params::Dict{Symbol,Float64})
    dispersion_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        C6_ij = sqrt(abs(params[Symbol(labels[i], :_C6_disp)]) * abs(params[Symbol(labels[j], :_C6_disp)]))
                        b_ij = sqrt(params[Symbol(labels[i], :_b_exch)] * params[Symbol(labels[j], :_b_exch)])

                        r_ij = norm(coords[j] - coords[i])
                        x = slater_damping_value(r_ij, b_ij)
                        dispersion_energy += -inc_gamma(x, 6) * C6_ij / r_ij^6
                    end
                end
            end
        end
    end
    return dispersion_energy
end

function total_two_and_three_body_dispersion_energy(coords::AbstractVector{MVector{3,Float64}}, labels::AbstractVector{String}, fragment_indices::AbstractVector{Vector{Int}}, params::Dict{Symbol,Float64})
    two_body_dispersion_energy = 0.0
    three_body_dispersion_energy = 0.0
    λ_3b = 1.2 #params[:λ_3B]
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    C6_i = abs(params[Symbol(labels[i], :_C6_disp)])
                    α_i = get_mean_polarizability(labels[i], params)
                    b_i = params[Symbol(labels[i], :_b_exch)]
                    for j in fragment_indices[j_frag]
                        C6_j = abs(params[Symbol(labels[j], :_C6_disp)])
                        α_j = get_mean_polarizability(labels[j], params)
                        b_j = params[Symbol(labels[j], :_b_exch)]
                        
                        C6_ij = sqrt(abs(params[Symbol(labels[i], :_C6_disp)]) * abs(params[Symbol(labels[j], :_C6_disp)]))
                        b_ij = sqrt(params[Symbol(labels[i], :_b_exch)] * params[Symbol(labels[j], :_b_exch)])

                        @views r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(r_ij_vec)
                        x_ij = slater_damping_value(r_ij, b_ij)
                        two_body_dispersion_energy += -inc_gamma(x_ij, 6) * C6_ij / r_ij^6
                        for k_frag in eachindex(fragment_indices)
                            if j_frag < k_frag
                                for k in fragment_indices[k_frag]
                                    C6_k = abs(params[Symbol(labels[k], :_C6_disp)])
                                    α_k = get_mean_polarizability(labels[k], params)
                                    b_k = params[Symbol(labels[k], :_b_exch)]
                                    b_ik = sqrt(b_i * b_k)
                                    b_jk = sqrt(b_j * b_k)
                        
                                    S_i = C6_i * α_j * α_k / α_i
                                    S_j = C6_j * α_i * α_k / α_j
                                    S_k = C6_k * α_i * α_j / α_k
                        
                                    C9_ijk = λ_3b * 2 * S_i * S_j * S_k * (S_i + S_j + S_k) / ((S_i + S_j) * (S_j + S_k) * (S_i + S_k))
                        
                                    @views r_ik_vec = coords[k] - coords[i]
                                    @views r_jk_vec = coords[k] - coords[j]
                                    r_ik = norm(r_ik_vec)
                                    r_jk = norm(r_jk_vec)
                                    x_ik = slater_damping_value(r_ik, b_ik)
                                    x_jk = slater_damping_value(r_jk, b_jk)
                                    f_ij = inc_gamma(x_ij, 3)
                                    f_ik = inc_gamma(x_ik, 3)
                                    f_jk = inc_gamma(x_jk, 3)
                        
                                    ϕ_i = r_ij_vec ⋅ r_ik_vec / (r_ij * r_ik)
                                    ϕ_j = -r_ij_vec ⋅ r_jk_vec / (r_ij * r_jk)
                                    ϕ_k = r_ik_vec ⋅ r_jk_vec / (r_ik * r_jk)
                                    three_body_dispersion_energy += f_ij * f_ik * f_jk * C9_ijk * (1 + 3 * ϕ_i * ϕ_j * ϕ_k) / (r_ij^3 * r_ik^3 * r_jk^3)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return two_body_dispersion_energy# + three_body_dispersion_energy
end

function total_dispersion_energy_with_charge_correction(coords::Vector{MVector{3,Float64}}, labels::AbstractVector{String}, fragment_indices::Vector{Vector{Int}}, params::Dict{Symbol,Float64}, Δq::AbstractVector{Float64})
    two_body_dispersion_energy = 0.0
    three_body_dispersion_energy = 0.0
    q_O_expected = -0.6
    q_H_expected =  0.3
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    C6_i = abs(params[Symbol(labels[i], :_C6_disp)])
                    α_i = get_mean_polarizability(labels[i], params)
                    b_i = params[Symbol(labels[i], :_b_exch)]
                    for j in fragment_indices[j_frag]
                        C6_j = abs(params[Symbol(labels[j], :_C6_disp)])
                        α_j = get_mean_polarizability(labels[j], params)
                        b_j = params[Symbol(labels[j], :_b_exch)]
                        
                        C6_ij = sqrt(abs(params[Symbol(labels[i], :_C6_disp)]) * abs(params[Symbol(labels[j], :_C6_disp)]))
                        b_ij = sqrt(params[Symbol(labels[i], :_b_exch)] * params[Symbol(labels[j], :_b_exch)])
                        
                        @views r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(r_ij_vec)
                        x_ij = slater_damping_value(r_ij, b_ij)
                        two_body_dispersion_energy += -inc_gamma(x_ij, 6) * C6_ij / r_ij^6
                    end
                end
            end
        end
    end
    return two_body_dispersion_energy + three_body_dispersion_energy
end

function total_c6_slater_dispersion_energy_and_grads!(coords::Vector{MVector{3,Float64}}, labels::AbstractVector{String}, fragment_indices::Vector{Vector{Int}}, params::Dict{Symbol,Float64}, grads::Vector{MVector{3,Float64}})
    dispersion_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        C6_ij = sqrt(abs(params[Symbol(labels[i], :_C6_disp)]) * abs(params[Symbol(labels[j], :_C6_disp)]))
                        b_ij = sqrt(params[Symbol(labels[i], :_b_exch)] * params[Symbol(labels[j], :_b_exch)])

                        r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(r_ij_vec)
                        x = slater_damping_value(r_ij, b_ij)
                        x_gradient_i = slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                        inc_gamma_gradient_prefactor = inc_gamma_derivative(x, 6)
                        dispersion_energy += -inc_gamma(x, 6) * C6_ij / r_ij^6
                        grads[i] += C6_ij * (6 * inc_gamma(x, 6) / r_ij^7 * -r_ij_vec / r_ij + r_ij^-6 * (-inc_gamma_gradient_prefactor * x_gradient_i))
                        grads[j] -= C6_ij * (6 * inc_gamma(x, 6) / r_ij^7 * -r_ij_vec / r_ij + r_ij^-6 * (-inc_gamma_gradient_prefactor * x_gradient_i))
                    end
                end
            end
        end
    end
    return dispersion_energy
end