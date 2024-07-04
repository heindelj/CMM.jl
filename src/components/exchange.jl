
function total_anisotropic_exchange_repulsion(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    local_axes::Vector{LocalAxes},
    params::Dict{Symbol,Float64},
)
    exchange_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    a_i_Y10 = 0.0
                    a_i_Y11 = 0.0
                    a_i_Y20 = 0.0
                    a_i_Y21 = 0.0
                    a_i_Y22 = 0.0
                    a_i = abs(params[Symbol(labels[i], :_a_exch)])
                    b_i = abs(params[Symbol(labels[i], :_b_exch)])
                    if haskey(params, Symbol(labels[i], :_a_exch_Y10))
                        a_i_Y10 = params[Symbol(labels[i], :_a_exch_Y10)]
                        a_i_Y11 = params[Symbol(labels[i], :_a_exch_Y11)]
                        a_i_Y20 = params[Symbol(labels[i], :_a_exch_Y20)]
                        a_i_Y21 = params[Symbol(labels[i], :_a_exch_Y21)]
                        a_i_Y22 = params[Symbol(labels[i], :_a_exch_Y22)]
                    end
                    for j in fragment_indices[j_frag]
                        a_j = abs(params[Symbol(labels[j], :_a_exch)])
                        a_j_Y10 = 0.0
                        a_j_Y11 = 0.0
                        a_j_Y20 = 0.0
                        a_j_Y21 = 0.0
                        a_j_Y22 = 0.0
                        if haskey(params, Symbol(labels[j], :_a_exch_Y10))
                            a_j_Y10 = params[Symbol(labels[j], :_a_exch_Y10)]
                            a_j_Y11 = params[Symbol(labels[j], :_a_exch_Y11)]
                            a_j_Y20 = params[Symbol(labels[j], :_a_exch_Y20)]
                            a_j_Y21 = params[Symbol(labels[j], :_a_exch_Y21)]
                            a_j_Y22 = params[Symbol(labels[j], :_a_exch_Y22)]
                        end
                        
                        b_j = abs(params[Symbol(labels[j], :_b_exch)])

                        @views r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(r_ij_vec)
                        
                        θ_i, θ_j, ϕ_i, ϕ_j = get_spherical_angles(coords[i], coords[j], local_axes[i], local_axes[j])

                        # get spherical harmonics for angular dependence
                        # of each exchange parameter
                        Y10_i = Y10(ϕ_i)
                        Y10_j = Y10(ϕ_j)
                        Y11_i = Y11(θ_i, ϕ_i)
                        Y11_j = Y11(θ_j, ϕ_j)
                        Y20_i = Y20(ϕ_i)
                        Y20_j = Y20(ϕ_j)
                        Y21_i = Y21(θ_i, ϕ_i)
                        Y21_j = Y21(θ_j, ϕ_j)
                        Y22_i = Y22(θ_i, ϕ_i)
                        Y22_j = Y22(θ_j, ϕ_j)

                        a_i_Ω = a_i * (1.0 + a_i_Y10 * Y10_i + a_i_Y11 * Y11_i + a_i_Y20 * Y20_i + a_i_Y21 * Y21_i + a_i_Y22 * Y22_i)
                        a_j_Ω = a_j * (1.0 + a_j_Y10 * Y10_j + a_j_Y11 * Y11_j + a_j_Y20 * Y20_j + a_j_Y21 * Y21_j + a_j_Y22 * Y22_j)

                        a_ij_Ω = a_i_Ω * a_j_Ω
                        b_ij = sqrt(b_i * b_j)
                        exchange_energy += a_ij_Ω * exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                    end
                end
            end
        end
    end
    return exchange_energy
end

function total_anisotropic_exchange_repulsion_and_grads!(
    coords::Vector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::Vector{Vector{Int}},
    local_axes::Vector{LocalAxes},
    params::Dict{Symbol,Float64},
    grads::Vector{MVector{3, Float64}}
)
    exchange_energy = 0.0
    spherical_grads = SVector{4, MMatrix{3, 4, Float64, 12}}([@MMatrix zeros(3, 4) for _ in 1:4])
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        a_i = abs(params[Symbol(labels[i], :_a_exch)])
                        a_j = abs(params[Symbol(labels[j], :_a_exch)])
                        a_i_Y10 = 0.0
                        a_j_Y10 = 0.0
                        a_i_Y11 = 0.0
                        a_j_Y11 = 0.0
                        a_i_Y20 = 0.0
                        a_j_Y20 = 0.0
                        a_i_Y21 = 0.0
                        a_j_Y21 = 0.0
                        a_i_Y22 = 0.0
                        a_j_Y22 = 0.0
                        if haskey(params, Symbol(labels[i], :_a_exch_Y10))
                            a_i_Y10 = params[Symbol(labels[i], :_a_exch_Y10)]
                            a_i_Y11 = params[Symbol(labels[i], :_a_exch_Y11)]
                            a_i_Y20 = params[Symbol(labels[i], :_a_exch_Y20)]
                            a_i_Y21 = params[Symbol(labels[i], :_a_exch_Y21)]
                            a_i_Y22 = params[Symbol(labels[i], :_a_exch_Y22)]
                        end
                        if haskey(params, Symbol(labels[j], :_a_exch_Y10))
                            a_j_Y10 = params[Symbol(labels[j], :_a_exch_Y10)]
                            a_j_Y11 = params[Symbol(labels[j], :_a_exch_Y11)]
                            a_j_Y20 = params[Symbol(labels[j], :_a_exch_Y20)]
                            a_j_Y21 = params[Symbol(labels[j], :_a_exch_Y21)]
                            a_j_Y22 = params[Symbol(labels[j], :_a_exch_Y22)]
                        end
                        
                        b_i = abs(params[Symbol(labels[i], :_b_exch)])
                        b_j = abs(params[Symbol(labels[j], :_b_exch)])

                        @views r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(r_ij_vec)
                        
                        θ_i, θ_j, ϕ_i, ϕ_j = get_spherical_angles_and_gradients!(coords[i], coords[j], local_axes[i], local_axes[j], spherical_grads)

                        # get spherical harmonics for angular dependence
                        # of each exchange parameter
                        Y10_i, Y10_i_grad_ϕ = Y10_and_grad(ϕ_i)
                        Y10_j, Y10_j_grad_ϕ = Y10_and_grad(ϕ_j)
                        Y11_i, Y11_i_grad_θ, Y11_i_grad_ϕ = Y11_and_grad(θ_i, ϕ_i)
                        Y11_j, Y11_j_grad_θ, Y11_j_grad_ϕ = Y11_and_grad(θ_j, ϕ_j)
                        Y20_i, Y20_i_grad_ϕ = Y20_and_grad(ϕ_i)
                        Y20_j, Y20_j_grad_ϕ = Y20_and_grad(ϕ_j)
                        Y21_i, Y21_i_grad_θ, Y21_i_grad_ϕ = Y21_and_grad(θ_i, ϕ_i)
                        Y21_j, Y21_j_grad_θ, Y21_j_grad_ϕ = Y21_and_grad(θ_j, ϕ_j)
                        Y22_i, Y22_i_grad_θ, Y22_i_grad_ϕ = Y22_and_grad(θ_i, ϕ_i)
                        Y22_j, Y22_j_grad_θ, Y22_j_grad_ϕ = Y22_and_grad(θ_j, ϕ_j)

                        # Y10 gradients
                        @views Y10_i_grad_i  = Y10_i_grad_ϕ * spherical_grads[3][:, 1]
                        @views Y10_i_grad_iz = Y10_i_grad_ϕ * spherical_grads[3][:, 2]
                        @views Y10_i_grad_ix = Y10_i_grad_ϕ * spherical_grads[3][:, 3]
                        @views Y10_i_grad_j  = Y10_i_grad_ϕ * spherical_grads[3][:, 4]

                        @views Y10_j_grad_j  = Y10_j_grad_ϕ * spherical_grads[4][:, 1]
                        @views Y10_j_grad_jz = Y10_j_grad_ϕ * spherical_grads[4][:, 2]
                        @views Y10_j_grad_jx = Y10_j_grad_ϕ * spherical_grads[4][:, 3]
                        @views Y10_j_grad_i  = Y10_j_grad_ϕ * spherical_grads[4][:, 4]

                        # Y20 gradients
                        @views Y20_i_grad_i  = Y20_i_grad_ϕ * spherical_grads[3][:, 1]
                        @views Y20_i_grad_iz = Y20_i_grad_ϕ * spherical_grads[3][:, 2]
                        @views Y20_i_grad_ix = Y20_i_grad_ϕ * spherical_grads[3][:, 3]
                        @views Y20_i_grad_j  = Y20_i_grad_ϕ * spherical_grads[3][:, 4]

                        @views Y20_j_grad_j  = Y20_j_grad_ϕ * spherical_grads[4][:, 1]
                        @views Y20_j_grad_jz = Y20_j_grad_ϕ * spherical_grads[4][:, 2]
                        @views Y20_j_grad_jx = Y20_j_grad_ϕ * spherical_grads[4][:, 3]
                        @views Y20_j_grad_i  = Y20_j_grad_ϕ * spherical_grads[4][:, 4]

                        # Y11 gradients
                        @views Y11_i_grad_i  = Y11_i_grad_θ * spherical_grads[1][:, 1] + Y11_i_grad_ϕ * spherical_grads[3][:, 1]
                        @views Y11_i_grad_iz = Y11_i_grad_θ * spherical_grads[1][:, 2] + Y11_i_grad_ϕ * spherical_grads[3][:, 2]
                        @views Y11_i_grad_ix = Y11_i_grad_θ * spherical_grads[1][:, 3] + Y11_i_grad_ϕ * spherical_grads[3][:, 3]
                        @views Y11_i_grad_j  = Y11_i_grad_θ * spherical_grads[1][:, 4] + Y11_i_grad_ϕ * spherical_grads[3][:, 4]

                        @views Y11_j_grad_j  = Y11_j_grad_θ * spherical_grads[2][:, 1] + Y11_j_grad_ϕ * spherical_grads[4][:, 1]
                        @views Y11_j_grad_jz = Y11_j_grad_θ * spherical_grads[2][:, 2] + Y11_j_grad_ϕ * spherical_grads[4][:, 2]
                        @views Y11_j_grad_jx = Y11_j_grad_θ * spherical_grads[2][:, 3] + Y11_j_grad_ϕ * spherical_grads[4][:, 3]
                        @views Y11_j_grad_i  = Y11_j_grad_θ * spherical_grads[2][:, 4] + Y11_j_grad_ϕ * spherical_grads[4][:, 4]

                        # Y21 gradients
                        @views Y21_i_grad_i  = Y21_i_grad_θ * spherical_grads[1][:, 1] + Y21_i_grad_ϕ * spherical_grads[3][:, 1]
                        @views Y21_i_grad_iz = Y21_i_grad_θ * spherical_grads[1][:, 2] + Y21_i_grad_ϕ * spherical_grads[3][:, 2]
                        @views Y21_i_grad_ix = Y21_i_grad_θ * spherical_grads[1][:, 3] + Y21_i_grad_ϕ * spherical_grads[3][:, 3]
                        @views Y21_i_grad_j  = Y21_i_grad_θ * spherical_grads[1][:, 4] + Y21_i_grad_ϕ * spherical_grads[3][:, 4]

                        @views Y21_j_grad_j  = Y21_j_grad_θ * spherical_grads[2][:, 1] + Y21_j_grad_ϕ * spherical_grads[4][:, 1]
                        @views Y21_j_grad_jz = Y21_j_grad_θ * spherical_grads[2][:, 2] + Y21_j_grad_ϕ * spherical_grads[4][:, 2]
                        @views Y21_j_grad_jx = Y21_j_grad_θ * spherical_grads[2][:, 3] + Y21_j_grad_ϕ * spherical_grads[4][:, 3]
                        @views Y21_j_grad_i  = Y21_j_grad_θ * spherical_grads[2][:, 4] + Y21_j_grad_ϕ * spherical_grads[4][:, 4]

                        # Y22 gradients
                        @views Y22_i_grad_i  = Y22_i_grad_θ * spherical_grads[1][:, 1] + Y22_i_grad_ϕ * spherical_grads[3][:, 1]
                        @views Y22_i_grad_iz = Y22_i_grad_θ * spherical_grads[1][:, 2] + Y22_i_grad_ϕ * spherical_grads[3][:, 2]
                        @views Y22_i_grad_ix = Y22_i_grad_θ * spherical_grads[1][:, 3] + Y22_i_grad_ϕ * spherical_grads[3][:, 3]
                        @views Y22_i_grad_j  = Y22_i_grad_θ * spherical_grads[1][:, 4] + Y22_i_grad_ϕ * spherical_grads[3][:, 4]

                        @views Y22_j_grad_j  = Y22_j_grad_θ * spherical_grads[2][:, 1] + Y22_j_grad_ϕ * spherical_grads[4][:, 1]
                        @views Y22_j_grad_jz = Y22_j_grad_θ * spherical_grads[2][:, 2] + Y22_j_grad_ϕ * spherical_grads[4][:, 2]
                        @views Y22_j_grad_jx = Y22_j_grad_θ * spherical_grads[2][:, 3] + Y22_j_grad_ϕ * spherical_grads[4][:, 3]
                        @views Y22_j_grad_i  = Y22_j_grad_θ * spherical_grads[2][:, 4] + Y22_j_grad_ϕ * spherical_grads[4][:, 4]

                        a_i_Ω = a_i * (1.0 + a_i_Y10 * Y10_i + a_i_Y11 * Y11_i + a_i_Y20 * Y20_i + a_i_Y21 * Y21_i + a_i_Y22 * Y22_i)
                        a_j_Ω = a_j * (1.0 + a_j_Y10 * Y10_j + a_j_Y11 * Y11_j + a_j_Y20 * Y20_j + a_j_Y21 * Y21_j + a_j_Y22 * Y22_j)

                        a_i_Ω_grad_i  = a_i * (a_i_Y10 * Y10_i_grad_i + a_i_Y11 * Y11_i_grad_i + a_i_Y20 * Y20_i_grad_i + a_i_Y21 * Y21_i_grad_i + a_i_Y22 * Y22_i_grad_i)
                        a_i_Ω_grad_iz = a_i * (a_i_Y10 * Y10_i_grad_iz + a_i_Y11 * Y11_i_grad_iz + a_i_Y20 * Y20_i_grad_iz + a_i_Y21 * Y21_i_grad_iz + a_i_Y22 * Y22_i_grad_iz)
                        a_i_Ω_grad_ix = a_i * (a_i_Y10 * Y10_i_grad_ix + a_i_Y11 * Y11_i_grad_ix + a_i_Y20 * Y20_i_grad_ix + a_i_Y21 * Y21_i_grad_ix + a_i_Y22 * Y22_i_grad_ix)
                        a_i_Ω_grad_j  = a_i * (a_i_Y10 * Y10_i_grad_j + a_i_Y11 * Y11_i_grad_j + a_i_Y20 * Y20_i_grad_j + a_i_Y21 * Y21_i_grad_j + a_i_Y22 * Y22_i_grad_j)

                        a_j_Ω_grad_j  = a_j * (a_j_Y10 * Y10_j_grad_j  + a_j_Y11 * Y11_j_grad_j  + a_j_Y20 * Y20_j_grad_j  + a_j_Y21 * Y21_j_grad_j  + a_j_Y22 * Y22_j_grad_j)
                        a_j_Ω_grad_jz = a_j * (a_j_Y10 * Y10_j_grad_jz + a_j_Y11 * Y11_j_grad_jz + a_j_Y20 * Y20_j_grad_jz + a_j_Y21 * Y21_j_grad_jz + a_j_Y22 * Y22_j_grad_jz)
                        a_j_Ω_grad_jx = a_j * (a_j_Y10 * Y10_j_grad_jx + a_j_Y11 * Y11_j_grad_jx + a_j_Y20 * Y20_j_grad_jx + a_j_Y21 * Y21_j_grad_jx + a_j_Y22 * Y22_j_grad_jx)
                        a_j_Ω_grad_i  = a_j * (a_j_Y10 * Y10_j_grad_i  + a_j_Y11 * Y11_j_grad_i  + a_j_Y20 * Y20_j_grad_i  + a_j_Y21 * Y21_j_grad_i  + a_j_Y22 * Y22_j_grad_i)

                        a_ij_Ω = a_i_Ω * a_j_Ω
                        b_ij = sqrt(b_i * b_j)

                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        slater_overlap_gradient_j = exp(-b_ij * r_ij) * (
                            (2 / 3 * b_ij^2 * r_ij + b_ij) +
                            -b_ij * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        ) * r_ij_vec / r_ij

                        exchange_energy += a_ij_Ω * slater_overlap
                        grads[i] += (
                            a_j_Ω * slater_overlap * a_i_Ω_grad_i +
                            a_i_Ω * slater_overlap * a_j_Ω_grad_i -
                            a_ij_Ω * slater_overlap_gradient_j
                        )
                        grads[local_axes[i].i_z] += a_j_Ω * slater_overlap * a_i_Ω_grad_iz
                        grads[local_axes[i].i_x] += a_j_Ω * slater_overlap * a_i_Ω_grad_ix

                        grads[j] += (
                            a_j_Ω * slater_overlap * a_i_Ω_grad_j +
                            a_i_Ω * slater_overlap * a_j_Ω_grad_j +
                            a_ij_Ω * slater_overlap_gradient_j
                        )
                        grads[local_axes[j].i_z] += a_i_Ω * slater_overlap * a_j_Ω_grad_jz
                        grads[local_axes[j].i_x] += a_i_Ω * slater_overlap * a_j_Ω_grad_jx
                    end
                end
            end
        end
    end
    return exchange_energy
end