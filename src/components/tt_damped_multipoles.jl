include("damped_multipoles.jl")

#
# Throughout this file, f_n represent the incomplete gamma function,
# inc_gamma(x, n). That is, the Tang-Toennies damping function of order
# n. f_n_deriv is always used to derivative of f_n w.r.t. x and x_gradient_i
# is the gradient of that value w.r.t. r_i. These are passed in since one will usually
# use this same value to compute many different damped multipole terms. So,
# it is much better to compute once and just reuse the value
#

@inline function get_tt_damped_electric_potential_charge(q::Float64, r_ij::Float64, f_n::Float64)
    return f_n * q * get_T(r_ij)
end

@inline function get_bare_and_tt_damped_electric_potential_charge(q::Float64, r_ij::Float64, f_n::Float64)
    return q * get_T(r_ij), f_n * q * get_T(r_ij)
end

@inline function get_tt_damped_electric_field_charge(q::Float64, r_ij::MVector{3,Float64}, f_n::Float64, f_n_deriv::Float64, x_gradient_i::MVector{3, Float64})
    E_field = @MVector zeros(3)
    r_ij_length = norm(r_ij)
    for α in 1:3
        E_field[α] = -q * (
            f_n * get_Tα(r_ij[α], r_ij_length) -
            get_T(r_ij_length) * f_n_deriv * x_gradient_i[α]
        )
    end
    return E_field
end

@inline function get_tt_damped_electric_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, x_gradient_i::MVector{3, Float64})
    r_ij_length = norm(r_ij)
    for α in 1:3
        E_field[α] += -q * (
            f_n * get_Tα(r_ij[α], r_ij_length) -
            get_T(r_ij_length) * f_n_deriv * x_gradient_i[α]
        )
    end
end

@inline function get_bare_and_tt_damped_electric_field_charge(q::Float64, r_ij::MVector{3,Float64}, f_n::Float64, f_n_deriv::Float64, x_gradient_i::MVector{3, Float64})
    E_field = @MVector zeros(3)
    E_field_damped = @MVector zeros(3)
    r_ij_length = norm(r_ij)
    for α in 1:3
        field = -q * get_Tα(r_ij[α], r_ij_length)
        E_field[α] += field
        E_field_damped[α] += f_n * field + q * get_T(r_ij_length) * f_n_deriv * x_gradient_i[α]
    end
    return E_field, E_field_damped
end

@inline function get_bare_and_tt_damped_electric_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64}, E_field_damped::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, x_gradient_i::MVector{3, Float64})
    r_ij_length = norm(r_ij)
    for α in 1:3
        field = -q * get_Tα(r_ij[α], r_ij_length)
        E_field[α] += field
        E_field_damped[α] += f_n * field + q * get_T(r_ij_length) * f_n_deriv * x_gradient_i[α]
    end
end

#@inline function get_tt_damped_electric_field_gradient_charge(q::Float64, r_ij::MVector{3,Float64}, f_n::Float64)
#    E_field_grad = @MMatrix zeros(3, 3)
#    r_ij_length = norm(r_ij)
#    for α in 1:3
#        for β in 1:3
#            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
#            E_field_grad[α, β] = -q * (λ5 * r5_term + λ3 * r3_term)
#        end
#    end
#    return E_field_grad
#end

#@inline function get_tt_damped_electric_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, f_n::Float64)
#    r_ij_length = norm(r_ij)
#    for α in 1:3
#        for β in 1:3
#            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
#            E_field_grad[α, β] += -q * (λ5 * r5_term + λ3 * r3_term)
#        end
#    end
#end

#@inline function get_bare_and_tt_damped_electric_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, E_field_grad_damped::MMatrix{3, 3, Float64, 9}, b_ij::Float64, n::Int=1)
#    r_ij_length = norm(r_ij)
#    for α in 1:3
#        for β in 1:3
#            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
#            E_field_grad[α, β] += -q * (r5_term + r3_term)
#            E_field_grad_damped[α, β] += -q * (λ5 * r5_term + λ3 * r3_term)
#        end
#    end
#end

@inline function get_tt_damped_electric_potential_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, x_gradient_i::MVector{3, Float64})
    ϕ = 0.0
    for α in 1:3
        ϕ -= μ[α] * (f_n * get_Tα(r_ij[α], norm(r_ij)) - get_T(norm(r_ij)) * f_n_deriv * x_gradient_i[α])
    end
    return ϕ
end

@inline function get_bare_and_tt_damped_electric_potential_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, x_gradient_i::MVector{3, Float64})
    ϕ_damped = 0.0
    ϕ = 0.0
    for α in 1:3
        potential = μ[α] * get_Tα(r_ij[α], norm(r_ij))
        ϕ -= potential
        ϕ_damped -= f_n * potential - μ[α] * get_T(norm(r_ij)) * f_n_deriv * x_gradient_i[α]
    end
    return ϕ, ϕ_damped
end

@inline function get_tt_damped_electric_field_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, f_n_second_deriv::Float64, x_gradient_i::MVector{3, Float64}, x_hess_i::MMatrix{3, 3, Float64, 9})
    E_field = @MVector zeros(3)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], norm(r_ij), α==β)
            E_field[α] += μ[β] * (
                f_n * (r5_term + r3_term) -
                get_Tα(r_ij[β], norm(r_ij)) * f_n_deriv * x_gradient_i[α] -
                get_Tα(r_ij[α], norm(r_ij)) * f_n_deriv * x_gradient_i[β] +
                get_T(norm(r_ij)) * f_n_second_deriv * x_gradient_i[α] * x_gradient_i[β] -
                get_T(norm(r_ij)) * f_n_deriv * x_hess_i[α, β]
            )
        end
    end
    return E_field
end

@inline function get_tt_damped_electric_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, f_n_second_deriv::Float64, x_gradient_i::MVector{3, Float64}, x_hess_i::MMatrix{3, 3, Float64, 9})
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], norm(r_ij), α==β)
            E_field[α] += μ[β] * (
                f_n * (r5_term + r3_term) -
                get_Tα(r_ij[β], norm(r_ij)) * f_n_deriv * x_gradient_i[α] -
                get_Tα(r_ij[α], norm(r_ij)) * f_n_deriv * x_gradient_i[β] +
                get_T(norm(r_ij)) * f_n_second_deriv * x_gradient_i[α] * x_gradient_i[β] -
                get_T(norm(r_ij)) * f_n_deriv * x_hess_i[α, β]
            )
        end
    end
end

@inline function get_bare_and_tt_damped_electric_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, E_field_damped::MVector{3, Float64}, f_n::Float64, f_n_deriv::Float64, f_n_second_deriv::Float64, x_gradient_i::MVector{3, Float64}, x_hess_i::MMatrix{3, 3, Float64, 9})
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], norm(r_ij), α==β)
            E_field[α] += μ[β] * (r5_term + r3_term)
            E_field_damped[α] += μ[β] * (
                f_n * (r5_term + r3_term) -
                get_Tα(r_ij[β], norm(r_ij)) * f_n_deriv * x_gradient_i[α] -
                get_Tα(r_ij[α], norm(r_ij)) * f_n_deriv * x_gradient_i[β] +
                get_T(norm(r_ij)) * f_n_second_deriv * x_gradient_i[α] * x_gradient_i[β] -
                get_T(norm(r_ij)) * f_n_deriv * x_hess_i[α, β]
            )
        end
    end
end

@inline function get_tt_damped_electric_field_gradient_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, λ5::Float64=1.0, λ7::Float64=1.0)
    E_field_grad = @MMatrix zeros(3, 3)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], norm(r_ij), α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * (λ7 * r7_term + λ5 * r5_term)
            end
        end
    end
    return E_field_grad
end

@inline function get_tt_damped_electric_field_gradient_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ5::Float64=1.0, λ7::Float64=1.0)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], norm(r_ij), α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * (λ7 * r7_term + λ5 * r5_term)
            end
        end
    end
end

@inline function get_bare_and_tt_damped_electric_field_gradient_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, E_field_grad_damped::MMatrix{3, 3, Float64, 9}, λ5::Float64, λ7::Float64)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], norm(r_ij), α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * (r7_term + r5_term)
                E_field_grad_damped[α, β] += μ[γ] * (λ7 * r7_term + λ5 * r5_term)
            end
        end
    end
end

function get_all_electrostatic_quantities!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole1},
    ϕ::AbstractVector{Float64},
    ϕ_damped::AbstractVector{Float64},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_damped::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64},
    damping_type::TangToenniesDamping
)
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                b_i = params[Symbol(labels[i], :_b_exch)]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    b_j = params[Symbol(labels[j], :_b_exch)]
                    b_ij = sqrt(b_i * b_j)

                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    x = born_mayer_damping_value(r_ij, b_ij)
                    x_gradient_i = born_mayer_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    x_hessian_i = born_mayer_damping_value_hessian(r_ij_vec, r_ij, b_ij)
                    f_n_1 = inc_gamma(x, 1)
                    f_n_2 = inc_gamma(x, 2)
                    f_n_2_deriv = inc_gamma_derivative(x, 2)
                    f_n_3 = inc_gamma(x, 3)
                    f_n_3_deriv = inc_gamma_derivative(x, 3)
                    f_n_3_second_deriv = inc_gamma_second_derivative(x, 3)

                    # electric potential
                    ϕ_j_q, ϕ_damped_j_q = get_bare_and_tt_damped_electric_potential_charge(M_i.q, r_ij, f_n_1)
                    ϕ_i_q, ϕ_damped_i_q = get_bare_and_tt_damped_electric_potential_charge(M_j.q, r_ij, f_n_1)
                    ϕ_j_μ, ϕ_damped_j_μ = get_bare_and_tt_damped_electric_potential_dipole(M_i.μ, r_ij_vec, f_n_2, f_n_2_deriv, x_gradient_i)
                    ϕ_i_μ, ϕ_damped_i_μ = get_bare_and_tt_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, f_n_2, f_n_2_deriv, -x_gradient_i)
                    ϕ[i] += ϕ_i_q + ϕ_i_μ
                    ϕ[j] += ϕ_j_q + ϕ_j_μ
                    ϕ_damped[i] += ϕ_damped_i_q + ϕ_damped_i_μ
                    ϕ_damped[j] += ϕ_damped_j_q + ϕ_damped_j_μ
                    
                    # electric field
                    get_bare_and_tt_damped_electric_field_charge!(M_i.q, r_ij_vec, E_field[j], E_field_damped[j], f_n_2, f_n_2_deriv, x_gradient_i)
                    get_bare_and_tt_damped_electric_field_charge!(M_j.q, -r_ij_vec, E_field[i], E_field_damped[i], f_n_2, f_n_2_deriv, -x_gradient_i)
                    get_bare_and_tt_damped_electric_field_dipole!(M_i.μ, r_ij_vec, E_field[j], E_field_damped[j], f_n_3, f_n_3_deriv, f_n_3_second_deriv, x_gradient_i, x_hessian_i)
                    get_bare_and_tt_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field[i], E_field_damped[i], f_n_3, f_n_3_deriv, f_n_3_second_deriv, -x_gradient_i, x_hessian_i)
                end
            end
        end
    end
end