
@inline function get_T6(r_ij::Float64)
    return 1.0 / r_ij^6
end

@inline function get_T6α(r_α::Float64, r_ij::Float64)
    return -6 * r_α / r_ij^8
end

@inline function get_T6αβ(r_α::Float64, r_β::Float64, r_ij::Float64, δ_αβ::Bool)
    return 48 * r_α * r_β / r_ij^10, -6 * δ_αβ / r_ij^8
end

# These are all of the potential, field, and field gradient terms for
# electrostatics which are damped using the same damping model as
# for the slater overlap dispersion model.

@inline function get_T(r_ij::Float64)
    return 1.0 / r_ij
end

@inline function get_Tα(r_α::Float64, r_ij::Float64)
    return -r_α / r_ij^3
end

@inline function get_Tαβ(r_α::Float64, r_β::Float64, r_ij::Float64, δ_αβ::Bool)
    return 3 * r_α * r_β / r_ij^5, -δ_αβ / r_ij^3
end

@inline function get_Tαβγ(r_α::Float64, r_β::Float64, r_γ::Float64, r_ij::Float64, δ_αβ::Bool, δ_αγ::Bool, δ_βγ::Bool)
    return -15 * r_α * r_β * r_γ / r_ij^7, 3 * (r_α * δ_βγ + r_β * δ_αγ + r_γ * δ_αβ) / r_ij^5
end

@inline function get_Tαβγη(r_α::Float64, r_β::Float64, r_γ::Float64, r_η::Float64, r_ij::Float64, δ_αβ::Bool, δ_αγ::Bool, δ_αη::Bool, δ_βγ::Bool, δ_βη::Bool, δ_γη::Bool)
    return (
        105 * r_α * r_β * r_γ * r_η / r_ij^9,
        -15 * (
            r_α * r_β * δ_γη +
            r_α * r_γ * δ_βη +
            r_α * r_η * δ_βγ +
            r_β * r_γ * δ_αη +
            r_β * r_η * δ_αγ +
            r_γ * r_η * δ_αβ
        ) / r_ij^7,
        3 * (δ_αβ * δ_γη + δ_αγ * δ_βη + δ_αη * δ_βγ) / r_ij^5
    )
end

@inline function get_Tαβγηϵ(r_α::Float64, r_β::Float64, r_γ::Float64, r_η::Float64, r_ϵ::Float64, r_ij::Float64, δ_αβ::Bool, δ_αγ::Bool, δ_αη::Bool, δ_αϵ::Bool, δ_βγ::Bool, δ_βη::Bool, δ_βϵ::Bool, δ_γη::Bool, δ_γϵ::Bool, δ_ηϵ::Bool)
    return (
        #-945 * r_α * r_β * r_γ * r_η * r_ϵ / r_ij^11,
        #0.0, 105 * (
        #    #r_α * r_β * r_γ * δ_ηϵ +
        #    #r_α * r_β * r_η * δ_γϵ +
        #    #r_α * r_γ * r_η * δ_βϵ +
        #    #r_β * r_γ * r_η * δ_αϵ +
        #    -(
        #        r_α * r_β * r_ϵ * δ_γη +
        #        r_α * r_γ * r_ϵ * δ_βη +
        #        r_α * r_η * r_ϵ * δ_βγ +
        #        r_β * r_γ * r_ϵ * δ_αη +
        #        r_β * r_η * r_ϵ * δ_αγ +
        #        r_γ * r_η * r_ϵ * δ_αβ
        #    )
        #) / r_ij^9,
        #0.0, 0.0, -15 * (
        #    r_ϵ * (δ_αβ * δ_γη + δ_αγ * δ_βη + δ_αη * δ_βγ) #+
        #    #(
        #    #    (r_α * δ_βϵ + r_β * δ_αϵ) * δ_γη +
        #    #    (r_α * δ_γϵ + r_γ * δ_αϵ) * δ_βη +
        #    #    (r_α * δ_ηϵ + r_η * δ_αϵ) * δ_βγ +
        #    #    (r_β * δ_γϵ + r_γ * δ_βϵ) * δ_αη +
        #    #    (r_β * δ_ηϵ + r_η * δ_βϵ) * δ_αγ +
        #    #    (r_γ * δ_ηϵ + r_η * δ_γϵ) * δ_αβ
        #    #)
        #) / r_ij^7
        0.0, 0.0, -15 * r_ϵ / r_ij^7
    )
end

@inline function get_damped_electric_potential_charge(q::Float64, r_ij::Float64, λ1::Float64=1.0)
    return λ1 * q * get_T(r_ij)
end

@inline function get_damped_dispersion_potential_charge(q::Float64, r_ij::Float64, λ1::Float64=1.0)
    return λ1 * q * get_T6(r_ij)
end

@inline function get_repulsion_potential_charge(q::Float64, r_ij::Float64, λ1::Float64)
    T = get_T(r_ij)
    return q * (1 - λ1) * T
end

@inline function get_ct_potential_charge(q::Float64, r_ij::Float64, λ1::Float64)
    T = get_T(r_ij)
    return q * (λ1 - 1) * T
end

@inline function get_electric_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64})
    r_ij_length = norm(r_ij)
    for α in 1:3
        E_field[α] += -q * get_Tα(r_ij[α], r_ij_length)
    end
end

@inline function get_damped_electric_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64}, λ3::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        E_field[α] += -q * λ3 * get_Tα(r_ij[α], r_ij_length)
    end
end

@inline function get_damped_dispersion_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64}, λ3::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        E_field[α] += -q * λ3 * get_T6α(r_ij[α], r_ij_length)
    end
end

@inline function get_repulsion_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64}, λ3::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        r3_term = get_Tα(r_ij[α], r_ij_length)
        E_field[α] += -q * (1 - λ3) * r3_term
    end
end

@inline function get_ct_field_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field::MVector{3, Float64}, λ3::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        r3_term = get_Tα(r_ij[α], r_ij_length)
        E_field[α] += -q * (λ3 - 1) * r3_term
    end
end

@inline function get_electric_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9})
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field_grad[α, β] += -q * (r5_term + r3_term)
        end
    end
end

@inline function get_damped_electric_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ3::Float64=1.0, λ5::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field_grad[α, β] += -q * (λ5 * r5_term + λ3 * r3_term)
        end
    end
end

@inline function get_damped_dispersion_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ3::Float64=1.0, λ5::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_T6αβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field_grad[α, β] += -q * (λ5 * r5_term + λ3 * r3_term)
        end
    end
end

@inline function get_repulsion_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ3::Float64, λ5::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field_grad[α, β] += -q * ((1 - λ5) * r5_term + (1 - λ3) * r3_term)
        end
    end
end

@inline function get_ct_field_gradient_charge!(q::Float64, r_ij::MVector{3,Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ3::Float64, λ5::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field_grad[α, β] += -q * ((λ5 - 1) * r5_term + (λ3 - 1) * r3_term)
        end
    end
end

@inline function get_damped_electric_potential_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64=1.0)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        ϕ -= λ3 * μ[α] * get_Tα(r_ij[α], r_ij_length)
    end
    return ϕ
end

@inline function get_damped_dispersion_potential_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64=1.0)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        ϕ -= λ3 * μ[α] * get_T6α(r_ij[α], r_ij_length)
    end
    return ϕ
end

@inline function get_repulsion_potential_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        r3_term = get_Tα(r_ij[α], r_ij_length)
        ϕ -= μ[α] * (1 - λ3) * r3_term
    end
    return ϕ
end

@inline function get_ct_potential_dipole(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        r3_term = get_Tα(r_ij[α], r_ij_length)
        ϕ -= μ[α] * (λ3 - 1) * r3_term
    end
    return ϕ
end

@inline function get_damped_electric_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ3::Float64=1.0, λ5::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field[α] += μ[β] * (λ5 * r5_term + λ3 * r3_term)
        end
    end
end

@inline function get_damped_dispersion_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ3::Float64=1.0, λ5::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_T6αβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field[α] += μ[β] * (λ5 * r5_term + λ3 * r3_term)
        end
    end
end

@inline function get_electric_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64})
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field[α] += μ[β] * (r5_term + r3_term)
        end
    end
end

@inline function get_repulsion_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ3::Float64, λ5::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field[α] += μ[β] * ((1 - λ5) * r5_term + (1 - λ3) * r3_term)
        end
    end
end

@inline function get_ct_field_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ3::Float64, λ5::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            E_field[α] += μ[β] * ((λ5 - 1) * r5_term + (λ3 - 1) * r3_term)
        end
    end
end

@inline function get_damped_electric_field_gradient_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ5::Float64=1.0, λ7::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * (λ7 * r7_term + λ5 * r5_term)
            end
        end
    end
end

@inline function get_electric_field_gradient_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64, 9})
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * (r7_term + r5_term)
            end
        end
    end
end

@inline function get_repulsion_field_gradient_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ5::Float64, λ7::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * ((1 - λ7) * r7_term + (1 - λ5) * r5_term)
            end
        end
    end
end

@inline function get_ct_field_gradient_dipole!(μ::MVector{3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64, 9}, λ5::Float64, λ7::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field_grad[α, β] += μ[γ] * ((λ7 - 1) * r7_term + (λ5 - 1) * r5_term)
            end
        end
    end
end

@inline function get_damped_electric_potential_quadrupole(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64=1.0, λ5::Float64=1.0)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            ϕ += Q[α, β] * (λ5 * r5_term + λ3 * r3_term)
        end
    end
    return ϕ / 3.0
end

@inline function get_repulsion_potential_quadrupole(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64, λ5::Float64)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            ϕ += Q[α, β] * ((1 - λ5) * r5_term + (1 - λ3) * r3_term)
        end
    end
    return ϕ / 3.0
end

@inline function get_ct_potential_quadrupole(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, λ3::Float64, λ5::Float64)
    ϕ = 0.0
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            r5_term, r3_term = get_Tαβ(r_ij[α], r_ij[β], r_ij_length, α==β)
            ϕ += Q[α, β] * ((λ5 - 1) * r5_term + (λ3 - 1) * r3_term)
        end
    end
    return ϕ / 3.0
end

@inline function get_damped_electric_field_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ5::Float64=1.0, λ7::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field[α] -=  Q[β, γ] * (λ7 * r7_term + λ5 * r5_term) / 3.0
            end
        end
    end
end

@inline function get_electric_field_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64})
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field[α] -=  Q[β, γ] * (r7_term + r5_term) / 3.0
            end
        end
    end
end

@inline function get_repulsion_field_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ5::Float64, λ7::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field[α] -=  Q[β, γ] * ((1 - λ7) * r7_term + (1 - λ5) * r5_term) / 3.0
            end
        end
    end
end

@inline function get_ct_field_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field::MVector{3, Float64}, λ5::Float64, λ7::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                r7_term, r5_term = get_Tαβγ(r_ij[α], r_ij[β], r_ij[γ], r_ij_length, α==β, α==γ, β==γ)
                E_field[α] -=  Q[β, γ] * ((λ7 - 1) * r7_term + (λ5 - 1) * r5_term) / 3.0
            end
        end
    end
end

@inline function get_damped_electric_field_gradient_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64}, λ5::Float64=1.0, λ7::Float64=1.0, λ9::Float64=1.0)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for η in 1:3
                    r9_term, r7_term, r5_term = get_Tαβγη(r_ij[α], r_ij[β], r_ij[γ], r_ij[η], r_ij_length, α==β, α==γ, α==η, β==γ, β==η, γ==η)
                    E_field_grad[α, β] -=  Q[γ, η] * (λ9 * r9_term + λ7 * r7_term + λ5 * r5_term) / 3.0
                end
            end
        end
    end
end

@inline function get_electric_field_gradient_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64})
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for η in 1:3
                    r9_term, r7_term, r5_term = get_Tαβγη(r_ij[α], r_ij[β], r_ij[γ], r_ij[η], r_ij_length, α==β, α==γ, α==η, β==γ, β==η, γ==η)
                    E_field_grad[α, β] -=  Q[γ, η] * (r9_term + r7_term + r5_term) / 3.0
                end
            end
        end
    end
end

@inline function get_repulsion_field_gradient_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64}, λ5::Float64, λ7::Float64, λ9::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for η in 1:3
                    r9_term, r7_term, r5_term = get_Tαβγη(r_ij[α], r_ij[β], r_ij[γ], r_ij[η], r_ij_length, α==β, α==γ, α==η, β==γ, β==η, γ==η)
                    E_field_grad[α, β] -=  Q[γ, η] * ((1 - λ9) * r9_term + (1 - λ7) * r7_term + (1 - λ5) * r5_term) / 3.0
                end
            end
        end
    end
end

@inline function get_ct_field_gradient_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field_grad::MMatrix{3, 3, Float64}, λ5::Float64, λ7::Float64, λ9::Float64)
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for η in 1:3
                    r9_term, r7_term, r5_term = get_Tαβγη(r_ij[α], r_ij[β], r_ij[γ], r_ij[η], r_ij_length, α==β, α==γ, α==η, β==γ, β==η, γ==η)
                    E_field_grad[α, β] -=  Q[γ, η] * ((λ9 - 1) * r9_term + (λ7 - 1) * r7_term + (λ5 - 1) * r5_term) / 3.0
                end
            end
        end
    end
end

@inline function get_electric_field_gradient_gradient_quadrupole!(Q::MMatrix{3, 3, Float64}, r_ij::MVector{3, Float64}, E_field_grad_grad::MArray{Tuple{3, 3, 3}, Float64, 3, 27})
    r_ij_length = norm(r_ij)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for η in 1:3
                    for ϵ in 1:3
                        r11_term, r9_term, r7_term = get_Tαβγηϵ(r_ij[α], r_ij[β], r_ij[γ], r_ij[η], r_ij[ϵ], r_ij_length, α==β, α==γ, α==η, α==ϵ, β==γ, β==η, β==ϵ, γ==η, γ==ϵ, η==ϵ)
                        E_field_grad_grad[α, β, γ] +=  Q[η, ϵ] * (r11_term + r9_term + r7_term) / 3.0
                    end
                end
            end
        end
    end
end

function get_all_induced_electrostatic_quantities!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{CSMultipole1},
    ϕ_induced::AbstractVector{Float64},
    E_field_induced::AbstractVector{MVector{3,Float64}},
    E_field_gradients_induced::AbstractVector{MMatrix{3,3,Float64,9}},
    params::Dict{Symbol, Float64},
    overlap_damping_type::AbstractDamping
)
    a = 1.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                b_i = abs(params[Symbol(labels[i], :_b_elec)])
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    b_j = abs(params[Symbol(labels[j], :_b_elec)])
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    b_ij = sqrt(b_i * b_j)
                    u_overlap = b_ij * r_ij
                    λ1_overlap = get_λ1(u_overlap, a, overlap_damping_type)
                    λ3_overlap = get_λ3(u_overlap, a, overlap_damping_type)
                    λ5_overlap = get_λ5(u_overlap, a, overlap_damping_type)
                    λ7_overlap = get_λ7(u_overlap, a, overlap_damping_type)

                    ### electric potential ###
                    # overlap electric potential #
                    ϕ_induced[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, λ1_overlap)
                    ϕ_induced[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, λ1_overlap)
                    ϕ_induced[j] += get_damped_electric_potential_dipole(M_i.μ,  r_ij_vec, λ3_overlap)
                    ϕ_induced[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, λ3_overlap)

                    ### electric field ###
                    # overlap electric field #
                    get_damped_electric_field_charge!(M_i.q_shell,  r_ij_vec, E_field_induced[j], λ3_overlap)
                    get_damped_electric_field_charge!(M_j.q_shell, -r_ij_vec, E_field_induced[i], λ3_overlap)
                    get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_induced[j], λ3_overlap, λ5_overlap)
                    get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_induced[i], λ3_overlap, λ5_overlap)

                    ### electric field gradient ###
                    # overlap electric field gradient #
                    #get_damped_electric_field_gradient_charge!(M_i.q,  r_ij_vec, E_field_gradients_induced[j], λ3_overlap, λ5_overlap)
                    #get_damped_electric_field_gradient_charge!(M_j.q, -r_ij_vec, E_field_gradients_induced[i], λ3_overlap, λ5_overlap)
                    #get_damped_electric_field_gradient_dipole!(M_i.μ,  r_ij_vec, E_field_gradients_induced[j], λ5_overlap, λ7_overlap)
                    #get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_induced[i], λ5_overlap, λ7_overlap)
                end
            end
        end
    end
end

function get_all_core_shell_electrostatic_quantities!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{CSMultipole1},
    ϕ_core::AbstractVector{Float64},
    ϕ_shell::AbstractVector{Float64},
    ϕ_overlap::AbstractVector{Float64},
    E_field_core::AbstractVector{MVector{3,Float64}},
    E_field_shell::AbstractVector{MVector{3,Float64}},
    E_field_overlap::AbstractVector{MVector{3,Float64}},
    E_field_gradients_core::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradients_shell::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradients_overlap::AbstractVector{MMatrix{3,3,Float64,9}},
    params::Dict{Symbol, Float64},
    shell_damping_type::AbstractDamping,
    overlap_damping_type::AbstractDamping
)

    a = get_a(params, shell_damping_type)
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                b_i = params[Symbol(labels[i], :_b_elec)]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    b_j = params[Symbol(labels[j], :_b_elec)]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    u_shell = get_u(r_ij, b_i, params, shell_damping_type)
                    u_overlap = get_u(r_ij, b_i, b_j, params, overlap_damping_type)
                    
                    ### electric potential ###
                    # core electric potential #
                    ϕ_core[j] += get_electric_potential_charge(M_i.Z, r_ij)
                    ϕ_core[i] += get_electric_potential_charge(M_j.Z, r_ij)

                    # shell electric potential #
                    ϕ_shell[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, get_λ1(u_shell, a, shell_damping_type))
                    ϕ_shell[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, get_λ1(u_shell, a, shell_damping_type))
                    ϕ_shell[j] += get_damped_electric_potential_dipole(M_i.μ, r_ij_vec, get_λ3(u_shell, a, shell_damping_type))
                    ϕ_shell[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, get_λ3(u_shell, a, shell_damping_type))

                    # overlap electric potential #
                    ϕ_overlap[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, get_λ1(u_overlap, a, overlap_damping_type))
                    ϕ_overlap[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, get_λ1(u_overlap, a, overlap_damping_type))
                    ϕ_overlap[j] += get_damped_electric_potential_dipole(M_i.μ, r_ij_vec, get_λ3(u_overlap, a, overlap_damping_type))
                    ϕ_overlap[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, get_λ3(u_overlap, a, overlap_damping_type))

                    ### electric field ###
                    # core electric field #
                    get_electric_field_charge!(M_i.Z,  r_ij_vec, E_field_core[j])
                    get_electric_field_charge!(M_j.Z, -r_ij_vec, E_field_core[i])

                    # shell electric field #
                    get_damped_electric_field_charge!(M_i.q,  r_ij_vec, E_field_shell[j], get_λ3(u_shell, a, shell_damping_type))
                    get_damped_electric_field_charge!(M_j.q, -r_ij_vec, E_field_shell[i], get_λ3(u_shell, a, shell_damping_type))
                    get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_shell[j], get_λ3(u_shell, a, shell_damping_type), get_λ5(u_shell, a, shell_damping_type))
                    get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_shell[i], get_λ3(u_shell, a, shell_damping_type), get_λ5(u_shell, a, shell_damping_type))

                    # overlap electric field #
                    get_damped_electric_field_charge!(M_i.q,  r_ij_vec, E_field_overlap[j], get_λ3(u_overlap, a, overlap_damping_type))
                    get_damped_electric_field_charge!(M_j.q, -r_ij_vec, E_field_overlap[i], get_λ3(u_overlap, a, overlap_damping_type))
                    get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_overlap[j], get_λ3(u_overlap, a, overlap_damping_type), get_λ5(u_overlap, a, overlap_damping_type))
                    get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_overlap[i], get_λ3(u_overlap, a, overlap_damping_type), get_λ5(u_overlap, a, overlap_damping_type))

                    ### electric field gradient ###
                    # core electric field gradient #
                    get_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_core[j])
                    get_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_core[i])

                    # shell electric field gradient #
                    get_damped_electric_field_gradient_charge!(M_i.q_shell,  r_ij_vec, E_field_gradients_shell[j], get_λ3(u_shell, a, shell_damping_type), get_λ5(u_shell, a, shell_damping_type))
                    get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_shell[i], get_λ3(u_shell, a, shell_damping_type), get_λ5(u_shell, a, shell_damping_type))
                    get_damped_electric_field_gradient_dipole!(M_i.μ,  r_ij_vec, E_field_gradients_shell[j], get_λ5(u_shell, a, shell_damping_type), get_λ7(u_shell, a, shell_damping_type))
                    get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_shell[i], get_λ5(u_shell, a, shell_damping_type), get_λ7(u_shell, a, shell_damping_type))

                    # overlap electric field gradient #
                    get_damped_electric_field_gradient_charge!(M_i.q_shell,  r_ij_vec, E_field_gradients_overlap[j], get_λ3(u_overlap, a, overlap_damping_type), get_λ5(u_overlap, a, overlap_damping_type))
                    get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_overlap[i], get_λ3(u_overlap, a, overlap_damping_type), get_λ5(u_overlap, a, overlap_damping_type))
                    get_damped_electric_field_gradient_dipole!(M_i.μ,  r_ij_vec, E_field_gradients_overlap[j], get_λ5(u_overlap, a, overlap_damping_type), get_λ7(u_overlap, a, overlap_damping_type))
                    get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_overlap[i], get_λ5(u_overlap, a, overlap_damping_type), get_λ7(u_overlap, a, overlap_damping_type))
                end
            end
        end
    end
end

function get_all_undamped_electrostatic_quantities!(
    coords::AbstractVector{MVector{3,Float64}},
    multipoles::AbstractVector{CSMultipole2},
    fragment_indices::AbstractVector{Vector{Int}},
    ϕ::AbstractVector{Float64},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_gradients::AbstractVector{MMatrix{3,3,Float64,9}}
)

    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    ### electric potential ###
                    # core electric potential #
                    ϕ[j] += get_electric_potential_charge(M_i.q_shell + M_i.Z, r_ij)
                    ϕ[i] += get_electric_potential_charge(M_j.q_shell + M_j.Z, r_ij)

                    ϕ[j] += get_electric_potential_dipole(M_i.μ, r_ij_vec)
                    ϕ[i] += get_electric_potential_dipole(M_j.μ, -r_ij_vec)
                    
                    ϕ[j] += get_electric_potential_quadrupole(M_i.Q, r_ij_vec)
                    ϕ[i] += get_electric_potential_quadrupole(M_j.Q, -r_ij_vec)

                    ### electric field ###
                    get_electric_field_charge!(M_i.q_shell + M_i.Z,  r_ij_vec, E_field[j])
                    get_electric_field_charge!(M_j.q_shell + M_j.Z, -r_ij_vec, E_field[i])

                    get_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field[j])
                    get_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field[i])
                    
                    get_electric_field_quadrupole!(M_i.Q, r_ij_vec, E_field[j])
                    get_electric_field_quadrupole!(M_j.Q, -r_ij_vec, E_field[i])

                    ### electric field gradient ###
                    # core electric field gradient #
                    get_electric_field_gradient_charge!(M_i.q_shell + M_i.Z,  r_ij_vec, E_field_gradients[j])
                    get_electric_field_gradient_charge!(M_j.q_shell + M_j.Z, -r_ij_vec, E_field_gradients[i])

                    get_electric_field_gradient_dipole!(M_i.μ,  r_ij_vec, E_field_gradients[j])
                    get_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients[i])
                    
                    get_electric_field_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradients[j])
                    get_electric_field_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradients[i])
                end
            end
        end
    end
end

function get_all_core_shell_electrostatic_quantities!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{CSMultipole2},
    ϕ_core::AbstractVector{Float64},
    ϕ_shell::AbstractVector{Float64},
    E_field_core::AbstractVector{MVector{3,Float64}},
    E_field_shell::AbstractVector{MVector{3,Float64}},
    E_field_gradients_core::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradients_shell::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradient_gradients_core::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
    E_field_gradient_gradients_shell::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
    params::Dict{Symbol, Float64},
    shell_damping_type::AbstractDamping,
    overlap_damping_type::AbstractDamping
)

    a = get_a(params, shell_damping_type)
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                b_i = abs(params[Symbol(labels[i], :_b_elec)])
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    b_j = abs(params[Symbol(labels[j], :_b_elec)])
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    u_shell_i = get_u(r_ij, b_i, params, shell_damping_type)
                    u_shell_j = get_u(r_ij, b_j, params, shell_damping_type)
                    b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_elec, params)
                    b_ij = abs(b_ij)
                    if !has_pairwise
                        b_ij = sqrt(b_i * b_j)
                    end
                    u_overlap = b_ij * r_ij
                    
                    λ1_shell_i = get_λ1(u_shell_i, a, shell_damping_type)
                    λ3_shell_i = get_λ3(u_shell_i, a, shell_damping_type)
                    λ5_shell_i = get_λ5(u_shell_i, a, shell_damping_type)
                    λ7_shell_i = get_λ7(u_shell_i, a, shell_damping_type)
                    λ9_shell_i = get_λ9(u_shell_i, a, shell_damping_type)
                    
                    λ1_shell_j = get_λ1(u_shell_j, a, shell_damping_type)
                    λ3_shell_j = get_λ3(u_shell_j, a, shell_damping_type)
                    λ5_shell_j = get_λ5(u_shell_j, a, shell_damping_type)
                    λ7_shell_j = get_λ7(u_shell_j, a, shell_damping_type)
                    λ9_shell_j = get_λ9(u_shell_j, a, shell_damping_type)

                    λ1_overlap = get_λ1(u_overlap, a, overlap_damping_type)
                    λ3_overlap = get_λ3(u_overlap, a, overlap_damping_type)
                    λ5_overlap = get_λ5(u_overlap, a, overlap_damping_type)
                    λ7_overlap = get_λ7(u_overlap, a, overlap_damping_type)
                    λ9_overlap = get_λ9(u_overlap, a, overlap_damping_type)
                    #λ11_overlap = get_λ11(u_overlap, a, overlap_damping_type)

                    ### electric potential ###
                    # core electric potential #
                    ϕ_core[j] += get_electric_potential_charge(M_i.Z, r_ij)
                    ϕ_core[i] += get_electric_potential_charge(M_j.Z, r_ij)

                    ϕ_core[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, λ1_shell_i)
                    ϕ_core[j] += get_damped_electric_potential_dipole(M_i.μ, r_ij_vec, λ3_shell_i)
                    ϕ_core[j] += get_damped_electric_potential_quadrupole(M_i.Q, r_ij_vec, λ3_shell_i, λ5_shell_i)

                    ϕ_core[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, λ1_shell_j)
                    ϕ_core[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, λ3_shell_j)
                    ϕ_core[i] += get_damped_electric_potential_quadrupole(M_j.Q, -r_ij_vec, λ3_shell_j, λ5_shell_j)

                    # shell electric potential #
                    # NOTE: the u_shell for the nuclear contribution is opposite of everything
                    # else since the potential felt by the shell due to a core is smeared
                    # over the density of that shell. It's not a bug.
                    ϕ_shell[j] += get_damped_electric_potential_charge(M_i.Z, r_ij, λ1_shell_j)
                    ϕ_shell[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, λ1_overlap)
                    ϕ_shell[j] += get_damped_electric_potential_dipole(M_i.μ, r_ij_vec, λ3_overlap)
                    ϕ_shell[j] += get_damped_electric_potential_quadrupole(M_i.Q, r_ij_vec, λ3_overlap, λ5_overlap)
                    
                    ϕ_shell[i] += get_damped_electric_potential_charge(M_j.Z, r_ij, λ1_shell_i)
                    ϕ_shell[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, λ1_overlap)
                    ϕ_shell[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, λ3_overlap)
                    ϕ_shell[i] += get_damped_electric_potential_quadrupole(M_j.Q, -r_ij_vec, λ3_overlap, λ5_overlap)

                    ### electric field ###
                    # core electric field #
                    get_electric_field_charge!(M_i.Z,  r_ij_vec, E_field_core[j])
                    get_electric_field_charge!(M_j.Z, -r_ij_vec, E_field_core[i])

                    get_damped_electric_field_charge!(M_i.q_shell,  r_ij_vec, E_field_core[j], λ3_shell_i)
                    get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_core[j], λ3_shell_i, λ5_shell_i)
                    get_damped_electric_field_quadrupole!(M_i.Q, r_ij_vec, E_field_core[j], λ5_shell_i, λ7_shell_i)
                    
                    get_damped_electric_field_charge!(M_j.q_shell, -r_ij_vec, E_field_core[i], λ3_shell_j)
                    get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_core[i], λ3_shell_j, λ5_shell_j)
                    get_damped_electric_field_quadrupole!(M_j.Q, -r_ij_vec, E_field_core[i], λ5_shell_j, λ7_shell_j)

                    # shell electric field #
                    get_damped_electric_field_charge!(M_i.Z,  r_ij_vec, E_field_shell[j], λ3_shell_j)
                    get_damped_electric_field_charge!(M_i.q_shell,  r_ij_vec, E_field_shell[j], λ3_overlap)
                    get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_shell[j], λ3_overlap, λ5_overlap)
                    get_damped_electric_field_quadrupole!(M_i.Q, r_ij_vec, E_field_shell[j], λ5_overlap, λ7_overlap)

                    get_damped_electric_field_charge!(M_j.Z, -r_ij_vec, E_field_shell[i], λ3_shell_i)
                    get_damped_electric_field_charge!(M_j.q_shell, -r_ij_vec, E_field_shell[i], λ3_overlap)
                    get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_shell[i], λ3_overlap, λ5_overlap)
                    get_damped_electric_field_quadrupole!(M_j.Q, -r_ij_vec, E_field_shell[i], λ5_overlap, λ7_overlap)

                    ### electric field gradient ###
                    # core electric field gradient #
                    get_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_core[j])
                    get_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_core[i])

                    get_damped_electric_field_gradient_charge!(M_i.q_shell, r_ij_vec, E_field_gradients_core[j], λ3_shell_i, λ5_shell_i)
                    get_damped_electric_field_gradient_dipole!(M_i.μ, r_ij_vec, E_field_gradients_core[j], λ5_shell_i, λ7_shell_i)
                    get_damped_electric_field_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradients_core[j], λ5_shell_i, λ7_shell_i, λ9_shell_i)

                    get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_core[i], λ3_shell_j, λ5_shell_j)
                    get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_core[i], λ5_shell_j, λ7_shell_j)
                    get_damped_electric_field_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradients_core[i], λ5_shell_j, λ7_shell_j, λ9_shell_j)
                    
                    # shell electric field gradient #
                    get_damped_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_shell[j], λ3_shell_j, λ5_shell_j)
                    get_damped_electric_field_gradient_charge!(M_i.q_shell, r_ij_vec, E_field_gradients_shell[j], λ3_overlap, λ5_overlap)
                    get_damped_electric_field_gradient_dipole!(M_i.μ, r_ij_vec, E_field_gradients_shell[j], λ5_overlap, λ7_overlap)
                    get_damped_electric_field_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradients_shell[j], λ5_overlap, λ7_overlap, λ9_overlap)

                    get_damped_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_shell[i], λ3_shell_i, λ5_shell_i)
                    get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_shell[i], λ3_overlap, λ5_overlap)
                    get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_shell[i], λ5_overlap, λ7_overlap)
                    get_damped_electric_field_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradients_shell[i], λ5_overlap, λ7_overlap, λ9_overlap)

                    # shell electric field gradient gradient #
                    # TODO: DAMPING FUNCTION ORDERS ARE WRONG i.e. should have λ11 for quads
                    #get_damped_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_shell[j], λ3_shell_j, λ5_shell_j)
                    #get_damped_electric_field_gradient_charge!(M_i.q_shell,  r_ij_vec, E_field_gradients_shell[j], λ3_overlap, λ5_overlap)
                    #get_damped_electric_field_gradient_dipole!(M_i.μ,  r_ij_vec, E_field_gradients_shell[j], λ5_overlap, λ7_overlap)
                    #get_electric_field_gradient_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradient_gradients_shell[j])#, λ5_overlap, λ7_overlap, λ9_overlap)

                    #get_damped_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_shell[i], λ3_shell_i, λ5_shell_i)
                    #get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_shell[i], λ3_overlap, λ5_overlap)
                    #get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_shell[i], λ5_overlap, λ7_overlap)
                    #get_electric_field_gradient_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradient_gradients_shell[i])#, λ5_overlap, λ7_overlap, λ9_overlap)
                end
            end
        end
    end
end

function get_all_core_shell_electrostatic_quantities!(
    dists::Distances,
    exclusion_list::BitVector,
    Z::Charges, q_shell::Charges,
    μ::Dipoles, Q::Quadrupoles,
    λ_shell_i::Matrix{Float64},
    λ_shell_j::Matrix{Float64},
    λ_overlap::Matrix{Float64},
    ϕ_core::AbstractVector{Float64},
    ϕ_shell::AbstractVector{Float64},
    E_field_core::AbstractVector{MVector{3,Float64}},
    E_field_shell::AbstractVector{MVector{3,Float64}},
    E_field_gradients_core::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradients_shell::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradient_gradients_core::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
    E_field_gradient_gradients_shell::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
)

    # Loop over all atomic pairs
    for i in eachindex(dists.r)
        # Skip any pair in the exclusion list
        if exclusion_list[i]
            continue
        end
        ### electric potential ###
        # core electric potential #
        ϕ_core[j] += get_electric_potential_charge(M_i.Z, r_ij)
        ϕ_core[i] += get_electric_potential_charge(M_j.Z, r_ij)

        ϕ_core[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, λ_shell_i[i, 1])
        ϕ_core[j] += get_damped_electric_potential_dipole(M_i.μ, r_ij_vec, λ_shell_i[i, 2])
        ϕ_core[j] += get_damped_electric_potential_quadrupole(M_i.Q, r_ij_vec, λ_shell_i[i, 2], λ_shell_i[i, 3])

        ϕ_core[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, λ_shell_j[i, 1])
        ϕ_core[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, λ_shell_j[i, 2])
        ϕ_core[i] += get_damped_electric_potential_quadrupole(M_j.Q, -r_ij_vec, λ_shell_j[i, 2], λ_shell_j[i, 3])

        # shell electric potential #
        # NOTE: the u_shell for the nuclear contribution is opposite of everything
        # else since the potential felt by the shell due to a core is smeared
        # over the density of that shell. It's not a bug.
        ϕ_shell[j] += get_damped_electric_potential_charge(M_i.Z, r_ij, λ_shell_j[i, 1])
        ϕ_shell[j] += get_damped_electric_potential_charge(M_i.q_shell, r_ij, λ_overlap[i, 1])
        ϕ_shell[j] += get_damped_electric_potential_dipole(M_i.μ, r_ij_vec, λ_overlap[i, 2])
        ϕ_shell[j] += get_damped_electric_potential_quadrupole(M_i.Q, r_ij_vec, λ_overlap[i, 2], λ_overlap[i, 2])
                    
        ϕ_shell[i] += get_damped_electric_potential_charge(M_j.Z, r_ij, λ_shell_i[i, 1])
        ϕ_shell[i] += get_damped_electric_potential_charge(M_j.q_shell, r_ij, λ_overlap[i, 1])
        ϕ_shell[i] += get_damped_electric_potential_dipole(M_j.μ, -r_ij_vec, λ_overlap[i, 2])
        ϕ_shell[i] += get_damped_electric_potential_quadrupole(M_j.Q, -r_ij_vec, λ_overlap[i, 2], λ_overlap[i, 2])

        ### electric field ###
        # core electric field #
        get_electric_field_charge!(M_i.Z,  r_ij_vec, E_field_core[j])
        get_electric_field_charge!(M_j.Z, -r_ij_vec, E_field_core[i])

        get_damped_electric_field_charge!(M_i.q_shell,  r_ij_vec, E_field_core[j], λ_shell_i[i, 2])
        get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_core[j], λ_shell_i[i, 2], λ_shell_i[i, 3])
        get_damped_electric_field_quadrupole!(M_i.Q, r_ij_vec, E_field_core[j], λ_shell_i[i, 3], λ_shell_i[i, 4])
                    
        get_damped_electric_field_charge!(M_j.q_shell, -r_ij_vec, E_field_core[i], λ_shell_j[i, 2])
        get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_core[i], λ_shell_j[i, 2], λ_shell_j[i, 3])
        get_damped_electric_field_quadrupole!(M_j.Q, -r_ij_vec, E_field_core[i], λ_shell_j[i, 3], λ_shell_j[i, 4])

        # shell electric field #
        get_damped_electric_field_charge!(M_i.Z,  r_ij_vec, E_field_shell[j], λ_shell_j[i, 2])
        get_damped_electric_field_charge!(M_i.q_shell,  r_ij_vec, E_field_shell[j], λ_overlap[i, 2])
        get_damped_electric_field_dipole!(M_i.μ,  r_ij_vec, E_field_shell[j], λ_overlap[i, 2], λ_overlap[i, 3])
        get_damped_electric_field_quadrupole!(M_i.Q, r_ij_vec, E_field_shell[j], λ_overlap[i, 3], λ_overlap[i, 4])

        get_damped_electric_field_charge!(M_j.Z, -r_ij_vec, E_field_shell[i], λ_shell_i[i, 2])
        get_damped_electric_field_charge!(M_j.q_shell, -r_ij_vec, E_field_shell[i], λ_overlap[i, 2])
        get_damped_electric_field_dipole!(M_j.μ, -r_ij_vec, E_field_shell[i], λ_overlap[i, 2], λ_overlap[i, 3])
        get_damped_electric_field_quadrupole!(M_j.Q, -r_ij_vec, E_field_shell[i], λ_overlap[i, 3], λ_overlap[i, 4])

        ### electric field gradient ###
        # core electric field gradient #
        get_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_core[j])
        get_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_core[i])

        get_damped_electric_field_gradient_charge!(M_i.q_shell, r_ij_vec, E_field_gradients_core[j], λ_shell_i[i, 2], λ_shell_i[i, 3])
        get_damped_electric_field_gradient_dipole!(M_i.μ, r_ij_vec, E_field_gradients_core[j], λ_shell_i[i, 3], λ_shell_i[i, 4])
        get_damped_electric_field_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradients_core[j], λ_shell_i[i, 3], λ_shell_i[i, 4], λ_shell_i[i, 5])

        get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_core[i], λ_shell_j[i, 2], λ_shell_j[i, 3])
        get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_core[i], λ_shell_j[i, 3], λ_shell_j[i, 4])
        get_damped_electric_field_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradients_core[i], λ_shell_j[i, 3], λ_shell_j[i, 4], λ_shell_j[i, 5])
                    
        # shell electric field gradient #
        get_damped_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_shell[j], λ_shell_j[i, 2], λ_shell_j[i, 3])
        get_damped_electric_field_gradient_charge!(M_i.q_shell, r_ij_vec, E_field_gradients_shell[j], λ_overlap[i, 2], λ_overlap[i, 3])
        get_damped_electric_field_gradient_dipole!(M_i.μ, r_ij_vec, E_field_gradients_shell[j], λ_overlap[i, 3], λ_overlap[i, 4])
        get_damped_electric_field_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradients_shell[j], λ_overlap[i, 3], λ_overlap[i, 4], λ_overlap[i, 5])

        get_damped_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_shell[i], λ3_shell_i, λ5_shell_i)
        get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_shell[i], λ_overlap[i, 2], λ_overlap[i, 3])
        get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_shell[i], λ_overlap[i, 3], λ_overlap[i, 4])
        get_damped_electric_field_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradients_shell[i], λ_overlap[i, 3], λ_overlap[i, 4], λ_overlap[i, 5])

        # shell electric field gradient gradient #
        # TODO: DAMPING FUNCTION ORDERS ARE WRONG i.e. should have λ11 for quads
        #get_damped_electric_field_gradient_charge!(M_i.Z,  r_ij_vec, E_field_gradients_shell[j], λ3_shell_j, λ5_shell_j)
        #get_damped_electric_field_gradient_charge!(M_i.q_shell,  r_ij_vec, E_field_gradients_shell[j], λ3_overlap, λ5_overlap)
        #get_damped_electric_field_gradient_dipole!(M_i.μ,  r_ij_vec, E_field_gradients_shell[j], λ5_overlap, λ7_overlap)
        #get_electric_field_gradient_gradient_quadrupole!(M_i.Q, r_ij_vec, E_field_gradient_gradients_shell[j])#, λ5_overlap, λ7_overlap, λ9_overlap)

        #get_damped_electric_field_gradient_charge!(M_j.Z, -r_ij_vec, E_field_gradients_shell[i], λ3_shell_i, λ5_shell_i)
        #get_damped_electric_field_gradient_charge!(M_j.q_shell, -r_ij_vec, E_field_gradients_shell[i], λ3_overlap, λ5_overlap)
        #get_damped_electric_field_gradient_dipole!(M_j.μ, -r_ij_vec, E_field_gradients_shell[i], λ5_overlap, λ7_overlap)
        #get_electric_field_gradient_gradient_quadrupole!(M_j.Q, -r_ij_vec, E_field_gradient_gradients_shell[i])#, λ5_overlap, λ7_overlap, λ9_overlap)
    end
end