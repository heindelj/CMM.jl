abstract type AbstractMultipole end

mutable struct Multipole0 <: AbstractMultipole
    q::Float64
end
Multipole0() = Multipole0(0.0)

mutable struct Multipole1 <: AbstractMultipole
    q::Float64
    μ::MVector{3,Float64}
end
Multipole1() = Multipole1(0.0, @MVector zeros(3))

mutable struct Multipole2 <: AbstractMultipole
    q::Float64
    μ::MVector{3,Float64}
    Q::MMatrix{3,3,Float64}
end
Multipole2() = Multipole2(0.0, zeros(3), zeros(3, 3))

struct Charges
    q::Vector{Float64}
end
Charges(natoms::Int) = Charges(zeros(natoms))
function Base.setindex!(q::Charges, v::Float64, i::Int)
    q.q[i] = v
end
function Base.getindex(q::Charges, i::Int)
    return q.q[i]
end

struct Dipoles
    μ::Vector{SVector{3, Float64}}
end
Dipoles(natoms::Int) = Dipoles([@SVector zeros(3) for _ in 1:natoms])
function Base.setindex!(μ::Dipoles, v::SVector{3, Float64}, i::Int)
    μ.μ[i] = v
end
function Base.getindex(μ::Dipoles, i::Int)
    return μ.μ[i]
end

struct Quadrupoles
    Q::Vector{SMatrix{3, 3, Float64, 9}}
end
Quadrupoles(natoms::Int) = Quadrupoles([@SMatrix zeros(3, 3) for _ in 1:natoms])
function Base.setindex!(Q::Quadrupoles, v::SMatrix{3, 3, Float64, 9}, i::Int)
    Q.Q[i] = v
end
function Base.getindex(Q::Quadrupoles, i::Int)
    return Q.Q[i]
end

struct DampingValues
    rank::Int
    λ::Matrix{Float64}
end
DampingValues(rank::Int, npairs::Int) = DampingValues(rank, zeros(npairs, rank+1))
# ^^^ Note that if you want forces, you have to ask for one rank higher than
# you are computing the energies at.

# This should be somewhere higher-level
struct Distances
    r_vec::Vector{SVector{3, Float64}}
    r::Vector{Float64}
    ij_pairs::Vector{Tuple{Int, Int}}
end
Distances(num_atoms::Int) = Distances(
    [@SVector zeros(3) for _ in 1:(num_atoms * (num_atoms-1) ÷ 2)],
    zeros(num_atoms * (num_atoms-1) ÷ 2),
    [(0, 0) for _ in 1:(num_atoms * (num_atoms-1) ÷ 2)]
)

### Core-Shell Multipoles ###
# Core-Shell Multipoles are the same as a regular multipole
# except that there is a point-like core charge.
# Z represents the core charge and q represents the shell charge.
# The total charge of the atom is the sum of these.
# The only reason to use this is if you are going to apply damping
# to the shells since this will effectively capture the charge
# penetration effect. If these Multipoles were not damped, the results
# would just be identical to an ordinary partial charge set to Z+q.
mutable struct CSMultipole0 <: AbstractMultipole
    Z::Float64
    q_shell::Float64
end
CSMultipole0() = CSMultipole0(0.0)

mutable struct CSMultipole1 <: AbstractMultipole
    Z::Float64
    q_shell::Float64
    μ::MVector{3,Float64}
end
CSMultipole1() = CSMultipole1(0.0, 0.0, @MVector zeros(3))

mutable struct CSMultipole2 <: AbstractMultipole
    Z::Float64
    q_shell::Float64
    μ::MVector{3,Float64}
    Q::MMatrix{3,3,Float64}
end
CSMultipole2() = CSMultipole2(0.0, 0.0, zeros(3), zeros(3, 3))

function fill_multipoles_with_precomputed_multipoles!(
    multipoles::Vector{CSMultipole2},
    core_charges::Charges,
    shell_charges::Charges,
    dipoles::Dipoles,
    quadrupoles::Quadrupoles
)
    # @SPEED: The assignment to qaudrupoles allocates each time.
    # I don't really know why but there is likely a way to avoid it.
    for i in eachindex(multipoles)
        core_charges.q[i] = multipoles[i].Z
        shell_charges.q[i] = multipoles[i].q_shell
        dipoles.μ[i] = SVector{3, Float64}(multipoles[i].μ)
        quadrupoles[i] = SMatrix{3, 3, Float64, 9}(multipoles[i].Q)
    end
end

include("damped_multipoles.jl")

function convert_spherical_dipole_to_cartesian_dipole(μ_spherical::MVector{3,Float64})
    return @MVector [μ_spherical[2], μ_spherical[3], μ_spherical[1]]
end

function convert_cartesian_dipole_to_spherical_dipole(μ_cartesian::MVector{3,Float64})
    return @MVector [μ_cartesian[3], μ_cartesian[1], μ_cartesian[2]]
end

function convert_spherical_quadrupole_to_cartesian_quadrupole(Q_spherical::MVector{5,Float64})
    Q = @MMatrix zeros(3, 3)
    # quadrupole moments in spherical multipole are given in the order:
    # Q20, Q21c, Q21s, Q22c, Q22s
    # Conversion factors come from Table E.2 of Anthony Stone book
    Q[1, 1] = -0.5 * Q_spherical[1] + sqrt(3) / 2 * Q_spherical[4]
    Q[2, 2] = -0.5 * Q_spherical[1] - sqrt(3) / 2 * Q_spherical[4]
    Q[3, 3] = Q_spherical[1]

    Q[1, 2] = sqrt(3) / 2 * Q_spherical[5]
    Q[2, 1] = sqrt(3) / 2 * Q_spherical[5]

    Q[1, 3] = sqrt(3) / 2 * Q_spherical[2]
    Q[3, 1] = sqrt(3) / 2 * Q_spherical[2]

    Q[2, 3] = sqrt(3) / 2 * Q_spherical[3]
    Q[3, 2] = sqrt(3) / 2 * Q_spherical[3]
    return Q
end

function convert_cartesian_quadrupole_to_spherical_quadrupole(Q::MMatrix{3,3,Float64})
    Q_spherical = @MVector zeros(5)

    # Conversion factors come from Table E.1 of Anthony Stone book
    Q_spherical[1] = Q[3, 3]
    Q_spherical[2] = 2 / sqrt(3) * Q[1, 3]
    Q_spherical[3] = 2 / sqrt(3) * Q[2, 3]
    Q_spherical[4] = 1 / sqrt(3) * (Q[1, 1] - Q[2, 2])
    Q_spherical[5] = 2 / sqrt(3) * Q[1, 2]
    return Q_spherical
end

"""
Electric potential due to a point charge.
"""
@inline function get_electric_potential_charge(q::Float64, r_ij::Float64)
    return q / r_ij
end

"""
Electric potential due to a dipole. r_ij is r_j - r_i where dipole is at i.
"""
@inline function get_electric_potential_dipole(μ_i::MVector{3,Float64}, r_ij::MVector{3,Float64})
    return μ_i ⋅ r_ij / norm(r_ij)^3
end

"""
Electric potential due to a quadrupole. r_ij is r_j - r_i where quadrupole is at i.
"""
@inline function get_electric_potential_quadrupole(Q_i::MMatrix{3,3,Float64}, r_ij::MVector{3,Float64})
    return r_ij' * Q_i * r_ij / norm(r_ij)^5
end

"""
Electric field due to charge. r_ij is r_j - r_i where charge is located at r_i.
"""
@inline function get_electric_field_charge(q::Float64, r_ij::MVector{3,Float64})
    return q * r_ij / norm(r_ij)^3
end

"""
Electric field due to dipole located at r_i. Field is evaluated at r_j.
r_ij is r_j - r_i where dipole is at r_i.
"""
@inline function get_electric_field_dipole(μ_i::MVector{3,Float64}, r_ij::MVector{3,Float64})
    return (3 * (μ_i ⋅ r_ij) * r_ij / norm(r_ij)^2 - μ_i) / norm(r_ij)^3
end

"""
Electric field due to a quadrupole at r_i. Field is evaluated at r_j.
"""
@inline function get_electric_field_quadrupole(Q_i::MMatrix{3,3,Float64}, r_ij::MVector{3,Float64})
    return (-2.0 * Q_i * r_ij + 5.0 * (r_ij' * Q_i * r_ij) * r_ij / (r_ij ⋅ r_ij)) / norm(r_ij)^5
end

"""
Electric field gradient due to a quadrupole at r_i. Field gradient is evaluated at r_j.
"""
function get_electric_field_gradient_charge(q_i::Float64, r_ij::MVector{3,Float64})
    E_field_grad = @MMatrix zeros(3, 3)
    for w1 in 1:3
        for w2 in 1:3
            l = 0
            m = 0
            n = 0
            if w1 == 1 l += 1 end
            if w2 == 1 l += 1 end

            if w1 == 2 m += 1 end
            if w2 == 2 m += 1 end

            if w1 == 3 n += 1 end
            if w2 == 3 n += 1 end

            E_field_grad[w1, w2] -= q_i * get_T_αβγ(r_ij, l, m, n)
        end
    end
    return E_field_grad
end

"""
Electric field gradient due to a quadrupole at r_i. Field gradient is evaluated at r_j.
"""
function get_electric_field_gradient_dipole(μ_i::MVector{3, Float64}, r_ij::MVector{3,Float64})
    E_field_grad = @MMatrix zeros(3, 3)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                l = 0
                m = 0
                n = 0
                if α == 1 l += 1 end
                if β == 1 l += 1 end
                if γ == 1 l += 1 end

                if α == 2 m += 1 end
                if β == 2 m += 1 end
                if γ == 2 m += 1 end

                if α == 3 n += 1 end
                if β == 3 n += 1 end
                if γ == 3 n += 1 end

                E_field_grad[α, β] += μ_i[γ] * get_T_αβγ(r_ij, l, m, n)
            end
        end
    end
    return E_field_grad
end

"""
Electric field gradient due to a quadrupole at r_i. Field gradient is evaluated at r_j.
"""
function get_electric_field_gradient_quadrupole(Q_i::MMatrix{3,3,Float64}, r_ij::MVector{3,Float64})
    E_field_grad = @MMatrix zeros(3, 3)
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for η in 1:3
                    l = 0
                    m = 0
                    n = 0
                    if α == 1 l += 1 end
                    if β == 1 l += 1 end
                    if γ == 1 l += 1 end
                    if η == 1 l += 1 end

                    if α == 2 m += 1 end
                    if β == 2 m += 1 end
                    if γ == 2 m += 1 end
                    if η == 2 m += 1 end

                    if α == 3 n += 1 end
                    if β == 3 n += 1 end
                    if γ == 3 n += 1 end
                    if η == 3 n += 1 end

                    E_field_grad[α, β] -= Q_i[γ, η] * get_T_αβγ(r_ij, l, m, n)
                end
            end
        end
    end
    return E_field_grad / 3.0 # account for tracelessness of quadrupoles
end

@inline function get_quadrupole_charge_interaction_gradient(Q_i::MMatrix{3,3,Float64}, q_j::Float64, r_ij::MVector{3,Float64})
    grad_Qq = @MVector zeros(3)
    for α in 1:3
        for β in 1:3
            for w in 1:3
                    l = 0
                    m = 0
                    n = 0
                    if α == 1 l += 1 end
                    if β == 1 l += 1 end
                    if w == 1 l += 1 end

                    if α == 2 m += 1 end
                    if β == 2 m += 1 end
                    if w == 2 m += 1 end

                    if α == 3 n += 1 end
                    if β == 3 n += 1 end
                    if w == 3 n += 1 end

                    grad_Qq[w] += Q_i[α, β] * get_T_αβγ(-r_ij, l, m, n) * q_j
            end
        end
    end
    return grad_Qq / 3.0
end

@inline function get_quadrupole_dipole_interaction_gradient(Q_i::MMatrix{3,3,Float64}, μ_j::MVector{3, Float64}, r_ij::MVector{3,Float64})
    grad_Qμ = @MVector zeros(3)
    for α in 1:3
        for β in 1:3
            for w in 1:3
                for γ in 1:3
                    l = 0
                    m = 0
                    n = 0
                    if α == 1 l += 1 end
                    if β == 1 l += 1 end
                    if γ == 1 l += 1 end
                    if w == 1 l += 1 end

                    if α == 2 m += 1 end
                    if β == 2 m += 1 end
                    if γ == 2 m += 1 end
                    if w == 2 m += 1 end

                    if α == 3 n += 1 end
                    if β == 3 n += 1 end
                    if γ == 3 n += 1 end
                    if w == 3 n += 1 end

                    grad_Qμ[w] += Q_i[α, β] * get_T_αβγ(-r_ij, l, m, n) * μ_j[γ]
                end
            end
        end
    end
    return grad_Qμ / 3.0
end

@inline function get_quadrupole_quadrupole_interaction_gradient(Q_i::MMatrix{3,3,Float64}, Q_j::MMatrix{3,3,Float64}, r_ij::MVector{3,Float64})
    grad_QQ = @MVector zeros(3)
    for α in 1:3
        for β in 1:3
            for w in 1:3
                for γ in 1:3
                    for δ in 1:3
                        l = 0
                        m = 0
                        n = 0
                        if α == 1 l += 1 end
                        if β == 1 l += 1 end
                        if γ == 1 l += 1 end
                        if δ == 1 l += 1 end
                        if w == 1 l += 1 end

                        if α == 2 m += 1 end
                        if β == 2 m += 1 end
                        if γ == 2 m += 1 end
                        if δ == 2 m += 1 end
                        if w == 2 m += 1 end

                        if α == 3 n += 1 end
                        if β == 3 n += 1 end
                        if γ == 3 n += 1 end
                        if δ == 3 n += 1 end
                        if w == 3 n += 1 end

                        grad_QQ[w] += Q_i[α, β] * get_T_αβγ(-r_ij, l, m, n) * Q_j[γ, δ]
                    end
                end
            end
        end
    end
    return grad_QQ / 9.0
end

@inline function get_quadrupole_quadrupole_interaction_gradient!(Q_i::MMatrix{3,3,Float64}, Q_j::MMatrix{3,3,Float64}, r_ij::MVector{3,Float64}, grad_QQ::MVector{3, Float64})
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for δ in 1:3
                    for w in 1:3
                        l = 0
                        m = 0
                        n = 0
                        if α == 1 l += 1 end
                        if β == 1 l += 1 end
                        if γ == 1 l += 1 end
                        if δ == 1 l += 1 end
                        if w == 1 l += 1 end

                        if α == 2 m += 1 end
                        if β == 2 m += 1 end
                        if γ == 2 m += 1 end
                        if δ == 2 m += 1 end
                        if w == 2 m += 1 end

                        if α == 3 n += 1 end
                        if β == 3 n += 1 end
                        if γ == 3 n += 1 end
                        if δ == 3 n += 1 end
                        if w == 3 n += 1 end

                        grad_QQ[w] += Q_i[α, β] * get_T_αβγ(-r_ij, l, m, n) * Q_j[γ, δ] / 9.0
                    end
                end
            end
        end
    end
end

function recursive_multipole_multipole_interaction(r_i::MVector{3,Float64}, M_i::Multipole0, r_j::MVector{3,Float64}, M_j::Multipole0)
    r_ji = r_i - r_j
    return M_i.q * get_T_αβγ(r_ji, 0, 0, 0) * M_j.q
end

function multipole_multipole_interaction(r_i::MVector{3,Float64}, M_i::Multipole0, r_j::MVector{3,Float64}, M_j::Multipole0)
    r_ij = r_j - r_i

    ϕ_i = get_electric_potential_charge(M_j.q, norm(r_ij))
    ϕ_j = get_electric_potential_charge(M_i.q, norm(r_ij))
    return 0.5 * (M_i.q * ϕ_i + M_j.q * ϕ_j)
end

"""
Computes the interaction energy between two multipoles,
M_i and M_j located at r_i and r_j, respectively.
"""
function multipole_multipole_interaction(r_i::MVector{3,Float64}, M_i::Multipole1, r_j::MVector{3,Float64}, M_j::Multipole1)
    r_ij = r_j - r_i

    ϕ_i = (
        get_damped_electric_potential_charge(M_j.q, norm(r_ij)) +
        get_damped_electric_potential_dipole(M_j.μ, -r_ij)
    )
    ϕ_j = (
        get_damped_electric_potential_charge(M_i.q, norm(r_ij)) +
        get_damped_electric_potential_dipole(M_i.μ, r_ij)
    )
    E_i = (
        get_damped_electric_field_charge(M_j.q, -r_ij) +
        get_damped_electric_field_dipole(M_j.μ, -r_ij)
    )
    E_j = (
        get_damped_electric_field_charge(M_i.q, r_ij) +
        get_damped_electric_field_dipole(M_i.μ, r_ij)
    )
    return 0.5 * (
        M_i.q * ϕ_i + M_j.q * ϕ_j -
        M_i.μ ⋅ E_i - M_j.μ ⋅ E_j
    )
end

function recursive_multipole_multipole_interaction(r_i::MVector{3,Float64}, M_i::Multipole1, r_j::MVector{3,Float64}, M_j::Multipole1)
    r_ji = r_i - r_j
    # charge - charge energy
    energy_qq = M_i.q * get_T_αβγ(r_ji, 0, 0, 0) * M_j.q

    # charge-dipole
    energy_qμ = -M_i.q * (
        get_T_αβγ(r_ji, 1, 0, 0) * M_j.μ[1] +
        get_T_αβγ(r_ji, 0, 1, 0) * M_j.μ[2] +
        get_T_αβγ(r_ji, 0, 0, 1) * M_j.μ[3]
    )

    energy_μq = (
        M_i.μ[1] * get_T_αβγ(r_ji, 1, 0, 0) +
        M_i.μ[2] * get_T_αβγ(r_ji, 0, 1, 0) +
        M_i.μ[3] * get_T_αβγ(r_ji, 0, 0, 1)
    ) * M_j.q

    # dipole-dipole
    energy_μμ = -(
        M_i.μ[1] * get_T_αβγ(r_ji, 2, 0, 0) * M_j.μ[1] +
        M_i.μ[1] * get_T_αβγ(r_ji, 1, 1, 0) * M_j.μ[2] +
        M_i.μ[1] * get_T_αβγ(r_ji, 1, 0, 1) * M_j.μ[3] +
        M_i.μ[2] * get_T_αβγ(r_ji, 1, 1, 0) * M_j.μ[1] +
        M_i.μ[2] * get_T_αβγ(r_ji, 0, 2, 0) * M_j.μ[2] +
        M_i.μ[2] * get_T_αβγ(r_ji, 0, 1, 1) * M_j.μ[3] +
        M_i.μ[3] * get_T_αβγ(r_ji, 1, 0, 1) * M_j.μ[1] +
        M_i.μ[3] * get_T_αβγ(r_ji, 0, 1, 1) * M_j.μ[2] +
        M_i.μ[3] * get_T_αβγ(r_ji, 0, 0, 2) * M_j.μ[3]
    )
    
    return energy_qq + energy_qμ + energy_μq + energy_μμ
end

"""
Computes the interaction energy between two multipoles,
M_i and M_j located at r_i and r_j, respectively.
"""
function multipole_multipole_interaction(r_i::MVector{3,Float64}, M_i::Multipole2, r_j::MVector{3,Float64}, M_j::Multipole2)
    r_ij = r_j - r_i

    # NOTE: Currently we double the potential and field contributions
    # from quadrupoles because the other interactions are all counted
    # twice. So, the final energy is multiplied by 0.5, but we are not
    # double counting the quadrupole interactions. That is, we aren't
    # counting the field gradient interaction with each quadrupole.

    ϕ_i = (
        get_electric_potential_charge(M_j.q, norm(r_ij)) +
        get_electric_potential_dipole(M_j.μ, -r_ij) +
        get_electric_potential_quadrupole(M_j.Q, r_ij)
    )
    ϕ_j = (
        get_electric_potential_charge(M_i.q, norm(r_ij)) +
        get_electric_potential_dipole(M_i.μ, r_ij) +
        get_electric_potential_quadrupole(M_i.Q, r_ij)
    )
    E_i = (
        get_electric_field_charge(M_j.q, r_ij) +
        get_electric_field_dipole(M_j.μ, r_ij) +
        get_electric_field_quadrupole(M_j.Q, -r_ij)
    )
    E_j = (
        get_electric_field_charge(M_i.q, -r_ij) +
        get_electric_field_dipole(M_i.μ, -r_ij) +
        get_electric_field_quadrupole(M_i.Q, r_ij)
    )

    E_grad_i = (
        get_electric_field_gradient_charge(M_j.q, r_ij) +
        get_electric_field_gradient_dipole(M_j.μ, -r_ij) +
        get_electric_field_gradient_quadrupole(M_j.Q, r_ij)
    )
    E_grad_j = (
        get_electric_field_gradient_charge(M_i.q, r_ij) +
        get_electric_field_gradient_dipole(M_i.μ, r_ij) +
        get_electric_field_gradient_quadrupole(M_i.Q, r_ij)
    )

    return 0.5 * (
        M_i.q * ϕ_i + M_j.q * ϕ_j -
        M_i.μ ⋅ E_i - M_j.μ ⋅ E_j +
        (M_i.Q ⋅ E_grad_i + M_j.Q ⋅ E_grad_j) / 3.0 # divide by 3 for traceless quadrupole
    )
end

function get_electrostatic_potential_on_grid(coords::Vector{MVector{3,Float64}}, multipoles::Vector{Multipole2}, grid_points::Vector{MVector{3, Float64}})
    potentials = zeros(length(grid_points))
    for i_grid in eachindex(grid_points)
        for i in eachindex(coords)
            r_ij = grid_points[i_grid] - coords[i]
            M_i = multipoles[i]
            potentials[i_grid] += (
                get_electric_potential_charge(M_i.q, norm(r_ij)) +
                get_electric_potential_dipole(M_i.μ, r_ij) +
                get_electric_potential_quadrupole(M_i.Q, r_ij)
            )
        end
    end
    return potentials
end

function total_multipole_interaction_energy(coords::Vector{MVector{3, Float64}}, multipoles::Vector{Multipole1}, fragment_indices::Vector{Vector{Int}})
    multipole_interaction_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        multipole_interaction_energy += multipole_multipole_interaction(coords[i], multipoles[i], coords[j], multipoles[j])
                    end
                end
            end
        end
    end
    return multipole_interaction_energy
end

function total_multipole_interaction_energy(coords::Vector{MVector{3, Float64}}, multipoles::Vector{Multipole2}, fragment_indices::Vector{Vector{Int}})
    multipole_interaction_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        multipole_interaction_energy += multipole_multipole_interaction(coords[i], multipoles[i], coords[j], multipoles[j])
                    end
                end
            end
        end
    end
    return multipole_interaction_energy
end

function recursive_quad_quad_interaction(r_ji::MVector{3, Float64}, Q1::MMatrix{3, 3, Float64}, Q2::MMatrix{3, 3, Float64})
    energy_QQ = 0.0
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for δ in 1:3
                    l = 0
                    m = 0
                    n = 0
                    if α == 1 l += 1 end
                    if β == 1 l += 1 end
                    if γ == 1 l += 1 end
                    if δ == 1 l += 1 end

                    if α == 2 m += 1 end
                    if β == 2 m += 1 end
                    if γ == 2 m += 1 end
                    if δ == 2 m += 1 end

                    if α == 3 n += 1 end
                    if β == 3 n += 1 end
                    if γ == 3 n += 1 end
                    if δ == 3 n += 1 end

                    energy_QQ += Q1[α, β] * get_T_αβγ(r_ji, l, m, n) * Q2[γ, δ]
                end
            end
        end
    end
    return energy_QQ / 9.0
end

function recursive_multipole_multipole_interaction(r_i::MVector{3,Float64}, M_i::Multipole2, r_j::MVector{3,Float64}, M_j::Multipole2)
    energy_charges_and_dipoles = recursive_multipole_multipole_interaction(r_i, Multipole1(M_i.q, M_i.μ), r_j, Multipole1(M_j.q, M_j.μ))

    r_ji = r_i - r_j
    # Divide by 3 because these are traceless quadrupoles
    # See: M. Challacombe et al. Chemical Physics Letters 241 (1995) 67-72

    energy_qQ = M_i.q * (
        get_T_αβγ(r_ji, 2, 0, 0) * M_j.Q[1, 1] +
        get_T_αβγ(r_ji, 1, 1, 0) * M_j.Q[1, 2] +
        get_T_αβγ(r_ji, 1, 0, 1) * M_j.Q[1, 3] +
        get_T_αβγ(r_ji, 1, 1, 0) * M_j.Q[2, 1] +
        get_T_αβγ(r_ji, 0, 2, 0) * M_j.Q[2, 2] +
        get_T_αβγ(r_ji, 0, 1, 1) * M_j.Q[2, 3] +
        get_T_αβγ(r_ji, 1, 0, 1) * M_j.Q[3, 1] +
        get_T_αβγ(r_ji, 0, 1, 1) * M_j.Q[3, 2] +
        get_T_αβγ(r_ji, 0, 0, 2) * M_j.Q[3, 3]
    )

    energy_Qq = (
        M_i.Q[1, 1] * get_T_αβγ(r_ji, 2, 0, 0) +
        M_i.Q[1, 2] * get_T_αβγ(r_ji, 1, 1, 0) +
        M_i.Q[1, 3] * get_T_αβγ(r_ji, 1, 0, 1) +
        M_i.Q[2, 1] * get_T_αβγ(r_ji, 1, 1, 0) +
        M_i.Q[2, 2] * get_T_αβγ(r_ji, 0, 2, 0) +
        M_i.Q[2, 3] * get_T_αβγ(r_ji, 0, 1, 1) +
        M_i.Q[3, 1] * get_T_αβγ(r_ji, 1, 0, 1) +
        M_i.Q[3, 2] * get_T_αβγ(r_ji, 0, 1, 1) +
        M_i.Q[3, 3] * get_T_αβγ(r_ji, 0, 0, 2)
    ) * M_j.q

    energy_Qq /= 3.0
    energy_qQ /= 3.0

    energy_Qμ = 0.0
    energy_μQ = 0.0
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                l = 0
                m = 0
                n = 0
                if α == 1 l += 1 end
                if β == 1 l += 1 end
                if γ == 1 l += 1 end

                if α == 2 m += 1 end
                if β == 2 m += 1 end
                if γ == 2 m += 1 end

                if α == 3 n += 1 end
                if β == 3 n += 1 end
                if γ == 3 n += 1 end

                T_lmn = get_T_αβγ(r_ji, l, m, n)
                energy_Qμ -= M_i.Q[α, β] * T_lmn * M_j.μ[γ]
                energy_μQ += M_i.μ[γ] * T_lmn * M_j.Q[α, β]
            end
        end
    end
    energy_Qμ /= 3.0
    energy_μQ /= 3.0

    energy_QQ = 0.0
    for α in 1:3
        for β in 1:3
            for γ in 1:3
                for δ in 1:3
                    l = 0
                    m = 0
                    n = 0
                    if α == 1 l += 1 end
                    if β == 1 l += 1 end
                    if γ == 1 l += 1 end
                    if δ == 1 l += 1 end

                    if α == 2 m += 1 end
                    if β == 2 m += 1 end
                    if γ == 2 m += 1 end
                    if δ == 2 m += 1 end

                    if α == 3 n += 1 end
                    if β == 3 n += 1 end
                    if γ == 3 n += 1 end
                    if δ == 3 n += 1 end

                    energy_QQ += M_i.Q[α, β] * get_T_αβγ(r_ji, l, m, n) * M_j.Q[γ, δ]
                end
            end
        end
    end
    energy_QQ /= 9.0

    return energy_charges_and_dipoles + energy_qQ + energy_Qq + energy_Qμ + energy_μQ + energy_QQ
end

"""
Get the electrostatic gradients based on pre-accumulated electric fields.
"""
function get_electrostatic_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    multipoles::AbstractVector{Multipole0},
    E_field::AbstractVector{MVector{3,Float64}},
    grads::AbstractVector{MVector{3,Float64}},
)
    for i in eachindex(coords)
        grads[i] -= multipoles[i].q * E_field[i]
    end
end

"""
Get the electrostatic gradients based on pre-accumulated electric fields.
"""
function get_electrostatic_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole1},
    E_field::AbstractVector{MVector{3,Float64}},
    grads::AbstractVector{MVector{3,Float64}}
)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]

                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)

                    #### gradient of dipole-charge interaction ###
                    grads[j] -= get_damped_electric_field_gradient_charge(M_i.q, r_ij_vec) * M_j.μ
                    grads[i] -= get_damped_electric_field_gradient_charge(M_j.q, -r_ij_vec) * M_i.μ

                    #### gradient of dipole-dipole interaction ###
                    dip_dip_grad_j = get_damped_electric_field_gradient_dipole(M_i.μ, r_ij_vec) * M_j.μ
                    grads[j] -= dip_dip_grad_j
                    grads[i] += dip_dip_grad_j
                end
            end
        end
    end
    # accumulate charge gradients.
    for i in eachindex(coords)
        grads[i] -= multipoles[i].q * E_field[i]
    end
end

function get_electrostatic_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole1},
    induced_multipoles::AbstractVector{Multipole1},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_damped::AbstractVector{MVector{3,Float64}},
    E_field_induced::AbstractVector{MVector{3,Float64}},
    grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64},
    damping_type::AbstractDamping
)
    a = get_a(params, damping_type)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                M_i_ind = induced_multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    M_j_ind = induced_multipoles[j]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    u = get_u(r_ij, labels[i], labels[j], params, damping_type)

                    ### gradient of dipole-charge interaction ###
                    # permanent dipole-charge interaction gradient
                    grads[j] -= get_damped_electric_field_gradient_charge(M_i.q, r_ij_vec) * M_j.μ
                    grads[i] -= get_damped_electric_field_gradient_charge(M_j.q, -r_ij_vec) * M_i.μ
                    grads[j] -= get_damped_electric_field_gradient_charge(M_i_ind.q, r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_j.μ
                    grads[i] -= get_damped_electric_field_gradient_charge(M_j_ind.q, -r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_i.μ
                    
                    # induced dipole-permanent charge interaction gradient
                    grads[j] -= get_damped_electric_field_gradient_charge(M_i.q, r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_j_ind.μ
                    grads[i] -= get_damped_electric_field_gradient_charge(M_j.q, -r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_i_ind.μ

                    ### gradient of dipole-dipole interaction ###
                    # permanent dipole-dipole
                    dip_dip_grad_perm_perm_j = get_damped_electric_field_gradient_dipole(M_i.μ, r_ij_vec) * M_j.μ
                    
                    # NOTE: We only calculate the unique contributions and then use
                    # Newton's third law to get the induced-permanent from permanent-induced

                    # permanent dipole i / induced dipole j interaction
                    dip_dip_grad_perm_ind_j = get_damped_electric_field_gradient_dipole(M_i.μ, r_ij_vec, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type)) * M_j_ind.μ
                    
                    # permanent dipole j / induced dipole i interaction
                    dip_dip_grad_perm_ind_i = get_damped_electric_field_gradient_dipole(M_j.μ, -r_ij_vec, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type)) * M_i_ind.μ
                    
                    # induced dipole - induced dipole
                    dip_dip_grad_ind_ind_j = get_damped_electric_field_gradient_dipole(M_i_ind.μ, r_ij_vec, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type)) * M_j_ind.μ

                    # minus sign on j in the gradient comes from the fact the
                    # energy is -μ⋅E.
                    grads[j] += -dip_dip_grad_perm_perm_j - dip_dip_grad_perm_ind_j + dip_dip_grad_perm_ind_i - dip_dip_grad_ind_ind_j
                    grads[i] +=  dip_dip_grad_perm_perm_j - dip_dip_grad_perm_ind_i + dip_dip_grad_perm_ind_j + dip_dip_grad_ind_ind_j
                end
            end
        end
    end
    # accumulate charge gradients
    for i in eachindex(coords)
        # permanent-permanent and permanent-induced
        grads[i] -= multipoles[i].q * (E_field[i] + E_field_induced[i])
        # induced-permanent
        grads[i] -= induced_multipoles[i].q * (E_field_damped[i] + E_field_induced[i])
    end
end

function get_electrostatic_and_polarization_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole1},
    induced_multipoles::AbstractVector{Multipole1},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_damped::AbstractVector{MVector{3,Float64}},
    E_field_induced::AbstractVector{MVector{3,Float64}},
    elec_grads::AbstractVector{MVector{3,Float64}},
    polarization_grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64},
    damping_type::AbstractDamping
)
    a = get_a(params, damping_type)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                M_i_ind = induced_multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    M_j_ind = induced_multipoles[j]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    u = get_u(r_ij, labels[i], labels[j], params, damping_type)

                    ### gradient of dipole-charge interaction ###
                    # permanent dipole-charge interaction gradient
                    elec_grads[j] -= get_damped_electric_field_gradient_charge(M_i.q, r_ij_vec) * M_j.μ
                    elec_grads[i] -= get_damped_electric_field_gradient_charge(M_j.q, -r_ij_vec) * M_i.μ
                    
                    # induced charge-permanent dipole interaction gradient
                    polarization_grads[j] -= get_damped_electric_field_gradient_charge(M_i_ind.q, r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_j.μ
                    polarization_grads[i] -= get_damped_electric_field_gradient_charge(M_j_ind.q, -r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_i.μ
                    
                    # induced charge - induced dipole interaction gradient
                    # mutual polarization only
                    #polarization_grads[j] -= get_damped_electric_field_gradient_charge(M_i_ind.q, r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_j_ind.μ
                    #polarization_grads[i] -= get_damped_electric_field_gradient_charge(M_j_ind.q, -r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_i_ind.μ

                    # induced dipole-permanent charge interaction gradient
                    polarization_grads[j] -= get_damped_electric_field_gradient_charge(M_i.q, r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_j_ind.μ
                    polarization_grads[i] -= get_damped_electric_field_gradient_charge(M_j.q, -r_ij_vec, get_λ3(u, a, damping_type), get_λ5(u, a, damping_type)) * M_i_ind.μ

                    ### gradient of dipole-dipole interaction ###
                    # permanent dipole-dipole
                    dip_dip_grad_perm_perm_j = get_damped_electric_field_gradient_dipole(M_i.μ, r_ij_vec) * M_j.μ
                    
                    # NOTE: We only calculate the unique contributions and then use
                    # Newton's third law to get the induced-permanent from permanent-induced

                    # permanent dipole i / induced dipole j interaction
                    dip_dip_grad_perm_ind_j = get_damped_electric_field_gradient_dipole(M_i.μ, r_ij_vec, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type)) * M_j_ind.μ
                    
                    # permanent dipole j / induced dipole i interaction
                    dip_dip_grad_perm_ind_i = get_damped_electric_field_gradient_dipole(M_j.μ, -r_ij_vec, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type)) * M_i_ind.μ

                    # induced dipole i / induced dipole j
                    dip_dip_grad_ind_ind_i = get_damped_electric_field_gradient_dipole(M_j_ind.μ, -r_ij_vec, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type)) * M_i_ind.μ

                    # minus sign on j in the gradient comes from the fact the
                    # energy is -μ⋅E.
                    elec_grads[j] += -dip_dip_grad_perm_perm_j
                    polarization_grads[j] += -dip_dip_grad_perm_ind_j + dip_dip_grad_perm_ind_i
                    elec_grads[i] +=  dip_dip_grad_perm_perm_j
                    polarization_grads[i] += -dip_dip_grad_perm_ind_i + dip_dip_grad_perm_ind_j
                    
                    # induced - induced only
                    #polarization_grads[i] -= dip_dip_grad_ind_ind_i
                    #polarization_grads[j] += dip_dip_grad_ind_ind_i
                end
            end
        end
    end
    # accumulate charge gradients
    for i in eachindex(coords)
        # permanent-permanent and permanent-induced
        elec_grads[i] -= multipoles[i].q * E_field[i]
        polarization_grads[i] -= multipoles[i].q * E_field_induced[i]
        # induced-permanent
        polarization_grads[i] -= induced_multipoles[i].q * E_field_damped[i]
        # induced-induced
        #polarization_grads[i] -= induced_multipoles[i].q * E_field_induced[i]
    end
end

function get_mutual_polarization_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    induced_multipoles::AbstractVector{Multipole1},
    grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64}
)
    E_field = @MVector zeros(3)
    E_field_gradient = @MMatrix zeros(3, 3)
    # accumulate gradients involving dipoles
    for i_frag in eachindex(fragment_indices)
        for j_frag in i_frag+1:length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i_ind = induced_multipoles[i]
                b_i = abs(params[Symbol(labels[i], :_b_exch)])
                for j in fragment_indices[j_frag]
                    if i < j
                        @views E_field[:] -= E_field[:]
                        @views E_field_gradient[:,:] -= E_field_gradient[:,:]

                        M_j_ind = induced_multipoles[j]
                        b_j = abs(params[Symbol(labels[j], :_b_exch)])
                        b_ij = sqrt(b_i * b_j)
                    
                        r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(r_ij_vec)
                        x = slater_damping_value(r_ij, b_ij)
                        x_gradient_i = slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)
                    
                        λ_qq = inc_gamma(x, 3)
                        λ_qq_deriv = inc_gamma_derivative(x, 3)
                        λ_qμ = inc_gamma(x, 5)
                        λ_qμ_deriv = inc_gamma_derivative(x, 5)
                        λ_μμ = inc_gamma(x, 7)
                        λ_μμ_deriv = inc_gamma_derivative(x, 7)

                        # induced charge-induced charge #
                        get_damped_electric_field_charge!(M_i_ind.q, r_ij_vec, E_field, λ_qq)
                        charge_charge_grad_ind_j = -(
                            E_field + 
                            M_i_ind.q * λ_qq_deriv * x_gradient_i / r_ij
                        ) * M_j_ind.q
                        grads[j] += charge_charge_grad_ind_j
                        grads[i] -= charge_charge_grad_ind_j
                        

                        # induced dipole-induced charge #
                        @views E_field[:] -= E_field[:]
                        get_damped_electric_field_dipole!(M_j_ind.μ, r_ij_vec, E_field, λ_qμ, λ_qμ)
                        charge_dipole_grad_ind_j = (
                            E_field + 
                            λ_qμ_deriv * M_j_ind.μ ⋅ r_ij_vec * x_gradient_i  / r_ij^3
                        ) * M_i_ind.q
                        grads[j] += charge_dipole_grad_ind_j
                        grads[i] -= charge_dipole_grad_ind_j
                        
                        @views E_field[:] -= E_field[:]
                        get_damped_electric_field_dipole!(M_i_ind.μ, -r_ij_vec, E_field, λ_qμ, λ_qμ)
                        charge_dipole_grad_ind_i = (
                            E_field +
                            λ_qμ_deriv * M_i_ind.μ ⋅ r_ij_vec * x_gradient_i  / r_ij^3
                        ) * M_j_ind.q
                        grads[j] -= charge_dipole_grad_ind_i
                        grads[i] += charge_dipole_grad_ind_i

                        # induced dipole - induced dipole #
                        get_damped_electric_field_gradient_dipole!(M_i_ind.μ, r_ij_vec, E_field_gradient, λ_μμ, λ_μμ)
                        dip_dip_grad_ind_j = -(
                            E_field_gradient * M_j_ind.μ -
                            M_i_ind.μ' * ((3 * r_ij_vec * r_ij_vec' / r_ij^2 - diagm(ones(3))) / r_ij^3) * M_j_ind.μ *
                            λ_μμ_deriv * x_gradient_i
                        )
                        grads[j] += dip_dip_grad_ind_j
                        grads[i] -= dip_dip_grad_ind_j
                    end
                end
            end
        end
    end
end

function get_variable_polarizability_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    induced_multipoles::AbstractVector{Multipole1},
    grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64}
)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i_ind = induced_multipoles[i]
                b_i = params[Symbol(labels[i], :_b_exch)]
                k_damp_i = abs(params[Symbol(labels[i], :_k_pol_damp)])
                for j in fragment_indices[j_frag]
                    k_damp_j = abs(params[Symbol(labels[j], :_k_pol_damp)])
                    k_damp_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :k_pol_damp, params)
                    k_damp_ij = abs(k_damp_ij)
                    if !has_pairwise
                        k_damp_ij = 0.5 * (k_damp_i + k_damp_j)
                    end
                    b_j = params[Symbol(labels[j], :_b_exch)]
                    b_ij = sqrt(b_i * b_j)

                    M_j_ind = induced_multipoles[j]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                    slater_overlap_gradient_j = exp(-b_ij * r_ij) * (
                        (2 / 3 * b_ij^2 * r_ij + b_ij) +
                        -b_ij * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                    ) * r_ij_vec / r_ij

                    grads[j] += 0.5 * k_damp_ij * (M_i_ind.μ ⋅ M_i_ind.μ + M_j_ind.μ ⋅ M_j_ind.μ) * slater_overlap_gradient_j
                    grads[i] -= 0.5 * k_damp_ij * (M_i_ind.μ ⋅ M_i_ind.μ + M_j_ind.μ ⋅ M_j_ind.μ) * slater_overlap_gradient_j
                end
            end
        end
    end
end

function get_electrostatic_and_polarization_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole2},
    induced_multipoles::AbstractVector{Multipole1},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_damped::AbstractVector{MVector{3,Float64}},
    E_field_induced::AbstractVector{MVector{3,Float64}},
    E_field_gradients::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradients_damped::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradients_induced::AbstractVector{MMatrix{3, 3, Float64, 9}},
    elec_grads::AbstractVector{MVector{3,Float64}},
    polarization_grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64},
    damping_type::AbstractDamping
)
    Q_damped_field_i = @MVector zeros(3)
    Q_damped_field_gradient_i = @MMatrix zeros(3, 3)
    Q_damped_field_j = @MVector zeros(3)
    Q_damped_field_gradient_j = @MMatrix zeros(3, 3)
    a = get_a(params, damping_type)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                M_i_ind = induced_multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    M_j_ind = induced_multipoles[j]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    u = get_u(r_ij, labels[i], labels[j], params, damping_type)

                    # permanent quadrupole-multipole interaction gradients #
                    grad_Qi_qj = get_quadrupole_charge_interaction_gradient(M_i.Q, M_j.q, r_ij_vec)
                    grad_qi_Qj = get_quadrupole_charge_interaction_gradient(M_j.Q, M_i.q, -r_ij_vec)
                    grad_Qi_μj = get_quadrupole_dipole_interaction_gradient(M_i.Q, M_j.μ, r_ij_vec)
                    grad_μi_Qj = get_quadrupole_dipole_interaction_gradient(M_j.Q, M_i.μ, -r_ij_vec)
                    grad_Qi_Qj = get_quadrupole_quadrupole_interaction_gradient(M_i.Q, M_j.Q, r_ij_vec)

                    elec_grads[i] += grad_Qi_qj
                    elec_grads[i] -= grad_Qi_μj
                    elec_grads[i] += grad_Qi_Qj

                    elec_grads[j] += grad_qi_Qj
                    elec_grads[j] -= grad_μi_Qj
                    elec_grads[j] -= grad_Qi_Qj

                    # NOTE: I get the quadrupole induced multipole
                    # interactions below.
                    # I do not manually accumulate the field gradient gradient
                    # (needed for quadrupole gradients), so here I am getting
                    # quadrupole field gradients and then taking the
                    # third law force pair to get the interaction in the
                    # other direction.
                    
                    # induced multipoles interacting with perm. quad. #
                    @views Q_damped_field_i[:] -= Q_damped_field_i
                    @views Q_damped_field_gradient_i[:, :] -= Q_damped_field_gradient_i
                    @views Q_damped_field_j[:] -= Q_damped_field_j
                    @views Q_damped_field_gradient_j[:, :] -= Q_damped_field_gradient_j
                    get_damped_electric_field_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_j, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type))
                    get_damped_electric_field_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_i, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type))
                    get_damped_electric_field_gradient_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_gradient_j, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type), get_λ9(u, a, damping_type))
                    get_damped_electric_field_gradient_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_gradient_i, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type), get_λ9(u, a, damping_type))

                    polarization_grads[j] += Q_damped_field_i * induced_multipoles[i].q
                    polarization_grads[j] += Q_damped_field_gradient_i * induced_multipoles[i].μ
                    polarization_grads[i] += Q_damped_field_j * induced_multipoles[j].q
                    polarization_grads[i] += Q_damped_field_gradient_j * induced_multipoles[j].μ
                end
            end
        end
    end
    ### NOTE: We only pre-accumulate the field gradients due to
    ###  permanent and induced charges/dipoles. Not quadrupoles.
    ### This is because we currently don't have analytic fifth-order
    ### interaction tensor elements. Hence, the damped quadrupole
    ### field gradients are accumulated in a pairwise fashion in the
    ### loop above, and then we use Newton's third law to get the
    ### induced multipole/quadrupole gradients.
    # accumulate charge and dipole gradients
    for i in eachindex(coords)
        ### charges ###
        # permanent-permanent and permanent-induced
        elec_grads[i] -= multipoles[i].q * E_field[i]
        polarization_grads[i] -= multipoles[i].q * E_field_induced[i]
        # induced-permanent
        polarization_grads[i] -= induced_multipoles[i].q * E_field_damped[i]
        # induced-induced
        #polarization_grads[i] -= induced_multipoles[i].q * E_field_induced[i]
        
        ### dipoles ###
        # permanent-permanent and permanent-induced
        elec_grads[i] -= E_field_gradients[i] * multipoles[i].μ
        polarization_grads[i] -= E_field_gradients_induced[i] * multipoles[i].μ
        # induced-permanent
        polarization_grads[i] -= E_field_gradients_damped[i] * induced_multipoles[i].μ
        # induced-induced
        #polarization_grads[i] -= E_field_gradients_induced[i] * induced_multipoles[i].μ
    end
end

function get_core_shell_electrostatic_and_polarization_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{CSMultipole2},
    induced_multipoles::AbstractVector{Multipole1},
    E_field_core::AbstractVector{MVector{3,Float64}},
    E_field_shell::AbstractVector{MVector{3,Float64}},
    E_field_induced::AbstractVector{MVector{3,Float64}},
    E_field_gradients_core::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradients_shell::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradients_induced::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradient_gradients_core::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
    E_field_gradient_gradients_shell::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
    E_field_gradient_gradients_induced::AbstractVector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}},
    elec_grads::AbstractVector{MVector{3,Float64}},
    polarization_grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64},
    shell_damping_type::AbstractDamping,
    overlap_damping_type::AbstractDamping
)
    Q_damped_field_i = @MVector zeros(3)
    Q_damped_field_gradient_i = @MMatrix zeros(3, 3)
    Q_damped_field_j = @MVector zeros(3)
    Q_damped_field_gradient_j = @MMatrix zeros(3, 3)
    a = get_a(params, shell_damping_type)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                M_i_ind = induced_multipoles[i]
                b_i = params[Symbol(labels[i], :_b_elec)]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    M_j_ind = induced_multipoles[j]
                    b_j = params[Symbol(labels[j], :_b_elec)]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    u_shell_i = get_u(r_ij, b_i, params, shell_damping_type)
                    u_shell_j = get_u(r_ij, b_j, params, shell_damping_type)
                    u_overlap = get_u(r_ij, labels[i], labels[j], params, overlap_damping_type)

                    # permanent quadrupole-multipole interaction gradients #
                    #grad_Qi_Zj = get_quadrupole_charge_interaction_gradient(M_i.Q, M_j.Z, r_ij_vec)
                    #grad_Zi_Qj = get_quadrupole_charge_interaction_gradient(M_j.Q, M_i.Z, -r_ij_vec)
                    #grad_Qi_qj = get_quadrupole_charge_interaction_gradient(M_i.Q, M_j.q_shell, r_ij_vec)
                    #grad_qi_Qj = get_quadrupole_charge_interaction_gradient(M_j.Q, M_i.q_shell, -r_ij_vec)
                    #grad_Qi_μj = get_quadrupole_dipole_interaction_gradient(M_i.Q, M_j.μ, r_ij_vec)
                    #grad_μi_Qj = get_quadrupole_dipole_interaction_gradient(M_j.Q, M_i.μ, -r_ij_vec)
                    #grad_Qi_Qj = get_quadrupole_quadrupole_interaction_gradient(M_i.Q, M_j.Q, r_ij_vec)

                    #elec_grads[i] += grad_Qi_Zj
                    #elec_grads[i] += grad_Qi_qj
                    #elec_grads[i] -= grad_Qi_μj
                    #elec_grads[i] += grad_Qi_Qj

                    #elec_grads[j] += grad_Zi_Qj
                    #elec_grads[j] += grad_qi_Qj
                    #elec_grads[j] -= grad_μi_Qj
                    #elec_grads[j] -= grad_Qi_Qj

                    #@views Q_damped_field_i[:] -= Q_damped_field_i
                    #@views Q_damped_field_gradient_i[:, :] -= Q_damped_field_gradient_i
                    #@views Q_damped_field_j[:] -= Q_damped_field_j
                    #@views Q_damped_field_gradient_j[:, :] -= Q_damped_field_gradient_j
                    #get_damped_electric_field_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_j, get_λ5(u_overlap, a, overlap_damping_type), get_λ7(u_overlap, a, overlap_damping_type))
                    #get_damped_electric_field_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_i, get_λ5(u_overlap, a, overlap_damping_type), get_λ7(u_overlap, a, overlap_damping_type))
                    #get_damped_electric_field_gradient_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_gradient_j, get_λ5(u_overlap, a, overlap_damping_type), get_λ7(u_overlap, a, overlap_damping_type), get_λ9(u_overlap, a, overlap_damping_type))
                    #get_damped_electric_field_gradient_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_gradient_i, get_λ5(u_overlap, a, overlap_damping_type), get_λ7(u_overlap, a, overlap_damping_type), get_λ9(u_overlap, a, overlap_damping_type))

                    #elec_grads[j] += Q_damped_field_i * M_i.q_shell
                    #elec_grads[j] += Q_damped_field_gradient_i * M_i.μ
                    #elec_grads[i] += Q_damped_field_j * M_j.q_shell
                    #elec_grads[i] += Q_damped_field_gradient_j * M_j.μ

                    # NOTE: I get the quadrupole induced multipole
                    # interactions below.
                    # I do not manually accumulate the field gradient gradient
                    # (needed for quadrupole gradients), so here I am getting
                    # quadrupole field gradients and then taking the
                    # third law force pair to get the interaction in the
                    # other direction.
                    
                    # induced multipoles interacting with perm. quad. #
                    #@views Q_damped_field_i[:] -= Q_damped_field_i
                    #@views Q_damped_field_gradient_i[:, :] -= Q_damped_field_gradient_i
                    #@views Q_damped_field_j[:] -= Q_damped_field_j
                    #@views Q_damped_field_gradient_j[:, :] -= Q_damped_field_gradient_j
                    #get_damped_electric_field_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_j, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type))
                    #get_damped_electric_field_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_i, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type))
                    #get_damped_electric_field_gradient_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_gradient_j, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type), get_λ9(u, a, damping_type))
                    #get_damped_electric_field_gradient_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_gradient_i, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type), get_λ9(u, a, damping_type))

                    #polarization_grads[j] += Q_damped_field_i * induced_multipoles[i].q
                    #polarization_grads[j] += Q_damped_field_gradient_i * induced_multipoles[i].μ
                    #polarization_grads[i] += Q_damped_field_j * induced_multipoles[j].q
                    #polarization_grads[i] += Q_damped_field_gradient_j * induced_multipoles[j].μ
                end
            end
        end
    end
    ### NOTE: We only pre-accumulate the field gradients due to
    ###  permanent and induced charges/dipoles. Not quadrupoles.
    ### This is because we currently don't have analytic fifth-order
    ### interaction tensor elements. Hence, the damped quadrupole
    ### field gradients are accumulated in a pairwise fashion in the
    ### loop above, and then we use Newton's third law to get the
    ### induced multipole/quadrupole gradients.
    # accumulate charge and dipole gradients
    for i in eachindex(coords)
        ### charges ###
        # permanent-permanent and permanent-induced
        elec_grads[i] -= multipoles[i].Z * E_field_core[i]
        elec_grads[i] -= multipoles[i].q_shell * E_field_shell[i]
        #polarization_grads[i] -= multipoles[i].q * E_field_induced[i]
        # induced-induced
        #polarization_grads[i] -= induced_multipoles[i].q * E_field_induced[i]
        
        ### dipoles ###
        # permanent-permanent and permanent-induced
        elec_grads[i] -= E_field_gradients_shell[i] * multipoles[i].μ
        #polarization_grads[i] -= E_field_gradients_induced[i] * multipoles[i].μ
        # induced-induced
        #polarization_grads[i] -= E_field_gradients_induced[i] * induced_multipoles[i].μ

        ### quadrupoles ###
        #@views elec_grads[i][1] -= E_field_gradient_gradients_shell[i][:, :, 1] ⋅ multipoles[i].Q  #/ 3.0
        #@views elec_grads[i][2] -= E_field_gradient_gradients_shell[i][:, :, 2] ⋅ multipoles[i].Q  #/ 3.0
        #@views elec_grads[i][3] -= E_field_gradient_gradients_shell[i][:, :, 3] ⋅ multipoles[i].Q  #/ 3.0
        #for w in 1:3
        #    for m in 1:3
        #        for n in 1:3
        #            elec_grads[i][w] -= multipoles[i].Q[n, m] * E_field_gradient_gradients_shell[i][m, n, w]
        #        end
        #    end
        #end
    end
end

function get_damped_and_undamped_electrostatic_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole2},
    induced_multipoles::AbstractVector{Multipole1},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_damped::AbstractVector{MVector{3,Float64}},
    E_field_induced::AbstractVector{MVector{3,Float64}},
    E_field_gradients::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradients_damped::AbstractVector{MMatrix{3, 3, Float64, 9}},
    E_field_gradients_induced::AbstractVector{MMatrix{3, 3, Float64, 9}},
    grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64},
    damping_type::AbstractDamping
)
    Q_damped_field_i = @MVector zeros(3)
    Q_damped_field_gradient_i = @MMatrix zeros(3, 3)
    Q_damped_field_j = @MVector zeros(3)
    Q_damped_field_gradient_j = @MMatrix zeros(3, 3)
    a = get_a(params, damping_type)
    # accumulate gradients involving dipoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                M_i_ind = induced_multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]
                    M_j_ind = induced_multipoles[j]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    u = get_u(r_ij, labels[i], labels[j], params, damping_type)

                    # permanent quadrupole-multipole interaction gradients
                    grad_Qi_qj = get_quadrupole_charge_interaction_gradient(M_i.Q, M_j.q, r_ij_vec)
                    grad_qi_Qj = get_quadrupole_charge_interaction_gradient(M_j.Q, M_i.q, -r_ij_vec)
                    grad_Qi_μj = get_quadrupole_dipole_interaction_gradient(M_i.Q, M_j.μ, r_ij_vec)
                    grad_μi_Qj = get_quadrupole_dipole_interaction_gradient(M_j.Q, M_i.μ, -r_ij_vec)
                    grad_Qi_Qj = get_quadrupole_quadrupole_interaction_gradient(M_i.Q, M_j.Q, r_ij_vec)

                    grads[i] += grad_Qi_qj
                    grads[i] -= grad_Qi_μj
                    grads[i] += grad_Qi_Qj

                    grads[j] += grad_qi_Qj
                    grads[j] -= grad_μi_Qj
                    grads[j] -= grad_Qi_Qj

                    # induced multipoles interacting with perm. quad.

                    # NOTE: I get the quadrupole induced multipole
                    # interactions in the loop below.
                    # I do not manually accumulate the field gradient gradient
                    # (needed for quadrupole gradients), so here I am getting
                    # quadrupole field gradients and then taking the
                    # third law force pair to get the interaction in the
                    # other direction.
                    @views Q_damped_field_i[:] -= Q_damped_field_i
                    @views Q_damped_field_gradient_i[:, :] -= Q_damped_field_gradient_i
                    @views Q_damped_field_j[:] -= Q_damped_field_j
                    @views Q_damped_field_gradient_j[:, :] -= Q_damped_field_gradient_j
                    get_damped_electric_field_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_j, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type))
                    get_damped_electric_field_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_i, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type))
                    get_damped_electric_field_gradient_quadrupole!(multipoles[i].Q,  r_ij_vec, Q_damped_field_gradient_j, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type), get_λ9(u, a, damping_type))
                    get_damped_electric_field_gradient_quadrupole!(multipoles[j].Q, -r_ij_vec, Q_damped_field_gradient_i, get_λ5(u, a, damping_type), get_λ7(u, a, damping_type), get_λ9(u, a, damping_type))

                    grads[j] += Q_damped_field_i * induced_multipoles[i].q
                    grads[j] += Q_damped_field_gradient_i * induced_multipoles[i].μ
                    grads[i] += Q_damped_field_j * induced_multipoles[j].q
                    grads[i] += Q_damped_field_gradient_j * induced_multipoles[j].μ
                end
            end
        end
    end


    ### NOTE: We only pre-accumulate the field gradients due to
    ###  permanent and induced charges/dipoles. Not quadrupoles.
    ### This is because we currently don't have analytic fifth-order
    ### interaction tensor elements. Hence, the damped quadrupole
    ### field gradients are accumulated in a pairwise fashion in the
    ### loop above, and then we use Newton's third law to get the
    ### induced multipole/quadrupole gradients.
    # accumulate charge and dipole gradients
    for i in eachindex(coords)
        ### charges ###
        # permanent-permanent and permanent-induced
        grads[i] -= multipoles[i].q * (E_field[i] + E_field_induced[i])
        # induced-permanent and induced-induced
        grads[i] -= induced_multipoles[i].q * (E_field_damped[i] + E_field_induced[i])
        
        ### dipoles ###
        # permanent-permanent and permanent-induced
        grads[i] -= (E_field_gradients[i] + E_field_gradients_induced[i]) * multipoles[i].μ
        # induced-permanent and induced-induced
        grads[i] -= (E_field_gradients_damped[i] + E_field_gradients_induced[i]) * induced_multipoles[i].μ
    end
end

"""
Get the electrostatic gradients based on pre-accumulated electric fields
and field gradients.
"""
function get_electrostatic_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    fragment_indices::AbstractVector{Vector{Int}},
    multipoles::AbstractVector{Multipole2},
    E_field::AbstractVector{MVector{3,Float64}},
    E_field_gradients::AbstractVector{MMatrix{3, 3, Float64, 9}},
    grads::AbstractVector{MVector{3,Float64}}
)
    # accumulate gradients involving quadrupoles
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = multipoles[i]
                for j in fragment_indices[j_frag]
                    M_j = multipoles[j]

                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)

                    grad_Qi_qj = get_quadrupole_charge_interaction_gradient(M_i.Q, M_j.q, r_ij_vec)
                    grad_qi_Qj = get_quadrupole_charge_interaction_gradient(M_j.Q, M_i.q, -r_ij_vec)
                    grad_Qi_μj = get_quadrupole_dipole_interaction_gradient(M_i.Q, M_j.μ, r_ij_vec)
                    grad_μi_Qj = get_quadrupole_dipole_interaction_gradient(M_j.Q, M_i.μ, -r_ij_vec)
                    grad_Qi_Qj = get_quadrupole_quadrupole_interaction_gradient(M_i.Q, M_j.Q, r_ij_vec)

                    grads[i] += grad_Qi_qj
                    grads[i] -= grad_Qi_μj
                    grads[i] += grad_Qi_Qj

                    grads[j] += grad_qi_Qj
                    grads[j] -= grad_μi_Qj
                    grads[j] -= grad_Qi_Qj # Note we only get Qi-Qj so this is Newton's third law.
                end
            end
        end
    end

    # accumulate charge and dipole gradients.
    for i in eachindex(coords)
        grads[i] -= multipoles[i].q * E_field[i]
        grads[i] += E_field_gradients[i] * multipoles[i].μ
    end
end

function torque_on_quadrupole(Q::MMatrix{3, 3, Float64, 9}, E_field_grad::MMatrix{3, 3, Float64, 9})
    τ = @MVector zeros(3)
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    τ[i] += ϵ_ijk(i,j,k) * Q[j, l] * E_field_grad[k, l]
                end
            end
        end
    end
    # Honestly, I don't know why this is a 2/3.
    # The reference I have looked at (Rev. Mex. F´ıs. 52 (6) (2006) 501–506)
    # has a 1/3. I don't know...
    return (2 / 3) * τ
end

function torque_on_quadrupole!(τ::MVector{3, Float64}, Q::MMatrix{3, 3, Float64, 9}, E_field_grad::MMatrix{3, 3, Float64, 9})
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    τ[i] += (2 / 3) * ϵ_ijk(i,j,k) * Q[j, l] * E_field_grad[k, l]
                end
            end
        end
    end
    return τ
end

"""
Computes the chain rule contribution to the gradients when charges
are allowed to vary with geometry. i.e. the gradient of q times the potential.

TODO: Cleanup! This assumes water molecules only and is an indexing nightmare.
"""
function add_variable_charge_gradients!(grads::AbstractVector{MVector{3,Float64}}, q_derivative::Vector{MArray{Tuple{3,3,3},Float64,3,27}}, ϕ::AbstractVector{Float64}, fragment_indices::Vector{Vector{Int}})
    i_O  = 1
    i_H1 = 2
    i_H2 = 3
    for i_frag in eachindex(fragment_indices)
        # CURRENTLY ASSUME WATER!!
        if length(fragment_indices[i_frag]) == 3
            @views grads[3*(i_frag-1)+1] .+= (
                q_derivative[i_frag][:, i_O,  i_O] * ϕ[3*(i_frag-1)+1] +    # phi(O)
                q_derivative[i_frag][:, i_H1, i_O] * ϕ[3*(i_frag-1)+2] +    # phi(h1)
                q_derivative[i_frag][:, i_H2, i_O] * ϕ[3*(i_frag-1)+3]      # phi(h2)
            )

            @views grads[3*(i_frag-1)+2] .+= (
                q_derivative[i_frag][:, i_O,  i_H1] * ϕ[3*(i_frag-1)+1] +    # phi(O)
                q_derivative[i_frag][:, i_H1, i_H1] * ϕ[3*(i_frag-1)+2] +    # phi(h1)
                q_derivative[i_frag][:, i_H2, i_H1] * ϕ[3*(i_frag-1)+3]      # phi(h2)
            )

            @views grads[3*(i_frag-1)+3] .+= (
                q_derivative[i_frag][:, i_O,  i_H2] * ϕ[3*(i_frag-1)+1] +    # phi(O)
                q_derivative[i_frag][:, i_H1, i_H2] * ϕ[3*(i_frag-1)+2] +    # phi(h1)
                q_derivative[i_frag][:, i_H2, i_H2] * ϕ[3*(i_frag-1)+3]      # phi(h2)
            )
        end
    end
end