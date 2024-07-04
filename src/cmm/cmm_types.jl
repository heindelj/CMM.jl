####################
### CMM STORAGE  ###
####################

struct CMMStorage <: AbstractForceFieldStorage
    num_fragments::Int
    shell_damping_type::SlaterShellDamping
    overlap_damping_type::SlaterOverlapDamping
    α_inv::Vector{MMatrix{3, 3, Float64, 9}}
    local_axes::Vector{LocalAxes}
    multipoles::Vector{CSMultipole2}
    induced_multipoles::Vector{CSMultipole1}
    ϕ_core::Vector{Float64}
    ϕ_shell::Vector{Float64}
    ϕ_induced::Vector{Float64}
    ϕ_exch_pol::Vector{Float64}
    ϕ_dispersion::Vector{Float64}
    ϕ_repulsion::Vector{Float64}
    ϕ_donor::Vector{Float64}
    ϕ_acceptor::Vector{Float64}
    E_field_core::Vector{MVector{3, Float64}}
    E_field_shell::Vector{MVector{3, Float64}}
    E_field_induced::Vector{MVector{3, Float64}}
    E_field_exch_pol::Vector{MVector{3, Float64}}
    E_field_dispersion::Vector{MVector{3, Float64}}
    E_field_repulsion::Vector{MVector{3, Float64}}
    E_field_donor::Vector{MVector{3, Float64}}
    E_field_acceptor::Vector{MVector{3, Float64}}
    E_field_gradients_core::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_shell::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_induced::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_exch_pol::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_dispersion::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_repulsion::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_donor::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradients_acceptor::Vector{MMatrix{3, 3, Float64, 9}}
    E_field_gradient_gradients_core::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_shell::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_induced::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_exch_pol::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_dispersion::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_repulsion::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_donor::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    E_field_gradient_gradients_acceptor::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}
    Δq_ct::Vector{Float64}
    b_vec::Vector{Float64}
    polarization_matrix::Matrix{Float64}
    solution_vector::Vector{Float64}
    applied_field::MVector{3, Float64}
    η_fq::Vector{Float64}
    include_exch_pol::Bool
    include_charge_transfer::Bool
    one_body_charge_grads::Union{Nothing, Vector{MArray{Tuple{3,3,3},Float64,3,27}}}
    torques::Union{Nothing, Vector{MVector{3, Float64}}}
    dα::Union{Nothing, SVector{3, MMatrix{3, 3, Float64, 9}}}
    deformation_grads::Union{Nothing, Vector{MVector{3, Float64}}}
    pauli_grads::Union{Nothing, Vector{MVector{3, Float64}}}
    dispersion_grads::Union{Nothing, Vector{MVector{3, Float64}}}
    electrostatic_grads::Union{Nothing, Vector{MVector{3, Float64}}}
    polarization_grads::Union{Nothing, Vector{MVector{3, Float64}}}
    charge_transfer_grads::Union{Nothing, Vector{MVector{3, Float64}}}
end

mutable struct CMM_FF <: AbstractForceField
    terms::Vector{Function}
    params::Dict{Symbol, Float64}
    storage::CMMStorage
    results::ForceFieldResults
end