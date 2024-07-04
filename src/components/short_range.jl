
function get_explicit_charge_transfer!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    Δq_ct::AbstractVector{Float64}
)
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        q_ij_forward = params[Symbol(labels[i], :_q_ct_donor)] * params[Symbol(labels[j], :_q_ct_acceptor)]
                        q_ij_backward = params[Symbol(labels[i], :_q_ct_acceptor)] * params[Symbol(labels[j], :_q_ct_donor)]
                        γ_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_γ_ct_slater, params)
                        γ_ij = abs(γ_ij)
                        if !has_pairwise
                            #@warn "Don't have pairwise CT energy to charge transferred parameter for this pair. Setting parameter to 1e15 to turn off CT for this pair."
                            γ_ij = 1e15
                        end
                        b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_ct, params)
                        b_ij = abs(b_ij)
                        if !has_pairwise
                            b_ij = sqrt(abs(params[Symbol(labels[i], :_b_ct)]) * abs(params[Symbol(labels[j], :_b_ct)]))
                        end

                        r_ij = norm(coords[i] - coords[j])
                        λ1_overlap = exp(-b_ij * r_ij) * ( 1.0 +
                            (11.0 / 16.0) * (b_ij * r_ij) +
                            (3.0 / 16.0)  * (b_ij * r_ij)^2 +
                            (1.0 / 48.0)  * (b_ij * r_ij)^3
                        )

                        # forward CT
                        Δq_ct_forward = q_ij_forward * λ1_overlap / r_ij / γ_ij
                        Δq_ct[i] += Δq_ct_forward
                        Δq_ct[j] -= Δq_ct_forward
                         
                        # backward CT
                        Δq_ct_backward = q_ij_backward * λ1_overlap / r_ij / γ_ij
                        Δq_ct[i] -= Δq_ct_backward
                        Δq_ct[j] += Δq_ct_backward
                    end
                end
            end
        end
    end
end

function exchange_polarizability_coupling!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    
)

end

function multipolar_dispersion!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    electric_multipoles::Vector{CSMultipole2},
    dispersion_charges::Vector{Float64},
    ϕ_dispersion::AbstractVector{Float64},
    E_field_dispersion::AbstractVector{MVector{3,Float64}},
    E_field_gradients_dispersion::AbstractVector{MMatrix{3,3,Float64,9}},
    dispersion_damping_type::AbstractDamping
)
    a = 1.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = electric_multipoles[i]
                q_i = dispersion_charges[i]
                b_i = abs(params[Symbol(labels[i], :_b_disp)])
                #K_i_μ = params[Symbol(labels[i], :_K_dispersion_μ)]
                #K_i_Q = params[Symbol(labels[i], :_K_dispersion_Q)]
                for j in fragment_indices[j_frag]
                    # NOTE(JOE): Currently there are no dipole or quadrupolar contributions to
                    # dispersion so those are just commented out.
                    M_j = electric_multipoles[j]
                    q_j = dispersion_charges[j]
                    b_j = abs(params[Symbol(labels[j], :_b_disp)])
                    #K_j_μ = params[Symbol(labels[j], :_K_dispersion_μ)]
                    #K_j_Q = params[Symbol(labels[j], :_K_dispersion_Q)]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_disp, params)
                    b_ij = abs(b_ij)
                    if !has_pairwise
                        b_ij = sqrt(b_i * b_j)
                    end
                    u_overlap = b_ij * r_ij

                    #λ1 = get_λ1(u_overlap, a, dispersion_damping_type)
                    #λ3 = get_λ3(u_overlap, a, dispersion_damping_type)
                    #λ5 = get_λ5(u_overlap, a, dispersion_damping_type)
                    λ7 = get_λ7(u_overlap, a, dispersion_damping_type)
                    λ9 = get_λ9(u_overlap, a, dispersion_damping_type)

                    ### Dispersion Potential ###
                    ϕ_dispersion[j] += get_damped_dispersion_potential_charge(q_i, r_ij, λ7)
                    ϕ_dispersion[i] += get_damped_dispersion_potential_charge(q_j, r_ij, λ7)
                    #ϕ_dispersion[j] += get_damped_dispersion_potential_dipole(K_i_μ * M_i.μ, r_ij_vec, λ3)
                    #ϕ_dispersion[i] += get_damped_dispersion_potential_dipole(K_j_μ * M_j.μ, -r_ij_vec, λ3)
                    #ϕ_dispersion[j] += get_damped_dispersion_potential_quadrupole(K_i_Q * M_i.Q, r_ij_vec, λ3, λ5)
                    #ϕ_dispersion[i] += get_damped_dispersion_potential_quadrupole(K_j_Q * M_j.Q, -r_ij_vec, λ3, λ5)

                    ### Dispersion Field ###
                    get_damped_dispersion_field_charge!(q_i,  r_ij_vec, E_field_dispersion[j], λ9)
                    get_damped_dispersion_field_charge!(q_j, -r_ij_vec, E_field_dispersion[i], λ9)
                    #get_damped_dispersion_field_dipole!(K_i_μ * M_i.μ,  r_ij_vec, E_field_dispersion[j], λ3, λ5)
                    #get_damped_dispersion_field_dipole!(K_j_μ * M_j.μ, -r_ij_vec, E_field_dispersion[i], λ3, λ5)
                    #get_damped_dispersion_field_quadrupole!(K_i_Q * M_i.Q, r_ij_vec, E_field_dispersion[j], λ5, λ7)
                    #get_damped_dispersion_field_quadrupole!(K_j_Q * M_j.Q, -r_ij_vec, E_field_dispersion[i], λ5, λ7)

                    ### Dispersion Field Gradients ###
                    #get_damped_dispersion_field_gradient_charge!(q_i,  r_ij_vec, E_field_gradients_dispersion[j], λ3, λ5)
                    #get_damped_dispersion_field_gradient_charge!(q_j, -r_ij_vec, E_field_gradients_dispersion[i], λ3, λ5)
                    #get_damped_dispersion_field_gradient_dipole!(K_i_μ * M_i.μ,  r_ij_vec, E_field_gradients_dispersion[j], λ5, λ7)
                    #get_damped_dispersion_field_gradient_dipole!(K_j_μ * M_j.μ, -r_ij_vec, E_field_gradients_dispersion[i], λ5, λ7)
                    #get_damped_dispersion_field_gradient_quadrupole!(K_i_Q * M_i.Q, r_ij_vec, E_field_gradients_dispersion[j], λ5, λ7, λ9)
                    #get_damped_dispersion_field_gradient_quadrupole!(K_j_Q * M_j.Q, -r_ij_vec, E_field_gradients_dispersion[i], λ5, λ7, λ9)
                end
            end
        end
    end
end

function multipolar_pauli_repulsion!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    electric_multipoles::Vector{CSMultipole2},
    repulsion_charges::Vector{Float64},
    ϕ_repulsion::AbstractVector{Float64},
    E_field_repulsion::AbstractVector{MVector{3,Float64}},
    E_field_gradients_repulsion::AbstractVector{MMatrix{3,3,Float64,9}},
    repulsion_damping_type::AbstractDamping
)
    a = 1.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = electric_multipoles[i]
                q_rep_i = repulsion_charges[i]
                b_i_rep = params[Symbol(labels[i], :_b_repulsion)]
                K_i_μ = params[Symbol(labels[i], :_K_repulsion_μ)]
                K_i_Q = params[Symbol(labels[i], :_K_repulsion_Q)]
                for j in fragment_indices[j_frag]
                    M_j = electric_multipoles[j]
                    q_rep_j = repulsion_charges[j]
                    b_j_rep = params[Symbol(labels[j], :_b_repulsion)]
                    
                    K_j_μ = params[Symbol(labels[j], :_K_repulsion_μ)]
                    K_j_Q = params[Symbol(labels[j], :_K_repulsion_Q)]
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    b_ij_rep, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_repulsion, params)
                    b_ij_rep = abs(b_ij_rep)
                    if !has_pairwise
                        b_ij_rep = sqrt(b_i_rep * b_j_rep)
                    end
                    u_overlap = b_ij_rep * r_ij

                    ### Repulsion Potential ###
                    ϕ_repulsion[j] += get_repulsion_potential_charge(q_rep_i, r_ij, get_λ1(u_overlap, a, repulsion_damping_type))
                    ϕ_repulsion[i] += get_repulsion_potential_charge(q_rep_j, r_ij, get_λ1(u_overlap, a, repulsion_damping_type))
                    ϕ_repulsion[j] += get_repulsion_potential_dipole(K_i_μ * M_i.μ, r_ij_vec, get_λ3(u_overlap, a, repulsion_damping_type))
                    ϕ_repulsion[i] += get_repulsion_potential_dipole(K_j_μ * M_j.μ, -r_ij_vec, get_λ3(u_overlap, a, repulsion_damping_type))
                    ϕ_repulsion[j] += get_repulsion_potential_quadrupole(K_i_Q * M_i.Q, r_ij_vec, get_λ3(u_overlap, a, repulsion_damping_type), get_λ5(u_overlap, a, repulsion_damping_type))
                    ϕ_repulsion[i] += get_repulsion_potential_quadrupole(K_j_Q * M_j.Q, -r_ij_vec, get_λ3(u_overlap, a, repulsion_damping_type), get_λ5(u_overlap, a, repulsion_damping_type))

                    ### Repulsion Field ###
                    get_repulsion_field_charge!(q_rep_i,  r_ij_vec, E_field_repulsion[j], get_λ3(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_charge!(q_rep_j, -r_ij_vec, E_field_repulsion[i], get_λ3(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_dipole!(K_i_μ * M_i.μ,  r_ij_vec, E_field_repulsion[j], get_λ3(u_overlap, a, repulsion_damping_type), get_λ5(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_dipole!(K_j_μ * M_j.μ, -r_ij_vec, E_field_repulsion[i], get_λ3(u_overlap, a, repulsion_damping_type), get_λ5(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_quadrupole!(K_i_Q * M_i.Q, r_ij_vec, E_field_repulsion[j], get_λ5(u_overlap, a, repulsion_damping_type), get_λ7(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_quadrupole!(K_j_Q * M_j.Q, -r_ij_vec, E_field_repulsion[i], get_λ5(u_overlap, a, repulsion_damping_type), get_λ7(u_overlap, a, repulsion_damping_type))

                    ### Repulsion Field Gradients ###
                    get_repulsion_field_gradient_charge!(q_rep_i,  r_ij_vec, E_field_gradients_repulsion[j], get_λ3(u_overlap, a, repulsion_damping_type), get_λ5(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_gradient_charge!(q_rep_j, -r_ij_vec, E_field_gradients_repulsion[i], get_λ3(u_overlap, a, repulsion_damping_type), get_λ5(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_gradient_dipole!(K_i_μ * M_i.μ,  r_ij_vec, E_field_gradients_repulsion[j], get_λ5(u_overlap, a, repulsion_damping_type), get_λ7(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_gradient_dipole!(K_j_μ * M_j.μ, -r_ij_vec, E_field_gradients_repulsion[i], get_λ5(u_overlap, a, repulsion_damping_type), get_λ7(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_gradient_quadrupole!(K_i_Q * M_i.Q, r_ij_vec, E_field_gradients_repulsion[j], get_λ5(u_overlap, a, repulsion_damping_type), get_λ7(u_overlap, a, repulsion_damping_type), get_λ9(u_overlap, a, repulsion_damping_type))
                    get_repulsion_field_gradient_quadrupole!(K_j_Q * M_j.Q, -r_ij_vec, E_field_gradients_repulsion[i], get_λ5(u_overlap, a, repulsion_damping_type), get_λ7(u_overlap, a, repulsion_damping_type), get_λ9(u_overlap, a, repulsion_damping_type))
                end
            end
        end
    end
end

function multipolar_exchange_polarization!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    exch_pol_charges::AbstractVector{Float64},
    ϕ_exch_pol::AbstractVector{Float64},
    E_field_exch_pol::AbstractVector{MVector{3,Float64}},
    repulsion_damping_type::AbstractDamping
)
    a = 1.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                q_i = exch_pol_charges[i]
                b_i = abs(params[Symbol(labels[i], :_b_exch_pol)])
                #K_i_μ = params[Symbol(labels[i], :_K_exch_pol_μ)]
                #K_i_Q = params[Symbol(labels[i], :_K_exch_pol_Q)]

                #μ_exch_pol_i = K_i_μ * M_i.μ
                #if is_ion(labels[i])
                #    μ_exch_pol_i = K_i_μ * normalize(E_field[i])
                #end
                for j in fragment_indices[j_frag]
                    q_j = exch_pol_charges[j]
                    b_j = abs(params[Symbol(labels[j], :_b_exch_pol)])
                    #K_j_μ = params[Symbol(labels[j], :_K_exch_pol_μ)]
                    #K_j_Q = params[Symbol(labels[j], :_K_exch_pol_Q)]

                    # Very possibly this dipole should always point in the field direction?
                    # Even for water. That direction should probably be the polarization
                    # direction?
                    #μ_exch_pol_j = K_j_μ * M_j.μ
                    #if is_ion(labels[j])
                    #    μ_exch_pol_j = K_j_μ * normalize(E_field[j])
                    #end
                    b_ij = sqrt(b_i * b_j)

                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)

                    br = b_ij * r_ij
                    u_overlap = br
                    λ1 = get_λ1(u_overlap, a, repulsion_damping_type)
                    λ3 = get_λ3(u_overlap, a, repulsion_damping_type)
                    λ5 = get_λ5(u_overlap, a, repulsion_damping_type)
                    #λ7 = get_λ7(u_overlap, a, repulsion_damping_type)

                    ### Exchange Polarization Potential ###
                    ϕ_exch_pol[j] += get_ct_potential_charge(q_i, r_ij, λ1)
                    ϕ_exch_pol[i] += get_ct_potential_charge(q_j, r_ij, λ1)
                    #ϕ_exch_pol[j] += get_ct_potential_dipole(K_i_μ * M_i.μ, r_ij_vec, λ3)
                    #ϕ_exch_pol[i] += get_ct_potential_dipole(K_j_μ * M_j.μ, -r_ij_vec, λ3)
                    #ϕ_exch_pol[j] += get_ct_potential_quadrupole(K_i_Q * M_i.Q, r_ij_vec, λ3, λ5)
                    #ϕ_exch_pol[i] += get_ct_potential_quadrupole(K_j_Q * M_j.Q, -r_ij_vec, λ3, λ5)

                    ### Exchange Polarization Field ###
                    get_ct_field_charge!(q_i,  r_ij_vec, E_field_exch_pol[j], λ3)
                    get_ct_field_charge!(q_j, -r_ij_vec, E_field_exch_pol[i], λ3)
                    #get_ct_field_dipole!(K_i_μ * M_i.μ,  r_ij_vec, E_field_exch_pol[j], λ3, λ5)
                    #get_ct_field_dipole!(K_j_μ * M_j.μ, -r_ij_vec, E_field_exch_pol[i], λ3, λ5)
                    #get_ct_field_quadrupole!(K_i_Q * M_i.Q, r_ij_vec, E_field_exch_pol[j], λ5, λ7)
                    #get_ct_field_quadrupole!(K_j_Q * M_j.Q, -r_ij_vec, E_field_exch_pol[i], λ5, λ7)
                end
            end
        end
    end
end

function multipolar_charge_transfer!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    electric_multipoles::Vector{CSMultipole2},
    E_field::AbstractVector{MVector{3, Float64}},
    ct_donor_charges::Vector{Float64},
    ct_acceptor_charges::Vector{Float64},
    ϕ_ct_donor::AbstractVector{Float64},
    ϕ_ct_acceptor::AbstractVector{Float64},
    E_field_ct_donor::AbstractVector{MVector{3,Float64}},
    E_field_ct_acceptor::AbstractVector{MVector{3,Float64}},
    E_field_gradients_ct_donor::AbstractVector{MMatrix{3,3,Float64,9}},
    E_field_gradients_ct_acceptor::AbstractVector{MMatrix{3,3,Float64,9}},
    ct_damping_type::AbstractDamping
)

    # The multipoles and arrays here are named acceptor and donor to
    # emphasize that there is usually a preferred direction for charge
    # transfer. With that being said, these are still multipoles so we
    # need to get the potential due to both charges. So, we accumulate
    # the potential for both donor and acceptor charges on both the
    # donor and acceptor atoms in each case. The reason we MUST do this
    # is Newton's third law. Anyways, if I just named
    # the parameters multipoles_1 and multipoles_2, the code would be less
    # confusing because that name is more illustrative as to what the
    # code actually does. However, a name like that obfuscates the
    # concept that the code is modelling. I wrote this comment cause
    # this is the first time I've understood why naming can be hard:
    # there are situations where choosing a name forces you to decide
    # between explaining the intent of the computation and the intent
    # of the program.
    # -Joe, 5/16/24
    a = 1.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                M_i = electric_multipoles[i]
                q_ct_donor_i = ct_donor_charges[i]
                q_ct_acceptor_i = ct_acceptor_charges[i]
                b_i_ct = abs(params[Symbol(labels[i], :_b_ct)])
                K_i_donor_μ = params[Symbol(labels[i], :_K_ct_donor_μ)]
                K_i_donor_Q = params[Symbol(labels[i], :_K_ct_donor_Q)]
                μ_donor_ct_i = K_i_donor_μ * M_i.μ
                Q_donor_ct_i = K_i_donor_Q * M_i.Q
                for j in fragment_indices[j_frag]
                    M_j = electric_multipoles[j]
                    q_ct_donor_j = ct_donor_charges[j]
                    q_ct_acceptor_j = ct_acceptor_charges[j]
                    b_j_ct = abs(params[Symbol(labels[j], :_b_ct)])
                    K_j_donor_μ = params[Symbol(labels[j], :_K_ct_donor_μ)]
                    K_j_donor_Q = params[Symbol(labels[j], :_K_ct_donor_Q)]
                    μ_donor_ct_j = K_j_donor_μ * M_j.μ
                    Q_donor_ct_j = K_j_donor_Q * M_j.Q
                    
                    b_ij_ct, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_ct, params)
                    b_ij_ct = abs(b_ij_ct)
                    if !has_pairwise
                        b_ij_ct = sqrt(b_i_ct * b_j_ct)
                    end
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    u_overlap = b_ij_ct * r_ij

                    λ1 = get_λ1(u_overlap, a, ct_damping_type)
                    λ3 = get_λ3(u_overlap, a, ct_damping_type)
                    λ5 = get_λ5(u_overlap, a, ct_damping_type)
                    λ7 = get_λ7(u_overlap, a, ct_damping_type)
                    λ9 = get_λ9(u_overlap, a, ct_damping_type)

                    # NOTE(JOE): I have chosen to only let the acceptor multipoles
                    # be represented by a charge, but this was a model decision.
                    # In the most general case, the acceptor can also have anisotropy
                    # which is captured by the higher-order multipoles. In the future,
                    # this would conceivably be a place to look for additional
                    # flexibility in the model.

                    ### CT Potential ###
                    # donor to acceptor #
                    ϕ_ct_acceptor[j] += get_ct_potential_charge(q_ct_donor_i, r_ij, λ1)
                    ϕ_ct_acceptor[i] += get_ct_potential_charge(q_ct_donor_j, r_ij, λ1)
                    ϕ_ct_acceptor[j] += get_ct_potential_dipole(μ_donor_ct_i, r_ij_vec, λ3)
                    ϕ_ct_acceptor[i] += get_ct_potential_dipole(μ_donor_ct_j, -r_ij_vec, λ3)
                    ϕ_ct_acceptor[j] += get_ct_potential_quadrupole(Q_donor_ct_i, r_ij_vec, λ3, λ5)
                    ϕ_ct_acceptor[i] += get_ct_potential_quadrupole(Q_donor_ct_j, -r_ij_vec, λ3, λ5)

                    # acceptor to donor #
                    ϕ_ct_donor[j] += get_ct_potential_charge(q_ct_acceptor_i, r_ij, λ1)
                    ϕ_ct_donor[i] += get_ct_potential_charge(q_ct_acceptor_j, r_ij, λ1)

                    ### CT Field ###
                    # donor to acceptor #
                    get_ct_field_charge!(q_ct_donor_i,  r_ij_vec, E_field_ct_acceptor[j], λ3)
                    get_ct_field_charge!(q_ct_donor_j, -r_ij_vec, E_field_ct_acceptor[i], λ3)
                    get_ct_field_dipole!(μ_donor_ct_i,  r_ij_vec, E_field_ct_acceptor[j], λ3, λ5)
                    get_ct_field_dipole!(μ_donor_ct_j, -r_ij_vec, E_field_ct_acceptor[i], λ3, λ5)
                    get_ct_field_quadrupole!(Q_donor_ct_i, r_ij_vec, E_field_ct_acceptor[j], λ5, λ7)
                    get_ct_field_quadrupole!(Q_donor_ct_j, -r_ij_vec, E_field_ct_acceptor[i], λ5, λ7)

                    # acceptor to donor #
                    get_ct_field_charge!(q_ct_acceptor_i,  r_ij_vec, E_field_ct_donor[j], λ3)
                    get_ct_field_charge!(q_ct_acceptor_j, -r_ij_vec, E_field_ct_donor[i], λ3)

                    ### CT Field Gradients ###
                    # donor to acceptor #
                    get_ct_field_gradient_charge!(q_ct_donor_i,  r_ij_vec, E_field_gradients_ct_acceptor[j], λ3, λ5)
                    get_ct_field_gradient_charge!(q_ct_donor_j, -r_ij_vec, E_field_gradients_ct_acceptor[i], λ3, λ5)
                    get_ct_field_gradient_dipole!(μ_donor_ct_i,  r_ij_vec, E_field_gradients_ct_acceptor[j], λ5, λ7)
                    get_ct_field_gradient_dipole!(μ_donor_ct_j, -r_ij_vec, E_field_gradients_ct_acceptor[i], λ5, λ7)
                    get_ct_field_gradient_quadrupole!(Q_donor_ct_i, r_ij_vec, E_field_gradients_ct_acceptor[j], λ5, λ7, λ9)
                    get_ct_field_gradient_quadrupole!(Q_donor_ct_j, -r_ij_vec, E_field_gradients_ct_acceptor[i], λ5, λ7, λ9)

                    # acceptor to donor #
                    get_ct_field_gradient_charge!(q_ct_acceptor_i,  r_ij_vec, E_field_gradients_ct_donor[j], λ3, λ5)
                    get_ct_field_gradient_charge!(q_ct_acceptor_j, -r_ij_vec, E_field_gradients_ct_donor[i], λ3, λ5)
                end
            end
        end
    end
end

function overlap_pauli!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64}
)
    pauli_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        a_ij, has_pairwise_1 = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_exch, params)
                        a_ij = abs(a_ij)
                        a_ij_sq, has_pairwise_2 = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_exch_sq, params)
                        a_ij_sq = abs(a_ij_sq)
                        if !(has_pairwise_1 | has_pairwise_2)
                            continue
                        end

                        b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_repulsion, params)
                        b_ij = abs(b_ij)
                        if !has_pairwise
                            b_ij = sqrt(abs(params[Symbol(labels[i], :_b_repulsion)]) * abs(params[Symbol(labels[j], :_b_repulsion)]))
                        end

                        r_ij = norm(coords[i] - coords[j])
                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)

                        pauli_energy += a_ij * slater_overlap + a_ij_sq * slater_overlap^2
                    end
                end
            end
        end
    end
    return pauli_energy
end

function overlap_squared_elec!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64}
)
    elec_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        a_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_elec_sq, params)
                        if !has_pairwise
                            continue
                        end

                        b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_elec, params)
                        b_ij = abs(b_ij)
                        if !has_pairwise
                            b_ij = sqrt(abs(params[Symbol(labels[i], :_b_elec)]) * abs(params[Symbol(labels[j], :_b_elec)]))
                        end

                        r_ij = norm(coords[i] - coords[j])
                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)

                        elec_energy += a_ij * slater_overlap^2
                    end
                end
            end
        end
    end
    return elec_energy
end

function overlap_squared_pol!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64}

)
    pol_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        if !is_ion(labels[i]) || !is_ion(labels[j])
                            continue
                        end
                        a_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_pol_sq, params)
                        if !has_pairwise
                            continue
                        end

                        b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_elec, params)
                        b_ij = abs(b_ij)
                        if !has_pairwise
                            b_ij = sqrt(abs(params[Symbol(labels[i], :_b_elec)]) * abs(params[Symbol(labels[j], :_b_elec)]))
                        end

                        r_ij = norm(coords[i] - coords[j])
                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)

                        pol_energy += a_ij * slater_overlap^2
                    end
                end
            end
        end
    end
    return pol_energy
end

function overlap_squared_ct!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64}

)
    ct_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag != j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        a_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_a_ct_sq, params)
                        if !has_pairwise || a_ij == 0.0
                            continue
                        end

                        b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_ct, params)
                        b_ij = abs(b_ij)
                        if !has_pairwise
                            b_ij = sqrt(abs(params[Symbol(labels[i], :_b_ct)]) * abs(params[Symbol(labels[j], :_b_ct)]))
                        end

                        r_ij = norm(coords[i] - coords[j])
                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)

                        ct_energy += a_ij * slater_overlap^2
                    end
                end
            end
        end
    end
    return ct_energy
end

function multipolar_pauli_repulsion_isotropic!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    repulsion_charges::Vector{Float64}
)
    exchange_energy = 0.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                q_rep_i = repulsion_charges[i]
                b_i_rep = abs(params[Symbol(labels[i], :_b_repulsion)])
                for j in fragment_indices[j_frag]
                    q_rep_j = repulsion_charges[j]
                    b_j_rep = abs(params[Symbol(labels[j], :_b_repulsion)])

                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    b_ij = sqrt(b_i_rep * b_j_rep)

                    f_repulsion = (1 + 0.5 * b_ij * r_ij + (0.5 * b_ij * r_ij)^2 / 3.0) * exp(-0.5 * b_ij * r_ij) / b_ij^3
                    exchange_energy += b_i_rep^3 * b_j_rep^3 * q_rep_i * q_rep_j * f_repulsion^2 / r_ij

                    #slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                    #exchange_energy += q_rep_i * q_rep_j * slater_overlap
                end
            end
        end
    end
    return exchange_energy
end

function triple_overlap_pauli_repulsion(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    ϕ_repulsion::Vector{Float64},
    params::Dict{Symbol,Float64}
)
    exchange_energy_3b = 0.0
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                if labels[i] == "X"
                    continue
                end
                b_i = params[Symbol(labels[i], :_b_repulsion)]
                Q_i = params[Symbol(labels[i], :_q_repulsion)]

                for j in fragment_indices[j_frag]
                    if labels[j] == "X"
                        continue
                    end
                    Q_j = params[Symbol(labels[i], :_q_repulsion)]

                    b_j = params[Symbol(labels[j], :_b_repulsion)]
                    b_ij = sqrt(b_i * b_j)

                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)

                    S_ij = exp(-0.5 * b_ij * r_ij) * (1 / 3 * (0.5 * b_ij * r_ij)^2 + 0.5 * b_ij * r_ij + 1.0)
                    
                    for k_frag in eachindex(fragment_indices)
                        for k in fragment_indices[k_frag]
                            if labels[k] == "X"
                                continue
                            end
                            b_k = params[Symbol(labels[k], :_b_repulsion)]
                            b_ik = sqrt(b_i * b_k)
                            b_jk = sqrt(b_j * b_k)
                                
                            r_ik_vec = coords[k] - coords[i]
                            r_ik = norm(r_ik_vec)
                            r_jk_vec = coords[k] - coords[j]
                            r_jk = norm(r_jk_vec)
                            S_ik = exp(-0.5 * b_ik * r_ik) * (1 / 3 * (0.5 * b_ik * r_ik)^2 + 0.5 * b_ik * r_ik + 1.0)
                            S_jk = exp(-0.5 * b_jk * r_jk) * (1 / 3 * (0.5 * b_jk * r_jk)^2 + 0.5 * b_jk * r_jk + 1.0)
                                
                            S_ijk = S_ij * S_jk
                            S_jik = S_ij * S_ik
                            if k != j && k != i

                                K_jk = 0.0
                                K_ik = 0.0
                                if labels[j] < labels[k]
                                    K_jk = params[Symbol(labels[j], labels[k], :_repulsion_flux)]
                                else
                                    K_jk = params[Symbol(labels[k], labels[j], :_repulsion_flux)]
                                end

                                if labels[i] < labels[k]
                                    K_ik = params[Symbol(labels[i], labels[k], :_repulsion_flux)]
                                else
                                    K_ik = params[Symbol(labels[k], labels[i], :_repulsion_flux)]
                                end

                                # get the correction to ij interaction from k
                                ΔQ_j = K_jk * S_ijk
                                ΔQ_i = K_ik * S_jik
                                exchange_energy_3b += ΔQ_j #* Q_i / r_ij
                                exchange_energy_3b += ΔQ_i #* Q_j / r_ij
                                #exchange_energy_3b += -ΔQ_j * Q_i / r_ik
                                #exchange_energy_3b += -ΔQ_i * Q_j / r_jk
                            end
                        end
                    end
                end
            end
        end
    end
    return exchange_energy_3b
end

function total_anisotropic_short_range_energy!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    local_axes::AbstractVector{LocalAxes},
    params::Dict{Symbol,Float64},
    Δq_ct::AbstractVector{Float64},
    C6::AbstractVector{Float64},
    include_charge_transfer::Bool
)
    exchange_energy = 0.0
    dispersion_energy = 0.0
    cp_energy = 0.0
    direct_ct_energy = 0.0
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    a_i_Y10_exch = 0.0
                    a_i_Y11_exch = 0.0
                    a_i_Y20_exch = 0.0
                    a_i_Y21_exch = 0.0
                    a_i_Y22_exch = 0.0
                    a_i_Y10_cp = 0.0
                    a_i_Y11_cp = 0.0
                    a_i_Y20_cp = 0.0
                    a_i_Y21_cp = 0.0
                    a_i_Y22_cp = 0.0
                    a_i_exch = abs(params[Symbol(labels[i], :_a_exch)])
                    a_i_exch_sq = params[Symbol(labels[i], :_a_exch_sq)]
                    a_i_cp = params[Symbol(labels[i], :_a_cp)]
                    a_i_cp_sq = params[Symbol(labels[i], :_a_cp_sq)]
                    b_i = abs(params[Symbol(labels[i], :_b_exch)])
                    b_i_ct = abs(params[Symbol(labels[i], :_b_ct)])
                    if haskey(params, Symbol(labels[i], :_a_exch_Y10))
                        a_i_Y10_exch = params[Symbol(labels[i], :_a_exch_Y10)]
                        a_i_Y11_exch = params[Symbol(labels[i], :_a_exch_Y11)]
                        a_i_Y20_exch = params[Symbol(labels[i], :_a_exch_Y20)]
                        a_i_Y21_exch = params[Symbol(labels[i], :_a_exch_Y21)]
                        a_i_Y22_exch = params[Symbol(labels[i], :_a_exch_Y22)]
                    end
                    if haskey(params, Symbol(labels[i], :_a_cp_Y10))
                        a_i_Y10_cp = params[Symbol(labels[i], :_a_cp_Y10)]
                        a_i_Y11_cp = params[Symbol(labels[i], :_a_cp_Y11)]
                        a_i_Y20_cp = params[Symbol(labels[i], :_a_cp_Y20)]
                        a_i_Y21_cp = params[Symbol(labels[i], :_a_cp_Y21)]
                        a_i_Y22_cp = params[Symbol(labels[i], :_a_cp_Y22)]
                    end
                    for j in fragment_indices[j_frag]
                        a_j_cp = params[Symbol(labels[j], :_a_cp)]
                        a_j_cp_sq = params[Symbol(labels[j], :_a_cp_sq)]
                        a_ij_forward = params[Symbol(labels[i], :_a_ct_donor_slater)] * params[Symbol(labels[j], :_a_ct_acceptor_slater)]
                        a_ij_backward = params[Symbol(labels[i], :_a_ct_acceptor_slater)] * params[Symbol(labels[j], :_a_ct_donor_slater)]
                        γ_forward  = abs(params[Symbol(labels[i], :_γ_ct_slater)])
                        γ_backward = abs(params[Symbol(labels[j], :_γ_ct_slater)])
                        γ_ij = 0.5 * (γ_forward + γ_backward)
                        
                        a_j_exch = abs(params[Symbol(labels[j], :_a_exch)])
                        a_j_exch_sq = params[Symbol(labels[j], :_a_exch_sq)]
                        a_j_Y10_exch = 0.0
                        a_j_Y11_exch = 0.0
                        a_j_Y20_exch = 0.0
                        a_j_Y21_exch = 0.0
                        a_j_Y22_exch = 0.0
                        a_j_Y10_cp = 0.0
                        a_j_Y11_cp = 0.0
                        a_j_Y20_cp = 0.0
                        a_j_Y21_cp = 0.0
                        a_j_Y22_cp = 0.0
                        if haskey(params, Symbol(labels[j], :_a_exch_Y10))
                            a_j_Y10_exch = params[Symbol(labels[j], :_a_exch_Y10)]
                            a_j_Y11_exch = params[Symbol(labels[j], :_a_exch_Y11)]
                            a_j_Y20_exch = params[Symbol(labels[j], :_a_exch_Y20)]
                            a_j_Y21_exch = params[Symbol(labels[j], :_a_exch_Y21)]
                            a_j_Y22_exch = params[Symbol(labels[j], :_a_exch_Y22)]
                        end
                        if haskey(params, Symbol(labels[j], :_a_cp_Y10))
                            a_j_Y10_cp = params[Symbol(labels[j], :_a_cp_Y10)]
                            a_j_Y11_cp = params[Symbol(labels[j], :_a_cp_Y11)]
                            a_j_Y20_cp = params[Symbol(labels[j], :_a_cp_Y20)]
                            a_j_Y21_cp = params[Symbol(labels[j], :_a_cp_Y21)]
                            a_j_Y22_cp = params[Symbol(labels[j], :_a_cp_Y22)]
                        end
                        #C6_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_C6_disp, params)
                        #C6_ij = abs(C6_ij)
                        #if !has_pairwise
                        #    C6_ij = sqrt(abs(params[Symbol(labels[i], :_C6_disp)]) * abs(params[Symbol(labels[j], :_C6_disp)]))
                        #end
                        C6_ij = sqrt(C6[i] * C6[j])
                        b_j = abs(params[Symbol(labels[j], :_b_exch)])
                        b_j_ct = abs(params[Symbol(labels[j], :_b_ct)])
                        b_ij = sqrt(b_i * b_j)
                        b_ij_ct = sqrt(b_i_ct * b_j_ct)

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

                        a_i_Ω_exch = a_i_exch * (1.0 + a_i_Y10_exch * Y10_i + a_i_Y11_exch * Y11_i + a_i_Y20_exch * Y20_i + a_i_Y21_exch * Y21_i + a_i_Y22_exch * Y22_i)
                        a_j_Ω_exch = a_j_exch * (1.0 + a_j_Y10_exch * Y10_j + a_j_Y11_exch * Y11_j + a_j_Y20_exch * Y20_j + a_j_Y21_exch * Y21_j + a_j_Y22_exch * Y22_j)
                        a_ij_Ω_exch = a_i_Ω_exch * a_j_Ω_exch
                        a_ij_exch_sq = a_i_exch_sq * a_j_exch_sq

                        a_i_Ω_cp = a_i_cp * (1.0 + a_i_Y10_cp * Y10_i + a_i_Y11_cp * Y11_i + a_i_Y20_cp * Y20_i + a_i_Y21_cp * Y21_i + a_i_Y22_cp * Y22_i)
                        a_j_Ω_cp = a_j_cp * (1.0 + a_j_Y10_cp * Y10_j + a_j_Y11_cp * Y11_j + a_j_Y20_cp * Y20_j + a_j_Y21_cp * Y21_j + a_j_Y22_cp * Y22_j)
                        a_ij_Ω_cp = a_i_Ω_cp * a_j_Ω_cp
                        a_ij_cp_sq = a_i_cp_sq * a_j_cp_sq

                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        slater_overlap_ct = exp(-b_ij_ct * r_ij) * (1 / 3 * (b_ij_ct * r_ij)^2 + b_ij_ct * r_ij + 1.0)
                        x = slater_damping_value(r_ij, b_ij)
                        
                        # exchange energy
                        exchange_energy += a_ij_Ω_exch * slater_overlap + a_ij_exch_sq * slater_overlap^2

                        # dispersion energy
                        dispersion_energy -= inc_gamma(x, 6) * C6_ij / r_ij^6

                        # charge penetration energy
                        cp_energy -= (a_ij_Ω_cp * slater_overlap + a_ij_cp_sq * slater_overlap^2) * inc_gamma(x, 3)

                        # forward CT
                        if include_charge_transfer
                            direct_ct_energy -= a_ij_forward * slater_overlap_ct
                            Δq_ct_forward = a_ij_forward * slater_overlap_ct / γ_ij
                            Δq_ct[i_frag] -= Δq_ct_forward
                            Δq_ct[j_frag] += Δq_ct_forward

                            # backward CT
                            direct_ct_energy -= a_ij_backward * slater_overlap_ct
                            Δq_ct_backward = a_ij_backward * slater_overlap_ct / γ_ij
                            Δq_ct[i_frag] += Δq_ct_backward
                            Δq_ct[j_frag] -= Δq_ct_backward
                        end
                    end
                end
            end
        end
    end
    fractional_charge_energy = 0.0
    return exchange_energy, dispersion_energy, cp_energy, direct_ct_energy, fractional_charge_energy
end

function total_anisotropic_short_range_energy_and_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    local_axes::AbstractVector{LocalAxes},
    params::Dict{Symbol,Float64},
    Δq_ct::AbstractVector{Float64},
    C6::AbstractVector{Float64},
    include_charge_transfer::Bool,
    pauli_grads::Vector{MVector{3, Float64}},
    disp_grads::Vector{MVector{3, Float64}},
    elec_grads::Vector{MVector{3, Float64}},
    ct_grads::Vector{MVector{3, Float64}},
)
    exchange_energy = 0.0
    dispersion_energy = 0.0
    cp_energy = 0.0
    direct_ct_energy = 0.0
    spherical_grads = SVector{4, MMatrix{3, 4, Float64, 12}}([@MMatrix zeros(3, 4) for _ in 1:4])
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    a_i_Y10_exch = 0.0
                    a_i_Y11_exch = 0.0
                    a_i_Y20_exch = 0.0
                    a_i_Y21_exch = 0.0
                    a_i_Y22_exch = 0.0
                    a_i_Y10_cp = 0.0
                    a_i_Y11_cp = 0.0
                    a_i_Y20_cp = 0.0
                    a_i_Y21_cp = 0.0
                    a_i_Y22_cp = 0.0
                    a_i_exch = abs(params[Symbol(labels[i], :_a_exch)])
                    a_i_exch_sq = params[Symbol(labels[i], :_a_exch_sq)]
                    a_i_cp = params[Symbol(labels[i], :_a_cp)]
                    a_i_cp_sq = params[Symbol(labels[i], :_a_cp_sq)]
                    b_i = abs(params[Symbol(labels[i], :_b_exch)])
                    if haskey(params, Symbol(labels[i], :_a_exch_Y10))
                        a_i_Y10_exch = params[Symbol(labels[i], :_a_exch_Y10)]
                        a_i_Y11_exch = params[Symbol(labels[i], :_a_exch_Y11)]
                        a_i_Y20_exch = params[Symbol(labels[i], :_a_exch_Y20)]
                        a_i_Y21_exch = params[Symbol(labels[i], :_a_exch_Y21)]
                        a_i_Y22_exch = params[Symbol(labels[i], :_a_exch_Y22)]
                    end
                    if haskey(params, Symbol(labels[i], :_a_cp_Y10))
                        a_i_Y10_cp = params[Symbol(labels[i], :_a_cp_Y10)]
                        a_i_Y11_cp = params[Symbol(labels[i], :_a_cp_Y11)]
                        a_i_Y20_cp = params[Symbol(labels[i], :_a_cp_Y20)]
                        a_i_Y21_cp = params[Symbol(labels[i], :_a_cp_Y21)]
                        a_i_Y22_cp = params[Symbol(labels[i], :_a_cp_Y22)]
                    end
                    for j in fragment_indices[j_frag]
                        #C6_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_C6_disp, params)
                        #C6_ij = abs(C6_ij)
                        #if !has_pairwise
                        #    C6_ij = sqrt(abs(params[Symbol(labels[i], :_C6_disp)]) * abs(params[Symbol(labels[j], :_C6_disp)]))
                        #end
                        C6_ij = sqrt(C6[i] * C6[j])
                        
                        a_j_cp = params[Symbol(labels[j], :_a_cp)]
                        a_j_cp_sq = params[Symbol(labels[j], :_a_cp_sq)]
                        a_ij_forward = params[Symbol(labels[i], :_a_ct_donor_slater)] * params[Symbol(labels[j], :_a_ct_acceptor_slater)]
                        a_ij_backward = params[Symbol(labels[i], :_a_ct_acceptor_slater)] * params[Symbol(labels[j], :_a_ct_donor_slater)]
                        γ_forward  = abs(params[Symbol(labels[i], :_γ_ct_slater)])
                        γ_backward = abs(params[Symbol(labels[j], :_γ_ct_slater)])
                        γ_ij = 0.5 * (γ_forward + γ_backward)
                        
                        a_j_exch = abs(params[Symbol(labels[j], :_a_exch)])
                        a_j_exch_sq = params[Symbol(labels[j], :_a_exch_sq)]
                        a_j_Y10_exch = 0.0
                        a_j_Y11_exch = 0.0
                        a_j_Y20_exch = 0.0
                        a_j_Y21_exch = 0.0
                        a_j_Y22_exch = 0.0
                        a_j_Y10_cp = 0.0
                        a_j_Y11_cp = 0.0
                        a_j_Y20_cp = 0.0
                        a_j_Y21_cp = 0.0
                        a_j_Y22_cp = 0.0
                        if haskey(params, Symbol(labels[j], :_a_exch_Y10))
                            a_j_Y10_exch = params[Symbol(labels[j], :_a_exch_Y10)]
                            a_j_Y11_exch = params[Symbol(labels[j], :_a_exch_Y11)]
                            a_j_Y20_exch = params[Symbol(labels[j], :_a_exch_Y20)]
                            a_j_Y21_exch = params[Symbol(labels[j], :_a_exch_Y21)]
                            a_j_Y22_exch = params[Symbol(labels[j], :_a_exch_Y22)]
                        end
                        if haskey(params, Symbol(labels[j], :_a_cp_Y10))
                            a_j_Y10_cp = params[Symbol(labels[j], :_a_cp_Y10)]
                            a_j_Y11_cp = params[Symbol(labels[j], :_a_cp_Y11)]
                            a_j_Y20_cp = params[Symbol(labels[j], :_a_cp_Y20)]
                            a_j_Y21_cp = params[Symbol(labels[j], :_a_cp_Y21)]
                            a_j_Y22_cp = params[Symbol(labels[j], :_a_cp_Y22)]
                        end
                        b_j = abs(params[Symbol(labels[j], :_b_exch)])
                        b_ij = sqrt(b_i * b_j)

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

                        a_i_Ω_exch = a_i_exch * (1.0 + a_i_Y10_exch * Y10_i + a_i_Y11_exch * Y11_i + a_i_Y20_exch * Y20_i + a_i_Y21_exch * Y21_i + a_i_Y22_exch * Y22_i)
                        a_j_Ω_exch = a_j_exch * (1.0 + a_j_Y10_exch * Y10_j + a_j_Y11_exch * Y11_j + a_j_Y20_exch * Y20_j + a_j_Y21_exch * Y21_j + a_j_Y22_exch * Y22_j)
                        a_ij_Ω_exch = a_i_Ω_exch * a_j_Ω_exch

                        # anisotropy gradients for exchange and CP
                        a_i_Ω_grad_exch_i  = (a_i_Y10_exch * Y10_i_grad_i  + a_i_Y11_exch * Y11_i_grad_i  + a_i_Y20_exch * Y20_i_grad_i  + a_i_Y21_exch * Y21_i_grad_i  + a_i_Y22_exch * Y22_i_grad_i)
                        a_i_Ω_grad_exch_iz = (a_i_Y10_exch * Y10_i_grad_iz + a_i_Y11_exch * Y11_i_grad_iz + a_i_Y20_exch * Y20_i_grad_iz + a_i_Y21_exch * Y21_i_grad_iz + a_i_Y22_exch * Y22_i_grad_iz)
                        a_i_Ω_grad_exch_ix = (a_i_Y10_exch * Y10_i_grad_ix + a_i_Y11_exch * Y11_i_grad_ix + a_i_Y20_exch * Y20_i_grad_ix + a_i_Y21_exch * Y21_i_grad_ix + a_i_Y22_exch * Y22_i_grad_ix)
                        a_i_Ω_grad_exch_j  = (a_i_Y10_exch * Y10_i_grad_j  + a_i_Y11_exch * Y11_i_grad_j  + a_i_Y20_exch * Y20_i_grad_j  + a_i_Y21_exch * Y21_i_grad_j  + a_i_Y22_exch * Y22_i_grad_j)

                        a_i_Ω_grad_cp_i  = (a_i_Y10_cp * Y10_i_grad_i  + a_i_Y11_cp * Y11_i_grad_i  + a_i_Y20_cp * Y20_i_grad_i  + a_i_Y21_cp * Y21_i_grad_i  + a_i_Y22_cp * Y22_i_grad_i)
                        a_i_Ω_grad_cp_iz = (a_i_Y10_cp * Y10_i_grad_iz + a_i_Y11_cp * Y11_i_grad_iz + a_i_Y20_cp * Y20_i_grad_iz + a_i_Y21_cp * Y21_i_grad_iz + a_i_Y22_cp * Y22_i_grad_iz)
                        a_i_Ω_grad_cp_ix = (a_i_Y10_cp * Y10_i_grad_ix + a_i_Y11_cp * Y11_i_grad_ix + a_i_Y20_cp * Y20_i_grad_ix + a_i_Y21_cp * Y21_i_grad_ix + a_i_Y22_cp * Y22_i_grad_ix)
                        a_i_Ω_grad_cp_j  = (a_i_Y10_cp * Y10_i_grad_j  + a_i_Y11_cp * Y11_i_grad_j  + a_i_Y20_cp * Y20_i_grad_j  + a_i_Y21_cp * Y21_i_grad_j  + a_i_Y22_cp * Y22_i_grad_j)

                        a_j_Ω_grad_exch_j  = (a_j_Y10_exch * Y10_j_grad_j  + a_j_Y11_exch * Y11_j_grad_j  + a_j_Y20_exch * Y20_j_grad_j  + a_j_Y21_exch * Y21_j_grad_j  + a_j_Y22_exch * Y22_j_grad_j)
                        a_j_Ω_grad_exch_jz = (a_j_Y10_exch * Y10_j_grad_jz + a_j_Y11_exch * Y11_j_grad_jz + a_j_Y20_exch * Y20_j_grad_jz + a_j_Y21_exch * Y21_j_grad_jz + a_j_Y22_exch * Y22_j_grad_jz)
                        a_j_Ω_grad_exch_jx = (a_j_Y10_exch * Y10_j_grad_jx + a_j_Y11_exch * Y11_j_grad_jx + a_j_Y20_exch * Y20_j_grad_jx + a_j_Y21_exch * Y21_j_grad_jx + a_j_Y22_exch * Y22_j_grad_jx)
                        a_j_Ω_grad_exch_i  = (a_j_Y10_exch * Y10_j_grad_i  + a_j_Y11_exch * Y11_j_grad_i  + a_j_Y20_exch * Y20_j_grad_i  + a_j_Y21_exch * Y21_j_grad_i  + a_j_Y22_exch * Y22_j_grad_i)

                        a_j_Ω_grad_cp_j  = (a_j_Y10_cp * Y10_j_grad_j  + a_j_Y11_cp * Y11_j_grad_j  + a_j_Y20_cp * Y20_j_grad_j  + a_j_Y21_cp * Y21_j_grad_j  + a_j_Y22_cp * Y22_j_grad_j)
                        a_j_Ω_grad_cp_jz = (a_j_Y10_cp * Y10_j_grad_jz + a_j_Y11_cp * Y11_j_grad_jz + a_j_Y20_cp * Y20_j_grad_jz + a_j_Y21_cp * Y21_j_grad_jz + a_j_Y22_cp * Y22_j_grad_jz)
                        a_j_Ω_grad_cp_jx = (a_j_Y10_cp * Y10_j_grad_jx + a_j_Y11_cp * Y11_j_grad_jx + a_j_Y20_cp * Y20_j_grad_jx + a_j_Y21_cp * Y21_j_grad_jx + a_j_Y22_cp * Y22_j_grad_jx)
                        a_j_Ω_grad_cp_i  = (a_j_Y10_cp * Y10_j_grad_i  + a_j_Y11_cp * Y11_j_grad_i  + a_j_Y20_cp * Y20_j_grad_i  + a_j_Y21_cp * Y21_j_grad_i  + a_j_Y22_cp * Y22_j_grad_i)

                        a_i_Ω_cp = a_i_cp * (1.0 + a_i_Y10_cp * Y10_i + a_i_Y11_cp * Y11_i + a_i_Y20_cp * Y20_i + a_i_Y21_cp * Y21_i + a_i_Y22_cp * Y22_i)
                        a_j_Ω_cp = a_j_cp * (1.0 + a_j_Y10_cp * Y10_j + a_j_Y11_cp * Y11_j + a_j_Y20_cp * Y20_j + a_j_Y21_cp * Y21_j + a_j_Y22_cp * Y22_j)
                        a_ij_Ω_cp = a_i_Ω_cp * a_j_Ω_cp

                        r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(coords[j] - coords[i])

                        x = slater_damping_value(r_ij, b_ij)
                        x_gradient_i = slater_damping_value_gradient(r_ij_vec, r_ij, b_ij)

                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        slater_overlap_gradient_j = exp(-b_ij * r_ij) * (
                            (2 / 3 * b_ij^2 * r_ij + b_ij) +
                            -b_ij * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        ) * r_ij_vec / r_ij

                        ### exchange energy ###
                        exchange_energy += a_ij_Ω_exch * slater_overlap
                        pauli_grads[i] += (
                            a_i_exch * a_j_Ω_exch * slater_overlap * a_i_Ω_grad_exch_i +
                            a_j_exch * a_i_Ω_exch * slater_overlap * a_j_Ω_grad_exch_i -
                            a_ij_Ω_exch * slater_overlap_gradient_j
                        )
                        if local_axes[i].type != :Global
                            pauli_grads[local_axes[i].i_z] += a_i_exch * a_j_Ω_exch * slater_overlap * a_i_Ω_grad_exch_iz
                            pauli_grads[local_axes[i].i_x] += a_i_exch * a_j_Ω_exch * slater_overlap * a_i_Ω_grad_exch_ix
                        end

                        pauli_grads[j] += (
                            a_i_exch * a_j_Ω_exch * slater_overlap * a_i_Ω_grad_exch_j +
                            a_j_exch * a_i_Ω_exch * slater_overlap * a_j_Ω_grad_exch_j +
                            a_ij_Ω_exch * slater_overlap_gradient_j
                        )
                        if local_axes[j].type != :Global
                            pauli_grads[local_axes[j].i_z] += a_j_exch * a_i_Ω_exch * slater_overlap * a_j_Ω_grad_exch_jz
                            pauli_grads[local_axes[j].i_x] += a_j_exch * a_i_Ω_exch * slater_overlap * a_j_Ω_grad_exch_jx
                        end

                        ### dispersion energy ###
                        dispersion_energy -= inc_gamma(x, 6) * C6_ij / r_ij^6
                        disp_grads[i] += C6_ij * (
                            6 * inc_gamma(x, 6) / r_ij^7 * -r_ij_vec / r_ij +
                            r_ij^-6 * (-inc_gamma_derivative(x, 6) * x_gradient_i)
                        )
                        disp_grads[j] -= C6_ij * (
                            6 * inc_gamma(x, 6) / r_ij^7 * -r_ij_vec / r_ij +
                            r_ij^-6 * (-inc_gamma_derivative(x, 6) * x_gradient_i)
                        )

                        ### charge penetration energy ###
                        cp_energy -= a_ij_Ω_cp * slater_overlap * inc_gamma(x, 3)
                        elec_grads[i] -= (
                            a_i_cp * a_j_Ω_cp * inc_gamma(x, 3) * slater_overlap * a_i_Ω_grad_cp_i +
                            a_j_cp * a_i_Ω_cp * inc_gamma(x, 3) * slater_overlap * a_j_Ω_grad_cp_i -
                            a_ij_Ω_cp * inc_gamma(x, 3) * slater_overlap_gradient_j + 
                            a_ij_Ω_cp * slater_overlap * inc_gamma_derivative(x, 3) * x_gradient_i
                        )
                        if local_axes[i].type != :Global
                            elec_grads[local_axes[i].i_z] -= a_j_Ω_cp * a_i_cp * inc_gamma(x, 3) * slater_overlap * a_i_Ω_grad_cp_iz
                            elec_grads[local_axes[i].i_x] -= a_j_Ω_cp * a_i_cp * inc_gamma(x, 3) * slater_overlap * a_i_Ω_grad_cp_ix
                        end
                        
                        elec_grads[j] -= (
                            a_i_cp * a_j_Ω_cp * inc_gamma(x, 3) * slater_overlap * a_i_Ω_grad_cp_j +
                            a_j_cp * a_i_Ω_cp * inc_gamma(x, 3) * slater_overlap * a_j_Ω_grad_cp_j +
                            a_ij_Ω_cp * inc_gamma(x, 3) * slater_overlap_gradient_j - 
                            a_ij_Ω_cp * slater_overlap * inc_gamma_derivative(x, 3) * x_gradient_i
                        )
                        if local_axes[j].type != :Global
                            elec_grads[local_axes[j].i_z] -= a_i_Ω_cp * a_j_cp * inc_gamma(x, 3) * slater_overlap * a_j_Ω_grad_cp_jz
                            elec_grads[local_axes[j].i_x] -= a_i_Ω_cp * a_j_cp * inc_gamma(x, 3) * slater_overlap * a_j_Ω_grad_cp_jx
                        end

                        ### charge transfer energy ###
                        # forward CT
                        if include_charge_transfer
                            direct_ct_energy -= a_ij_forward * slater_overlap
                            Δq_ct_forward = a_ij_forward * slater_overlap / γ_ij
                            Δq_ct[i_frag] -= Δq_ct_forward
                            Δq_ct[j_frag] += Δq_ct_forward
                            ct_grads[i] += a_ij_forward * slater_overlap_gradient_j
                            ct_grads[j] -= a_ij_forward * slater_overlap_gradient_j

                            # backward CT
                            direct_ct_energy -= a_ij_backward * slater_overlap
                            Δq_ct_backward = a_ij_backward * slater_overlap / γ_ij
                            Δq_ct[i_frag] += Δq_ct_backward
                            Δq_ct[j_frag] -= Δq_ct_backward
                            ct_grads[i] += a_ij_backward * slater_overlap_gradient_j
                            ct_grads[j] -= a_ij_backward * slater_overlap_gradient_j
                        end
                    end
                end
            end
        end
    end
    return exchange_energy, dispersion_energy, cp_energy, direct_ct_energy
end

function get_variable_C6_gradients!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::Vector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    C6::Vector{Float64},
    grads::AbstractVector{MVector{3,Float64}},
    params::Dict{Symbol, Float64}
)
    for i_frag in 1:(length(fragment_indices)-1)
        for j_frag in (i_frag+1):length(fragment_indices)
            for i in fragment_indices[i_frag]
                b_i = params[Symbol(labels[i], :_b_exch)]
                k_damp_C6_i = abs(params[Symbol(labels[i], :_C6_damp)])
                C6_0_i = abs(params[Symbol(labels[i], :_C6_disp)])
                for j in fragment_indices[j_frag]
                    k_damp_C6_j = abs(params[Symbol(labels[j], :_C6_damp)])
                    C6_0_j = abs(params[Symbol(labels[j], :_C6_disp)])
                    k_damp_C6_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :k_C6_damp, params)
                    k_damp_C6_ij = abs(k_damp_C6_ij)
                    if !has_pairwise
                        k_damp_C6_ij = 0.5 * (k_damp_C6_i + k_damp_C6_j)
                    end
                    C6_ij = sqrt(C6[i] * C6[j])
                    b_j = params[Symbol(labels[j], :_b_exch)]
                    b_ij = sqrt(b_i * b_j)
                    
                    r_ij_vec = coords[j] - coords[i]
                    r_ij = norm(r_ij_vec)
                    
                    x = slater_damping_value(r_ij, b_ij)
                    slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                    slater_overlap_gradient_j = exp(-b_ij * r_ij) * (
                        (2 / 3 * b_ij^2 * r_ij + b_ij) +
                        -b_ij * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                    ) * r_ij_vec / r_ij
                    if i == 1 && j == 4
                    grads[j] -= inc_gamma(x, 6) / (C6_ij * r_ij^6) * (
                        -C6_0_i * k_damp_C6_ij - C6_0_j * k_damp_C6_ij + 2 * k_damp_C6_ij^2 * slater_overlap
                    ) * slater_overlap_gradient_j
                    grads[i] += inc_gamma(x, 6) / (C6_ij * r_ij^6) * (
                        -C6_0_i * k_damp_C6_ij - C6_0_j * k_damp_C6_ij + 2 * k_damp_C6_ij^2 * slater_overlap
                    ) * slater_overlap_gradient_j
                    end
                    for k_frag in eachindex(fragment_indices)
                        if k_frag != i_frag
                            for k in fragment_indices[k_frag]
                                C6_ij = sqrt(C6[i] * C6[j])
                                b_k = params[Symbol(labels[k], :_b_exch)]
                                b_ik = sqrt(b_i * b_k)
                                b_ik = sqrt(b_j * b_k)
                    
                                r_ik_vec = coords[k] - coords[i]
                                r_ik = norm(r_ik_vec)
                                r_jk_vec = coords[k] - coords[j]
                                r_jk = norm(r_jk_vec)
                    
                                x_ik = slater_damping_value(r_ik, b_ik)
                                x_jk = slater_damping_value(r_jk, b_jk)
                                slater_overlap_ik = exp(-b_ik * r_ik) * (1 / 3 * (b_ik * r_ik)^2 + b_ik * r_ik + 1.0)
                                slater_overlap_jk = exp(-b_jk * r_jk) * (1 / 3 * (b_jk * r_jk)^2 + b_jk * r_jk + 1.0)
                                slater_overlap_gradient_ik = exp(-b_ik * r_ik) * (
                                    (2 / 3 * b_ik^2 * r_ik + b_ik) +
                                    -b_ik * (1 / 3 * (b_ik * r_ik)^2 + b_ik * r_ik + 1.0)
                                ) * r_ik_vec / r_ik
                                slater_overlap_gradient_jk = exp(-b_jk * r_jk) * (
                                    (2 / 3 * b_jk^2 * r_jk + b_jk) +
                                    -b_jk * (1 / 3 * (b_jk * r_jk)^2 + b_jk * r_jk + 1.0)
                                ) * r_jk_vec / r_jk
                                k_damp_C6_k = abs(params[Symbol(labels[k], :_C6_damp)])
                                k_damp_C6_ik, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[k], :k_C6_damp, params)
                                k_damp_C6_ik = abs(k_damp_C6_ik)
                                k_damp_C6_jk, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[k], :k_C6_damp, params)
                                k_damp_C6_jk = abs(k_damp_C6_jk)
                                if !has_pairwise
                                    k_damp_C6_ik = 0.5 * (k_damp_C6_i + k_damp_C6_k)
                                    k_damp_C6_jk = 0.5 * (k_damp_C6_j + k_damp_C6_k)
                                end
                                grads[k] -= inc_gamma(x, 6) / (C6_ij * r_ij^6) * (
                                    k_damp_C6_ij * k_damp_C6_jk * slater_overlap
                                ) * slater_overlap_gradient_j
                                #grads[i] += 0.5 * inc_gamma(x, 6) / (C6_ij * r_ij^6) * (
                                #   -C6_0_j * k_damp_C6_ik - (C6_0_i - C6[i]) * k_damp_C6_ik
                                #) * slater_overlap_gradient_k
                            end
                        end
                    end
                end
            end
        end
    end
end

"""
Takes lagrange multipliers determined after evaluating polarization
and computes the charge transfer rearrangement contribution to the
gradients.
"""
function evaluate_Δq_grads!(
    coords::AbstractVector{MVector{3,Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    params::Dict{Symbol,Float64},
    λ::AbstractVector{Float64},
    grads::AbstractVector{MVector{3, Float64}}
)
    for i_frag in eachindex(fragment_indices)
        for j_frag in eachindex(fragment_indices)
            if i_frag < j_frag
                for i in fragment_indices[i_frag]
                    for j in fragment_indices[j_frag]
                        a_ij_forward = params[Symbol(labels[i], :_a_ct_donor_slater)] * params[Symbol(labels[j], :_a_ct_acceptor_slater)]
                        a_ij_backward = params[Symbol(labels[i], :_a_ct_acceptor_slater)] * params[Symbol(labels[j], :_a_ct_donor_slater)]
                        γ_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_γ_ct, params)
                        γ_ij = abs(γ_ij)
                        if !has_pairwise
                            γ_forward  = abs(params[Symbol(labels[i], :_γ_ct_slater)])
                            γ_backward = abs(params[Symbol(labels[j], :_γ_ct_slater)])
                            γ_ij = 0.5 * (γ_forward + γ_backward)
                        end
                        b_ij = sqrt(params[Symbol(labels[i], :_b_exch)] * params[Symbol(labels[j], :_b_exch)])
                        
                        r_ij_vec = coords[j] - coords[i]
                        r_ij = norm(coords[j] - coords[i])

                        slater_overlap = exp(-b_ij * r_ij) * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        slater_overlap_gradient_j = exp(-b_ij * r_ij) * (
                            (2 / 3 * b_ij^2 * r_ij + b_ij) +
                            -b_ij * (1 / 3 * (b_ij * r_ij)^2 + b_ij * r_ij + 1.0)
                        ) * r_ij_vec / r_ij

                        ### forward CT ###
                        grads[i] += λ[i_frag] * a_ij_forward * slater_overlap_gradient_j / γ_ij
                        grads[j] -= λ[i_frag] * a_ij_forward * slater_overlap_gradient_j / γ_ij
                        grads[j] += λ[j_frag] * a_ij_forward * slater_overlap_gradient_j / γ_ij
                        grads[i] -= λ[j_frag] * a_ij_forward * slater_overlap_gradient_j / γ_ij

                        ### backward CT ###
                        grads[i] -= λ[i_frag] * a_ij_backward * slater_overlap_gradient_j / γ_ij
                        grads[j] += λ[i_frag] * a_ij_backward * slater_overlap_gradient_j / γ_ij
                        grads[j] -= λ[j_frag] * a_ij_backward * slater_overlap_gradient_j / γ_ij
                        grads[i] += λ[j_frag] * a_ij_backward * slater_overlap_gradient_j / γ_ij
                    end
                end
            end
        end
    end
end