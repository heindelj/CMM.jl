
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

                    b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_exch_pol, params)
                    b_ij = abs(b_ij)
                    if !has_pairwise
                        b_ij = sqrt(b_i * b_j)
                    end

                    # Very possibly this dipole should always point in the field direction?
                    # Even for water. That direction should probably be the polarization
                    # direction?
                    #μ_exch_pol_j = K_j_μ * M_j.μ
                    #if is_ion(labels[j])
                    #    μ_exch_pol_j = K_j_μ * normalize(E_field[j])
                    #end

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