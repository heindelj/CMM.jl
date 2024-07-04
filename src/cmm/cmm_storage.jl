struct CMM_DampingValues
    λ_elec_1c::DampingValues # core-shell
    λ_elec_2c::DampingValues # shell-shell
    λ_pauli_2c::DampingValues
    λ_disp_2c::DampingValues
    λ_pol_2c::DampingValues
    λ_exch_pol_2c::DampingValues
    λ_ct_d_to_a_2c::DampingValues
    λ_ct_a_to_d_2c::DampingValues
end

struct DampingExponents
    # Atom-specific exponents #
    b_i_elec::Vector{Float64}
    b_i_pauli::Vector{Float64}
    b_i_disp::Vector{Float64}
    b_i_pol::Vector{Float64}
    b_i_ct::Vector{Float64}
    # Pair-specific exponents #
    b_ij_elec::Vector{Float64}
    b_ij_pauli::Vector{Float64}
    b_ij_disp::Vector{Float64}
    b_ij_pol::Vector{Float64}
    b_ij_ct::Vector{Float64}
end
DampingExponents(natoms::Int) = DampingExponents(
    zeros(natoms), zeros(natoms), zeros(natoms),
    zeros(natoms), zeros(natoms),
    zeros(natoms * (natoms-1)÷2), zeros(natoms * (natoms-1)÷2), zeros(natoms * (natoms-1)÷2),
    zeros(natoms * (natoms-1)÷2), zeros(natoms * (natoms-1)÷2)
)

# Return to this later...
struct CMMStorage
    num_atoms::Int
    # Indices for each atom to loop over intermolecular
    # and intramolecular interactions
    inter_indices::Vector{Vector{Int}}
    intra_indices::Vector{Vector{Int}}
    # Atom types defined by integer index
    atom_types::Vector{Int}
end

function fill_atom_types!(labels::Vector{String}, atom_types::Vector{Int})
    @assert length(atom_types) == length(labels) "Size of atom type array and labels array do not match! Exiting."
    for i in eachindex(labels)
        if labels[i] == "O"
            atom_types[i] = 1
        elseif labels[i] == "H"
            atom_types[i] = 2
        elseif labels[i] == "F"
            atom_types[i] = 3
        elseif labels[i] == "Cl"
            atom_types[i] = 4
        elseif labels[i] == "Br"
            atom_types[i] = 5
        elseif labels[i] == "I"
            atom_types[i] = 6
        elseif labels[i] == "Li"
            atom_types[i] = 7
        elseif labels[i] == "Na"
            atom_types[i] = 8
        elseif labels[i] == "K"
            atom_types[i] = 9
        elseif labels[i] == "Rb"
            atom_types[i] = 10
        elseif labels[i] == "Cs"
            atom_types[i] = 11
        elseif labels[i] == "Mg"
            atom_types[i] = 12
        elseif labels[i] == "Ca"
            atom_types[i] = 13
        end
    end
end

function fill_distances!(coords::Vector{SVector{3, Float64}}, dists::Distances)
    @assert length(dists.r) == length(coords) * (length(coords)-1) / 2 "Dists object has not been properly pre-allocated before trying to fill it!"
    
    pair_index = 1
    for i in 1:length(coords)-1
        for j in (i+1):length(coords)
            dists.r_vec[pair_index] = coords[j] - coords[i]
            dists.r[pair_index] = norm(dists.r_vec[pair_index])
            dists.ij_pairs[pair_index] = (i, j)
            pair_index += 1
        end
    end
end

function fill_damping_exponents!(labels::Vector{String}, params::Dict{Symbol, Float64}, damping_exponents::DampingExponents)
    # NOTE(JOE): Currently this does 10 allocations PER LOOP to fill in the parameters array.
    # IT does this when creating the symbol to look up the parameter.
    # I am not too worried about this since for a fixed system, these values are the same
    # no matter what so this operation is only done once per cluster.
    # Obviously, it would be great to eliminate these allocations but I'll leave it for the future.
    # 5/26/24

    for i in eachindex(labels)
        damping_exponents.b_i_elec[i] = params[Symbol(labels[i], :_b_elec)]
        damping_exponents.b_i_pauli[i] = params[Symbol(labels[i], :_b_repulsion)]
        damping_exponents.b_i_disp[i] = params[Symbol(labels[i], :_b_disp)]
        damping_exponents.b_i_pol[i] = params[Symbol(labels[i], :_b_pol)]
        damping_exponents.b_i_ct[i] = params[Symbol(labels[i], :_b_ct)]
    end

    pair_index = 1
    for i in 1:length(labels)-1
        for j in (i+1):length(labels)
            # Electrostatics #
            #b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_elec, params)
            #if !has_pairwise
                damping_exponents.b_ij_elec[pair_index] = sqrt(damping_exponents.b_i_elec[i] * damping_exponents.b_i_elec[j])
            #else
            #    damping_exponents.b_ij_elec[pair_index] = b_ij
            #end

            # Pauli #
            #b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_repulsion, params)
            #if !has_pairwise
                damping_exponents.b_ij_pauli[pair_index] = sqrt(damping_exponents.b_i_pauli[i] * damping_exponents.b_i_pauli[j])
            #else
            #    damping_exponents.b_ij_pauli[pair_index] = b_ij
            #end

            # Dispersion #
            #b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_disp, params)
            #if !has_pairwise
                damping_exponents.b_ij_disp[pair_index] = sqrt(damping_exponents.b_i_disp[i] * damping_exponents.b_i_disp[j])
            #else
            #    damping_exponents.b_ij_disp[pair_index] = b_ij
            #end

            # Exchange Polarization #
            #b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_pol, params)
            #if !has_pairwise
                damping_exponents.b_ij_pol[pair_index] = sqrt(damping_exponents.b_i_pol[i] * damping_exponents.b_i_pol[j])
            #else
            #    damping_exponents.b_ij_pol[pair_index] = b_ij
            #end

            # Charge Transfer #
            #b_ij, has_pairwise = maybe_get_pairwise_parameter(labels[i], labels[j], :_b_ct, params)
            #if !has_pairwise
                damping_exponents.b_ij_ct[pair_index] = sqrt(damping_exponents.b_i_ct[i] * damping_exponents.b_i_ct[j])
            #else
            #    damping_exponents.b_ij_ct[pair_index] = b_ij
            #end

            pair_index += 1
        end
    end
end

