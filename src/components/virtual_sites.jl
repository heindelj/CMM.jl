function get_coords_and_electrostatic_types_including_virtual_sites!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    elec_types::AbstractVector{ElecType},
    γ_vs::Float64
)
    # NOTE: The number of virtual sites needs to be determined before this and
    # an appropriately sized array is passed in. This has to be done based on
    # the atom label information when building the force field storage object.

    coords_with_vs = [@MVector zeros(3) for _ in eachindex(elec_types)]
    labels_with_vs = String[]
    fragment_indices_with_vs = Vector{Int}[]
    num_vs_seen = 0
    for i_frag in eachindex(fragment_indices)
        frag_indices_i = Int[]
        for i in fragment_indices[i_frag]
            coords_with_vs[i+num_vs_seen] = coords[i]
            push!(labels_with_vs, labels[i])
            if labels[i] == "O"
                # Oxygen site #
                push!(frag_indices_i, i + num_vs_seen)
                elec_types[i+num_vs_seen] = CORESHELL
                
                # Oxygen virtual site #
                num_vs_seen += 1
                push!(frag_indices_i, i + num_vs_seen)
                elec_types[i+num_vs_seen] = CORESHELL
                
                r_OH1 = coords[i+1] - coords[i]
                r_OH2 = coords[i+2] - coords[i]
                coords_vs = coords[i] + 0.5 * γ_vs * (r_OH1 + r_OH2)
                coords_with_vs[i+num_vs_seen] = coords_vs
                push!(labels_with_vs, "X")
            else
                # everything else has type CoreShell
                push!(frag_indices_i, i + num_vs_seen)
                elec_types[i+num_vs_seen] = CORESHELL
            end
        end
        push!(fragment_indices_with_vs, frag_indices_i)
    end
    return coords_with_vs, labels_with_vs, fragment_indices_with_vs
end

function get_coords_including_virtual_sites!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    γ_vs::Float64
)
    # NOTE: The number of virtual sites needs to be determined before this and
    # an appropriately sized array is passed in. This has to be done based on
    # the atom label information when building the force field storage object.

    num_sites = length(labels) + count(==("O"), labels)
    coords_with_vs = [@MVector zeros(3) for _ in 1:num_sites]
    labels_with_vs = String[]
    fragment_indices_with_vs = Vector{Int}[]
    num_vs_seen = 0
    for i_frag in eachindex(fragment_indices)
        frag_indices_i = Int[]
        for i in fragment_indices[i_frag]
            coords_with_vs[i+num_vs_seen] = coords[i]
            push!(labels_with_vs, labels[i])
            if labels[i] == "O"
                # Oxygen core site #
                push!(frag_indices_i, i + num_vs_seen)
                
                # Oxygen shell site #
                num_vs_seen += 1
                push!(frag_indices_i, i + num_vs_seen)
                
                r_OH1 = coords[i+1] - coords[i]
                r_OH2 = coords[i+2] - coords[i]
                coords_vs = coords[i] + 0.5 * γ_vs * (r_OH1 + r_OH2)
                coords_with_vs[i+num_vs_seen] = coords_vs
                push!(labels_with_vs, "X")
            else
                # everything else has type CoreShell
                push!(frag_indices_i, i + num_vs_seen)
            end
        end
        push!(fragment_indices_with_vs, frag_indices_i)
    end
    return coords_with_vs, labels_with_vs, fragment_indices_with_vs
end