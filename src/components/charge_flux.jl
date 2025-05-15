
function get_charges_with_charge_flux!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    charges::AbstractVector{Float64},
    params::Dict{Symbol, Float64},
    charge_grads::Union{Nothing, Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}}=nothing
)
    # NOTE: This is not a generic implementation. Instead,
    # it works for water and ions and assumes that waters
    # are given in OHH order. Since there is no charge
    # flux for ions, we simply skip anything without three
    # atoms.
    for i_frag in eachindex(fragment_indices)
        if length(fragment_indices[i_frag]) == 3
            if labels[fragment_indices[i_frag]] != ["O", "H", "H"]
                display(labels[fragment_indices[i_frag]])
                display(coords[fragment_indices[i_frag]])
                display(labels)
                display(coords * 0.529177)
                @assert false "Water atoms are in the wrong order! Must be OHH."
            end
            i_O  = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]
            
            j_OH  = params[:j_OH]
            j_OH_bb  = params[:j_OH_bb]
            j_HOH = params[:j_HOH]
            r_OH_eq = params[:re_water]
            θ_HOH_eq = acos(params[:cos_angle_eq_water])

            r_OH1 = coords[i_H1] - coords[i_O]
            r_OH2 = coords[i_H2] - coords[i_O]
            r_OH1_length = norm(r_OH1)
            r_OH2_length = norm(r_OH2)
            θ = acos(r_OH1 ⋅ r_OH2 / (r_OH1_length * r_OH2_length))

            dq_H1 = (
                j_HOH * (θ - θ_HOH_eq) +
                j_OH * (r_OH1_length - r_OH_eq) +
                j_OH_bb * (r_OH2_length - r_OH_eq)
            )
            dq_H2 = (
                j_HOH * (θ - θ_HOH_eq) +
                j_OH * (r_OH2_length - r_OH_eq) +
                j_OH_bb * (r_OH1_length - r_OH_eq)
            )
            dq_O  = -(dq_H1 + dq_H2) 

            charges[i_O]  = params[:q_O] + dq_O
            charges[i_H1] = -0.5 * params[:q_O] + dq_H1
            charges[i_H2] = -0.5 * params[:q_O] + dq_H2

            if charge_grads !== nothing
                idx_O  = 1
                idx_H1 = 2
                idx_H2 = 3
                grad_θ_H1_H1 = normalize(cross(r_OH1, cross(r_OH1, r_OH2))) / norm(r_OH1)
                grad_θ_H1_H2 = normalize(cross(r_OH2, cross(r_OH1, r_OH2))) / norm(r_OH2)
                grad_θ_H2_H2 = normalize(cross(r_OH2, cross(r_OH2, r_OH1))) / norm(r_OH2)
                grad_θ_H2_H1 = normalize(cross(r_OH1, cross(r_OH2, r_OH1))) / norm(r_OH1)
                
                @views charge_grads[i_frag][:, idx_H1, idx_H1] = j_OH * r_OH1 / r_OH1_length + j_HOH * grad_θ_H1_H1
                @views charge_grads[i_frag][:, idx_H1, idx_H2] = j_OH_bb * r_OH2 /  r_OH2_length - j_HOH * grad_θ_H1_H2
                @views charge_grads[i_frag][:, idx_H1, idx_O]  = -(charge_grads[i_frag][:, idx_H1, idx_H2] + charge_grads[i_frag][:, idx_H1, idx_H1])

                @views charge_grads[i_frag][:, idx_H2, idx_H1] =  j_OH_bb * r_OH1 /  r_OH1_length - j_HOH * grad_θ_H2_H1
                @views charge_grads[i_frag][:, idx_H2, idx_H2] =  j_OH * r_OH2 / r_OH2_length + j_HOH * grad_θ_H2_H2
                @views charge_grads[i_frag][:, idx_H2, idx_O]  = -(charge_grads[i_frag][:, idx_H2, idx_H2] + charge_grads[i_frag][:, idx_H2, idx_H1])

                @views charge_grads[i_frag][:, idx_O, idx_H1] = -(charge_grads[i_frag][:, idx_H1, idx_H1] + charge_grads[i_frag][:, idx_H2, idx_H1])
                @views charge_grads[i_frag][:, idx_O, idx_H2] = -(charge_grads[i_frag][:, idx_H1, idx_H2] + charge_grads[i_frag][:, idx_H2, idx_H2])
                @views charge_grads[i_frag][:, idx_O, idx_O]  = -(charge_grads[i_frag][:, idx_H1, idx_O] + charge_grads[i_frag][:, idx_H2, idx_O])
            end
        elseif length(fragment_indices[i_frag]) != 1
            n = length(fragment_indices[i_frag])
            @warn "Got fragment with $n atoms. This is not water or an ion. Either the implementation needs to be updated or the provided input is wrong."
        end
    end
end

function update_repulsion_charges_with_charge_flux!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    charges::AbstractVector{Float64},
    params::Dict{Symbol, Float64},
    charge_grads::Union{Nothing, Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}}=nothing
)
    # NOTE: This is not a generic implementation. Instead,
    # it works for water and ions and assumes that waters
    # are given in OHH order. Since there is no charge
    # flux for ions, we simply skip anything without three
    # atoms.
    for i_frag in eachindex(fragment_indices)
        if length(fragment_indices[i_frag]) == 3
            if labels[fragment_indices[i_frag]] != ["O", "H", "H"]
                display(labels[fragment_indices[i_frag]])
                display(coords[fragment_indices[i_frag]])
                @assert false "Water atoms are in the wrong order! Must be OHH."
            end
            i_O  = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]
            
            j_OH_pauli  = params[:j_OH_pauli]
            r_OH_eq = params[:re_water]

            r_OH1 = coords[i_H1] - coords[i_O]
            r_OH2 = coords[i_H2] - coords[i_O]
            r_OH1_length = norm(r_OH1)
            r_OH2_length = norm(r_OH2)

            dq_H1 = (
                j_OH_pauli * (r_OH1_length - r_OH_eq)
            )
            dq_H2 = (
                j_OH_pauli * (r_OH2_length - r_OH_eq)
            )
            dq_O  = -(dq_H1 + dq_H2)

            charges[i_O]  += dq_O
            charges[i_H1] += dq_H1
            charges[i_H2] += dq_H2

            if charge_grads !== nothing
                idx_O  = 1
                idx_H1 = 2
                idx_H2 = 3
                grad_θ_H1_H1 = normalize(cross(r_OH1, cross(r_OH1, r_OH2))) / norm(r_OH1)
                grad_θ_H1_H2 = normalize(cross(r_OH2, cross(r_OH1, r_OH2))) / norm(r_OH2)
                grad_θ_H2_H2 = normalize(cross(r_OH2, cross(r_OH2, r_OH1))) / norm(r_OH2)
                grad_θ_H2_H1 = normalize(cross(r_OH1, cross(r_OH2, r_OH1))) / norm(r_OH1)
                
                @views charge_grads[i_frag][:, idx_H1, idx_H1] = j_OH * r_OH1 / r_OH1_length + j_HOH * grad_θ_H1_H1
                @views charge_grads[i_frag][:, idx_H1, idx_H2] = j_OH_bb * r_OH2 /  r_OH2_length - j_HOH * grad_θ_H1_H2
                @views charge_grads[i_frag][:, idx_H1, idx_O]  = -(charge_grads[i_frag][:, idx_H1, idx_H2] + charge_grads[i_frag][:, idx_H1, idx_H1])

                @views charge_grads[i_frag][:, idx_H2, idx_H1] =  j_OH_bb * r_OH1 /  r_OH1_length - j_HOH * grad_θ_H2_H1
                @views charge_grads[i_frag][:, idx_H2, idx_H2] =  j_OH * r_OH2 / r_OH2_length + j_HOH * grad_θ_H2_H2
                @views charge_grads[i_frag][:, idx_H2, idx_O]  = -(charge_grads[i_frag][:, idx_H2, idx_H2] + charge_grads[i_frag][:, idx_H2, idx_H1])

                @views charge_grads[i_frag][:, idx_O, idx_H1] = -(charge_grads[i_frag][:, idx_H1, idx_H1] + charge_grads[i_frag][:, idx_H2, idx_H1])
                @views charge_grads[i_frag][:, idx_O, idx_H2] = -(charge_grads[i_frag][:, idx_H1, idx_H2] + charge_grads[i_frag][:, idx_H2, idx_H2])
                @views charge_grads[i_frag][:, idx_O, idx_O]  = -(charge_grads[i_frag][:, idx_H1, idx_O] + charge_grads[i_frag][:, idx_H2, idx_O])
            end

        elseif length(fragment_indices[i_frag]) != 1
            n = length(fragment_indices[i_frag])
            @warn "Got fragment with $n atoms. This is not water or an ion. Either the implementation needs to be updated or the provided input is wrong."
        end
    end
end

function get_geometry_dependent_atomic_hardness!(
    coords::AbstractVector{MVector{3, Float64}},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    η_fq::AbstractVector{Float64},
    params::Dict{Symbol, Float64}
)
    for i_frag in eachindex(fragment_indices)
        if length(fragment_indices[i_frag]) == 3
            @assert labels[fragment_indices[i_frag]] == ["O", "H", "H"] "Water atoms are in the wrong order! Must be OHH."
            i_O  = fragment_indices[i_frag][1]
            i_H1 = fragment_indices[i_frag][2]
            i_H2 = fragment_indices[i_frag][3]
            
            η_fq[i_O]  = abs(params[Symbol(labels[i_O], :_η_fq)])
            η_fq[i_H1] = abs(params[Symbol(labels[i_H1], :_η_fq)])
            η_fq[i_H2] = abs(params[Symbol(labels[i_H2], :_η_fq)])

            k_OH_η  = abs(params[:k_OH_η])
            k_OH_bb_η  = abs(params[:k_OH_bb_η])
            k_OH_θ_η = params[:k_OH_θ_η]
            r_OH_eq = params[:re_water]
            θ_HOH_eq = acos(params[:cos_angle_eq_water])

            r_OH1 = coords[i_H1] - coords[i_O]
            r_OH2 = coords[i_H2] - coords[i_O]
            r_OH1_length = norm(r_OH1)
            r_OH2_length = norm(r_OH2)
            θ = acos(r_OH1 ⋅ r_OH2 / (r_OH1_length * r_OH2_length))
            η_fq[i_H1] *= ((r_OH_eq / r_OH1_length)^k_OH_η) * ((r_OH_eq / r_OH2_length)^k_OH_bb_η)
            η_fq[i_H2] *= ((r_OH_eq / r_OH2_length)^k_OH_η) * ((r_OH_eq / r_OH1_length)^k_OH_bb_η)
            η_fq[i_H1] += k_OH_θ_η * (θ - θ_HOH_eq)
            η_fq[i_H2] += k_OH_θ_η * (θ - θ_HOH_eq)

        elseif length(fragment_indices[i_frag]) != 1
            n = length(fragment_indices[i_frag])
            @warn "Got fragment with $n atoms. This is not water or an ion. Either the implementation needs to be updated or the provided input is wrong."
        else
            # If we are here, this is an ion so there is only one index for the fragment.
            i_X = fragment_indices[i_frag][1]
            η_fq[i_X] = abs(params[Symbol(labels[i_X], :_η_fq)])
        end
    end
end